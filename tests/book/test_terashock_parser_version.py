"""Regression tests for review finding F-13 (dev-loop 20261020-dlcobra-ce5a).

design.md says the recorded Terashock source identity is (path, size, mtime,
entry count, parser version) and that a mismatch triggers a reimport. The
parser version was carried in a dataclass but never stored or compared, so
changing the parser (or the packed move layout) left a stale terashock_entry
index in use against an unchanged book file.
"""

from __future__ import annotations

import dataclasses
import io

import cshogi
import pytest

from dlshogi.book import terashock as T
from dlshogi.book.book_db import BookDBPrinter, TerashockEntry, TerashockMove
from dlshogi.book.node_store import NodeStore
from tests.book.test_interop import _INITIAL_SFEN, _drop_book_tables, _make_config

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _write_book(path) -> str:
    legal = [int(cshogi.move16(m)) for m in cshogi.Board().legal_moves][:3]
    buf = io.StringIO()
    printer = BookDBPrinter()
    printer.write_header(buf, 1)
    printer.write_entry(
        buf,
        TerashockEntry(sfen=_INITIAL_SFEN, moves=[TerashockMove(m, None, 100, 20, 5) for m in legal]),
    )
    path.write_text(buf.getvalue())
    return str(path)


async def _imported(pg_scratch_database, pg_conn, tmp_path):
    await _drop_book_tables(pg_conn)
    book = _write_book(tmp_path / "book.db")
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), terashock_book=book
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    await T.import_terashock(store.pool, cfg)
    return cfg, store


async def test_import_records_the_parser_version(pg_scratch_database, pg_conn, tmp_path):
    cfg, store = await _imported(pg_scratch_database, pg_conn, tmp_path)
    try:
        async with store.pool.acquire() as conn:
            recorded = await conn.fetchval("SELECT terashock_parser_version FROM book_meta")
        assert recorded == T.PARSER_VERSION
    finally:
        await store.close()


async def test_a_parser_bump_triggers_a_reimport_for_an_unchanged_file(
    pg_scratch_database, pg_conn, tmp_path, monkeypatch
):
    cfg, store = await _imported(pg_scratch_database, pg_conn, tmp_path)
    try:
        assert await T.check_terashock_source(store.pool, cfg) is False  # unchanged file, unchanged parser
        monkeypatch.setattr(T, "PARSER_VERSION", T.PARSER_VERSION + 1)
        # _get_file_identity builds its identity from the module constant at call time.
        assert await T.check_terashock_source(store.pool, cfg) is True

        await T.import_terashock(store.pool, cfg)  # re-imports and re-records
        assert await T.check_terashock_source(store.pool, cfg) is False
    finally:
        await store.close()


async def test_a_database_imported_before_the_column_existed_is_treated_as_v1(
    pg_scratch_database, pg_conn, tmp_path
):
    """NULL means parser v1. The current parser is v2 (it adds flipped entries), so a v1 index reimports."""
    cfg, store = await _imported(pg_scratch_database, pg_conn, tmp_path)
    try:
        async with store.pool.acquire() as conn:
            await conn.execute("UPDATE book_meta SET terashock_parser_version = NULL")
        assert T.PARSER_VERSION == 2 and T._LEGACY_PARSER_VERSION == 1
        assert await T.check_terashock_source(store.pool, cfg) is True
    finally:
        await store.close()


async def test_repair_adds_the_column_to_an_older_database(pg_scratch_database, pg_conn, tmp_path):
    """A database created before this change is upgraded in place, rows untouched."""
    cfg, store = await _imported(pg_scratch_database, pg_conn, tmp_path)
    try:
        async with store.pool.acquire() as conn:
            await conn.execute("ALTER TABLE book_meta DROP COLUMN terashock_parser_version")
            before = await conn.fetchrow("SELECT terashock_source_path, terashock_entry_count FROM book_meta")
        await store.ensure_schema()
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT terashock_source_path, terashock_entry_count, terashock_parser_version FROM book_meta"
            )
        assert (row["terashock_source_path"], row["terashock_entry_count"]) == tuple(before.values())
        assert row["terashock_parser_version"] is None  # legacy: read as v1
        assert await T.check_terashock_source(store.pool, cfg) is True  # v1 index is older than parser v2
    finally:
        await store.close()


async def test_file_change_still_triggers_a_reimport(pg_scratch_database, pg_conn, tmp_path):
    """The existing size/mtime/path checks are unaffected."""
    cfg, store = await _imported(pg_scratch_database, pg_conn, tmp_path)
    try:
        with open(cfg.terashock_book, "a") as f:
            f.write("# trailing comment\n")
        assert await T.check_terashock_source(store.pool, cfg) is True
    finally:
        await store.close()
