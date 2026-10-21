"""Regression tests for review finding F-15 (dev-loop 20261020-dlcobra-ce5a).

``position_key_from_sfen`` does not raise on an SFEN it cannot parse: it
returns the all-zero key. ``import_terashock`` only skipped entries when the
call raised, so every such entry was filed under key (0, 0), the last one
winning and the rest counted as duplicates, with nothing in the log.
"""

from __future__ import annotations

import dataclasses
import io
import logging

import cshogi
import pytest

from dlshogi.book import terashock as T
from dlshogi.book.book_db import BookDBPrinter, TerashockEntry, TerashockMove
from dlshogi.book.node_store import NodeStore
from tests.book.test_interop import _INITIAL_SFEN, _drop_book_tables, _make_config

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

# Parses as a line, and the binding maps it to key (0, 0). Entries carry no
# move lines: a bad SFEN *with* move lines aborts the parser itself (F-21).
_BAD_SFENS = ["9/9/9 b - 1", "9/9/9/9 b - 1"]


def _book(path, sfens, with_moves_for=(_INITIAL_SFEN,)) -> str:
    legal = [int(cshogi.move16(m)) for m in cshogi.Board().legal_moves][:2]
    buf = io.StringIO()
    printer = BookDBPrinter()
    printer.write_header(buf, len(sfens))
    for sfen in sfens:
        moves = [TerashockMove(m, None, 100, 20, 5) for m in legal] if sfen in with_moves_for else []
        printer.write_entry(buf, TerashockEntry(sfen=sfen, moves=moves))
    path.write_text(buf.getvalue())
    return str(path)


async def _import(pg_scratch_database, pg_conn, tmp_path, sfens):
    await _drop_book_tables(pg_conn)
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN),
        terashock_book=_book(tmp_path / "book.db", sfens),
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    return cfg, store


def test_the_fixtures_really_map_to_the_zero_key():
    for sfen in _BAD_SFENS:
        assert tuple(T.position_key_from_sfen(sfen)) == (0, 0)
    assert tuple(T.position_key_from_sfen(_INITIAL_SFEN)) != (0, 0)


async def test_unindexable_entries_are_skipped_counted_and_logged(
    pg_scratch_database, pg_conn, tmp_path, caplog
):
    cfg, store = await _import(pg_scratch_database, pg_conn, tmp_path, [_INITIAL_SFEN, *_BAD_SFENS])
    try:
        with caplog.at_level(logging.WARNING, logger="dlshogi.book.terashock"):
            stats = await T.import_terashock(store.pool, cfg)
        async with store.pool.acquire() as conn:
            rows = await conn.fetch("SELECT key_hi, key_lo, sfen FROM terashock_entry")

        assert [r["sfen"] for r in rows] == [_INITIAL_SFEN]
        assert not any(r["key_hi"] == 0 and r["key_lo"] == 0 for r in rows)
        assert stats.entry_count == 1
        assert stats.unindexable_count == len(_BAD_SFENS)
        assert stats.duplicate_count == 0  # they used to collide with each other at (0, 0)
        warning = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("not indexed" in m and "2 of 3" in m for m in warning), warning
    finally:
        await store.close()


async def test_a_clean_book_reports_nothing_unindexable(
    pg_scratch_database, pg_conn, tmp_path, caplog
):
    cfg, store = await _import(pg_scratch_database, pg_conn, tmp_path, [_INITIAL_SFEN])
    try:
        with caplog.at_level(logging.WARNING, logger="dlshogi.book.terashock"):
            stats = await T.import_terashock(store.pool, cfg)
        assert stats.unindexable_count == 0
        assert not [r for r in caplog.records if "not indexed" in r.getMessage()]
    finally:
        await store.close()


async def test_a_book_with_only_unindexable_entries_still_exits(
    pg_scratch_database, pg_conn, tmp_path
):
    """R7.8: zero indexable entries is the existing report-and-exit case."""
    cfg, store = await _import(pg_scratch_database, pg_conn, tmp_path, _BAD_SFENS)
    try:
        with pytest.raises(SystemExit):
            await T.import_terashock(store.pool, cfg)
    finally:
        await store.close()
