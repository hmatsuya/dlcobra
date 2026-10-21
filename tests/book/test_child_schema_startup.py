"""Regression tests for review finding F-12 (dev-loop 20261020-dlcobra-ce5a).

A search child process is started after the parent has created and repaired
the schema. ``_connect_and_ensure_schema(read_only=True)`` documented that the
child skips creation and repair, but the flag was never read: every child ran
the full repair DDL, whose ``ALTER TABLE ... SET STORAGE`` takes an ACCESS
EXCLUSIVE lock on ``book_node`` and so stalled every running process's reads
each time a child started.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from dlshogi.book import node_store as ns
from dlshogi.book.node_store import (
    NodeStore,
    SchemaAbsentError,
    SchemaVersionMismatchError,
)
from tests.book.test_interop import _INITIAL_SFEN, _drop_book_tables, _make_config

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _parent(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), worker_count=4
    )
    return cfg, await NodeStore.connect_and_prepare(cfg, role="search")


async def test_read_only_startup_sends_no_ddl(pg_scratch_database, pg_conn):
    cfg, parent = await _parent(pg_scratch_database, pg_conn)
    child = NodeStore(cfg, role="search")
    await child.connect()
    try:
        ddl = []
        real_repair = child._repair_schema

        async def spy_repair(conn):
            ddl.append("repair")
            return await real_repair(conn)

        child._repair_schema = spy_repair
        await child.ensure_schema(read_only=True)
        assert ddl == [], "a read-only startup must not run the repair DDL"
        assert child._root_sfen is not None  # still loaded the recorded root
    finally:
        await child.close()
        await parent.close()


async def test_full_startup_still_repairs(pg_scratch_database, pg_conn):
    """Control: the non-read-only path (the parent) keeps repairing."""
    cfg, parent = await _parent(pg_scratch_database, pg_conn)
    other = NodeStore(cfg, role="search")
    await other.connect()
    try:
        ddl = []
        real_repair = other._repair_schema

        async def spy_repair(conn):
            ddl.append("repair")
            return await real_repair(conn)

        other._repair_schema = spy_repair
        await other.ensure_schema()
        assert ddl == ["repair"]
    finally:
        await other.close()
        await parent.close()


async def test_read_only_startup_does_not_block_running_readers(
    pg_scratch_database, pg_conn, monkeypatch
):
    """The symptom: while a child starts, a running process's reads must not stall.

    The repair transaction is stretched with a trailing pg_sleep so that, were
    it to run, the ACCESS EXCLUSIVE lock from ALTER TABLE would still be held
    while the other process reads.
    """
    cfg, parent = await _parent(pg_scratch_database, pg_conn)
    child = NodeStore(cfg, role="search")
    await child.connect()
    try:
        monkeypatch.setattr(ns, "_REPAIR_DDL", (*ns._REPAIR_DDL, "SELECT pg_sleep(2)"))
        startup = asyncio.ensure_future(child.ensure_schema(read_only=True))
        await asyncio.sleep(0.5)  # the child is now past its checks

        async with parent.pool.acquire() as conn:
            count = await asyncio.wait_for(conn.fetchval("SELECT count(*) FROM book_node"), 1)
        assert count >= 0
        await startup
    finally:
        await child.close()
        await parent.close()


async def test_read_only_startup_still_checks_the_schema_version(pg_scratch_database, pg_conn):
    """Skipping repair must not skip the version/fingerprint check (R2.8)."""
    cfg, parent = await _parent(pg_scratch_database, pg_conn)
    child = NodeStore(cfg, role="search")
    await child.connect()
    try:
        async with parent.pool.acquire() as conn:
            await conn.execute("UPDATE book_meta SET schema_version = 'not-this-one'")
        with pytest.raises(SchemaVersionMismatchError):
            await child.ensure_schema(read_only=True)
    finally:
        await child.close()
        await parent.close()


async def test_read_only_startup_refuses_to_create_a_schema(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)
    child = NodeStore(cfg, role="search")
    await child.connect()
    try:
        with pytest.raises(SchemaAbsentError):
            await child.ensure_schema(read_only=True)
        async with child.pool.acquire() as conn:
            tables = await conn.fetchval(
                "SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename='book_node'"
            )
        assert tables == 0, "a child must never create the schema"
    finally:
        await child.close()


async def test_connect_helper_passes_the_flag_and_exits_cleanly_on_absent_schema(
    pg_scratch_database, pg_conn
):
    from dlshogi.book.__main__ import _connect_and_ensure_schema

    await _drop_book_tables(pg_conn)
    cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)
    with pytest.raises(SystemExit) as exc:
        await _connect_and_ensure_schema(cfg, role="search", read_only=True)
    assert exc.value.code == 1
