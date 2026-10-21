"""Regression tests for review finding F-20 (dev-loop 20261020-dlcobra-ce5a).

``NodeStore.get()`` must show every uncommitted visit (Requirement 4.5):

- a ``backup()`` merged while the SELECT is in flight used to be missed
  (pending was read before the await);
- while ``flush()``'s transaction was open the delta was in neither the
  pending map nor the database, and a read in that window cached the
  pre-commit row, which then stayed stale after the commit;
- a flush that started and ended during a SELECT left the read with a
  pre-commit row and no delta anywhere.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from dlshogi.book.keys import position_key_from_sfen
from dlshogi.book.node_store import BackupDelta, NodeStore
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _drop_book_tables,
    _make_config,
    _run_small_search,
)

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

_ROOT = position_key_from_sfen(_INITIAL_SFEN)
_N = 5  # visits in every test delta


async def _store(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    # Worker_Count sizes the pool; a reader needs a second connection while a
    # flush transaction holds the first.
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), worker_count=4
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    await _run_small_search(store, cfg, max_descents=30)
    _, view = await store.get(_ROOT)
    store._cache.invalidate(_ROOT)
    return store, int(view.edges[0]["move16"]), view.visit_count


def _delta(move16: int) -> BackupDelta:
    return BackupDelta(
        key=_ROOT,
        node_visit_delta=_N,
        node_value_delta=2.0,
        edge_deltas={move16: (_N, 2.0)},
    )


async def _visits(store) -> int:
    _, view = await store.get(_ROOT)
    return view.visit_count


async def test_backup_merged_during_the_select_is_visible(pg_scratch_database, pg_conn):
    store, m16, base = await _store(pg_scratch_database, pg_conn)
    try:
        real_run = store._run

        async def run(fn):
            result = await real_run(fn)
            store.backup([_delta(m16)])  # lands after the SELECT, before get() resumes
            return result

        store._run = run
        assert await _visits(store) == base + _N
    finally:
        await store.close()


async def test_read_while_the_flush_transaction_is_open(pg_scratch_database, pg_conn):
    """Visible once during the flush, not cached stale, exactly once after the commit."""
    store, m16, base = await _store(pg_scratch_database, pg_conn)
    try:
        entered, gate = asyncio.Event(), asyncio.Event()
        real_apply = store._apply_backup_row_fallback

        async def slow_apply(conn, delta):
            await real_apply(conn, delta)  # UPDATE issued, transaction still open
            entered.set()
            await gate.wait()

        store._apply_backup_row_fallback = slow_apply
        store.backup([_delta(m16)])
        flush = asyncio.ensure_future(store.flush())
        await entered.wait()

        assert await _visits(store) == base + _N  # delta lives only in the flush snapshot
        gate.set()
        await flush
        assert await _visits(store) == base + _N  # committed once; no stale cache, no double count
    finally:
        await store.close()


async def test_flush_that_starts_and_ends_during_a_select(pg_scratch_database, pg_conn):
    store, m16, base = await _store(pg_scratch_database, pg_conn)
    try:
        real_run = store._run
        state = {"flushed": False}

        async def run(fn):
            result = await real_run(fn)  # the SELECT returns the pre-commit row
            if not state["flushed"]:
                state["flushed"] = True
                store.backup([_delta(m16)])
                await store.flush()  # commits and clears the snapshot before get() resumes
            return result

        store._run = run
        assert await _visits(store) == base + _N
    finally:
        await store.close()


async def test_failed_flush_keeps_the_delta_visible_once(pg_scratch_database, pg_conn):
    store, m16, base = await _store(pg_scratch_database, pg_conn)
    try:
        async def boom(conn, delta):
            raise RuntimeError("disk full")

        real_apply = store._apply_backup_row_fallback
        store._apply_backup_row_fallback = boom
        store.backup([_delta(m16)])
        with pytest.raises(RuntimeError):
            await store.flush()

        assert await _visits(store) == base + _N  # re-merged into pending, counted once
        store._apply_backup_row_fallback = real_apply
        await store.flush()
        store._cache.invalidate(_ROOT)
        assert await _visits(store) == base + _N
    finally:
        await store.close()


async def test_read_that_starts_during_the_commit_waits_for_it(pg_scratch_database, pg_conn):
    """A SELECT issued while the flush COMMIT is in flight can see either state.

    The delta stays in the snapshot until the commit is acknowledged, so a
    read that races the commit must not be trusted: get() holds it until the
    commit finishes, then reads the committed row exactly once.
    """
    store, m16, base = await _store(pg_scratch_database, pg_conn)
    try:
        selects_during_commit = []
        real_run = store._run

        async def watched_run(fn):
            if store._commits_in_flight:
                selects_during_commit.append(1)
            return await real_run(fn)

        store._run = watched_run
        store.backup([_delta(m16)])
        flush = asyncio.ensure_future(store.flush())

        # flush() raises _commits_in_flight in the step before COMMIT is sent;
        # start the read the moment it does.
        reader = None
        for _ in range(100_000):
            if store._commits_in_flight:
                reader = asyncio.ensure_future(_visits(store))
                break
            await asyncio.sleep(0)
        assert reader is not None, "never observed the commit window"

        await flush
        assert await asyncio.wait_for(reader, 10) == base + _N
        assert not selects_during_commit, "a SELECT was issued while the commit was in flight"
    finally:
        await store.close()
