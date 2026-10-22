"""Regression tests for review finding F-23 (dev-loop 20261020-dlcobra-ce5a).

``NodeStore.start_flusher`` / ``start_rss_sampler`` had no caller in production:
``SearchCoordinator.run()`` flushed once, in its ``finally`` after the stop
request. Every backup delta (visit counts, value sums) therefore lived only in
process memory for the whole run, so a crash or a ``kill -9`` lost all of them
(the expansions were already durable). design.md has the flusher write every
200 ms and on stop; R10 and R15.4 assume it.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from dlshogi.book.evaluator import Evaluator
from dlshogi.book.node_store import NodeStore
from dlshogi.book.search import SearchCoordinator
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _StubInferenceSession,
    _drop_book_tables,
    _make_config,
)

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _running(pg_scratch_database, pg_conn, **overrides):
    await _drop_book_tables(pg_conn)
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN),
        worker_count=4,
        max_book_ply=4,
        **overrides,
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    evaluator = Evaluator(
        _StubInferenceSession(0.5), batch_size=cfg.batch_size, batch_timeout_ms=cfg.batch_timeout
    )
    evaluator.start()
    coordinator = SearchCoordinator(cfg, store, evaluator, None)
    return cfg, store, evaluator, coordinator


async def _root_visits(pg_conn) -> int:
    return int(await pg_conn.fetchval("SELECT coalesce(max(visit_count), 0) FROM book_node"))


async def test_visit_counts_reach_the_database_while_the_search_runs(pg_scratch_database, pg_conn):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    try:
        for _ in range(100):  # up to 5 s
            await asyncio.sleep(0.05)
            if await _root_visits(pg_conn) > 0:
                break
        in_db_before_stop = await _root_visits(pg_conn)
        assert in_db_before_stop > 0, "no visit count was persisted before the stop request"
        assert coordinator.descents_completed >= in_db_before_stop
    finally:
        coordinator._request_stop()
        await asyncio.wait_for(run, 60)
        await evaluator.stop()
        await store.close()


async def test_the_final_flush_leaves_nothing_in_memory(pg_scratch_database, pg_conn):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(1.0)
    coordinator._request_stop()
    await asyncio.wait_for(run, 60)
    try:
        assert store._pending_backup == {}
        assert store._flushing == {}
        assert store._flusher_task is None
        assert store._rss_task is None
        # R4.5 once everything is flushed: nothing was lost between memory and the database.
        assert await _root_visits(pg_conn) == coordinator.descents_completed
    finally:
        await evaluator.stop()
        await store.close()


async def test_run_starts_both_background_tasks_and_stops_them(pg_scratch_database, pg_conn):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if store._flusher_task is not None and store._rss_task is not None:
                break
        assert store._flusher_task is not None and not store._flusher_task.done()
        assert store._rss_task is not None and not store._rss_task.done()
    finally:
        coordinator._request_stop()
        await asyncio.wait_for(run, 60)
        await evaluator.stop()
        await store.close()
    assert store._flusher_task is None and store._rss_task is None


async def test_a_forced_cancel_of_run_still_stops_the_flusher(pg_scratch_database, pg_conn):
    """run() is cancelled (e.g. its caller times out): no orphaned flusher keeps using a closed pool."""
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(0.5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    try:
        assert store._flusher_task is None or store._flusher_task.done()
    finally:
        await evaluator.stop()
        await store.close()


async def test_a_cancel_right_after_the_background_tasks_start_does_not_orphan_them(
    pg_scratch_database, pg_conn
):
    """The start calls sit inside run()'s try, so the finally stops them.

    Injected at the line right after start_flusher(): before this was inside
    the try, a CancelledError here skipped the finally and left the flusher
    running (and failing every 200 ms once the pool was closed).
    """
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    real_start = store.start_rss_sampler

    def cancel_after_start(*args, **kwargs):
        real_start(*args, **kwargs)
        raise asyncio.CancelledError()

    store.start_rss_sampler = cancel_after_start
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(coordinator.run(), 10)
        assert store._flusher_task is None or store._flusher_task.done()
        assert store._rss_task is None or store._rss_task.done()
    finally:
        await evaluator.stop()
        await store.close()


async def test_a_failing_final_flush_is_recorded_for_the_exit_status(pg_scratch_database, pg_conn):
    """run.sh only stops before export on a non-zero search status, so a final flush
    that did not reach the database must not look like a clean run."""
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(0.8)

    async def boom():
        raise RuntimeError("PostgreSQL went away")

    store.stop_flusher = boom
    coordinator._request_stop()
    await asyncio.wait_for(run, 60)
    try:
        assert coordinator.final_flush_failed is True
    finally:
        await evaluator.stop()
        await store.close()


async def test_a_clean_run_does_not_flag_the_final_flush(pg_scratch_database, pg_conn):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(0.8)
    coordinator._request_stop()
    await asyncio.wait_for(run, 60)
    try:
        assert coordinator.final_flush_failed is False
    finally:
        await evaluator.stop()
        await store.close()


async def test_a_failing_reaper_stop_does_not_skip_the_flusher_stop(pg_scratch_database, pg_conn):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)

    async def boom():
        raise RuntimeError("reaper died")

    coordinator._reaper.stop = boom
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(0.8)
    coordinator._request_stop()
    await asyncio.wait_for(run, 60)
    try:
        assert store._flusher_task is None and store._rss_task is None
        assert store._pending_backup == {}
    finally:
        await evaluator.stop()
        await store.close()


@pytest.mark.parametrize("workers, expected", [(1, 2), (8, 9), (63, 64), (64, 64), (1024, 64)])
async def test_the_pool_has_a_connection_beyond_the_workers_up_to_the_cap(
    pg_scratch_database, pg_conn, workers, expected
):
    """Worker_Count + 1 connections below the cap of 64 (it is a size, not a reservation)."""
    await _drop_book_tables(pg_conn)
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), worker_count=workers
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    try:
        assert store.pool.get_max_size() == expected
    finally:
        await store.close()
