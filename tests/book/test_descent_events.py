"""Regression tests for review finding F-06 (dev-loop 20261020-dlcobra-ce5a).

Requirements 4.9 and 15.6 say an abandoned Selection_Descent and a descent
timeout must be reported to the Progress_Reporter. The coordinator only kept
private counters, and the CPU-deadline check inside ``_descent`` counted
nothing at all, so the progress record could not show either.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest

from dlshogi.book import search as search_module
from dlshogi.book.keys import PositionKey, position_key_from_sfen
from dlshogi.book.evaluator import Evaluator
from dlshogi.book.node_store import NodeStore
from dlshogi.book.report import ProgressReporter
from dlshogi.book.search import SearchCoordinator
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _StubInferenceSession,
    _drop_book_tables,
    _make_config,
)


def _reporter() -> ProgressReporter:
    return ProgressReporter(report_interval=1, throughput_floor=0.01, throughput_grace_period=60)


class _SlowSession(_StubInferenceSession):
    def run(self, features1, features2):
        time.sleep(0.5)  # blocks the loop, like a CPU-bound stall
        return super().run(features1, features2)


class TestCounting:
    """The two helpers, without a database."""

    def test_abandoned_and_timeout_reach_the_reporter(self):
        rep = _reporter()
        coordinator = SearchCoordinator(None, None, None, None, rep)
        coordinator._count_abandoned()
        coordinator._count_abandoned()
        coordinator._count_timeout()
        assert (coordinator.descents_abandoned, rep.descents_abandoned) == (2, 2)
        assert (coordinator.descents_timed_out, rep.descent_timeouts) == (1, 1)

    def test_without_a_reporter_the_private_counters_still_move(self):
        coordinator = SearchCoordinator(None, None, None, None, None)
        coordinator._count_abandoned()
        coordinator._count_timeout()
        assert (coordinator.descents_abandoned, coordinator.descents_timed_out) == (1, 1)

    def test_progress_record_carries_both_counts(self):
        rep = ProgressReporter(
            report_interval=1,
            throughput_floor=0.01,
            throughput_grace_period=60,
            time_func=iter([0.0, 1.0, 1.0, 1.0, 1.0]).__next__,
        )
        rep.tick()
        rep.record_event("descent_abandoned")
        rep.record_event("descent_timeout")
        rep.record_event("descent_timeout")
        record = rep.tick(dt=1.0)
        assert record is not None
        assert (record.to_dict()["descents_abandoned"], record.to_dict()["descent_timeouts"]) == (1, 2)

    def test_unknown_category_is_still_rejected(self):
        with pytest.raises(ValueError):
            _reporter().record_event("descent_abandonded")


@pytest.mark.db
@pytest.mark.asyncio
class TestThroughTheSearch:
    async def _build(self, pg_scratch_database, pg_conn, session, **overrides):
        await _drop_book_tables(pg_conn)
        cfg = dataclasses.replace(
            _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), max_book_ply=2, **overrides
        )
        store = await NodeStore.connect_and_prepare(cfg, role="search")
        evaluator = Evaluator(session, batch_size=cfg.batch_size, batch_timeout_ms=cfg.batch_timeout)
        evaluator.start()
        reporter = _reporter()
        return store, evaluator, reporter, SearchCoordinator(cfg, store, evaluator, None, reporter)

    async def test_workers_colliding_on_one_node_report_abandoned_descents(
        self, pg_scratch_database, pg_conn
    ):
        store, evaluator, reporter, coordinator = await self._build(
            pg_scratch_database, pg_conn, _StubInferenceSession(0.5), worker_count=16
        )
        todo = 300

        async def worker(i):
            nonlocal todo
            while todo > 0:
                todo -= 1
                await coordinator._run_descent_task(i)

        try:
            await asyncio.gather(*(worker(i) for i in range(16)))
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.descents_abandoned > 0
        assert reporter.descents_abandoned == coordinator.descents_abandoned
        assert reporter.descent_timeouts == 0

    async def test_all_edges_excluded_is_abandoned_and_reported(self, pg_scratch_database, pg_conn):
        """R4.9: every edge of the current node leads into an in-flight child."""
        store, evaluator, reporter, coordinator = await self._build(
            pg_scratch_database, pg_conn, _StubInferenceSession(0.5)
        )
        try:
            await coordinator._descent(worker_id=0, claimed=[])  # expands the root
            _, root = await store.get(position_key_from_sfen(_INITIAL_SFEN))
            for key in coordinator._get_child_keys(root, root.edges):
                assert coordinator.in_flight.test_and_add(
                    PositionKey(int(key["hi"]), int(key["lo"])), 99
                )

            before = (coordinator.descents_completed, root.visit_count)
            await coordinator._descent(worker_id=0, claimed=[])
            _, after = await store.get(position_key_from_sfen(_INITIAL_SFEN))
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.descents_abandoned == 1
        assert reporter.descents_abandoned == 1
        # R4.9: an abandoned descent changes no visit count and applies no backup.
        assert (coordinator.descents_completed, after.visit_count) == before

    async def test_wait_for_timeout_is_reported(self, pg_scratch_database, pg_conn, monkeypatch):
        monkeypatch.setattr(search_module, "_DESCENT_TIMEOUT_S", 0.2)
        store, evaluator, reporter, coordinator = await self._build(
            pg_scratch_database, pg_conn, _SlowSession(0.5)
        )
        try:
            await coordinator._run_descent_task(0)
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.descents_timed_out == 1
        assert reporter.descent_timeouts == 1
        assert coordinator.descents_completed == 0  # the backup was never applied

    async def test_cpu_deadline_inside_the_descent_is_reported(
        self, pg_scratch_database, pg_conn, monkeypatch
    ):
        """A stall that never yields is caught by the in-loop check, not by wait_for."""
        monkeypatch.setattr(search_module, "_DESCENT_TIMEOUT_S", 0.0)
        store, evaluator, reporter, coordinator = await self._build(
            pg_scratch_database, pg_conn, _StubInferenceSession(0.5)
        )
        try:
            await coordinator._descent(worker_id=0, claimed=[])
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.descents_timed_out == 1
        assert reporter.descent_timeouts == 1
        assert coordinator.descents_completed == 0
