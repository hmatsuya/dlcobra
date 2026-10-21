"""Regression tests for review finding F-05 (dev-loop 20261020-dlcobra-ce5a).

The progress record carries Terashock injection / illegal-move and Evaluator
failure counts (Requirements 14.2, 14.6, 7.4, 5.6), but nothing in the search
ever reported them, so they read 0 however many occurred.
"""

from __future__ import annotations

import dataclasses

import cshogi
import numpy as np
import pytest

from dlshogi.book.evaluator import Evaluator
from dlshogi.book.node_store import NodeStore, WriteOutcome, WriteResult
from dlshogi.book.packed_edge import PACKED_TS_MOVE
from dlshogi.book.report import ProgressReporter
from dlshogi.book.search import _FLAG_TERASHOCK_PRESENT, SearchCoordinator
from dlshogi.book.terashock import TerashockLookupResult
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _StubInferenceSession,
    _drop_book_tables,
    _make_config,
)

_LEGAL = sorted(int(cshogi.move16(m)) for m in cshogi.Board().legal_moves)
_ILLEGAL = [m for m in (40001, 40002, 40003) if m not in _LEGAL]


def _reporter() -> ProgressReporter:
    return ProgressReporter(report_interval=1, throughput_floor=0.01, throughput_grace_period=60)


def _lookup(*move16s: int) -> TerashockLookupResult:
    moves = np.zeros(len(move16s), dtype=PACKED_TS_MOVE)
    for i, m in enumerate(move16s):
        moves[i]["move16"], moves[i]["eval"], moves[i]["depth"] = m, 100, 20
    return TerashockLookupResult(sfen=_INITIAL_SFEN, moves=moves)


def _edges(*move16s: int, terashock: tuple[int, ...] = ()) -> list[dict]:
    return [
        {"move16": m, "flags": _FLAG_TERASHOCK_PRESENT if m in terashock else 0}
        for m in move16s
    ]


class _FakeIndex:
    def __init__(self, result):
        self._result = result

    async def lookup(self, key):
        return self._result


class _FailingSession:
    def run(self, features1, features2):
        raise RuntimeError("simulated inference failure")


class TestCountTerashock:
    """The counting rule, without a database."""

    def _coordinator(self, reporter):
        return SearchCoordinator(None, None, None, None, reporter)

    def test_one_injection_and_each_illegal_move_counted(self):
        rep = _reporter()
        legal = _LEGAL[0]
        self._coordinator(rep)._count_terashock(
            _edges(*_LEGAL[:3], terashock=(legal,)), _lookup(legal, *_ILLEGAL[:2])
        )
        assert (rep.terashock_injections, rep.terashock_illegal_discards) == (1, 2)

    def test_injection_is_counted_per_node_not_per_move(self):
        """R14.2 counts Book_Nodes: three Terashock moves on one node is one injection."""
        rep = _reporter()
        three = tuple(_LEGAL[:3])
        self._coordinator(rep)._count_terashock(
            _edges(*_LEGAL[:5], terashock=three), _lookup(*three)
        )
        assert (rep.terashock_injections, rep.terashock_illegal_discards) == (1, 0)

    def test_all_moves_illegal_is_not_an_injection(self):
        rep = _reporter()
        self._coordinator(rep)._count_terashock(_edges(*_LEGAL[:3]), _lookup(*_ILLEGAL))
        assert (rep.terashock_injections, rep.terashock_illegal_discards) == (0, len(_ILLEGAL))

    def test_no_entry_for_the_position_counts_nothing(self):
        rep = _reporter()
        self._coordinator(rep)._count_terashock(_edges(*_LEGAL[:3]), None)
        assert (rep.terashock_injections, rep.terashock_illegal_discards) == (0, 0)

    def test_without_a_reporter_it_is_a_no_op(self):
        self._coordinator(None)._count_terashock(_edges(*_LEGAL[:3]), _lookup(*_ILLEGAL))


@pytest.mark.db
@pytest.mark.asyncio
class TestThroughTheSearch:
    async def _store(self, pg_scratch_database, pg_conn, **overrides):
        await _drop_book_tables(pg_conn)
        cfg = dataclasses.replace(
            _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN),
            terashock_prior_weight=0.5,
            **overrides,
        )
        return cfg, await NodeStore.connect_and_prepare(cfg, role="search")

    async def test_expansion_reports_injection_and_illegal_moves(self, pg_scratch_database, pg_conn):
        cfg, store = await self._store(pg_scratch_database, pg_conn)
        evaluator = Evaluator(
            _StubInferenceSession(0.5), batch_size=cfg.batch_size, batch_timeout_ms=cfg.batch_timeout
        )
        evaluator.start()
        rep = _reporter()
        index = _FakeIndex(_lookup(_LEGAL[0], *_ILLEGAL))
        coordinator = SearchCoordinator(cfg, store, evaluator, index, rep)
        try:
            await coordinator._descent(worker_id=0, claimed=[])  # expands the root only
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.expansions_committed == 1
        assert rep.terashock_injections == 1
        assert rep.terashock_illegal_discards == len(_ILLEGAL)

    async def test_duplicate_expansion_is_not_counted_twice(self, pg_scratch_database, pg_conn):
        cfg, store = await self._store(pg_scratch_database, pg_conn)
        evaluator = Evaluator(
            _StubInferenceSession(0.5), batch_size=cfg.batch_size, batch_timeout_ms=cfg.batch_timeout
        )
        evaluator.start()
        rep = _reporter()
        coordinator = SearchCoordinator(
            cfg, store, evaluator, _FakeIndex(_lookup(_LEGAL[0], *_ILLEGAL)), rep
        )

        async def lost_race(write):
            return WriteResult(outcome=WriteOutcome.DUPLICATE, key=write.key)

        store.insert_expansion = lost_race
        try:
            await coordinator._descent(worker_id=0, claimed=[])
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.expansions_duplicate == 1
        assert (rep.terashock_injections, rep.terashock_illegal_discards) == (0, 0)

    async def test_evaluator_failure_is_reported_once_per_descent(self, pg_scratch_database, pg_conn):
        cfg, store = await self._store(pg_scratch_database, pg_conn)
        evaluator = Evaluator(
            _FailingSession(), batch_size=cfg.batch_size, batch_timeout_ms=cfg.batch_timeout
        )
        evaluator.start()
        rep = _reporter()
        coordinator = SearchCoordinator(cfg, store, evaluator, None, rep)
        try:
            for _ in range(3):
                await coordinator._run_descent_task(0)
        finally:
            await evaluator.stop()
            await store.close()
        assert coordinator.descents_failed == 3
        assert rep.evaluator_failures == 3
        assert coordinator.expansions_committed == 0
