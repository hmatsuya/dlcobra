"""Regression tests: terminal and Max_Book_Ply leaves persist as Book_Node rows.

Covers the defect in ``dlshogi.book.search._descent``'s termination rules 1
(Max_Book_Ply leaf) and 2 (terminal leaf): both used to ``break`` without
writing any row, so a mated child (or a ply-limit leaf) was absent from
``book_node``. An absent child makes ``propagate.py`` treat it as an
unevaluated leaf (Draw_Value) and makes ``export.py`` drop every edge into
it, so mating moves silently vanished from the exported book.

Also covers the companion defect in ``dlshogi.book.propagate.propagate``:
the above-threshold "terminal child" branch computed a contribution for the
parent edge but never wrote a ``prop_value`` for the terminal child itself,
so export's freshness filter (``prop_value IS NULL`` / stale
``prop_epoch``) still excluded the mating edge even after the leaf row
existed.

Test (a) drives a real mate-in-1 position through search -> propagate ->
export and checks the mated child's row, the root's propagated best move,
and the exported YaneuraOu ``.db`` text. Test (b) checks that Max_Book_Ply
leaves are deliberately NOT persisted (an edgeless row with eval_win_rate
would never expand after a resume with a larger Max_Book_Ply, F-16).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cshogi
import pytest

from dlshogi.book.config import BookConfig
from dlshogi.book.export import export_book
from dlshogi.book.keys import (
    fold_u64_to_i64,
    position_key,
    position_key_from_sfen,
)
from dlshogi.book.node_store import NodeStore, WriteOutcome, WriteResult
from dlshogi.book.propagate import propagate
from dlshogi.book.search import SearchCoordinator

from tests.book.test_interop import (
    _INITIAL_SFEN,
    _drop_book_tables,
    _make_config,
    _run_small_search,
)

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

_MATE_ROOT = (
    "3g1kpnb/2s2s2g/lpnp1+P1+Sl/p1p1N2Pp/4p1P1P/1P4+r2/P1PPP1G1L/LB1G1K1p1/1NS6 b Pr 75"
)
_MATE_MOVE_USI = "4c4b"


def _config(pg_info, *, root_sfen: str, max_book_ply: int, **overrides) -> BookConfig:
    return dataclasses.replace(
        _make_config(pg_info, root_sfen=root_sfen),
        max_book_ply=max_book_ply,
        **overrides,
    )


class TestTerminalLeafPersistence:
    """Mate-in-1 root: the mated child must become a real Book_Node row."""

    @pytest.mark.parametrize("threshold", [0, 1000])
    async def test_mated_child_persisted_and_exported(
        self, pg_scratch_database, pg_conn, tmp_path: Path, threshold: int
    ):
        """threshold 0 exercises the above-threshold terminal branch of the
        propagator; 1000 (above every edge's visit count) the below-threshold
        one (R9.15 puts the terminal rule before criteria 12/13)."""
        await _drop_book_tables(pg_conn)

        cfg = _config(
            pg_scratch_database,
            root_sfen=_MATE_ROOT,
            max_book_ply=6,
            propagation_visit_threshold=threshold,
        )

        search_store = await NodeStore.connect_and_prepare(cfg, role="search")
        try:
            coordinator = await _run_small_search(
                search_store, cfg, max_descents=400
            )
            assert coordinator.expansions_committed > 0

            # The mated child's row: terminal == LOSS_FOR_STM (2), no edges.
            board = cshogi.Board(_MATE_ROOT)
            mate_move = [
                m for m in board.legal_moves
                if cshogi.move_to_usi(m) == _MATE_MOVE_USI
            ][0]
            move16 = cshogi.move16(mate_move)
            board.push(mate_move)
            child_key = position_key(board)

            async with search_store.pool.acquire() as conn:
                child_row = await conn.fetchrow(
                    "SELECT terminal, edge_count, eval_win_rate FROM book_node "
                    "WHERE key_hi=$1 AND key_lo=$2",
                    fold_u64_to_i64(child_key.hi),
                    fold_u64_to_i64(child_key.lo),
                )
            assert child_row is not None, (
                "mated child has no Book_Node row; termination rule 2 did "
                "not persist the leaf"
            )
            assert child_row["terminal"] == 2
            assert child_row["edge_count"] == 0

            await search_store.bump_search_write_seq()
        finally:
            await search_store.close()

        prop_store = await NodeStore.connect_and_prepare(cfg, role="propagate")
        try:
            root_key = position_key_from_sfen(cfg.root_position)
            stats = await propagate(prop_store, root_key, cfg)
            assert stats.nodes > 0

            async with prop_store.pool.acquire() as conn:
                root_row = await conn.fetchrow(
                    "SELECT prop_value, prop_best_move16 FROM book_node "
                    "WHERE key_hi=$1 AND key_lo=$2",
                    fold_u64_to_i64(root_key.hi),
                    fold_u64_to_i64(root_key.lo),
                )
            assert root_row is not None
            assert root_row["prop_value"] == pytest.approx(1.0)
            assert root_row["prop_best_move16"] == move16
        finally:
            await prop_store.close()

        yaneuraou_out = tmp_path / "book.db"
        export_store = await NodeStore.connect_and_prepare(cfg, role="export")
        try:
            counts = await export_book(export_store, cfg, None, yaneuraou_out)
            assert counts.yaneuraou_entries > 0
        finally:
            await export_store.close()

        text = yaneuraou_out.read_text()
        move_lines = [
            ln for ln in text.splitlines()
            if ln.split() and ln.split()[0] == _MATE_MOVE_USI
        ]
        assert move_lines, (
            "the mating move does not appear as an edge in the exported "
            "YaneuraOu .db"
        )


class TestMaxBookPlyLeafStaysExpandable:
    """F-16: a Max_Book_Ply leaf must not be frozen as an edgeless row, so a
    resumed run with a larger Max_Book_Ply can still grow the book."""

    async def test_resume_with_larger_ply_limit_expands(
        self, pg_scratch_database, pg_conn
    ):
        await _drop_book_tables(pg_conn)
        cfg1 = _config(
            pg_scratch_database, root_sfen=_INITIAL_SFEN, max_book_ply=1
        )
        store = await NodeStore.connect_and_prepare(cfg1, role="search")
        try:
            await _run_small_search(store, cfg1, max_descents=120)
            async with store.pool.acquire() as conn:
                n1 = await conn.fetchval("SELECT count(*) FROM book_node")
                frozen = await conn.fetchval(
                    "SELECT count(*) FROM book_node "
                    "WHERE edge_count = 0 AND terminal = 0"
                )
            assert frozen == 0, "a non-terminal ply-limit leaf was persisted edgeless"

            cfg2 = _config(
                pg_scratch_database, root_sfen=_INITIAL_SFEN, max_book_ply=3
            )
            await _run_small_search(store, cfg2, max_descents=300)
            async with store.pool.acquire() as conn:
                n2 = await conn.fetchval("SELECT count(*) FROM book_node")
        finally:
            await store.close()

        assert n2 > n1 + 10, f"book did not grow after raising Max_Book_Ply ({n1} -> {n2})"


class TestCountWrite:
    """The shared insert_expansion outcome ladder used by rule 3 and the terminal leaf."""

    @pytest.mark.parametrize(
        "outcome, keeps_going, counter",
        [
            (WriteOutcome.COMMITTED, True, "expansions_committed"),
            (WriteOutcome.DUPLICATE, True, "expansions_duplicate"),
            (WriteOutcome.COLLISION, True, "expansions_collision"),
            (WriteOutcome.FAILED, False, "expansions_failed"),
        ],
    )
    def test_outcomes(self, outcome, keeps_going, counter):
        coord = SearchCoordinator(None, None, None)
        result = WriteResult(outcome=outcome, key=position_key_from_sfen(_INITIAL_SFEN))
        assert coord._count_write(result, 3) is keeps_going
        assert getattr(coord, counter) == 1
