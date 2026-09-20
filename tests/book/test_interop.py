"""End-to-end book interoperability integration test (task 19.4).

This is the only test that proves an exported book file works with the
*game-time* engine, which is the whole point of the export. It drives a
small but real slice of the whole pipeline against a scratch PostgreSQL
database, GPU-free:

    seed the Root_Position
      -> run a bounded number of real Selection_Descents
         (dlshogi.book.search.SearchCoordinator._descent, GPU-free via an
          injected stub InferenceSession returning drawn/uniform arrays)
      -> flush the coalescing backup accumulator
      -> run one real Value_Propagation pass (dlshogi.book.propagate.propagate)
      -> export both book.bin (Apery) and book.db (YaneuraOu)
         (dlshogi.book.export.export_book)

and then reads the two exported files back through *independent* readers
and confirms they agree on keys, moves, counts, and scores:

  * ``dlshogi/utils/book.py``'s ``np.fromfile(path, cshogi.BookEntry)``
    (the existing production Apery reader);
  * ``cshogi.BookEntry`` dtype directly, plus ``cshogi.Board.book_key()``
    -- the game-time key. Requirement 12.1 says the exported ``key`` field
    must equal ``Board.book_key()`` for the position; this test confirms
    that on the *real* positions produced by the small search;
  * ``tests/book/reference/book_db.py`` -- the independently-written
    YaneuraOu ``.db`` parser (task 5.1), which imports nothing from
    ``dlshogi.book``. The exported ``.db`` is re-parsed through it and
    checked for agreement.

The test is deliberately **not** a hypothesis property test: it is a
single, hermetic, deterministic integration run. It carries the ``db``
marker (needs PostgreSQL) and is optional (task 19.4 is a ``*`` task). It
needs no GPU and runs at small scale, so it is fully runnable in this
environment.

Requirements exercised: 12.1 (Apery record mapping incl. key ==
Board.book_key()), 12.2 (Apery ordering ascending by unsigned key), 12.3
(within-key ordering by score/count/fromToPro), 12.4 (YaneuraOu ordering
ascending by byte-wise SFEN), 12.6 (score via Eval_Coef), 12.7 (exported
moves legal), 12.11 (reported counts), 12.12 (both Black-to-move and
White-to-move nodes present in the one file).
"""

from __future__ import annotations

import math
from pathlib import Path

import cshogi
import numpy as np
import pytest

from dlshogi.book.config import BookConfig
from dlshogi.book.evaluator import Evaluator
from dlshogi.book.export import export_book
from dlshogi.book.keys import (
    fold_u64_to_i64,
    position_key_from_sfen,
    side_to_move_is_white,
)
from dlshogi.book.node_store import GetResult, NodeStore
from dlshogi.book.propagate import propagate
from dlshogi.book.search import SearchCoordinator

# The reference YaneuraOu .db reader, written from Requirement 6's text
# alone and importing nothing from dlshogi.book (task 5.1).
from tests.book.reference.book_db import ref_parse_book_db

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

POLICY_DIM = 9 * 9 * 27  # 2187, dlshogi policy head width.

# Initial Shogi position; Black (sente) to move.
_INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


# ---------------------------------------------------------------------------
# GPU-free stub inference session (design.md: "a stub returning drawn arrays
# satisfies the whole property suite with no GPU").
# ---------------------------------------------------------------------------


class _StubInferenceSession:
    """A deterministic, GPU-free ``InferenceSession`` (structural protocol).

    Returns, for a batch of ``n`` Board_States:

    * ``policy_logits`` of shape ``(n, POLICY_DIM)`` -- all zeros, which
      the Evaluator decodes into a uniform softmax over the legal moves
      of each position (every legal move gets an equal prior). Uniform,
      finite, and never degenerate, so no Requirement 5.6 failure and no
      Requirement 5.7 substitution is triggered.
    * ``values`` of shape ``(n, 1)`` -- a fixed win rate of 0.5 (a drawn
      evaluation). Finite, so no Requirement 5.6 failure.

    Uniform priors plus a drawn value make the search's PUCT selection
    explore breadth-first-ish across the legal moves, which is exactly
    what produces a graph spanning both Black-to-move and White-to-move
    nodes (Requirement 12.12) within a small descent budget.
    """

    def __init__(self, win_rate: float = 0.5) -> None:
        self._win_rate = float(win_rate)

    def run(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = features1.shape[0]
        policy_logits = np.zeros((n, POLICY_DIM), dtype=np.float32)
        values = np.full((n, 1), self._win_rate, dtype=np.float32)
        return policy_logits, values


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _make_config(pg_info, *, root_sfen: str) -> BookConfig:
    """Build a validated small-scale BookConfig for the scratch database."""
    # BookConfig.from_mapping consumes the RANGE_TABLE names, so build the
    # raw mapping with exactly those names. The PostgreSQL connection
    # settings are five separate names in the range table.
    raw = {
        "PostgreSQL host": pg_info.host,
        "PostgreSQL port": str(pg_info.port),
        "PostgreSQL user": pg_info.user,
        # BookConfig requires a non-empty password text (>= 1 char). Local
        # socket peer/trust auth ignores the password entirely, and
        # NodeStore._connect_kwargs only forwards a truthy password to
        # asyncpg, so this placeholder is inert against a peer-auth socket
        # (the conftest connects with no password at all successfully).
        "PostgreSQL password": pg_info.password or "unused-placeholder",
        "PostgreSQL database": pg_info.database,
        "Cache_Budget": 268_435_456,  # 256 MiB, the minimum permitted
        "Worker_Count": 1,
        "Batch_Size": 8,
        "Batch_Timeout": 20,  # ms
        "Virtual_Loss": 1,
        "Terashock_Prior_Weight": 0.0,  # no Terashock book in this test
        "Eval_Coef": 600.0,
        "Draw_Value_Black": 0.5,
        "Draw_Value_White": 0.5,
        "Max_Book_Ply": 8,
        "Propagation_Visit_Threshold": 0,  # negamax over every edge
        "Export_Visit_Threshold": 0.0,  # keep every visited edge
        "Connection_Retry_Limit": 3,
        "Root_Position": root_sfen,
        "Report_Interval": 1,
        "Throughput_Floor": 0.01,
        "Throughput_Grace_Period": 60,
        # Terashock_Book path deliberately absent (Requirement 13.8).
    }
    return BookConfig.load(raw)


async def _drop_book_tables(conn) -> None:
    """Drop every Node_Store-owned table so connect_and_prepare recreates them.

    Matches the bootstrap used by ``tests/book/test_node_store.py``: the
    session-scoped scratch-DB fixture applies ``schema.sql`` (DDL only),
    but ``book_meta``'s seed row is written by NodeStore's own creation
    path, so a clean creation must start from an empty database.
    """
    await conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, "
        "book_meta CASCADE"
    )


# ---------------------------------------------------------------------------
# The small end-to-end run
# ---------------------------------------------------------------------------


async def _run_small_search(
    store: NodeStore,
    cfg: BookConfig,
    *,
    max_descents: int,
) -> SearchCoordinator:
    """Drive a bounded number of real Selection_Descents, then flush.

    Rather than ``SearchCoordinator.run()`` (which installs signal
    handlers and runs an unbounded supervisor loop -- the design says the
    run has no descent budget), this drives ``_descent`` directly for a
    fixed count, which is the bounded "small real search" this task asks
    for. Each ``_descent`` creates the root node when absent, selects via
    PUCT, expands leaves through the (stubbed) Evaluator, and applies the
    value backup through the coalescing accumulator, exactly as the
    supervisor's replaced tasks would.
    """
    evaluator = Evaluator(
        _StubInferenceSession(win_rate=0.5),
        batch_size=cfg.batch_size,
        batch_timeout_ms=cfg.batch_timeout,
    )
    evaluator.start()
    coordinator = SearchCoordinator(cfg, store, evaluator, terashock_index=None)
    try:
        for _ in range(max_descents):
            claimed: list = []
            await coordinator._descent(worker_id=0, claimed=claimed)
            # Mirror _run_descent_task's finally: release any claim held.
            for key in claimed:
                coordinator.in_flight.discard(key)
        # Make every backup delta durable before propagation reads them.
        await store.flush()
    finally:
        await evaluator.stop()
    return coordinator


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


class TestEndToEndBookInterop:
    """Task 19.4: small search -> propagate -> export -> independent re-read."""

    async def test_export_round_trips_through_independent_readers(
        self, pg_scratch_database, pg_conn, tmp_path: Path
    ):
        # The session-scoped fixture applies schema.sql, but the book_meta
        # seed row (schema_version, Zobrist fingerprint, Root_Position) is
        # written only by NodeStore's own creation path. Drop the
        # pre-applied tables so connect_and_prepare recreates and seeds
        # them properly -- the same bootstrap the node_store db tests use.
        await _drop_book_tables(pg_conn)

        cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)

        # -- Connect a propagation/export-role store (synchronous_commit=on)
        #    and a search-role store. One database, two roles.
        search_store = await NodeStore.connect_and_prepare(cfg, role="search")
        try:
            # Record the Root_Position so search's root-match check passes
            # and propagate can find the root. connect_and_prepare's
            # ensure_schema seeds book_meta; the root node itself is
            # created by the first descent.
            coordinator = await _run_small_search(
                search_store, cfg, max_descents=200
            )

            # The search must have committed at least a handful of
            # expansions for the rest of the test to be meaningful.
            assert coordinator.expansions_committed > 0, (
                "small search committed no expansions; nothing to export"
            )

            # bump_search_write_seq so the stale-propagation bookkeeping
            # reflects that search writes happened before propagation.
            await search_store.bump_search_write_seq()
        finally:
            await search_store.close()

        # -- Propagation pass (role=propagate: synchronous_commit=on) ------
        prop_store = await NodeStore.connect_and_prepare(cfg, role="propagate")
        try:
            root_key = position_key_from_sfen(cfg.root_position)
            stats = await propagate(prop_store, root_key, cfg)
            assert stats.nodes > 0, "propagation visited no nodes"
            # Mark the pass complete so export's freshness test
            # (prop_epoch == propagation_done_seq) accepts these values.
            await prop_store.mark_propagation_done(await _current_prop_seq(prop_store))
        finally:
            await prop_store.close()

        # -- Export both formats (role=export: synchronous_commit=on) ------
        apery_out = tmp_path / "book.bin"
        yaneuraou_out = tmp_path / "book.db"

        export_store = await NodeStore.connect_and_prepare(cfg, role="export")
        try:
            counts = await export_book(
                export_store, cfg, apery_out, yaneuraou_out
            )
        finally:
            await export_store.close()

        # The export must have written something to check.
        assert counts.apery_records > 0, "export wrote no Apery records"
        assert counts.yaneuraou_entries > 0, "export wrote no YaneuraOu entries"
        assert apery_out.exists()
        assert yaneuraou_out.exists()

        # ================================================================
        # Independent reader 1: dlshogi/utils/book.py's np.fromfile reader.
        # ================================================================
        book = np.fromfile(str(apery_out), cshogi.BookEntry)
        assert book.dtype == cshogi.BookEntry
        assert len(book) == counts.apery_records, (
            "the Apery file read back through the production reader has a "
            "different record count than the exporter reported"
        )

        # ================================================================
        # 12.2 / 12.3: file ordering is total and content-determined.
        # ================================================================
        _assert_apery_total_order(book)

        # ================================================================
        # 12.1: every exported key equals Board.book_key() for its parent
        #       position, and every exported move is legal there (12.7).
        #       We reconstruct the (parent sfen -> apery key, legal moves)
        #       mapping straight from the graph in the database, keyed by
        #       the game-time key, then confirm every file record matches.
        # ================================================================
        parent_index = await _load_parent_index(cfg)

        # Group the file's records by key (they are already key-sorted).
        both_colors_seen = {"black": False, "white": False}
        # Rebuild the expected (key, move16) -> (count, score) mapping from
        # the graph so we can compare the file against an independent
        # recomputation of counts and scores (12.1 count, 12.6 score).
        expected = await _expected_records(cfg)

        seen_pairs = set()
        for rec in book:
            key = int(rec["key"])
            move16 = int(rec["fromToPro"])
            count = int(rec["count"])
            score = int(rec["score"])

            assert key in parent_index, (
                f"exported key {key} does not correspond to any Book_Node's "
                f"Board.book_key(); Requirement 12.1 violated"
            )
            parent = parent_index[key]

            # 12.1: the game-time key. cshogi's Board.book_key() must equal
            # the exported key for the parent position.
            board = cshogi.Board(parent["sfen"])
            assert board.book_key() == key, (
                f"exported key {key} != Board.book_key() "
                f"{board.book_key()} for sfen {parent['sfen']!r}"
            )

            # 12.7: the move is legal from the parent position.
            legal_move16s = {cshogi.move16(m) for m in board.legal_moves}
            assert move16 in legal_move16s, (
                f"exported move {cshogi.move_to_usi(move16)} is not legal "
                f"at {parent['sfen']!r}; Requirement 12.7 violated"
            )

            # 12.12: record which side to move this parent had.
            if side_to_move_is_white(parent["key"]):
                both_colors_seen["white"] = True
            else:
                both_colors_seen["black"] = True

            # 12.1 count / 12.6 score: independent recomputation agreement.
            exp = expected.get((key, move16))
            assert exp is not None, (
                f"file has a record for (key={key}, move={move16}) that the "
                f"graph recomputation did not expect"
            )
            assert count == exp["count"], (
                f"count mismatch for (key={key}, move={move16}): "
                f"file={count}, expected={exp['count']}"
            )
            assert score == exp["score"], (
                f"score mismatch for (key={key}, move={move16}): "
                f"file={score}, expected={exp['score']}"
            )
            seen_pairs.add((key, move16))

        # Every expected record appears in the file, and no extras.
        assert seen_pairs == set(expected.keys()), (
            "the set of (key, move16) pairs in the Apery file differs from "
            "the independent recomputation from the graph"
        )

        # ================================================================
        # 12.12: both Black-to-move and White-to-move nodes are present in
        #        the one exported file.
        # ================================================================
        assert both_colors_seen["black"], (
            "no Black-to-move node was exported (Requirement 12.12)"
        )
        assert both_colors_seen["white"], (
            "no White-to-move node was exported (Requirement 12.12)"
        )

        # ================================================================
        # Independent reader 2: the YaneuraOu .db reference parser (task
        # 5.1), which imports nothing from dlshogi.book.
        # ================================================================
        db_text = yaneuraou_out.read_text(encoding="utf-8")
        ref_book = ref_parse_book_db(db_text)

        # 12.4: one entry per Book_Node with >= 1 exported edge, ordered
        # ascending by byte-wise SFEN.
        assert len(ref_book.entries) == counts.yaneuraou_entries, (
            "the .db re-parsed through the independent reference reader has "
            "a different entry count than the exporter reported"
        )
        ref_sfens = [e.sfen for e in ref_book.entries]
        ref_sfen_bytes = [s.encode("utf-8") for s in ref_sfens]
        assert ref_sfen_bytes == sorted(ref_sfen_bytes), (
            "YaneuraOu .db entries are not ordered ascending by byte-wise "
            "SFEN (Requirement 12.4)"
        )

        # Every entry's SFEN is a real parent position that was exported,
        # every move within it is legal there, and its moves are ordered
        # descending by evaluation value (Requirement 12.4).
        exported_parent_sfens = {parent["sfen"] for parent in parent_index.values()}
        for entry in ref_book.entries:
            assert entry.sfen in exported_parent_sfens, (
                f".db entry sfen {entry.sfen!r} is not an exported parent "
                f"position"
            )
            board = cshogi.Board(entry.sfen)
            legal_usi = {cshogi.move_to_usi(cshogi.move16(m)) for m in board.legal_moves}
            evals = [m.eval for m in entry.moves]
            assert evals == sorted(evals, reverse=True), (
                f".db entry {entry.sfen!r} moves not ordered descending by "
                f"eval (Requirement 12.4): {evals}"
            )
            for m in entry.moves:
                assert m.move in legal_usi, (
                    f".db move {m.move!r} is not legal at {entry.sfen!r} "
                    f"(Requirement 12.7)"
                )

        # ================================================================
        # 12.11: the reported excluded-edge count is a non-negative int and
        #        the reported record/entry counts match what both readers
        #        actually found (already asserted above per-reader).
        # ================================================================
        assert counts.excluded_edges >= 0


# ---------------------------------------------------------------------------
# Helpers that recompute expectations straight from the graph in the DB,
# independently of export.py's own phase-1 emission logic.
# ---------------------------------------------------------------------------


def _apery_score(child_prop_value: float, eval_coef: float) -> int:
    """Recompute the Apery score independently of export.py.

    score = clip(round(-log(1/v - 1) * Eval_Coef), INT32) with
    v = 1 - child_prop_value (parent's perspective), matching design.md.
    """
    v = 1.0 - child_prop_value
    v = max(1e-15, min(1.0 - 1e-15, v))
    logit = -math.log(1.0 / v - 1.0)
    raw = round(logit * eval_coef)
    return max(-(2**31), min(2**31 - 1, raw))


async def _current_prop_seq(store: NodeStore) -> int:
    """Read book_meta.propagation_seq (the pass id propagate() just used)."""
    async with store.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT propagation_seq FROM book_meta WHERE id = 1")
    return int(row["propagation_seq"])


async def _load_parent_index(cfg: BookConfig) -> dict:
    """Map apery_key (unsigned) -> {sfen, key} for every Book_Node."""
    from dlshogi.book.keys import unfold_i64_to_u64, PositionKey

    store = await NodeStore.connect_and_prepare(cfg, role="export")
    try:
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key_hi, key_lo, sfen, apery_key FROM book_node"
            )
    finally:
        await store.close()

    index = {}
    for r in rows:
        apery_key = unfold_i64_to_u64(r["apery_key"])
        key = PositionKey(
            hi=unfold_i64_to_u64(r["key_hi"]),
            lo=unfold_i64_to_u64(r["key_lo"]),
        )
        index[apery_key] = {"sfen": r["sfen"], "key": key}
    return index


async def _expected_records(cfg: BookConfig) -> dict:
    """Independently recompute the (apery_key, move16) -> {count, score} map.

    Walks every Book_Node's packed edges, applies the same exclusion rules
    Requirement 12.5/12.9 state, and computes count/score from the child's
    propagated value -- all straight from the database, without calling
    export.py's phase-1 code. Used to confirm the exported file agrees on
    counts and scores.
    """
    from dlshogi.book.keys import (
        unfold_i64_to_u64,
        fold_u64_to_i64,
        position_keys_after,
        PositionKey,
    )
    from dlshogi.book.packed_edge import decode_edges

    store = await NodeStore.connect_and_prepare(cfg, role="export")
    try:
        async with store.pool.acquire() as conn:
            done_row = await conn.fetchrow(
                "SELECT propagation_done_seq FROM book_meta WHERE id = 1"
            )
            done_seq = int(done_row["propagation_done_seq"])

            node_rows = await conn.fetch(
                "SELECT key_hi, key_lo, sfen, apery_key, visit_count, edges "
                "FROM book_node"
            )

            expected: dict = {}
            for nr in node_rows:
                edges_blob = nr["edges"]
                if not edges_blob:
                    continue
                edges = decode_edges(edges_blob)
                if len(edges) == 0:
                    continue
                node_visit = int(nr["visit_count"])
                if node_visit == 0:
                    continue
                apery_key = unfold_i64_to_u64(nr["apery_key"])
                sfen = nr["sfen"]

                child_keys = position_keys_after(sfen, edges["move16"])
                for i in range(len(edges)):
                    move16 = int(edges[i]["move16"])
                    vc = int(edges[i]["visit_count"])
                    ratio = vc / node_visit
                    if cfg.export_visit_threshold > 0 and ratio < cfg.export_visit_threshold:
                        continue

                    ck_hi = fold_u64_to_i64(int(child_keys[i]["hi"]))
                    ck_lo = fold_u64_to_i64(int(child_keys[i]["lo"]))
                    child = await conn.fetchrow(
                        "SELECT prop_value, prop_epoch FROM book_node "
                        "WHERE key_hi = $1 AND key_lo = $2",
                        ck_hi,
                        ck_lo,
                    )
                    if child is None or child["prop_value"] is None:
                        continue
                    if child["prop_epoch"] != done_seq:
                        continue

                    score = _apery_score(float(child["prop_value"]), cfg.eval_coef)
                    expected[(apery_key, move16)] = {
                        "count": min(vc, 65535),
                        "score": score,
                    }
    finally:
        await store.close()
    return expected


# ---------------------------------------------------------------------------
# Ordering assertion
# ---------------------------------------------------------------------------


def _assert_apery_total_order(book: np.ndarray) -> None:
    """Assert the Apery file is in the total order of Requirements 12.2/12.3.

    Ascending key (unsigned u8), then descending score, then descending
    count, then ascending fromToPro -- fully content-determined.
    """
    prev = None
    for rec in book:
        key = int(rec["key"])  # dtype field is <u8, so already unsigned
        cur = (
            key,
            -int(rec["score"]),
            -int(rec["count"]),
            int(rec["fromToPro"]),
        )
        if prev is not None:
            assert prev <= cur, (
                f"Apery file out of order: {prev} followed by {cur} "
                f"(Requirements 12.2, 12.3)"
            )
        prev = cur
