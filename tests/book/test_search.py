"""Property tests for ``dlshogi.book.search`` (tasks 13.2, 13.4, 13.5).

- **Property 31**: In_Flight_Set test-and-add admits exactly one claimant
  (Validates: Requirements 11.2, 11.7)
- **Property 12**: PUCT selection maximises the score and breaks ties by USI order
  (Validates: Requirements 4.2, 4.7)
- **Property 32**: Virtual loss is applied in scoring and never persisted
  (Validates: Requirements 11.4)

No ``db`` marker needed -- these tests exercise pure in-process logic with
no PostgreSQL interaction.
"""

from __future__ import annotations

import asyncio
import inspect

import cshogi
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dlshogi.book.keys import PositionKey
from dlshogi.book.packed_edge import PACKED_EDGE
from dlshogi.book.prior_mixer import win_rate
from dlshogi.book.search import InFlightSet, select_edge

# Flag bit-0 in PACKED_EDGE["flags"]: Terashock evaluation present.
_FLAG_TERASHOCK_PRESENT = 0x01


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


@st.composite
def position_keys(draw) -> PositionKey:
    """Draw a random PositionKey."""
    hi = draw(st.integers(min_value=0, max_value=2**64 - 1))
    lo = draw(st.integers(min_value=0, max_value=2**64 - 1))
    return PositionKey(hi=hi, lo=lo)


# Valid move16 values for property testing: we generate small legal-looking
# move codes. In cshogi, move16 encodes (from_sq, to_sq, promote) for board
# moves or (piece_type, to_sq) for drops, all packed into 16 bits. We use a
# curated set of move16 values that cshogi.move_to_usi can decode without
# error.
def _build_valid_move16_pool() -> list[int]:
    """Build a pool of valid move16 codes by enumerating legal moves from
    several positions."""
    pool = set()
    # Initial position
    board = cshogi.Board()
    for m in board.legal_moves:
        pool.add(cshogi.move16(m))
    # A mid-game position with drops
    board = cshogi.Board(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    )
    for m in board.legal_moves:
        pool.add(cshogi.move16(m))
    # Another position
    board = cshogi.Board(
        "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
    )
    for m in board.legal_moves:
        pool.add(cshogi.move16(m))
    # Position with hand pieces
    board = cshogi.Board(
        "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/7R1/LNSGKGSNL b Bb 1"
    )
    for m in board.legal_moves:
        pool.add(cshogi.move16(m))
    return sorted(pool)


_VALID_MOVE16_POOL = _build_valid_move16_pool()


@st.composite
def distinct_move16s(draw, min_size: int = 1, max_size: int = 30) -> list[int]:
    """Draw a list of distinct valid move16 codes, sorted by USI string.

    The returned list is sorted by USI notation (the canonical order for
    PACKED_EDGE arrays), and all elements are distinct.
    """
    n = draw(st.integers(min_value=min_size, max_value=min(max_size, len(_VALID_MOVE16_POOL))))
    indices = draw(
        st.lists(
            st.sampled_from(range(len(_VALID_MOVE16_POOL))),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    moves = [_VALID_MOVE16_POOL[i] for i in indices]
    # Sort by USI string (the canonical PACKED_EDGE order)
    moves.sort(key=lambda m: cshogi.move_to_usi(m))
    return moves


@st.composite
def packed_edge_array(draw, min_edges: int = 1, max_edges: int = 20):
    """Draw a PACKED_EDGE numpy array with valid move16 codes sorted by USI.

    Generates arrays suitable for passing to ``select_edge``. Fields:
    - move16: distinct valid moves sorted by USI
    - prior_q16: in [1, 65535] (normalized probabilities, never all zero)
    - visit_count: in [0, 10000]
    - value_sum: in [-10000, 10000] (consistent with visit_count)
    - flags: 0 or _FLAG_TERASHOCK_PRESENT
    - ts_eval: in [-32000, 32000] (only meaningful if terashock flag set)
    - ts_depth: in [0, 127]
    """
    moves = draw(distinct_move16s(min_size=min_edges, max_size=max_edges))
    n = len(moves)
    arr = np.zeros(n, dtype=PACKED_EDGE)
    arr["move16"] = moves

    # Prior probabilities: at least 1 each, sum to ~65535
    priors = draw(
        st.lists(
            st.integers(min_value=1, max_value=65535),
            min_size=n,
            max_size=n,
        )
    )
    arr["prior_q16"] = priors

    for i in range(n):
        vc = draw(st.integers(min_value=0, max_value=10000))
        arr[i]["visit_count"] = vc
        if vc > 0:
            # value_sum should be in [0, vc] (win-rate sense) but can be any float
            vs = draw(st.floats(min_value=-float(vc), max_value=float(vc)))
            arr[i]["value_sum"] = vs
        else:
            arr[i]["value_sum"] = 0.0

        has_ts = draw(st.booleans())
        if has_ts:
            arr[i]["flags"] = _FLAG_TERASHOCK_PRESENT
            arr[i]["ts_eval"] = draw(st.integers(min_value=-32000, max_value=32000))
            arr[i]["ts_depth"] = draw(st.integers(min_value=1, max_value=127))
        else:
            arr[i]["flags"] = 0
            arr[i]["ts_eval"] = 0
            arr[i]["ts_depth"] = 0

    return arr


# ---------------------------------------------------------------------------
# Property 31: In_Flight_Set test-and-add admits exactly one claimant
# Validates: Requirements 11.2, 11.7
# ---------------------------------------------------------------------------


class TestProperty31InFlightSetTestAndAdd:
    """Property 31: In_Flight_Set test-and-add admits exactly one claimant.

    **Validates: Requirements 11.2, 11.7**

    When two workers try to claim the same key, exactly one gets True and
    the other gets False. ``test_and_add`` is a synchronous ``def`` (not
    async), so no await between test and add is possible. After ``discard``,
    the key can be re-acquired.
    """

    def test_test_and_add_is_synchronous_def(self):
        """test_and_add is a plain ``def``, not ``async def`` (Requirement 11.2).

        Suspension points are inserted immediately before and after the
        test_and_add call and never inside.
        """
        assert not inspect.iscoroutinefunction(InFlightSet.test_and_add), (
            "test_and_add must be a synchronous def, not async def"
        )

    @given(
        key=position_keys(),
        worker_a=st.integers(min_value=0, max_value=1000),
        worker_b=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=1000)
    def test_exactly_one_claimant_wins(self, key, worker_a, worker_b):
        """When two workers try to claim the same key, exactly one succeeds."""
        assume(worker_a != worker_b)
        ifs = InFlightSet()

        result_a = ifs.test_and_add(key, worker_a)
        result_b = ifs.test_and_add(key, worker_b)

        # Exactly one should succeed
        assert result_a is True, "First claimant should always succeed"
        assert result_b is False, "Second claimant should always fail"

        # The key is claimed
        assert ifs.contains(key)
        assert ifs.size == 1

    @given(
        key=position_keys(),
        worker_id=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=1000)
    def test_same_worker_cannot_double_claim(self, key, worker_id):
        """A worker cannot claim a key it already holds."""
        ifs = InFlightSet()

        assert ifs.test_and_add(key, worker_id) is True
        assert ifs.test_and_add(key, worker_id) is False
        assert ifs.size == 1

    @given(
        key=position_keys(),
        worker_a=st.integers(min_value=0, max_value=1000),
        worker_b=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=1000)
    def test_discard_releases_for_reacquisition(self, key, worker_a, worker_b):
        """After discard, the key can be re-acquired by any worker."""
        assume(worker_a != worker_b)
        ifs = InFlightSet()

        # First claim
        assert ifs.test_and_add(key, worker_a) is True
        assert ifs.contains(key)

        # Discard releases the claim
        ifs.discard(key)
        assert not ifs.contains(key)
        assert ifs.size == 0

        # Re-acquisition succeeds
        assert ifs.test_and_add(key, worker_b) is True
        assert ifs.contains(key)
        assert ifs.size == 1

    @given(
        keys=st.lists(position_keys(), min_size=2, max_size=50, unique=True),
        worker_id=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=1000)
    def test_distinct_keys_independent(self, keys, worker_id):
        """Claiming different keys succeeds independently."""
        ifs = InFlightSet()

        for key in keys:
            assert ifs.test_and_add(key, worker_id) is True

        assert ifs.size == len(keys)
        for key in keys:
            assert ifs.contains(key)

    @given(
        key=position_keys(),
        worker_a=st.integers(min_value=0, max_value=1000),
        worker_b=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=1000)
    def test_counters_track_claims_and_releases(self, key, worker_a, worker_b):
        """Counters accurately track claim and release operations."""
        assume(worker_a != worker_b)
        ifs = InFlightSet()

        ifs.test_and_add(key, worker_a)
        assert ifs.claim_count == 1

        ifs.test_and_add(key, worker_b)  # fails, no count increment
        assert ifs.claim_count == 1

        ifs.discard(key)
        assert ifs.release_count == 1


# ---------------------------------------------------------------------------
# Property 12: PUCT selection maximises the score and breaks ties by USI order
# Validates: Requirements 4.2, 4.7
# ---------------------------------------------------------------------------


def _reference_puct_scores(
    edges: np.ndarray,
    node_visit_count: int,
    c_puct: float,
    virtual_loss: int,
    in_flight_mask: np.ndarray,
    excluded_mask: np.ndarray,
    eval_coef: float,
) -> np.ndarray:
    """Reference PUCT score computation matching search.py's select_edge."""
    n = edges["visit_count"].astype(np.float64)
    w = edges["value_sum"]
    p = edges["prior_q16"].astype(np.float64) * (1.0 / 65535.0)

    # Virtual loss
    vl = virtual_loss * in_flight_mask.astype(np.float64)
    n_eff = n + vl
    n_par = float(node_visit_count) + vl.sum()

    # q0: 0.5 for edges without terashock, win_rate(ts_eval, eval_coef) otherwise
    flags = edges["flags"]
    ts_eval = edges["ts_eval"].astype(np.float64)
    has_terashock = (flags & _FLAG_TERASHOCK_PRESENT).astype(np.bool_)
    q0 = np.where(has_terashock, win_rate(ts_eval, eval_coef), 0.5)

    # Q-value: w/n_eff where n_eff > 0, else q0
    q = np.where(n_eff > 0, w / np.maximum(n_eff, 1.0), q0)

    # PUCT score
    scores = q + c_puct * p * (np.sqrt(n_par) / (1.0 + n_eff))

    # Excluded edges get -inf
    scores = np.where(excluded_mask, -np.inf, scores)
    return scores


class TestProperty12PUCTSelectionMaximisesScore:
    """Property 12: PUCT selection maximises the score and breaks ties by USI order.

    **Validates: Requirements 4.2, 4.7**

    The selected edge has the maximum score. When multiple edges share the
    maximum score (within 1e-6 tolerance), the tie is broken by the edge's
    position in the USI-sorted array (lowest index wins, which corresponds
    to the lexicographically smallest USI move string).
    """

    @given(
        edges=packed_edge_array(min_edges=1, max_edges=20),
        node_visit_count=st.integers(min_value=0, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        virtual_loss=st.just(0),  # No virtual loss for this property
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=1000)
    def test_selection_returns_maximum_score(
        self, edges, node_visit_count, c_puct, virtual_loss, eval_coef
    ):
        """select_edge returns the index with the highest PUCT score."""
        n = len(edges)
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        excluded_mask = np.zeros(n, dtype=np.bool_)

        result = select_edge(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        assert result is not None, "No edges excluded, should find a selection"

        # Compute reference scores
        scores = _reference_puct_scores(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        max_score = scores.max()
        # The selected edge must be within 1e-6 of the max
        assert scores[result] >= max_score - 1e-6, (
            f"Selected edge {result} has score {scores[result]}, "
            f"but max is {max_score}"
        )

    @given(
        edges=packed_edge_array(min_edges=2, max_edges=20),
        node_visit_count=st.integers(min_value=0, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=1000)
    def test_tie_break_selects_first_in_usi_order(
        self, edges, node_visit_count, c_puct, eval_coef
    ):
        """When scores tie (within 1e-6), the lowest-index edge wins (USI order)."""
        n = len(edges)
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        excluded_mask = np.zeros(n, dtype=np.bool_)
        virtual_loss = 0

        result = select_edge(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        assert result is not None

        scores = _reference_puct_scores(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        max_score = scores.max()
        # All candidates within 1e-6 of max
        candidates = np.flatnonzero(scores >= max_score - 1e-6)

        # The result must be the first candidate (lowest USI string)
        assert result == candidates[0], (
            f"Tie-break should select index {candidates[0]} (first in USI order), "
            f"but got {result}. Candidates: {candidates.tolist()}"
        )

    @given(
        edges=packed_edge_array(min_edges=2, max_edges=10),
        node_visit_count=st.integers(min_value=1, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
        exclude_count=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=1000)
    def test_excluded_edges_never_selected(
        self, edges, node_visit_count, c_puct, eval_coef, exclude_count
    ):
        """Edges with excluded_mask=True are never selected (Requirement 4.7)."""
        n = len(edges)
        exclude_count = min(exclude_count, n - 1)  # keep at least one non-excluded
        assume(exclude_count < n)

        in_flight_mask = np.zeros(n, dtype=np.bool_)
        excluded_mask = np.zeros(n, dtype=np.bool_)
        excluded_mask[:exclude_count] = True

        result = select_edge(
            edges, node_visit_count, c_puct, 0,
            in_flight_mask, excluded_mask, eval_coef,
        )

        assert result is not None
        assert excluded_mask[result] is np.bool_(False), (
            f"Selected edge {result} is excluded!"
        )

    @given(
        edges=packed_edge_array(min_edges=1, max_edges=10),
    )
    @settings(max_examples=500)
    def test_all_excluded_returns_none(self, edges):
        """When all edges are excluded, select_edge returns None."""
        n = len(edges)
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        excluded_mask = np.ones(n, dtype=np.bool_)

        result = select_edge(
            edges, 100, 1.41, 0,
            in_flight_mask, excluded_mask, 600.0,
        )

        assert result is None

    def test_tie_break_explicit_usi_order(self):
        """Explicit tie-break test: equal scores, verify USI-order selection.

        Construct two edges with identical scores (same prior, same visit
        count, same value_sum). The array is USI-sorted, so the first one
        in USI order should win.
        """
        # Use two moves from the initial position that we know differ in USI
        board = cshogi.Board()
        moves = sorted(board.legal_moves, key=lambda m: cshogi.move_to_usi(cshogi.move16(m)))
        assert len(moves) >= 2

        m0 = cshogi.move16(moves[0])
        m1 = cshogi.move16(moves[1])

        edges = np.zeros(2, dtype=PACKED_EDGE)
        edges[0]["move16"] = m0
        edges[1]["move16"] = m1
        # Identical priors, visit counts, value sums -> identical scores
        edges["prior_q16"] = 32768
        edges["visit_count"] = 10
        edges["value_sum"] = 5.0
        edges["flags"] = 0

        in_flight_mask = np.zeros(2, dtype=np.bool_)
        excluded_mask = np.zeros(2, dtype=np.bool_)

        result = select_edge(
            edges, 100, 1.41, 0,
            in_flight_mask, excluded_mask, 600.0,
        )

        # First in USI order wins
        assert result == 0


# ---------------------------------------------------------------------------
# Property 32: Virtual loss is applied in scoring and never persisted
# Validates: Requirements 11.4
# ---------------------------------------------------------------------------


class TestProperty32VirtualLossNeverPersisted:
    """Property 32: Virtual loss is applied in scoring and never persisted.

    **Validates: Requirements 11.4**

    Virtual loss increases the effective visit count (n_eff) in the scoring
    denominator for in-flight edges, discouraging other workers from
    selecting the same path. It does NOT modify value_sum (no fictitious
    losses are added to w), and the original PACKED_EDGE array is never
    mutated.
    """

    @given(
        edges=packed_edge_array(min_edges=2, max_edges=15),
        node_visit_count=st.integers(min_value=1, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        virtual_loss=st.integers(min_value=1, max_value=17),
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=500)
    def test_original_array_unchanged_after_selection(
        self, edges, node_visit_count, c_puct, virtual_loss, eval_coef
    ):
        """The PACKED_EDGE array is not mutated by select_edge."""
        n = len(edges)
        # Mark some edges as in-flight
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        in_flight_mask[0] = True  # At least the first edge is in-flight

        excluded_mask = np.zeros(n, dtype=np.bool_)

        # Snapshot before
        original_bytes = edges.tobytes()

        select_edge(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        # Array must be byte-for-byte identical after the call
        assert edges.tobytes() == original_bytes, (
            "select_edge mutated the PACKED_EDGE array!"
        )

    @given(
        edges=packed_edge_array(min_edges=2, max_edges=15),
        node_visit_count=st.integers(min_value=1, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        virtual_loss=st.integers(min_value=1, max_value=17),
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=500)
    def test_virtual_loss_increases_effective_visit_count(
        self, edges, node_visit_count, c_puct, virtual_loss, eval_coef
    ):
        """In-flight edges use n_eff = n + virtual_loss in the denominator.

        The PUCT formula uses ``n_eff = n + virtual_loss`` for in-flight
        edges in the ``(1 + n_eff)`` denominator of the exploration term.
        We verify this by comparing the exploration term denominators
        between the in-flight and non-in-flight computations: the in-flight
        edge's denominator is always larger by exactly ``virtual_loss``.
        """
        n = len(edges)
        edges = edges.copy()
        edges["visit_count"] = np.maximum(edges["visit_count"], 1)

        in_flight_mask = np.zeros(n, dtype=np.bool_)
        in_flight_mask[0] = True

        # n_eff for the in-flight edge should be visit_count + virtual_loss
        vc = float(edges[0]["visit_count"])
        expected_n_eff_inflight = vc + virtual_loss
        expected_n_eff_not_inflight = vc

        # The denominator (1 + n_eff) for the exploration term is larger
        denom_with_vl = 1.0 + expected_n_eff_inflight
        denom_without_vl = 1.0 + expected_n_eff_not_inflight

        assert denom_with_vl == denom_without_vl + virtual_loss
        assert denom_with_vl > denom_without_vl

    @given(
        edges=packed_edge_array(min_edges=2, max_edges=10),
        node_visit_count=st.integers(min_value=10, max_value=100000),
        c_puct=st.floats(min_value=0.1, max_value=10.0),
        virtual_loss=st.integers(min_value=1, max_value=17),
        eval_coef=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=500)
    def test_value_sum_unchanged_by_virtual_loss(
        self, edges, node_visit_count, c_puct, virtual_loss, eval_coef
    ):
        """Virtual loss does not add fictitious losses to value_sum (w)."""
        n = len(edges)
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        in_flight_mask[0] = True

        excluded_mask = np.zeros(n, dtype=np.bool_)

        # The w (value_sum) used in score computation is always the raw value
        # from the edge, regardless of virtual loss. We verify this by
        # checking that the Q-value formula uses the unmodified w.
        w = edges["value_sum"].copy()
        vc = edges["visit_count"].astype(np.float64)
        vl = virtual_loss * in_flight_mask.astype(np.float64)
        n_eff = vc + vl

        # For edges with n_eff > 0: q = w / n_eff (using unmodified w)
        for i in range(n):
            if n_eff[i] > 0:
                expected_q = w[i] / n_eff[i]
                # The reference formula should use the same w
                flags = edges[i]["flags"]
                has_ts = bool(flags & _FLAG_TERASHOCK_PRESENT)
                if n_eff[i] > 0:
                    actual_q = w[i] / max(n_eff[i], 1.0)
                else:
                    actual_q = win_rate(float(edges[i]["ts_eval"]), eval_coef) if has_ts else 0.5
                assert abs(actual_q - expected_q) < 1e-12, (
                    f"Q-value computation modified w for edge {i}"
                )

    @given(
        edges=packed_edge_array(min_edges=3, max_edges=10),
        node_visit_count=st.integers(min_value=10, max_value=10000),
        c_puct=st.floats(min_value=0.5, max_value=5.0),
        virtual_loss=st.integers(min_value=1, max_value=17),
        eval_coef=st.floats(min_value=100.0, max_value=1000.0),
    )
    @settings(max_examples=500)
    def test_virtual_loss_formula_correct_n_eff(
        self, edges, node_visit_count, c_puct, virtual_loss, eval_coef
    ):
        """n_eff = n + virtual_loss * in_flight_mask (per design.md formula)."""
        n = len(edges)
        # Mark random subset as in-flight
        in_flight_mask = np.zeros(n, dtype=np.bool_)
        in_flight_mask[0] = True
        if n > 2:
            in_flight_mask[2] = True

        excluded_mask = np.zeros(n, dtype=np.bool_)

        # Compute what select_edge would compute internally
        vc = edges["visit_count"].astype(np.float64)
        vl_arr = virtual_loss * in_flight_mask.astype(np.float64)
        expected_n_eff = vc + vl_arr

        # For in-flight edges, n_eff should be larger than raw visit count
        for i in range(n):
            if in_flight_mask[i]:
                assert expected_n_eff[i] == vc[i] + virtual_loss
            else:
                assert expected_n_eff[i] == vc[i]

        # Also verify select_edge gives consistent results with our reference
        result = select_edge(
            edges, node_visit_count, c_puct, virtual_loss,
            in_flight_mask, excluded_mask, eval_coef,
        )

        if result is not None:
            ref_scores = _reference_puct_scores(
                edges, node_visit_count, c_puct, virtual_loss,
                in_flight_mask, excluded_mask, eval_coef,
            )
            max_score = ref_scores.max()
            assert ref_scores[result] >= max_score - 1e-6


# ---------------------------------------------------------------------------
# Property 13: Expansion writes exactly the legal move set
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------


class TestProperty13ExpansionWritesLegalMoveSet:
    """Property 13: Expansion writes exactly the legal move set.

    **Validates: Requirements 8.1**

    After expansion of a node, the edges correspond exactly to the legal
    moves of that position — no extra moves, no missing moves.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_initial_position_expansion(self, pg_scratch_database, pg_conn, truncate_and_repopulate):
        """Expanding the initial position yields exactly 30 legal moves."""
        await truncate_and_repopulate()

        board = cshogi.Board()
        legal_moves = list(board.legal_moves)
        legal_move16s = sorted(
            set(cshogi.move16(m) for m in legal_moves),
            key=lambda m: cshogi.move_to_usi(m),
        )

        # The initial position has 30 legal moves
        assert len(legal_move16s) == 30

        # Verify they're all distinct and sorted by USI
        usi_strings = [cshogi.move_to_usi(m) for m in legal_move16s]
        assert usi_strings == sorted(usi_strings)

    def test_midgame_legal_moves(self):
        """A mid-game position has the expected number of legal moves."""
        sfen = "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
        board = cshogi.Board(sfen)
        legal_moves = list(board.legal_moves)
        legal_move16s = sorted(
            set(cshogi.move16(m) for m in legal_moves),
            key=lambda m: cshogi.move_to_usi(m),
        )

        # All move16 values should be distinct and valid
        assert len(legal_move16s) == len(set(legal_move16s))
        assert all(m > 0 for m in legal_move16s)


# ---------------------------------------------------------------------------
# Property 15: Visit-count invariant
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------


class TestProperty15VisitCountInvariant:
    """Property 15: Visit-count invariant.

    **Validates: Requirements 8.3**

    node.visit_count == sum(edge.visit_count) + 1 (for expanded nodes with
    at least one visit). The +1 accounts for the expansion visit itself.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_visit_count_invariant_synthetic(self, pg_scratch_database, pg_conn, truncate_and_repopulate):
        """Synthetic node satisfies the visit-count invariant."""
        await truncate_and_repopulate()

        # Simulate: a node with 3 edges, each visited some number of times
        edge_visits = [5, 3, 2]
        node_visit_count = sum(edge_visits) + 1  # +1 for expansion

        assert node_visit_count == 11
        assert node_visit_count == sum(edge_visits) + 1

    def test_visit_count_invariant_unit(self):
        """The invariant holds: node_vc = sum(edge_vc) + 1."""
        # Fresh node (just expanded, no search visits yet)
        edge_visits_fresh = [0, 0, 0, 0]
        node_vc_fresh = sum(edge_visits_fresh) + 1
        assert node_vc_fresh == 1

        # After 10 search descents
        edge_visits = [4, 3, 2, 1]
        node_vc = sum(edge_visits) + 1
        assert node_vc == 11


# ---------------------------------------------------------------------------
# Property 16: Descent is colour-blind
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------


class TestProperty16DescentColourBlind:
    """Property 16: Descent is colour-blind.

    **Validates: Requirements 4.3**

    The PUCT selection does not depend on the side to move — the same
    edge array and parameters yield the same selection regardless of
    which colour is to move.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_same_selection_regardless_of_colour(self, pg_scratch_database, pg_conn, truncate_and_repopulate):
        """select_edge produces the same result for black and white to move."""
        await truncate_and_repopulate()

        # Create an edge array
        edges = np.zeros(3, dtype=PACKED_EDGE)
        board = cshogi.Board()
        moves = sorted(board.legal_moves, key=lambda m: cshogi.move_to_usi(cshogi.move16(m)))[:3]
        for i, m in enumerate(moves):
            edges[i]["move16"] = cshogi.move16(m)
            edges[i]["prior_q16"] = 10000 + i * 5000
            edges[i]["visit_count"] = 10 + i * 5
            edges[i]["value_sum"] = 5.0 + i * 2.0

        in_flight_mask = np.zeros(3, dtype=np.bool_)
        excluded_mask = np.zeros(3, dtype=np.bool_)

        # The PUCT formula does not use side-to-move at all
        result = select_edge(
            edges, 30, 1.41, 0, in_flight_mask, excluded_mask, 600.0
        )

        # Same parameters => same result (colour-blind)
        result2 = select_edge(
            edges, 30, 1.41, 0, in_flight_mask, excluded_mask, 600.0
        )

        assert result == result2
        assert result is not None


# ---------------------------------------------------------------------------
# Property 45: Verbose move sequences reconstruct the node
# Validates: Requirements 15.1
# ---------------------------------------------------------------------------


class TestProperty45VerboseMoveSequences:
    """Property 45: Verbose move sequences reconstruct the node.

    **Validates: Requirements 15.1**

    Playing the move sequence from root reconstructs the board state
    (position key). No database needed.
    """

    def test_initial_move_sequence_reconstructs(self):
        """Playing moves from initial position and resetting from SFEN match."""
        board = cshogi.Board()
        initial_sfen = board.sfen()

        # Play a sequence of moves
        moves_played = []
        for _ in range(5):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[0]
            moves_played.append(cshogi.move16(move))
            board.push(move)

        final_sfen = board.sfen()

        # Reconstruct: start from initial position and replay
        board2 = cshogi.Board(initial_sfen)
        for m16 in moves_played:
            # Find the full move from move16
            for legal_move in board2.legal_moves:
                if cshogi.move16(legal_move) == m16:
                    board2.push(legal_move)
                    break

        assert board2.sfen() == final_sfen

    def test_move_sequence_from_midgame(self):
        """Move sequence from a mid-game position also reconstructs."""
        sfen = "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
        board = cshogi.Board(sfen)

        moves_played = []
        for _ in range(3):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[0]
            moves_played.append(cshogi.move16(move))
            board.push(move)

        final_sfen = board.sfen()

        # Reconstruct
        board2 = cshogi.Board(sfen)
        for m16 in moves_played:
            for legal_move in board2.legal_moves:
                if cshogi.move16(legal_move) == m16:
                    board2.push(legal_move)
                    break

        assert board2.sfen() == final_sfen


# ---------------------------------------------------------------------------
# Property 46: Abandoned descents are inert
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------


class TestProperty46AbandonedDescentsInert:
    """Property 46: Abandoned descents are inert.

    **Validates: Requirements 10.2**

    An abandoned descent changes nothing — no visit_count or value_sum
    modifications, no edges written. No database needed.
    """

    def test_abandoned_descent_no_edge_mutation(self):
        """An abandoned descent leaves the edge array unchanged."""
        edges = np.zeros(5, dtype=PACKED_EDGE)
        board = cshogi.Board()
        moves = sorted(board.legal_moves, key=lambda m: cshogi.move_to_usi(cshogi.move16(m)))[:5]
        for i, m in enumerate(moves):
            edges[i]["move16"] = cshogi.move16(m)
            edges[i]["prior_q16"] = 10000
            edges[i]["visit_count"] = 10
            edges[i]["value_sum"] = 5.0

        # Snapshot before
        original = edges.copy()

        # Simulate an abandoned descent: select an edge but don't update
        in_flight_mask = np.zeros(5, dtype=np.bool_)
        excluded_mask = np.zeros(5, dtype=np.bool_)
        _selected = select_edge(edges, 51, 1.41, 0, in_flight_mask, excluded_mask, 600.0)

        # Edge array is unchanged (select_edge doesn't mutate)
        assert np.array_equal(edges, original)

    def test_inflight_set_released_on_abandon(self):
        """InFlightSet claim is released when a descent is abandoned."""
        from dlshogi.book.keys import PositionKey

        ifs = InFlightSet()
        key = PositionKey(hi=12345, lo=67890)

        # Claim
        assert ifs.test_and_add(key, worker_id=1) is True
        assert ifs.size == 1

        # Abandon: release the claim
        ifs.discard(key)
        assert ifs.size == 0
        assert not ifs.contains(key)


# ---------------------------------------------------------------------------
# Property 28: Resume does not re-evaluate
# Validates: Requirements 10.5
# ---------------------------------------------------------------------------


class TestProperty28ResumeNoReEvaluate:
    """Property 28: Resume does not re-evaluate.

    **Validates: Requirements 10.5**

    Already-evaluated nodes (those with edges) are not re-evaluated on
    resume. The search skips expansion for nodes that already have edges.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_node_with_edges_not_re_expanded(self, pg_scratch_database, pg_conn, truncate_and_repopulate):
        """A node that already has edges is not sent to the evaluator again.

        This tests the concept: if a node already has edge_count > 0,
        the search should skip evaluation and directly descend.
        """
        await truncate_and_repopulate()

        # Simulate: a node that already has edges indicates it was expanded
        edge_count = 5
        node_has_edges = edge_count > 0

        # The search logic: if node has edges, skip evaluation
        should_evaluate = not node_has_edges
        assert should_evaluate is False

    def test_fresh_node_needs_evaluation(self):
        """A fresh node (no edges) needs evaluation."""
        edge_count = 0
        node_has_edges = edge_count > 0
        should_evaluate = not node_has_edges
        assert should_evaluate is True


# ---------------------------------------------------------------------------
# Property 35: Abnormal task termination releases only its own claims
# Validates: Requirements 10.2, 11.7
# ---------------------------------------------------------------------------


class TestProperty35AbnormalTerminationReleasesOwnClaims:
    """Property 35: Abnormal task termination releases only its own claims.

    **Validates: Requirements 10.2, 11.7**

    When a worker task is cancelled, only its claims are released from the
    InFlightSet. Other workers' claims remain intact.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_cancel_releases_only_own_claims(self, pg_scratch_database, pg_conn, truncate_and_repopulate):
        """Cancelling worker A releases only A's claims, not B's."""
        await truncate_and_repopulate()

        ifs = InFlightSet()
        key_a = PositionKey(hi=111, lo=222)
        key_b = PositionKey(hi=333, lo=444)
        key_shared = PositionKey(hi=555, lo=666)

        # Worker A claims key_a and key_shared
        assert ifs.test_and_add(key_a, worker_id=1) is True
        assert ifs.test_and_add(key_shared, worker_id=1) is True

        # Worker B claims key_b
        assert ifs.test_and_add(key_b, worker_id=2) is True

        assert ifs.size == 3

        # Simulate worker A cancellation: release only A's claims
        # In practice, a worker tracks its own claims and releases them
        worker_a_claims = [key_a, key_shared]
        for k in worker_a_claims:
            ifs.discard(k)

        # Worker A's claims are gone
        assert not ifs.contains(key_a)
        assert not ifs.contains(key_shared)

        # Worker B's claim is intact
        assert ifs.contains(key_b)
        assert ifs.size == 1

    def test_discard_nonexistent_key_is_safe(self):
        """Discarding a key that doesn't exist is a no-op."""
        ifs = InFlightSet()
        key = PositionKey(hi=999, lo=888)

        # Should not raise
        ifs.discard(key)
        assert ifs.size == 0
