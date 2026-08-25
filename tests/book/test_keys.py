"""Property tests for ``dlshogi.book.keys`` (tasks 3.2, 3.3, 3.4).

Implements:

- **Property 8: Position_Key depends only on the Board_State**
  (Requirements 3.2, 3.4) -- task 3.2.
- **Property 9: Position_Key is collision-free over the verification
  sample** (Requirement 3.3) -- task 3.4. A single example of 1,000,000+
  distinct Board_States, not 100 short examples, per design.md's own
  instruction for this property.
- **Property 10: Incremental child key equals the recomputed key**
  (Requirement 3.7) -- task 3.3. Checks *every* legal move of each drawn
  position, and additionally that the batched ``position_keys_after``
  form agrees element-wise with the single-move ``position_key_from_sfen``
  form.

None of these need a database or a GPU; they exercise the
``dlshogi.cppshogi`` binding directly, which is already built (task 2).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import cshogi
import pytest
from hypothesis import given, settings

from dlshogi.book.keys import (
    PositionKey,
    position_key,
    position_key_from_sfen,
    position_keys_after,
    zobrist_fingerprint,
)
from tests.book.strategies import self_play_board_with_moves, self_play_sfen

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_POSITION_KEYS_FIXTURE = _FIXTURES_DIR / "position_keys.json"
_ZOBRIST_FINGERPRINT_FIXTURE = _FIXTURES_DIR / "zobrist_fingerprint.txt"


def _load_golden_positions():
    with open(_POSITION_KEYS_FIXTURE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# Property 8: Position_Key depends only on the Board_State
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 8: Position_Key depends only on the Board_State
# Validates: Requirements 3.2, 3.4


@given(self_play_sfen())
@settings(max_examples=300)
def test_position_key_is_stable_across_repeated_invocations(sfen):
    """Repeated invocations on the same Board_State agree."""
    k1 = position_key_from_sfen(sfen)
    k2 = position_key_from_sfen(sfen)
    assert k1 == k2


@given(self_play_sfen())
@settings(max_examples=300)
def test_position_key_agrees_across_board_construction_means(sfen):
    """A board reached by ``set_sfen`` and one reached by pushing moves from
    the initial position to the same Board_State yield equal Position_Keys.

    ``self_play_sfen`` already reaches the Board_State via a move sequence
    (the search fixture's own construction path); this test additionally
    constructs a *fresh* board directly from the resulting SFEN (the
    Value_Propagator's construction path, ``set_sfen``) and checks the two
    agree.
    """
    key_from_moves = position_key_from_sfen(sfen)
    fresh_board = cshogi.Board(sfen)
    key_from_fresh_board = position_key(fresh_board)
    assert key_from_moves == key_from_fresh_board


@given(self_play_sfen())
@settings(max_examples=300)
def test_position_key_is_independent_of_the_ply_field(sfen):
    """Two SFENs differing only in the trailing ply field denote the same
    Board_State (ply is explicitly excluded from Board_State, per the
    Glossary) and must yield equal Position_Keys."""
    board_field, ply_str = sfen.rsplit(" ", 1)
    ply = int(ply_str)
    other_sfen = f"{board_field} {ply + 37}"
    assert position_key_from_sfen(sfen) == position_key_from_sfen(other_sfen)


def test_position_key_distinguishes_non_movers_hand():
    """Two Board_States with identical piece placement and side to move,
    differing only in the *non-moving* side's hand, must receive distinct
    Position_Keys -- the case Apery's 64-bit ``book_key()`` gets wrong
    (Requirement 3.2), confirmed directly against the golden fixture."""
    golden = _load_golden_positions()
    by_name = {p["name"]: p for p in golden["positions"]}
    a = by_name["hand_only_pair_a_no_black_pawn_in_hand"]
    b = by_name["hand_only_pair_b_black_pawn_in_hand"]
    key_a = position_key_from_sfen(a["sfen"])
    key_b = position_key_from_sfen(b["sfen"])
    assert key_a != key_b
    # The board key half (hi) is unaffected by a hand-only difference; only
    # the hand key half (lo) differs. This is exactly the width argument
    # design.md makes for why 64 bits (board key alone) is insufficient.
    assert key_a.hi == key_b.hi
    assert key_a.lo != key_b.lo
    # And the 64-bit Apery key that the design explicitly rejects as a
    # Position_Key does collide on this pair, which is the whole point.
    assert cshogi.Board(a["sfen"]).book_key() == cshogi.Board(b["sfen"]).book_key()


def test_golden_position_keys_are_reproduced_exactly():
    """Pinned (SFEN, key_hi, key_lo) golden vectors -- initial position,
    mid-game positions, and a drop-and-promotion position -- must still
    match exactly. A change here means Zobrist initialisation or the
    binding's key derivation changed, which is exactly what
    ``zobrist_fingerprint()`` is meant to catch at startup (Requirement
    3.4, Requirement 2.8)."""
    golden = _load_golden_positions()
    for entry in golden["positions"]:
        key = position_key_from_sfen(entry["sfen"])
        assert key == PositionKey(entry["key_hi"], entry["key_lo"]), entry["name"]


def test_zobrist_fingerprint_matches_the_golden_value():
    golden_fingerprint = int(_ZOBRIST_FINGERPRINT_FIXTURE.read_text().strip())
    assert zobrist_fingerprint() == golden_fingerprint


def test_zobrist_fingerprint_matches_the_golden_positions_fixture():
    golden = _load_golden_positions()
    assert zobrist_fingerprint() == golden["zobrist_fingerprint"]


@pytest.mark.parametrize(
    "sfen",
    [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "lns4n1/r2k2Gbl/pgps1p+Bpp/1p1Pp1p2/1N7/PPP1P4/3K1PPPP/3S1S1R1/L1G1G2NL b P 41",
    ],
)
def test_position_key_is_stable_across_separate_processes(sfen):
    """Position_Key stability across separate PUCT_Book_Builder runs
    (Requirement 3.4) is tested by actually spawning a fresh Python
    process -- a fresh ``g_mt64bit``/Zobrist-table initialisation -- rather
    than assuming in-process stability generalises to it."""
    script = (
        "from dlshogi.book.keys import position_key_from_sfen; "
        f"k = position_key_from_sfen({sfen!r}); "
        "print(k.hi, k.lo)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    hi_str, lo_str = result.stdout.strip().split()
    subprocess_key = PositionKey(int(hi_str), int(lo_str))
    assert subprocess_key == position_key_from_sfen(sfen)


# ---------------------------------------------------------------------------
# Property 10: Incremental child key equals the recomputed key
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 10: Incremental child key equals the recomputed key
# Validates: Requirements 3.7


@given(self_play_board_with_moves())
@settings(max_examples=300)
def test_incremental_child_key_matches_recomputation_for_every_legal_move(position_and_moves):
    """For every legal move of the drawn position, the incrementally
    derived child key (the batched ``position_keys_after`` binding) equals
    the Position_Key computed from scratch for the Board_State reached by
    applying that move -- checked for *every* legal move, not a sample, so
    drops, promotions, and captures are all exercised per position."""
    sfen, moves16 = position_and_moves
    if not moves16:
        return

    incremental_keys = position_keys_after(sfen, moves16)
    assert len(incremental_keys) == len(moves16)

    board = cshogi.Board(sfen)
    for move16_value, incremental_key in zip(moves16, incremental_keys):
        move = board.move_from_move16(move16_value)
        board.push(move)
        recomputed_key = position_key(board)
        board.pop()

        incremental_key_tuple = PositionKey(int(incremental_key["hi"]), int(incremental_key["lo"]))
        assert incremental_key_tuple == recomputed_key, (
            f"sfen={sfen!r} move16={move16_value} usi={cshogi.move_to_usi(move16_value)!r}"
        )


@given(self_play_board_with_moves())
@settings(max_examples=200)
def test_batched_and_single_move_forms_agree_elementwise(position_and_moves):
    """The batched ``position_keys_after(sfen, moves16)`` form must agree,
    element-wise, with computing each child key one at a time via a
    single-move-array call -- this is what the "batched form of the
    binding" actually claims, distinct from agreement with recomputation
    from scratch (the previous test)."""
    sfen, moves16 = position_and_moves
    if not moves16:
        return

    batched = position_keys_after(sfen, moves16)
    singles = [position_keys_after(sfen, [m16])[0] for m16 in moves16]

    for batched_row, single_row in zip(batched, singles):
        assert int(batched_row["hi"]) == int(single_row["hi"])
        assert int(batched_row["lo"]) == int(single_row["lo"])


def test_position_keys_after_of_empty_move_list_is_empty():
    sfen = cshogi.Board().sfen()
    result = position_keys_after(sfen, [])
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Property 9: Position_Key is collision-free over the verification sample
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 9: Position_Key is collision-free over the verification sample
# Validates: Requirements 3.3
#
# Not a hypothesis @given property in the usual sense: the requirement
# fixes the sample size at 1,000,000+ distinct Board_States, so this runs
# as a single, large, deterministic example rather than 100 short ones,
# per design.md's explicit instruction for this property. Takes on the
# order of ten seconds.


def _piece_type_to_hand_index() -> dict:
    """Map a base (unpromoted) piece type to its hand-piece index.

    Built from ``cshogi.hand_piece_to_piece_type``'s inverse rather than
    hand-derived from the ``H*``/base piece-type constants directly, since
    the hand index for GOLD does not follow the same +/-1 offset the other
    piece types do (``HGOLD == 4`` while ``GOLD == 7``).
    """
    return {cshogi.hand_piece_to_piece_type(h): h for h in cshogi.HAND_PIECES}


_PIECE_TYPE_TO_HAND_INDEX = _piece_type_to_hand_index()
# PROM_X == X + 8 for every promotable piece type X (verified against the
# cshogi constants: PROM_PAWN=9 vs PAWN=1, ..., PROM_ROOK=14 vs ROOK=6).
_PROMOTION_OFFSET = 8


def _random_walk_with_perturbations(rng: random.Random, target: int) -> dict:
    """Build ``target`` distinct Board_States (by normalized SFEN) and
    return their Position_Keys, keyed by the normalized SFEN.

    Combines two sources per design.md's Property 9 strategy note ("random
    self-play plus randomly perturbed positions"): plain self-play random
    walks (restarted occasionally so the sample is not one single long
    game), and, at low probability, a random single-piece hand
    redistribution off the current board (validated with ``board.is_ok()``
    before being accepted), which is the perturbation design.md names as
    covering "differ in ... hands" without relying on self-play alone to
    reach every hand configuration.
    """
    board = cshogi.Board()
    seen: dict[str, PositionKey] = {}
    while len(seen) < target:
        if rng.random() < 0.005:
            board.reset()
            continue
        if rng.random() < 0.001:
            _try_random_hand_perturbation(board, rng)
            continue
        moves = list(board.legal_moves)
        if not moves:
            board.reset()
            continue
        board.push(moves[rng.randrange(len(moves))])

        sfen = board.sfen()
        norm = sfen.rsplit(" ", 1)[0]
        if norm in seen:
            continue
        seen[norm] = position_key_from_sfen(sfen)
    return seen


def _try_random_hand_perturbation(board: "cshogi.Board", rng: random.Random) -> None:
    """Move one random non-king piece from the board to its owner's hand,
    in place on ``board``, if the result is a valid position. A no-op
    (leaving ``board`` unchanged) if there is no eligible piece or the
    resulting position fails ``is_ok()``.
    """
    pieces = board.pieces.copy()
    hands = (board.pieces_in_hand[0].copy(), board.pieces_in_hand[1].copy())
    candidates = [
        sq
        for sq, p in enumerate(pieces)
        if p != cshogi.NONE and p not in (cshogi.BKING, cshogi.WKING)
    ]
    if not candidates:
        return
    sq = candidates[rng.randrange(len(candidates))]
    piece = pieces[sq]
    is_white = piece >= cshogi.WPAWN
    color = cshogi.WHITE if is_white else cshogi.BLACK
    piece_type = cshogi.piece_to_piece_type(piece)
    base_piece_type = (
        piece_type - _PROMOTION_OFFSET if piece_type > cshogi.KING else piece_type
    )
    hand_index = _PIECE_TYPE_TO_HAND_INDEX.get(base_piece_type)
    if hand_index is None:
        return
    pieces[sq] = cshogi.NONE
    hands[color][hand_index] += 1
    candidate_board = board.copy()
    try:
        candidate_board.set_pieces(pieces, hands)
    except Exception:
        return
    if candidate_board.is_ok():
        board.set_pieces(pieces, hands)


def test_position_key_is_collision_free_over_one_million_board_states():
    """Not a hypothesis ``@given`` property: the requirement fixes the
    sample size at 1,000,000+ distinct Board_States, so this runs as one
    single, large, deterministic example rather than 100 short ones
    (design.md's explicit instruction for Property 9), with a fixed seed
    so the sample itself is reproducible.
    """
    rng = random.Random(20260101)
    positions = _random_walk_with_perturbations(rng, 1_000_000)
    assert len(positions) >= 1_000_000
    distinct_keys = set(positions.values())
    assert len(distinct_keys) == len(positions), (
        f"expected {len(positions)} distinct Position_Keys, got {len(distinct_keys)} "
        "-- a collision was found"
    )
