"""Colour-swap fixture for the Property 16 colour-blindness metamorphic test.

The transform is cshogi's canonical 180-degree board rotation:

* ``cshogi.rotate_sfen(sfen)`` -- the colour-swapped image of a position: the
  board is mirrored across both axes, every piece flips side, the two hands are
  exchanged, and the turn is swapped. It is an involution (rotating twice is
  the identity) whose fixed points are symmetric positions.
* ``cshogi.move_rotate(m16)`` -- the corresponding move in move16 space.

The defining property (verified in the self-check below): for every legal
position ``P``, ``{move_rotate(m) for m in legal_moves(P)}`` equals
``legal_moves(rotate_sfen(P))``. Because the descent edge selection is a pure
function of the side-independent decision fields (visit_count, value_sum,
prior) over a USI-ordered buffer, a colour-blind implementation selects, on the
rotated image, exactly the rotated version of the edge it selected on ``P``.
"""
from __future__ import annotations

import cshogi


def rotate_sfen(sfen: str) -> str:
    """The colour-swapped (180-degree rotated) image of ``sfen``."""
    return cshogi.rotate_sfen(sfen)


def rotate_move16(m16: int) -> int:
    """The colour-swapped (180-degree rotated) version of a move16."""
    return cshogi.move_rotate(m16)


def legal_move16(sfen: str) -> frozenset[int]:
    """The set of legal move16 from ``sfen`` (empty if unparseable)."""
    try:
        board = cshogi.Board(sfen)
    except Exception:
        return frozenset()
    if not board.is_ok():
        return frozenset()
    return frozenset(cshogi.move16(m) for m in board.legal_moves)


def legal_moves_correspond(sfen: str) -> bool:
    """Return True iff the legal-move set of ``sfen`` maps, via
    :func:`rotate_move16`, onto the legal-move set of its rotated image."""
    left = {rotate_move16(m) for m in legal_move16(sfen)}
    return left == legal_move16(rotate_sfen(sfen))


if __name__ == "__main__":
    import random

    random.seed(1234)
    board = cshogi.Board()
    checked = 0
    for _ in range(40):
        sfen = board.sfen()
        if not legal_moves_correspond(sfen):
            print(f"MISMATCH: {sfen} -> {cshogi.rotate_sfen(sfen)}")
            raise SystemExit(1)
        checked += 1
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push_usi(cshogi.move_to_usi(random.choice(moves)))
    print(f"OK: {checked} positions; rotate_sfen + move_rotate correspondence holds.")
