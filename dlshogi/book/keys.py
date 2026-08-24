"""Position_Key_Function: 128-bit position identity for Book_Node lookup.

A Position_Key is the pair ``(hi, lo)`` where ``hi`` is Apery's board key
(``Position::getBoardKey()``, which XORs in ``zobTurn_ == 1`` when White is
to move) and ``lo`` is the hand key (``Position::getHandKey()``). Together
they cover piece placement, both hands, and the side to move, and exclude
ply, which is exactly the Board_State definition (Requirement 3.2).

The values themselves are produced by the ``dlshogi.cppshogi`` Cython
binding added for this feature (task 2), which wraps
``Position::getBoardKey()``, ``getHandKey()``, and
``getKeyAndBoardKeyAfter()`` in `cppshogi/python_module.cpp`. See
design.md's "Position_Key from Python, and the binding prerequisite"
section for the full derivation and the reasons `Board.book_key()` and
`Board.zobrist_hash()` cannot serve as the Position_Key.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dlshogi import cppshogi

# 128-bit Position_Key as a numpy structured dtype: 8 bytes of board key
# ("hi") followed by 8 bytes of hand key ("lo"), both little-endian
# unsigned 64-bit. This is the dtype the Cython binding's buffers are typed
# as, and the dtype used for a Book_Node's PostgreSQL bigint columns
# key_hi/key_lo once folded (see fold_u64_to_i64 below).
POSITION_KEY = np.dtype([("hi", "<u8"), ("lo", "<u8")])
assert POSITION_KEY.itemsize == 16


class PositionKey(NamedTuple):
    """A single Position_Key as a plain Python value.

    ``hi`` is the board key (piece placement plus the side-to-move bit at
    bit 0), ``lo`` is the hand key. Both are unsigned 64-bit integers in
    [0, 2**64 - 1].
    """

    hi: int
    lo: int


def position_key(board) -> PositionKey:
    """Return the Position_Key of a cshogi ``Board``'s current position.

    Uses ``board.sfen()`` and the ``position_key_from_sfen`` binding rather
    than any incremental board state, so the result depends only on the
    Board_State the board currently holds (Requirement 3.2).
    """
    return position_key_from_sfen(board.sfen())


def position_key_from_sfen(sfen: str) -> PositionKey:
    """Return the Position_Key of the Board_State denoted by ``sfen``."""
    ndkey = np.empty(1, dtype=POSITION_KEY)
    cppshogi.position_key_from_sfen(sfen, ndkey)
    return PositionKey(int(ndkey[0]["hi"]), int(ndkey[0]["lo"]))


def position_keys_after(sfen: str, moves16) -> np.ndarray:
    """Return the Position_Key of the Board_State after each of ``moves16``.

    ``moves16`` is any array-like of 16-bit move codes (``cshogi.move16``
    values) legal in the Board_State denoted by ``sfen``. The Board_State
    is set from ``sfen`` exactly once and ``getKeyAndBoardKeyAfter`` is
    called once per move (the batched form of the binding), which is the
    shape both the PUCT descent's expansion step and the propagation
    prefetch of design.md's "Value propagation under a packed edge list"
    want. The result is a ``POSITION_KEY``-dtype array of the same length
    as ``moves16``, in the same order.
    """
    ndmoves16 = np.asarray(moves16, dtype="<u2")
    ndkeys = np.empty(len(ndmoves16), dtype=POSITION_KEY)
    if len(ndmoves16) > 0:
        cppshogi.position_keys_after(sfen, ndmoves16, ndkeys)
    return ndkeys


def apery_book_key(sfen: str) -> int:
    """Return the 64-bit Apery ``Book::bookKey`` of the Board_State ``sfen``.

    This is the key stored in ``book_node.apery_key`` for export
    (Requirement 12.1); it is not a Position_Key and must not be used as
    one, because it hashes only the mover's hand (see design.md's
    "Position_Key from Python" section).
    """
    return int(cppshogi.apery_book_key_from_sfen(sfen))


def zobrist_fingerprint() -> int:
    """Return the fingerprint of the running process's Zobrist tables.

    Recorded in ``book_meta`` at schema creation and compared on every
    subsequent start (Requirement 2.8, Requirement 3.4); a mismatch means
    Zobrist initialisation changed and the Book_Graph's keys are no longer
    trustworthy.
    """
    return int(cppshogi.zobrist_fingerprint())


def side_to_move_is_white(key: PositionKey) -> int:
    """Return whether the side to move at ``key`` is White.

    Every entry of the Zobrist board-key table has bit 0 cleared by
    construction (``initZobrist()`` fills each entry with
    ``g_mt64bit.random() & ~1``), and ``zobTurn_ == 1`` is XORed into the
    board key only when White is to move. A sum of table entries therefore
    has bit 0 equal to 0 when Black is to move and 1 when White is to
    move, so ``key.hi & 1`` recovers the side to move with no extra column
    and no SFEN parsing. This is used wherever a value needs to be
    oriented by side to move: the export score perspective (Requirement
    12.6) and the choice between Draw_Value_Black and Draw_Value_White
    (Requirements 8.2, 9.5, 9.10, 9.13).
    """
    return key.hi & 1


def fold_u64_to_i64(u: int) -> int:
    """Fold an unsigned 64-bit integer into PostgreSQL's signed bigint range.

    ``book_node.key_hi``, ``key_lo``, and ``apery_key`` are ``bigint``,
    i.e. signed, while a Position_Key half and the Apery book key are
    unsigned 64-bit. This performs the two's-complement fold needed on the
    way in: values at or above 2**63 wrap to their negative signed
    representation, matching how PostgreSQL (and asyncpg) will store and
    return the same bit pattern.
    """
    u &= 0xFFFFFFFFFFFFFFFF
    return u - 0x10000000000000000 if u >= 0x8000000000000000 else u


def unfold_i64_to_u64(i: int) -> int:
    """Recover the unsigned 64-bit value from a signed bigint reinterpreted key.

    The inverse of ``fold_u64_to_i64``: negative values read back from a
    ``bigint`` column are reinterpreted as unsigned by adding 2**64. This is
    the same reinterpretation ``dlshogi/utils/book.py`` performs implicitly
    by reading its ``key`` field as ``<u8``.
    """
    return i & 0xFFFFFFFFFFFFFFFF
