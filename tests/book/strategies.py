"""Shared hypothesis strategies for the ``tests/book`` property suite (task 3.7).

This module currently provides the self-play position strategy that the
Position_Key properties (8, 9, 10; tasks 3.2-3.4) and the packed-edge
codec property (task 3.6) all need: a way to draw a real, legal
``cshogi.Board`` (as its SFEN string) by playing a random walk of legal
moves from the initial position. Later property tasks (the node-field
strategy of Property 1, the PACKED_EDGE edge-array strategy of Property
12, the random-DAG graph strategy of Property 26, the Terashock
entry-list strategy of Property 20) extend this module as those tasks are
implemented; only the self-play strategy and its direct dependents exist
so far.

Design.md's Property 8 write-up names this "a `@composite` self-play
strategy that draws a move count from `st.integers(0, 200)` and then, at
each ply, a move index from `st.integers()` mapped over the legal move
list -- this is the graph/position generator that several later
properties reuse." That is exactly what ``self_play_sfen`` below does.
"""

from __future__ import annotations

import random
from typing import List

import cshogi
from hypothesis import strategies as st

MAX_SELF_PLAY_PLY = 200


def _random_walk_sfen(rng: random.Random, max_ply: int) -> str:
    """Play a random legal-move walk from the initial position; return its SFEN.

    Stops early if the walk reaches a position with no legal moves
    (checkmate or stalemate-equivalent), since a real game simply ends
    there. ``rng`` is a plain ``random.Random`` seeded from a hypothesis
    integer draw, which is what lets hypothesis shrink a failing example
    by shrinking that single seed while still producing full, structurally
    varied self-play games (drops, promotions, captures, and, at low
    probability, real repetitions and mates).
    """
    board = cshogi.Board()
    for _ in range(max_ply):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = moves[rng.randrange(len(moves))]
        board.push(move)
    return board.sfen()


@st.composite
def self_play_sfen(draw, max_ply: int = MAX_SELF_PLAY_PLY) -> str:
    """Draw the SFEN of a Board_State reached by self-play from the initial position.

    Draws a move count from ``st.integers(0, max_ply)`` and a walk seed,
    then plays that many random legal moves (or until the game ends
    early). The result always denotes a legal, reachable Board_State,
    including the initial position itself (move count 0).
    """
    ply = draw(st.integers(min_value=0, max_value=max_ply))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = random.Random(seed)
    return _random_walk_sfen(rng, ply)


@st.composite
def self_play_board_with_moves(draw, max_ply: int = MAX_SELF_PLAY_PLY):
    """Draw ``(sfen, legal_move16_list)`` for a self-play Board_State.

    ``legal_move16_list`` is the ``cshogi.move16`` encoding of every legal
    move of the drawn Board_State, in the order ``board.legal_moves``
    yields them (no particular sort order). Positions with zero legal
    moves (mate) are excluded, since several consumers -- the incremental
    child key property (task 3.3) chief among them -- need at least one
    move to check.
    """
    sfen = draw(self_play_sfen(max_ply=max_ply))
    board = cshogi.Board(sfen)
    moves = list(board.legal_moves)
    return sfen, [cshogi.move16(m) for m in moves]


def legal_move16_list(sfen: str) -> List[int]:
    """Return the ``move16`` code of every legal move of the Board_State ``sfen``."""
    board = cshogi.Board(sfen)
    return [cshogi.move16(m) for m in board.legal_moves]


__all__ = [
    "MAX_SELF_PLAY_PLY",
    "self_play_sfen",
    "self_play_board_with_moves",
    "legal_move16_list",
]
