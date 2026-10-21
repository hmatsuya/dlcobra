"""Repetition_Resolver: repetition and terminal-state classification.

Implements Requirements 8.1 through 8.11:

- Four-fold repetition detection using the resolver's own
  ``path_occurrences: dict[PositionKey, int]`` counter, incremented on
  push and decremented on pop.  cshogi's ``Board.is_draw()`` is
  consulted only as a corroborating signal and never drives the
  classification (design.md: "同一局面4回をきちんと数えていない").
- The four-way repetition truth table (8.1/8.2/8.3/8.4/8.9):
  neither/both sides checking -> draw; mover-only checking -> loss;
  opponent-only checking -> win.
- Terminal states: zero legal moves (8.5), declaration win (8.6),
  with an independent WCSC predicate as cross-check.
- Cyclic_Flag placement on the repeated node, the leaf, and every
  node between them on the descent path, for repetition-derived
  values only (8.7).
- Non-repetition pass-through (8.10); terminal nodes get no edges
  (8.11).

The module is importable without side effects and usable by later
tasks (Search_Coordinator/descent) and by property tests 11.2-11.4.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cshogi

from dlshogi.book.keys import PositionKey, side_to_move_is_white
from dlshogi.book.node_store import Terminal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification results
# ---------------------------------------------------------------------------


class RepetitionClass(enum.Enum):
    """The classification outcome of a single descent position."""

    NON_REPETITION = "non_repetition"
    DRAW = "draw"
    LOSS_FOR_MOVER = "loss_for_mover"
    WIN_FOR_MOVER = "win_for_mover"


@dataclass(frozen=True)
class RepetitionResult:
    """Result of ``RepetitionResolver.classify()``.

    Attributes
    ----------
    classification : RepetitionClass
        The four-way classification (or non-repetition).
    value : float or None
        The descent value (win rate for the side to move at the repeated
        Board_State): ``None`` for non-repetition (value comes from the
        Evaluator or a terminal state -- Requirement 8.10).
    cyclic_indices : tuple of int
        The path indices (inclusive) on which the Cyclic_Flag must be
        placed.  Empty for non-repetition and terminal states.  For a
        repetition, this spans from the first occurrence index of the
        repeated position to the current (fourth occurrence) index,
        inclusive -- covering the repeated node, the leaf, and every
        node between them (Requirement 8.7).
    """

    classification: RepetitionClass
    value: Optional[float] = None
    cyclic_indices: Tuple[int, ...] = ()


@dataclass(frozen=True)
class TerminalResult:
    """Result of ``check_terminal()``.

    Attributes
    ----------
    terminal : Terminal
        The terminal state classification (NONE if not terminal).
    value : float or None
        The descent value (win rate for the side to move): 1 for win,
        0 for loss, ``None`` when not terminal.
    """

    terminal: Terminal
    value: Optional[float] = None


# ---------------------------------------------------------------------------
# WCSC declaration-win predicate (independent of cshogi's is_nyugyoku)
# ---------------------------------------------------------------------------


# cshogi piece constants (from cshogi module):
# BLACK=0, WHITE=1
# Piece types: PAWN..KING, PPAWN..PBISHOP (promoted)
# Hand piece indices: HPAWN=0, HLANCE=1, HKNIGHT=2, HSILVER=3, HGOLD=4,
#                     HBISHOP=5, HROOK=6

# Pieces on the board are represented as integers; white pieces have
# offset +16 (cshogi.WPAWN = cshogi.BPAWN + 16 = 17 typically, but we
# use cshogi's own constants for safety).

# Squares in opponent's three ranks (enemy territory).
# cshogi square numbering: sq = (file-1)*9 + (rank-1), file 1..9 (right-to-left),
# rank 1..9 (top-to-bottom, i.e. rank 1 = rank 'a' = White's back rank).
#   For Black entering (opponent's ranks 1-3): rank_index in {0, 1, 2}
#   For White entering (opponent's ranks 7-9): rank_index in {6, 7, 8}

_BLACK_TERRITORY_SQUARES = frozenset(
    f * 9 + r for f in range(9) for r in range(3)
)  # 27 squares in ranks 1-3 (White's camp)
_WHITE_TERRITORY_SQUARES = frozenset(
    f * 9 + r for f in range(9) for r in range(6, 9)
)  # 27 squares in ranks 7-9 (Black's camp)

# Big pieces (worth 5 points each, whether promoted or not):
# Rook, Dragon (promoted rook), Bishop, Horse (promoted bishop)


def wcsc_declaration_win(board) -> bool:
    """Independent WCSC declaration-win predicate (Requirement 8.6).

    The six conditions from the requirement text:
    1. It is the declaring side's turn (implicit -- we check the current
       side to move).
    2. The declaring side's king stands within the opponent's three ranks.
    3. The declaring side is not in check.
    4. At least 10 of the declaring side's pieces other than its king
       stand within the opponent's three ranks.
    5. The declaring side's point total (rook/bishop = 5, all other
       non-king pieces = 1, counting pieces in the opponent's three
       ranks plus hand pieces) is at least 28 for Black, 27 for White.
    6. The declaring side's time is remaining (always true in book
       building context).

    This predicate is written independently from the Requirement 8.6 text
    and compared against ``board.is_nyugyoku()`` in
    ``check_terminal()``. If the two disagree, Property 24 will catch it.
    """
    us = board.turn  # 0 = BLACK, 1 = WHITE

    # Condition 3: not in check
    if board.is_check():
        return False

    pieces = board.pieces  # length-81 array of piece codes (0 = NONE)
    territory = _BLACK_TERRITORY_SQUARES if us == cshogi.BLACK else _WHITE_TERRITORY_SQUARES

    # Condition 2: king in opponent's three ranks
    our_king = cshogi.BKING if us == cshogi.BLACK else cshogi.WKING
    king_in_territory = False
    for sq in territory:
        if pieces[sq] == our_king:
            king_in_territory = True
            break
    if not king_in_territory:
        return False

    # Condition 4 and 5: count own non-king pieces in territory, compute points
    # Identify which board-piece codes belong to us.
    # Big pieces (rook, promoted rook, bishop, promoted bishop) are worth 5 each.
    if us == cshogi.BLACK:
        big_pieces_on_board = frozenset(
            {cshogi.BROOK, cshogi.BPROM_ROOK, cshogi.BBISHOP, cshogi.BPROM_BISHOP}
        )
        own_piece_min = cshogi.BPAWN  # 1
        own_piece_max = cshogi.BPROM_ROOK  # 14 (highest Black piece code)
    else:
        big_pieces_on_board = frozenset(
            {cshogi.WROOK, cshogi.WPROM_ROOK, cshogi.WBISHOP, cshogi.WPROM_BISHOP}
        )
        own_piece_min = cshogi.WPAWN  # 17
        own_piece_max = cshogi.WPROM_ROOK  # 30 (highest White piece code)

    own_king_code = our_king
    piece_count_in_territory = 0
    big_count_in_territory = 0

    for sq in territory:
        p = pieces[sq]
        if p == cshogi.NONE or p == own_king_code:
            continue
        if own_piece_min <= p <= own_piece_max:
            piece_count_in_territory += 1
            if p in big_pieces_on_board:
                big_count_in_territory += 1

    # Condition 4: at least 10 non-king own pieces in territory
    if piece_count_in_territory < 10:
        return False

    # Condition 5: point total
    small_count_in_territory = piece_count_in_territory - big_count_in_territory
    # Points from pieces on board in territory
    points = small_count_in_territory + big_count_in_territory * 5

    # Points from hand pieces
    hand = board.pieces_in_hand[us]
    # Hand indices: HPAWN=0, HLANCE=1, HKNIGHT=2, HSILVER=3, HGOLD=4,
    #              HBISHOP=5, HROOK=6
    points += hand[cshogi.HPAWN]
    points += hand[cshogi.HLANCE]
    points += hand[cshogi.HKNIGHT]
    points += hand[cshogi.HSILVER]
    points += hand[cshogi.HGOLD]
    points += hand[cshogi.HBISHOP] * 5
    points += hand[cshogi.HROOK] * 5

    threshold = 28 if us == cshogi.BLACK else 27
    if points < threshold:
        return False

    return True


# ---------------------------------------------------------------------------
# Terminal-state detection
# ---------------------------------------------------------------------------


def check_terminal(board) -> TerminalResult:
    """Check whether ``board``'s position is a terminal state.

    Terminal states (Requirement 8.5, 8.6, 8.11):
    - Zero legal moves: loss for the side to move (value 0).
    - WCSC declaration win: win for the side to move (value 1).

    When ``board.is_nyugyoku()`` and ``wcsc_declaration_win(board)``
    disagree, a warning is logged and the independent predicate's answer
    is used (per design.md: "If that property fails, the independent
    predicate becomes the implementation and cshogi's answer becomes the
    corroborating signal").

    Returns ``TerminalResult(Terminal.NONE, None)`` when the position is
    not terminal.
    """
    # Check zero legal moves first (mate or stalemate = loss for stm)
    if not list(board.legal_moves):
        return TerminalResult(terminal=Terminal.LOSS_FOR_STM, value=0.0)

    # Declaration win check
    independent = wcsc_declaration_win(board)
    cshogi_answer = board.is_nyugyoku()

    if independent != cshogi_answer:
        logger.warning(
            "Declaration-win disagreement at sfen=%r: "
            "independent=%s, cshogi.is_nyugyoku=%s; "
            "using independent predicate.",
            board.sfen(),
            independent,
            cshogi_answer,
        )

    if independent:
        return TerminalResult(terminal=Terminal.WIN_FOR_STM, value=1.0)

    return TerminalResult(terminal=Terminal.NONE, value=None)


# ---------------------------------------------------------------------------
# cshogi is_draw() corroboration constants
# ---------------------------------------------------------------------------

# cshogi repetition constants (imported from cshogi module)
_REPETITION_DRAW = cshogi.REPETITION_DRAW
_REPETITION_WIN = cshogi.REPETITION_WIN
_REPETITION_LOSE = cshogi.REPETITION_LOSE
_NOT_REPETITION = cshogi.NOT_REPETITION


# ---------------------------------------------------------------------------
# Repetition Resolver
# ---------------------------------------------------------------------------


@dataclass
class _PathEntry:
    """One ply on the descent path."""

    key: PositionKey
    is_in_check: bool


class RepetitionResolver:
    """Stateful per-descent repetition resolver (Requirement 8.1-8.11).

    Usage pattern for a single Selection_Descent::

        resolver = RepetitionResolver(draw_value_black=cfg.draw_value_black,
                                      draw_value_white=cfg.draw_value_white)

        # At each ply of the descent:
        resolver.push(key, is_in_check)
        result = resolver.classify()
        if result.classification != RepetitionClass.NON_REPETITION:
            # Handle repetition: use result.value, mark cyclic_indices
            ...

        # When backtracking:
        resolver.pop()

    The resolver maintains:
    - ``path``: the sequence of (key, is_in_check) entries on the current
      descent, representing the plies from the Root_Position onward.
    - ``path_occurrences``: a dict mapping PositionKey -> count of how
      many times that key appears on the current path. Incremented on
      push, decremented on pop.
    - ``key_first_indices``: a dict mapping PositionKey -> list of path
      indices where that key occurs, used to identify the four occurrence
      plies for the check-history scan.
    """

    def __init__(
        self,
        draw_value_black: float,
        draw_value_white: float,
    ) -> None:
        self._draw_value_black = draw_value_black
        self._draw_value_white = draw_value_white
        self._path: List[_PathEntry] = []
        self._path_occurrences: dict[PositionKey, int] = {}
        self._key_indices: dict[PositionKey, List[int]] = {}

    @property
    def path_length(self) -> int:
        """Current number of plies on the descent path."""
        return len(self._path)

    @property
    def path_occurrences(self) -> dict[PositionKey, int]:
        """Read-only view of the occurrence counter (for testing)."""
        return self._path_occurrences

    def push(self, key: PositionKey, is_in_check: bool) -> None:
        """Record a new ply on the descent path.

        ``key`` is the Position_Key of the Board_State after the move.
        ``is_in_check`` is whether the side to move at that Board_State is
        in check (i.e., ``board.is_check()`` *after* the move was pushed).

        Increments ``path_occurrences[key]`` and records the ply index.
        """
        idx = len(self._path)
        self._path.append(_PathEntry(key=key, is_in_check=is_in_check))
        self._path_occurrences[key] = self._path_occurrences.get(key, 0) + 1
        if key not in self._key_indices:
            self._key_indices[key] = []
        self._key_indices[key].append(idx)

    def pop(self) -> None:
        """Remove the most recent ply from the descent path.

        Decrements ``path_occurrences[key]`` and removes the index record.
        Raises ``IndexError`` if the path is empty.
        """
        if not self._path:
            raise IndexError("Cannot pop from an empty path")
        entry = self._path.pop()
        key = entry.key
        self._path_occurrences[key] -= 1
        if self._path_occurrences[key] == 0:
            del self._path_occurrences[key]
        self._key_indices[key].pop()
        if not self._key_indices[key]:
            del self._key_indices[key]

    def classify(self, board=None) -> RepetitionResult:
        """Classify the current (most recently pushed) position.

        If the current position's occurrence count on the path has reached
        4, the four-way truth table is applied. Otherwise, the result is
        NON_REPETITION with no value (Requirement 8.10: value comes from
        the Evaluator or terminal state).

        If ``board`` is provided, ``board.is_draw()`` is consulted as a
        corroborating signal (design.md). Discrepancies are logged but
        never override the resolver's own classification.
        """
        if not self._path:
            return RepetitionResult(classification=RepetitionClass.NON_REPETITION)

        entry = self._path[-1]
        key = entry.key
        count = self._path_occurrences.get(key, 0)

        if count < 4:
            return RepetitionResult(classification=RepetitionClass.NON_REPETITION)

        # Four-fold repetition reached -- classify using truth table.
        # Find all indices where this key occurs on the path.
        indices = self._key_indices[key]
        # We need the four occurrence indices.
        assert len(indices) >= 4, (
            f"path_occurrences says count={count} but "
            f"key_indices has only {len(indices)} entries"
        )
        # Use the first four occurrences for the check-history scan.
        four_indices = indices[:4]

        # Who gave check (Requirements 8.1, 8.3, 8.4, 8.9). The side to move
        # is the same at all four occurrences (it is part of the key); call
        # it the mover. Positions alternate between the mover's turn and the
        # opponent's turn, and is_in_check at a ply says whether the side
        # that just moved gave check to the side now to move:
        #
        #   offset from first_idx    turn       is_in_check means
        #   0, 2, 4, ...             mover      the opponent gave check
        #   1, 3, 5, ...             opponent   the mover gave check
        #
        # Perpetual check by a side = every one of its moves between the
        # first and fourth occurrence was a check (the rule for sennichite
        # by continuous check). design.md: "a scan of that list between the
        # first and fourth occurrence indices".
        first_idx = four_indices[0]
        last_idx = four_indices[3]

        # Mover gave check on every one of its moves: all odd offsets.
        mover_gave_check_all = all(
            self._path[i].is_in_check for i in range(first_idx + 1, last_idx + 1, 2)
        )
        # Opponent gave check on every one of its moves: all even offsets,
        # starting at the first occurrence itself.
        opponent_gave_check_all = all(
            self._path[i].is_in_check for i in range(first_idx, last_idx + 1, 2)
        )

        # Apply the truth table:
        # | mover_check | opponent_check | classification | value |
        # | no          | no             | draw           | draw_value |
        # | yes         | no             | loss for mover | 0 |
        # | no          | yes            | win for mover  | 1 |
        # | yes         | yes            | draw           | draw_value |

        if mover_gave_check_all and not opponent_gave_check_all:
            classification = RepetitionClass.LOSS_FOR_MOVER
            value = 0.0
        elif not mover_gave_check_all and opponent_gave_check_all:
            classification = RepetitionClass.WIN_FOR_MOVER
            value = 1.0
        else:
            # Neither or both: draw
            classification = RepetitionClass.DRAW
            # Draw value selected by side to move at the repeated position
            is_white = side_to_move_is_white(key)
            value = (
                self._draw_value_white if is_white else self._draw_value_black
            )

        # Cyclic_Flag indices: from first occurrence to the fourth
        # (current), inclusive (Requirement 8.7)
        cyclic_indices = tuple(range(first_idx, last_idx + 1))

        # Corroboration with cshogi's is_draw() if board is provided
        if board is not None:
            self._corroborate(board, classification)

        return RepetitionResult(
            classification=classification,
            value=value,
            cyclic_indices=cyclic_indices,
        )

    def _corroborate(self, board, classification: RepetitionClass) -> None:
        """Compare our classification against cshogi's is_draw() as a signal.

        Discrepancies are logged but never override our classification.
        """
        cshogi_draw = board.is_draw()

        if classification == RepetitionClass.LOSS_FOR_MOVER:
            # Mover perpetual check -> loss for mover.
            # cshogi should report REPETITION_LOSE (the side to move loses)
            if cshogi_draw == _REPETITION_WIN:
                logger.warning(
                    "Repetition corroboration discrepancy: resolver says "
                    "LOSS_FOR_MOVER but cshogi says REPETITION_WIN at "
                    "sfen (path length=%d)",
                    len(self._path),
                )
        elif classification == RepetitionClass.WIN_FOR_MOVER:
            # Opponent perpetual check -> win for mover.
            # cshogi should report REPETITION_WIN (the side to move wins)
            if cshogi_draw == _REPETITION_LOSE:
                logger.warning(
                    "Repetition corroboration discrepancy: resolver says "
                    "WIN_FOR_MOVER but cshogi says REPETITION_LOSE at "
                    "sfen (path length=%d)",
                    len(self._path),
                )
        elif classification == RepetitionClass.DRAW:
            # Draw: cshogi should report REPETITION_DRAW (or possibly
            # NOT_REPETITION if it triggered earlier at the 2nd occurrence)
            if cshogi_draw in (_REPETITION_WIN, _REPETITION_LOSE):
                logger.warning(
                    "Repetition corroboration discrepancy: resolver says "
                    "DRAW but cshogi says %s at sfen (path length=%d)",
                    "REPETITION_WIN"
                    if cshogi_draw == _REPETITION_WIN
                    else "REPETITION_LOSE",
                    len(self._path),
                )

    def reset(self) -> None:
        """Clear all state, making this resolver reusable for a new descent."""
        self._path.clear()
        self._path_occurrences.clear()
        self._key_indices.clear()
