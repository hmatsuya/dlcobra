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

        # "The moving side delivered check at all four occurrences" means:
        # at each of the four positions where this key appears, was the
        # side to move at that position in check? Since the key encodes
        # the side to move and the position, "the moving side delivered
        # check" means the *opponent* of the side to move is delivering
        # check, which is what board.is_check() reports (it reports
        # whether the *current* side to move's king is in check, i.e.
        # whether the side that just moved delivered check).
        #
        # Wait -- design.md says: "the moving side delivered check at all
        # four occurrences" is answered by scanning board.is_check() at
        # each ply. board.is_check() after a push tells whether the side
        # to move (the one whose turn it now is) is in check -- meaning
        # the side that just moved (the "moving side" that made the last
        # move to reach this position) delivered check.
        #
        # So: is_in_check at an index means "the side that moved INTO
        # this position delivered check to the now-to-move side".
        #
        # The "moving side" in the truth table refers to the side whose
        # turn it is at the REPEATED position (who will next make a move
        # from there). At a repetition, the side to move is the same at
        # all four occurrences (because side-to-move is part of
        # Position_Key).
        #
        # "The moving side delivered check at all four occurrences" means:
        # at each of those four positions, did the side-to-move (the
        # "mover") deliver check? The mover delivers check by playing a
        # checking move -- that check would show up at the NEXT ply
        # (the position after the mover's move). So we look at the ply
        # *after* each occurrence.
        #
        # Actually, re-reading the design more carefully:
        # "did one side deliver check at all four occurrences?" -- the
        # occurrences are the positions themselves. At each occurrence,
        # is_in_check tells us if the mover (side to move at that
        # position) is being checked (by the opponent's last move).
        #
        # In shogi repetition rules:
        # - "Moving side is giving perpetual check" = the mover's moves
        #   from that position result in check. Since we're at the same
        #   position four times and the mover keeps giving check to reach
        #   the next occurrence.
        #
        # The standard interpretation: look at the positions BETWEEN the
        # four occurrences. If at every ply between the first and fourth
        # occurrence, the mover (the side to move at the repeated
        # position) delivered check, then it's perpetual check by the
        # mover.
        #
        # Per design.md: "Each task records board.is_check() at every ply
        # on the path, so 'the moving side delivered check at all four
        # occurrences' is a scan of that list between the first and
        # fourth occurrence indices."
        #
        # The "moving side" is the side to move at the repeated position.
        # Between the four occurrences, plies alternate. The mover
        # delivers check when, after the mover's move, the opponent is
        # in check -- which is the is_in_check at the ply right after
        # each of the mover's moves.
        #
        # More precisely: at each occurrence index i, the side to move is
        # the "mover". The mover then plays a move, leading to position
        # at index i+1 (if it exists). If is_in_check at i+1 is True,
        # the mover delivered check.
        #
        # So "mover checked at all four occurrences" means: for each of
        # the first three occurrences (the fourth has no subsequent move
        # within the repetition), the ply right after has is_in_check
        # == True. For the fourth occurrence itself, the mover just
        # arrived at the position that completes the repetition -- the
        # mover's check status for the fourth occurrence is whether they
        # delivered check to reach it, which is the is_in_check at the
        # fourth occurrence itself (the opponent moved to create this
        # position, and if is_in_check is True, the opponent delivered
        # check TO the mover).
        #
        # Actually, the simplest and standard interpretation in Japanese
        # shogi rules: check if every move between the first and last
        # occurrence (inclusive of the moves that created each occurrence)
        # by a given side was a checking move.
        #
        # Let me use the standard approach:
        # - "Side X gave check at all four occurrences" means that at
        #   each of the four occurrence positions, the position was
        #   reached by a checking move from side X.
        # - is_in_check at a position = True means the side that moved
        #   INTO that position gave check (to the current stm).
        #
        # At each occurrence, the side to move is the same (it's part of
        # the key). Let's call that side "STM" (side to move). The side
        # that moved into each occurrence is the opponent of STM.
        #
        # So "opponent checked at all four occurrences" =
        #   is_in_check at all four occurrence indices.
        # "mover checked at all four occurrences" =
        #   is_in_check at the ply AFTER each of the first three
        #   occurrences (i.e., after STM played from that position).
        #   For the 4th occurrence there's no "after" within the cycle.
        #
        # Wait, but the standard shogi rule checks moves BETWEEN the
        # occurrences, looking at ALL intervening plies, not just the
        # ply immediately after each occurrence.
        #
        # The standard rule (千日手): "同一手順が繰り返された場合、
        # 連続王手の千日手は王手をかけている側の負け"
        # = If the same sequence repeats, perpetual check (continuous
        # check by one side) makes the checking side lose.
        #
        # "Continuous check" means: between the positions that form the
        # repetition, EVERY move by that side was a checking move.
        #
        # So between first and fourth occurrence, for every ply where
        # it's the mover's turn to have just moved (i.e., odd-offset
        # plies from each occurrence), is_in_check should be True.
        #
        # Simpler: between the first and fourth occurrence (inclusive),
        # at every ply where the is_in_check field reflects a check
        # delivered by the mover:
        #   - Plies at odd offsets from the occurrence (the mover has
        #     just played) -> is_in_check = mover gave check
        #   - Plies at even offsets from the occurrence (the opponent
        #     has just played) -> is_in_check = opponent gave check
        #
        # Design.md says "a scan of that list between the first and
        # fourth occurrence indices" -- the key insight is that the
        # position repeats every N plies in a cycle of length N.
        #
        # Let me implement the straightforward interpretation:
        # Between first_idx and last_idx (the fourth occurrence),
        # scan every ply. The plies alternate between mover-to-play
        # and opponent-to-play. At a mover-to-play ply (same key parity
        # as the repeated position), is_in_check means the opponent just
        # gave check. At an opponent-to-play ply, is_in_check means the
        # mover just gave check.
        #
        # "Mover checked at all four" = at every ply between first and
        # fourth occurrence where it's the opponent's turn (the mover
        # just moved), is_in_check is True.
        #
        # "Opponent checked at all four" = at every ply between first and
        # fourth occurrence where it's the mover's turn (the opponent
        # just moved), is_in_check is True.

        first_idx = four_indices[0]
        last_idx = four_indices[3]

        # Determine which plies are "mover just moved" vs "opponent just
        # moved". At the occurrence indices, it's the mover's turn (to
        # play). So:
        # - Even offset from first_idx (0, 2, 4, ...) = mover's turn
        #   -> is_in_check here means opponent just gave check to mover
        # - Odd offset from first_idx (1, 3, 5, ...) = opponent's turn
        #   -> is_in_check here means mover just gave check to opponent

        mover_gave_check_all = True  # Mover gave check continuously
        opponent_gave_check_all = True  # Opponent gave check continuously

        # Scan plies from first_idx+1 to last_idx (inclusive).
        # At first_idx itself, we know it's a mover-turn ply; is_in_check
        # there means the opponent checked the mover to reach that
        # position.
        #
        # For "mover gave check": check all plies where opponent is to
        # move (odd offsets from first_idx): first_idx+1, first_idx+3, ...
        # For "opponent gave check": check all plies where mover is to
        # move (even offsets from first_idx, excluding first_idx itself):
        # first_idx+2, first_idx+4, ...
        # Plus the is_in_check at occurrence indices themselves (even
        # offsets) for opponent giving check.

        # Actually the simplest correct approach based on Japanese rules:
        # - Mover gives perpetual check if at EVERY position on the path
        #   between first and fourth occurrence where the opponent is the
        #   side to move, that position was reached by a checking move
        #   (is_in_check is True).
        # - Opponent gives perpetual check if at EVERY position on the
        #   path between first and fourth occurrence where the mover is
        #   the side to move, that position was reached by a checking
        #   move (is_in_check is True).

        # Plies from first_idx to last_idx:
        # first_idx: mover's turn. is_in_check => opponent checked mover
        # first_idx+1: opponent's turn. is_in_check => mover checked opp
        # first_idx+2: mover's turn. is_in_check => opponent checked mover
        # ...
        # At even offset: mover's turn, is_in_check = opponent gave check
        # At odd offset: opponent's turn, is_in_check = mover gave check

        # "Mover checked at all four occurrences" in the shogi repetition
        # rule context means: every move by the mover between the
        # repetitions was a checking move. This is detected by: every
        # odd-offset ply from first_idx has is_in_check == True.

        # "Opponent checked at all four occurrences" means: every move by
        # the opponent was a checking move. This is detected by: every
        # even-offset ply from first_idx (the occurrence positions
        # themselves) has is_in_check == True.

        # Check mover's continuous check: all odd-offset plies
        for i in range(first_idx + 1, last_idx + 1, 2):
            if not self._path[i].is_in_check:
                mover_gave_check_all = False
                break

        # Check opponent's continuous check: all even-offset plies
        # (from first_idx itself -- if the mover is in check at ALL
        # occurrences, the opponent gave continuous check)
        for i in range(first_idx, last_idx + 1, 2):
            if not self._path[i].is_in_check:
                opponent_gave_check_all = False
                break

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
