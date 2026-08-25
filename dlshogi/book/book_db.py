"""Book_DB_Parser / Book_DB_Printer: YaneuraOu ``.db`` text format I/O.

Implements Requirement 6 (YaneuraOu Book Format Parsing and Printing): the
``TerashockMove`` / ``TerashockEntry`` / ``TerashockBook`` data model, a
line-oriented parser (``BookDBParser``) that turns YaneuraOu ``.db`` text
into a ``TerashockBook``, and a printer (``BookDBPrinter``) that turns a
``TerashockBook`` back into that text. See design.md's "Book_DB_Parser and
Book_DB_Printer data model" section for the full derivation; this module
follows it directly.

**Line classification order (design.md, followed literally):** blank
(discard silently, Requirement 6.12); ``#YANEURAOU-DB2016 <version>`` with
a 1-16 character version (record, replacing any previous, Requirement
6.10); any other ``#`` line (comment, Requirement 6.4); ``sfen `` prefix
(start an entry if the trimmed remainder is 1-256 characters, else reject
per Requirement 6.5); exactly five whitespace-separated fields (candidate
move, but rejected with a report if no entry is open, Requirement 6.11, or
if any field is out of range, Requirement 6.5); anything else (reject,
report line number and content, keep everything parsed so far, continue,
Requirement 6.5). The parser's two states -- *before any sfen line* and
*inside an entry* -- collapse to "``self.entries`` is empty" versus
"non-empty", since a candidate-move line always attaches to
``self.entries[-1]``.

**Moves are stored as ``move16``, not text (Requirement 1.2's rationale
mirrored here).** Design.md states this plainly: "Moves are stored as
``move16`` -- obtained with ``board.move_from_usi`` on a board set from
the entry's SFEN, then ``cshogi.move16`` -- rather than as text, so that
the parser is also the validator, and so that the printer's output is
canonical, which is what makes the round trips of Requirements 6.8 and 6.9
provable rather than approximately true." Concretely: the move field is
resolved with ``board.move_from_usi(move_str)`` on a board set from the
current Terashock_Entry's SFEN; an unresolved move (``move_from_usi``
returns ``cshogi.MOVE_NONE == 0`` for anything it cannot make on that
board -- unparseable syntax and geometrically or positionally illegal
moves alike) is treated as "matches none of the recognized line forms"
and the whole line is rejected under Requirement 6.5, exactly as an
out-of-range numeric field would be. The opponent-reply field is resolved
the same way on the position *after* the candidate move is applied (it is
the expected reply to that move), and the literal ``none`` is recognized
first and recorded as absent (Requirement 6.3) without ever touching the
board. The printer needs no board at all: ``cshogi.move_to_usi(move16)``
determines the USI string with no position context (also stated in
design.md, and the reason Requirement 1.2's "no child Position_Key" and
"``move16`` instead of the USI string" design choices for ``PACKED_EDGE``
carry over unchanged here).

**Board construction is defensive, not exception-free.** Setting a
``cshogi.Board`` from an arbitrary SFEN string can raise ``RuntimeError``
on malformed input (confirmed by ``config.py``'s docstring for
Root_Position, which documents the same underlying cshogi behaviour and
the same reason full SFEN legality is not attempted at the configuration
layer). Because Requirement 6.1 accepts *any* 1-256 character trimmed
remainder as an Entry's SFEN with no legality check, a Terashock_Entry's
SFEN may not denote a legal Board_State at all. This module therefore
sets the board for the *current* entry lazily -- only when the first
candidate-move line under it needs to resolve a move -- and wraps that
setup, and every subsequent ``move_from_usi`` call, in ``except
Exception``, so a ``RuntimeError`` from an unparseable SFEN rejects the
candidate-move line under Requirement 6.5 rather than propagating out of
the parser, while every other Terashock_Entry and Terashock_Move parsed
before or after it is retained unchanged, per Requirement 6.5's "keep
everything parsed so far". (A small number of malformed inputs are known
to abort the process outright rather than raise -- the same documented
cshogi limitation ``config.py`` already accepts for Root_Position -- and
that risk is unmitigated here for the same reason: there is no Python-level
guard against it.)

**Progress_Reporter integration.** ``report.py`` (task 16.1) does not
exist yet, so ``BookDBParser`` accepts an optional ``on_rejected_line``
callback, ``Callable[[int, str], None]``, invoked with the 1-based line
number and the exact line content for every rejected line (Requirement
6.5's "report the line number and the line content to the Progress_
Reporter"). Independently of that callback, every rejection is always
appended to ``self.rejected_lines`` so a caller -- including this task's
own future property tests (5.3, 5.4) -- can inspect what was rejected
without needing a concrete Progress_Reporter. The eventual
``Progress_Reporter`` need only supply a matching callable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cshogi

# ---------------------------------------------------------------------------
# Data model (design.md's "Book_DB_Parser and Book_DB_Printer data model")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerashockMove:
    """One candidate move of a Terashock_Entry (Requirement 6.2).

    ``move16`` and ``reply16`` are Apery ``proFromAndTo`` 16-bit move
    codes (``cshogi.move16``), not text, so that the printer's output is
    canonical (see the module docstring). ``reply16`` is ``None`` for the
    literal ``none`` opponent reply (Requirement 6.3).
    """

    move16: int
    reply16: Optional[int]
    eval: int  # -32000..32000
    depth: int  # 0..127
    count: int  # 0..4294967295


@dataclass
class TerashockEntry:
    """One parsed position from a Terashock_Book (Requirement 6.1).

    ``sfen`` is the trimmed remainder of a ``sfen `` line, 1 to 256
    characters, stored exactly as read with no legality check. ``moves``
    holds the entry's Terashock_Move records in file order (Requirement
    6.2's "preserving the order in which the candidate-move lines
    appear").
    """

    sfen: str
    moves: List[TerashockMove] = field(default_factory=list)


@dataclass
class TerashockBook:
    """A whole parsed Terashock_Book: a header version and entry list.

    ``header_version`` is the ``<version>`` token of the most recently
    read ``#YANEURAOU-DB2016 <version>`` line (Requirement 6.10), or ``""``
    if no such line has been read. This is purely a record of what the
    *input* declared; ``BookDBPrinter`` always writes the fixed literal
    ``#YANEURAOU-DB2016 1.00`` header (Requirement 6.6) regardless of this
    field's value.
    """

    header_version: str = ""
    entries: List[TerashockEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Field bounds (Requirement 6.1, 6.2, 6.10)
# ---------------------------------------------------------------------------

MAX_SFEN_LEN = 256
MAX_HEADER_VERSION_LEN = 16
MIN_MOVE_FIELD_LEN = 1
MAX_MOVE_FIELD_LEN = 7
EVAL_MIN, EVAL_MAX = -32000, 32000
DEPTH_MIN, DEPTH_MAX = 0, 127
COUNT_MIN, COUNT_MAX = 0, 4294967295

_NONE_REPLY = "none"
_HEADER_LINE_RE = re.compile(r"^#YANEURAOU-DB2016 (\S{1,%d})$" % MAX_HEADER_VERSION_LEN)

OnRejectedLine = Callable[[int, str], None]


def _parse_int_in_range(text: str, low: int, high: int) -> Optional[int]:
    """Parse ``text`` as a base-10 integer within ``[low, high]``, or ``None``."""
    try:
        value = int(text)
    except ValueError:
        return None
    if low <= value <= high:
        return value
    return None


# ---------------------------------------------------------------------------
# Book_DB_Parser
# ---------------------------------------------------------------------------


class BookDBParser:
    """Line-oriented state machine parsing YaneuraOu ``.db`` text.

    Feed lines one at a time with `parse_line`, or a whole text with
    `parse_text`; either way, the accumulated result is available at any
    point as `book` (or via `parse_text`'s return value). The two states
    the design calls *before any sfen line* and *inside an entry* are not
    tracked by an explicit enum: they are exactly "``self.entries`` is
    empty" and "non-empty", since a candidate-move line always attaches to
    the most recently started entry, `self.entries[-1]`.
    """

    def __init__(self, on_rejected_line: Optional[OnRejectedLine] = None) -> None:
        self.on_rejected_line = on_rejected_line
        self.header_version: str = ""
        self.entries: List[TerashockEntry] = []
        self.rejected_lines: List[Tuple[int, str]] = []
        # Lazily (re)initialised to the current entry's SFEN on first use;
        # see the module docstring's "Board construction is defensive" note.
        self._board = cshogi.Board()
        self._board_ready_for_current_entry = False
        self._board_error_for_current_entry = False

    @property
    def book(self) -> TerashockBook:
        """The `TerashockBook` accumulated so far."""
        return TerashockBook(header_version=self.header_version, entries=self.entries)

    def parse_text(self, text: str) -> TerashockBook:
        """Parse every line of ``text`` and return the resulting `TerashockBook`.

        ``text.splitlines()`` is used rather than a manual ``\\n`` split,
        so ``\\n``, ``\\r\\n``, and bare ``\\r`` line endings are all
        handled with no stray ``\\r`` left on a line, and line numbers are
        1-based to match what an Operator would see in a text editor.
        """
        for line_number, line in enumerate(text.splitlines(), start=1):
            self.parse_line(line_number, line)
        return self.book

    def parse_line(self, line_number: int, line: str) -> None:
        """Classify and process one line, per the module docstring's ordering."""
        if not line.strip():
            # Requirement 6.12: blank line, discard silently, no report.
            return

        if line.startswith("#"):
            match = _HEADER_LINE_RE.match(line)
            if match:
                # Requirement 6.10: record, replacing any previous.
                self.header_version = match.group(1)
            # else: an ordinary comment line (Requirement 6.4). Either way,
            # a line starting with "#" is never rejected.
            return

        if line.startswith("sfen "):
            self._handle_sfen_line(line_number, line)
            return

        fields = line.split()
        if len(fields) != 5:
            # Requirement 6.5: non-comment line whose field count isn't five.
            self._reject(line_number, line)
            return

        if not self.entries:
            # Requirement 6.11: candidate-move line before any sfen line.
            self._reject(line_number, line)
            return

        self._handle_candidate_move_line(line_number, line, fields)

    # -- line handlers ------------------------------------------------

    def _handle_sfen_line(self, line_number: int, line: str) -> None:
        remainder = line[len("sfen "):]
        trimmed = remainder.strip()
        if not (1 <= len(trimmed) <= MAX_SFEN_LEN):
            # Requirement 6.5: empty or >256-character trimmed remainder.
            self._reject(line_number, line)
            return
        # Requirement 6.1: start a new Terashock_Entry with an empty move list.
        self.entries.append(TerashockEntry(sfen=trimmed))
        self._board_ready_for_current_entry = False
        self._board_error_for_current_entry = False

    def _handle_candidate_move_line(
        self, line_number: int, line: str, fields: List[str]
    ) -> None:
        move_str, reply_str, eval_str, depth_str, count_str = fields

        if not (MIN_MOVE_FIELD_LEN <= len(move_str) <= MAX_MOVE_FIELD_LEN):
            self._reject(line_number, line)
            return

        eval_value = _parse_int_in_range(eval_str, EVAL_MIN, EVAL_MAX)
        depth_value = _parse_int_in_range(depth_str, DEPTH_MIN, DEPTH_MAX)
        count_value = _parse_int_in_range(count_str, COUNT_MIN, COUNT_MAX)
        if eval_value is None or depth_value is None or count_value is None:
            # Requirement 6.5: an out-of-bounds or non-integer numeric field.
            self._reject(line_number, line)
            return

        resolved = self._resolve_move_and_reply(move_str, reply_str)
        if resolved is None:
            self._reject(line_number, line)
            return
        move16, reply16 = resolved

        self.entries[-1].moves.append(
            TerashockMove(
                move16=move16,
                reply16=reply16,
                eval=eval_value,
                depth=depth_value,
                count=count_value,
            )
        )

    def _resolve_move_and_reply(
        self, move_str: str, reply_str: str
    ) -> Optional[Tuple[int, Optional[int]]]:
        """Resolve the move and opponent-reply fields to ``move16`` codes.

        Returns ``None`` if the current entry's SFEN cannot be set on a
        board, if ``move_str`` is not a move `cshogi` can make from that
        board, or (when ``reply_str`` is not the literal ``none``) if
        ``reply_str`` is not a move `cshogi` can make from the position
        reached by applying the resolved move. See the module docstring's
        "Moves are stored as ``move16``" and "Board construction is
        defensive" notes.
        """
        if not self._ensure_board_ready():
            return None

        try:
            move = self._board.move_from_usi(move_str)
        except Exception:
            return None
        if move == cshogi.MOVE_NONE:
            return None
        move16 = cshogi.move16(move)

        if reply_str == _NONE_REPLY:
            return move16, None

        try:
            self._board.push(move)
            try:
                reply_move = self._board.move_from_usi(reply_str)
            finally:
                self._board.pop()
        except Exception:
            return None
        if reply_move == cshogi.MOVE_NONE:
            return None
        return move16, cshogi.move16(reply_move)

    def _ensure_board_ready(self) -> bool:
        """Set ``self._board`` from the current entry's SFEN, once, lazily."""
        if self._board_ready_for_current_entry:
            return True
        if self._board_error_for_current_entry:
            return False
        try:
            self._board.set_sfen(self.entries[-1].sfen)
        except Exception:
            self._board_error_for_current_entry = True
            return False
        self._board_ready_for_current_entry = True
        return True

    def _reject(self, line_number: int, content: str) -> None:
        self.rejected_lines.append((line_number, content))
        if self.on_rejected_line is not None:
            self.on_rejected_line(line_number, content)


# ---------------------------------------------------------------------------
# Book_DB_Printer
# ---------------------------------------------------------------------------


class BookDBPrinter:
    """Formats a `TerashockBook` back into YaneuraOu ``.db`` text.

    Needs no board: `cshogi.move_to_usi` determines a move's USI string
    from its ``move16`` code with no position context (see the module
    docstring). Supports both an incremental, stream-oriented interface
    (`write_header` / `write_entry`, for a future exporter that knows the
    entry count before it has the entries -- design.md's Terashock_Index
    and Book_Exporter sections both rely on this) and a whole-text
    convenience method (`format_book`).
    """

    HEADER_VERSION = "1.00"

    def write_header(self, stream, entry_count: int) -> None:
        """Write the two-line header: Requirement 6.6.

        The version is always the fixed literal `HEADER_VERSION`,
        regardless of any header version a parser may have recorded from
        its input; see `TerashockBook.header_version`'s docstring.
        """
        stream.write(f"#YANEURAOU-DB2016 {self.HEADER_VERSION}\n")
        stream.write(f"# NOE:{entry_count}\n")

    def write_entry(self, stream, entry: TerashockEntry) -> None:
        """Write one Terashock_Entry: its ``sfen`` line and its move lines (Requirement 6.7)."""
        stream.write(f"sfen {entry.sfen}\n")
        for move in entry.moves:
            stream.write(self.format_move(move))
            stream.write("\n")

    @staticmethod
    def format_move(move: TerashockMove) -> str:
        """Format one Terashock_Move as its five-field candidate-move line.

        Fields are separated by a single space character, in the order
        move, opponent reply, evaluation value, search depth, selection
        count (Requirement 6.2's field order), with the literal ``none``
        for an absent opponent reply (Requirement 6.3, 6.7).
        """
        move_usi = cshogi.move_to_usi(move.move16)
        reply_usi = _NONE_REPLY if move.reply16 is None else cshogi.move_to_usi(move.reply16)
        return f"{move_usi} {reply_usi} {move.eval} {move.depth} {move.count}"

    def format_book(self, book: TerashockBook) -> str:
        """Return the whole ``book`` formatted as YaneuraOu ``.db`` text."""
        import io

        buf = io.StringIO()
        self.write_header(buf, len(book.entries))
        for entry in book.entries:
            self.write_entry(buf, entry)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def parse_book_db(text: str, on_rejected_line: Optional[OnRejectedLine] = None) -> TerashockBook:
    """Parse ``text`` as YaneuraOu ``.db`` content and return the `TerashockBook`."""
    return BookDBParser(on_rejected_line=on_rejected_line).parse_text(text)


def format_book_db(book: TerashockBook) -> str:
    """Format ``book`` as YaneuraOu ``.db`` text."""
    return BookDBPrinter().format_book(book)


__all__ = [
    "TerashockMove",
    "TerashockEntry",
    "TerashockBook",
    "BookDBParser",
    "BookDBPrinter",
    "parse_book_db",
    "format_book_db",
    "MAX_SFEN_LEN",
    "MAX_HEADER_VERSION_LEN",
    "MIN_MOVE_FIELD_LEN",
    "MAX_MOVE_FIELD_LEN",
    "EVAL_MIN",
    "EVAL_MAX",
    "DEPTH_MIN",
    "DEPTH_MAX",
    "COUNT_MIN",
    "COUNT_MAX",
]
