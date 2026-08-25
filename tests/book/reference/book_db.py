"""Independent reference parser/printer for the YaneuraOu ``.db`` format.

Written directly from Requirement 6's text alone (see
``.kiro/specs/puct-book-builder/requirements.md``), and reviewed against
that text rather than against ``dlshogi.book.book_db``. This module
imports nothing from ``dlshogi.book``.

**Deliberate scope difference from ``dlshogi.book.book_db``.** The real
implementation additionally resolves each candidate move against a
``cshogi.Board`` set from the entry's own SFEN, and rejects a
candidate-move line whose move field is not an actually legal move from
that position (a decision design.md makes explicitly, in service of
storing moves as canonical ``move16`` codes rather than text). Requirement
6 criterion 2's text, read on its own, imposes no such legality check on
the move field -- it says only that "the move field is 1 to 7
non-whitespace characters". This reference therefore performs that
narrower, purely textual check: 1 to 7 non-whitespace characters, with no
attempt to determine whether the string denotes a move that is legal (or
even syntactically a well-formed USI move) from any particular position.
Moves and opponent replies are kept as plain text here, not resolved to
``move16`` codes, for the same reason: doing so would require the same
legality check design.md adds and the requirement text does not.

Property tests that cross-check this reference against the real
implementation therefore restrict their move-field generator to strings
that are actually legal USI moves of the entry's own (self-play-generated)
SFEN, so the real implementation's additional legality check never has
occasion to reject anything -- the two implementations are being compared
within the region of the input space where Requirement 6's text and
design.md's stricter behaviour agree, which is the only region where
comparing them is meaningful in the first place. Property 21's noise
tolerance cases (malformed field counts, out-of-range numeric fields, and
so on) are unaffected by this restriction, since both implementations
reject that noise for the same reasons.

Line classification order (Requirement 6, read literally, matching
criteria 1 through 5, 10, 11, 12): blank (6.12); ``#YANEURAOU-DB2016
<version>`` (6.10); any other ``#`` line (6.4); ``sfen `` prefix (6.1,
rejected per 6.5 if the trimmed remainder is empty or over 256 characters);
exactly five whitespace-separated fields (6.2, rejected per 6.11 if no
entry has been started yet); anything else (rejected per 6.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MAX_SFEN_LEN = 256
MAX_HEADER_VERSION_LEN = 16
MIN_MOVE_FIELD_LEN = 1
MAX_MOVE_FIELD_LEN = 7
EVAL_MIN, EVAL_MAX = -32000, 32000
DEPTH_MIN, DEPTH_MAX = 0, 127
COUNT_MIN, COUNT_MAX = 0, 4294967295

_NONE_REPLY = "none"
_HEADER_LINE_RE = re.compile(r"^#YANEURAOU-DB2016 (\S{1,%d})$" % MAX_HEADER_VERSION_LEN)


@dataclass(frozen=True)
class RefTerashockMove:
    """One candidate move, kept as plain text (Requirement 6 criteria 2, 3)."""

    move: str
    reply: Optional[str]  # None for the literal "none" (criterion 3)
    eval: int
    depth: int
    count: int


@dataclass
class RefTerashockEntry:
    sfen: str
    moves: List[RefTerashockMove] = field(default_factory=list)


@dataclass
class RefTerashockBook:
    header_version: str = ""
    entries: List[RefTerashockEntry] = field(default_factory=list)


def _parse_int_in_range(text: str, low: int, high: int) -> Optional[int]:
    try:
        value = int(text)
    except ValueError:
        return None
    return value if low <= value <= high else None


class RefBookDBParser:
    """Line-oriented reference parser, following Requirement 6's text alone."""

    def __init__(self) -> None:
        self.header_version = ""
        self.entries: List[RefTerashockEntry] = []
        self.rejected_lines: List[Tuple[int, str]] = []

    @property
    def book(self) -> RefTerashockBook:
        return RefTerashockBook(header_version=self.header_version, entries=self.entries)

    def parse_text(self, text: str) -> RefTerashockBook:
        for line_number, line in enumerate(text.splitlines(), start=1):
            self.parse_line(line_number, line)
        return self.book

    def parse_line(self, line_number: int, line: str) -> None:
        if not line.strip():
            # Requirement 6.12
            return

        if line.startswith("#"):
            match = _HEADER_LINE_RE.match(line)
            if match:
                # Requirement 6.10
                self.header_version = match.group(1)
            # else: an ordinary comment (Requirement 6.4). Neither case rejects.
            return

        if line.startswith("sfen "):
            remainder = line[len("sfen "):]
            trimmed = remainder.strip()
            if not (1 <= len(trimmed) <= MAX_SFEN_LEN):
                self._reject(line_number, line)
                return
            # Requirement 6.1
            self.entries.append(RefTerashockEntry(sfen=trimmed))
            return

        fields = line.split()
        if len(fields) != 5:
            # Requirement 6.5: field count other than five.
            self._reject(line_number, line)
            return

        if not self.entries:
            # Requirement 6.11
            self._reject(line_number, line)
            return

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

        reply = None if reply_str == _NONE_REPLY else reply_str
        self.entries[-1].moves.append(
            RefTerashockMove(
                move=move_str, reply=reply, eval=eval_value, depth=depth_value, count=count_value
            )
        )

    def _reject(self, line_number: int, content: str) -> None:
        self.rejected_lines.append((line_number, content))


class RefBookDBPrinter:
    """Reference printer, following Requirement 6 criteria 6 and 7 alone."""

    HEADER_VERSION = "1.00"

    def format_book(self, book: RefTerashockBook) -> str:
        lines = [
            f"#YANEURAOU-DB2016 {self.HEADER_VERSION}",
            f"# NOE:{len(book.entries)}",
        ]
        for entry in book.entries:
            lines.append(f"sfen {entry.sfen}")
            for move in entry.moves:
                reply_text = _NONE_REPLY if move.reply is None else move.reply
                lines.append(f"{move.move} {reply_text} {move.eval} {move.depth} {move.count}")
        return "\n".join(lines) + "\n"


def ref_parse_book_db(text: str) -> RefTerashockBook:
    return RefBookDBParser().parse_text(text)


def ref_format_book_db(book: RefTerashockBook) -> str:
    return RefBookDBPrinter().format_book(book)


__all__ = [
    "RefTerashockMove",
    "RefTerashockEntry",
    "RefTerashockBook",
    "RefBookDBParser",
    "RefBookDBPrinter",
    "ref_parse_book_db",
    "ref_format_book_db",
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
