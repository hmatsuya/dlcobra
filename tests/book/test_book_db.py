"""Property tests for ``dlshogi.book.book_db`` (tasks 5.3, 5.4).

Implements:

- **Property 20: YaneuraOu `.db` printer/parser round trip**
  (Requirements 6.1, 6.2, 6.3, 6.6, 6.7, 6.8) -- task 5.3.
- **Property 21: `.db` text round trip and noise tolerance**
  (Requirements 6.4, 6.5, 6.9, 6.10, 6.11, 6.12) -- task 5.4.

Both properties cross-check the real implementation
(``dlshogi.book.book_db``) against the independent reference
(``tests/book/reference/book_db.py``), not only against each
implementation's own inverse -- per design.md's explicit instruction for
both properties.

The move-field generator draws from actual legal moves of self-play
positions (see ``tests/book/reference/book_db.py``'s module docstring for
why): the real implementation resolves and legality-checks every
candidate move against a board set from the entry's SFEN, which
Requirement 6's text does not itself demand, so a generator that produced
arbitrary 1-7 character move-field text would exercise a real rejection
path the reference implementation has no matching concept of, making the
two non-comparable on those inputs. Property 21's noise-line generator is
unaffected, since noise (wrong field count, out-of-range numbers, ...) is
rejected by both implementations for the same textual reasons.
"""

from __future__ import annotations

import random

import cshogi
from hypothesis import given, settings
from hypothesis import strategies as st

from dlshogi.book.book_db import (
    BookDBParser,
    BookDBPrinter,
    TerashockBook,
    TerashockEntry,
    TerashockMove,
    format_book_db,
    parse_book_db,
)
from tests.book.reference.book_db import (
    RefTerashockBook,
    RefTerashockEntry,
    RefTerashockMove,
    ref_format_book_db,
    ref_parse_book_db,
)
from tests.book.strategies import self_play_sfen

# ---------------------------------------------------------------------------
# Entry-list generation: legal-move-backed entries, so the real
# implementation's board-based legality check never rejects a
# well-formed line the reference implementation would accept.
# ---------------------------------------------------------------------------


@st.composite
def _entry_strategy(draw):
    sfen = draw(self_play_sfen(max_ply=40))
    board = cshogi.Board(sfen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return TerashockEntry(sfen=sfen, moves=[]), RefTerashockEntry(sfen=sfen, moves=[])

    n_moves = draw(st.integers(min_value=0, max_value=min(len(legal_moves), 6)))
    chosen_moves = draw(
        st.lists(st.sampled_from(legal_moves), min_size=n_moves, max_size=n_moves)
    )

    real_moves = []
    ref_moves = []
    for move in chosen_moves:
        move_usi = cshogi.move_to_usi(move)
        move16 = cshogi.move16(move)

        has_reply = draw(st.booleans())
        reply_usi = None
        reply16 = None
        if has_reply:
            child_board = board.copy()
            child_board.push(move)
            reply_moves = list(child_board.legal_moves)
            if reply_moves:
                reply_move = draw(st.sampled_from(reply_moves))
                reply_usi = cshogi.move_to_usi(reply_move)
                reply16 = cshogi.move16(reply_move)

        eval_value = draw(
            st.sampled_from([-32000, -1, 0, 1, 32000]) | st.integers(-32000, 32000)
        )
        depth_value = draw(st.sampled_from([0, 1, 127]) | st.integers(0, 127))
        count_value = draw(
            st.sampled_from([0, 1, 2**32 - 1]) | st.integers(0, 2**32 - 1)
        )

        real_moves.append(
            TerashockMove(
                move16=move16, reply16=reply16, eval=eval_value, depth=depth_value, count=count_value
            )
        )
        ref_moves.append(
            RefTerashockMove(
                move=move_usi, reply=reply_usi, eval=eval_value, depth=depth_value, count=count_value
            )
        )

    return TerashockEntry(sfen=sfen, moves=real_moves), RefTerashockEntry(sfen=sfen, moves=ref_moves)


@st.composite
def _entry_list_strategy(draw, max_entries=8):
    n = draw(st.integers(min_value=0, max_value=max_entries))
    pairs = draw(st.lists(_entry_strategy(), min_size=n, max_size=n))
    real_entries = [p[0] for p in pairs]
    ref_entries = [p[1] for p in pairs]
    return real_entries, ref_entries


def _terashock_moves_equal(a: TerashockMove, b: TerashockMove) -> bool:
    return (a.move16, a.reply16, a.eval, a.depth, a.count) == (
        b.move16,
        b.reply16,
        b.eval,
        b.depth,
        b.count,
    )


def _terashock_entries_equal(a: TerashockEntry, b: TerashockEntry) -> bool:
    if a.sfen != b.sfen or len(a.moves) != len(b.moves):
        return False
    return all(_terashock_moves_equal(x, y) for x, y in zip(a.moves, b.moves))


def _terashock_books_equal(a: TerashockBook, b: TerashockBook) -> bool:
    if len(a.entries) != len(b.entries):
        return False
    return all(_terashock_entries_equal(x, y) for x, y in zip(a.entries, b.entries))


# ---------------------------------------------------------------------------
# Property 20: YaneuraOu .db printer/parser round trip
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 20: YaneuraOu .db printer/parser round trip
# Validates: Requirements 6.1, 6.2, 6.3, 6.6, 6.7, 6.8


@given(_entry_list_strategy())
@settings(max_examples=60)
def test_printer_output_starts_with_the_fixed_header(entries_pair):
    real_entries, _ = entries_pair
    book = TerashockBook(entries=real_entries)
    text = format_book_db(book)
    lines = text.splitlines()
    assert lines[0] == "#YANEURAOU-DB2016 1.00"
    assert lines[1] == f"# NOE:{len(real_entries)}"


@given(_entry_list_strategy())
@settings(max_examples=60)
def test_printer_output_has_five_single_space_separated_fields_per_move(entries_pair):
    real_entries, _ = entries_pair
    text = format_book_db(TerashockBook(entries=real_entries))
    for line in text.splitlines()[2:]:
        if line.startswith("sfen "):
            continue
        fields = line.split(" ")
        assert len(fields) == 5, f"line {line!r} did not have exactly five space-separated fields"
        assert fields[1]  # the opponent-reply field, "none" or a USI move, is never empty


@given(_entry_list_strategy())
@settings(max_examples=60)
def test_real_printer_then_parser_round_trips_to_an_equal_entry_list(entries_pair):
    real_entries, _ = entries_pair
    book = TerashockBook(entries=real_entries)
    text = format_book_db(book)
    reparsed = parse_book_db(text)
    assert _terashock_books_equal(reparsed, book)


@given(_entry_list_strategy())
@settings(max_examples=60)
def test_real_implementation_cross_checked_against_the_independent_reference(entries_pair):
    """The core of Property 20: printing with the real printer and parsing
    with the independent reference parser (and vice versa) must agree with
    the real implementation's own round trip -- not just with itself."""
    real_entries, ref_entries = entries_pair
    real_book = TerashockBook(entries=real_entries)
    ref_book = RefTerashockBook(entries=ref_entries)

    real_text = format_book_db(real_book)
    ref_text = ref_format_book_db(ref_book)

    # The two printers must agree byte for byte on well-formed input: both
    # follow the same fixed header, "sfen "-prefix, and five-field-with-
    # single-space format (Requirement 6.6, 6.7), and every move field the
    # generator produces is realisable by both (it comes from a real legal
    # move, and "none" is spelled identically).
    assert real_text == ref_text

    # And parsing either printer's output with either parser reproduces
    # the original entry list (allowing for the move16-vs-text
    # representation difference, checked via the USI string).
    real_reparsed = parse_book_db(ref_text)
    for real_entry, ref_entry in zip(real_reparsed.entries, ref_entries):
        assert real_entry.sfen == ref_entry.sfen
        assert len(real_entry.moves) == len(ref_entry.moves)
        for real_move, ref_move in zip(real_entry.moves, ref_entry.moves):
            assert cshogi.move_to_usi(real_move.move16) == ref_move.move
            if ref_move.reply is None:
                assert real_move.reply16 is None
            else:
                assert cshogi.move_to_usi(real_move.reply16) == ref_move.reply
            assert real_move.eval == ref_move.eval
            assert real_move.depth == ref_move.depth
            assert real_move.count == ref_move.count


# ---------------------------------------------------------------------------
# Property 21: .db text round trip and noise tolerance
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 21: .db text round trip and noise tolerance
# Validates: Requirements 6.4, 6.5, 6.9, 6.10, 6.11, 6.12


@given(_entry_list_strategy())
@settings(max_examples=60)
def test_well_formed_text_round_trips_through_parse_then_print(entries_pair):
    real_entries, _ = entries_pair
    original_text = format_book_db(TerashockBook(entries=real_entries))
    parsed = parse_book_db(original_text)
    reprinted = format_book_db(parsed)
    reparsed_again = parse_book_db(reprinted)
    assert _terashock_books_equal(parse_book_db(original_text), reparsed_again)


def _noise_lines():
    """Lines that Requirement 6.5 rejects, plus comments and header lines
    that Requirement 6.4/6.10 accept without an entry list changing."""
    return st.one_of(
        st.just(""),  # blank (6.12), never reported
        st.just("   "),  # whitespace-only (6.12)
        st.just("# a plain comment"),  # 6.4
        st.just("#YANEURAOU-DB2016 9.99"),  # 6.10, records but doesn't reject
        st.just("wrong field count here"),  # 6.5: 4 fields
        st.just("a b c d e f"),  # 6.5: 6 fields
        st.just("mv none 99999 5 10"),  # 6.5: eval out of range
        st.just("mv none 0 999 10"),  # 6.5: depth out of range
        st.just("mv none 0 5 99999999999"),  # 6.5: count out of range
        st.just("mv none notanumber 5 10"),  # 6.5: non-integer eval
    )


@given(
    entries_pair=_entry_list_strategy(max_entries=10),
    noise_specs=st.lists(
        st.tuples(st.integers(min_value=0, max_value=100), _noise_lines()), max_size=15
    ),
)
@settings(max_examples=60)
def test_noise_insertions_leave_the_parsed_entry_list_unchanged(entries_pair, noise_specs):
    real_entries, _ = entries_pair
    clean_text = format_book_db(TerashockBook(entries=real_entries))
    clean_lines = clean_text.splitlines()

    noisy_lines = list(clean_lines)
    # Insert noise lines at clamped positions, latest-index-first so
    # earlier insertion indices remain valid after each insert.
    for index, noise_line in sorted(noise_specs, key=lambda t: -t[0]):
        pos = min(index, len(noisy_lines))
        noisy_lines.insert(pos, noise_line)
    noisy_text = "\n".join(noisy_lines) + "\n"

    clean_parsed = parse_book_db(clean_text)
    noisy_parsed = parse_book_db(noisy_text)

    assert _terashock_books_equal(clean_parsed, noisy_parsed)


def test_candidate_move_line_before_any_sfen_line_is_rejected_and_reported():
    text = "mv none 0 5 10\nsfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\n"
    parser = BookDBParser()
    parser.parse_text(text)
    assert (1, "mv none 0 5 10") in parser.rejected_lines
    assert len(parser.entries) == 1


def test_header_version_records_the_last_seen_token():
    text = (
        "#YANEURAOU-DB2016 1.00\n"
        "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\n"
        "#YANEURAOU-DB2016 2.50\n"
    )
    book = parse_book_db(text)
    assert book.header_version == "2.50"


def test_blank_lines_are_never_reported():
    text = "\n   \n\t\nsfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\n\n"
    parser = BookDBParser()
    parser.parse_text(text)
    assert parser.rejected_lines == []
    assert len(parser.entries) == 1


def test_on_rejected_line_callback_is_invoked_with_line_number_and_content():
    seen = []
    parser = BookDBParser(on_rejected_line=lambda n, c: seen.append((n, c)))
    parser.parse_line(1, "not five fields")
    assert seen == [(1, "not five fields")]
