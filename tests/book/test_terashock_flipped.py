"""Tests for the 180-degree flipped entries ``import_terashock`` adds (parser version 2).

The 2025 petabook lists Black-to-move positions only; YaneuraOu reaches the White-to-move
side through ``FlippedBook``. The PUCT search looks positions up by exact key, so the
importer stores a flipped copy of each entry. The pure-function tests need no database.
"""

from __future__ import annotations

import dataclasses
import io

import cshogi
import pytest

from dlshogi.book import terashock as T
from dlshogi.book.book_db import BookDBPrinter, TerashockEntry, TerashockMove

_START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
# Black holds a knight (34 legal drops) and a bishop can promote (2 promotion moves), so both
# the drop and the promotion encodings are exercised. test_middlegame_really_has_drops_and_promotions
# guards that claim.
_MIDDLE = "l1sgkgsnl/1r5b1/p1pppp1pp/1p4p2/9/2P4P1/PP1PPPP1P/1B5R1/LNSGKGSNL b N 9"


def _entry(sfen: str, n: int = 400) -> TerashockEntry:
    legal = [int(cshogi.move16(m)) for m in cshogi.Board(sfen).legal_moves][:n]
    return TerashockEntry(sfen=sfen, moves=[TerashockMove(m, None, 50 - i, 20, 0) for i, m in enumerate(legal)])


def test_middlegame_really_has_drops_and_promotions():
    # The entries below use the first 400 legal moves; both kinds must be inside that slice.
    legal = [cshogi.move_to_usi(m) for m in list(cshogi.Board(_MIDDLE).legal_moves)[:400]]
    assert any(u[1] == "*" for u in legal) and any(u.endswith("+") for u in legal)


def test_flipped_moves_are_the_rotated_moves_including_drops_and_promotions():
    # move_rotate on a bare move16 must give the move16 of the rotated move, for every legal move.
    for sfen in (_START, _MIDDLE):
        board = cshogi.Board(sfen)
        flipped = T.flipped_entry(_entry(sfen))
        expected = [cshogi.move16(cshogi.move_rotate(m)) for m in board.legal_moves]
        assert [m.move16 for m in flipped.moves] == expected


def test_a_flipped_entry_is_a_white_to_move_position_whose_moves_are_all_legal():
    for sfen in (_START, _MIDDLE):
        flipped = T.flipped_entry(_entry(sfen))
        board = cshogi.Board(flipped.sfen)
        assert board.turn == cshogi.WHITE
        assert flipped.moves
        for move in flipped.moves:
            assert board.move_from_move16(move.move16) in board.legal_moves


def test_a_single_flip_keeps_eval_depth_and_count_and_rotates_the_reply():
    # eval is a score for the side to move, so it must not change sign (YaneuraOu copies it too).
    legal = [int(cshogi.move16(m)) for m in cshogi.Board(_START).legal_moves]
    reply = int(cshogi.move16(next(iter(cshogi.Board("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").legal_moves))))
    original = TerashockEntry(sfen=_START, moves=[TerashockMove(legal[0], reply, 250, 33, 7)])
    flipped = T.flipped_entry(original).moves[0]
    assert (flipped.eval, flipped.depth, flipped.count) == (250, 33, 7)
    assert flipped.move16 == cshogi.move_rotate(legal[0]) and flipped.reply16 == cshogi.move_rotate(reply)
    assert T.flipped_entry(TerashockEntry(sfen=_START, moves=[TerashockMove(legal[0], None, 1, 1, 1)])).moves[0].reply16 is None


def test_flipping_twice_returns_the_original_moves_and_values():
    original = _entry(_MIDDLE)
    back = T.flipped_entry(T.flipped_entry(original))
    assert [m.move16 for m in back.moves] == [m.move16 for m in original.moves]
    assert [(m.eval, m.depth) for m in back.moves] == [(m.eval, m.depth) for m in original.moves]


def test_importer_input_gains_one_flipped_entry_per_black_entry():
    entries = [_entry(_START), _entry(_MIDDLE)]
    out = list(T._entries_with_flips(entries))
    assert len(out) == 4
    # Flips come first and the book's own entries last, so staging (last row wins) prefers the book's own.
    assert [cshogi.Board(e.sfen).turn for e in out] == [cshogi.WHITE, cshogi.WHITE, cshogi.BLACK, cshogi.BLACK]


def _staged(entries):
    """What the import's staging step keeps: the last row per Position_Key."""
    kept = {}
    for e in T._entries_with_flips(entries):
        pk = T.position_key_from_sfen(e.sfen)
        kept[(pk.hi, pk.lo)] = e
    return kept


def test_an_entry_the_book_already_holds_beats_a_flip_of_another_entry():
    black = _entry(_START)
    flipped = T.flipped_entry(black)
    own = TerashockEntry(sfen=flipped.sfen, moves=[flipped.moves[0]])   # the book's own White entry, one move
    kept = _staged([black, own])
    assert len(kept) == 2
    white = [e for e in kept.values() if cshogi.Board(e.sfen).turn == cshogi.WHITE]
    assert len(white) == 1 and len(white[0].moves) == 1                  # the own entry, not the 40-move flip


def test_duplicate_positions_flip_from_the_entry_the_import_keeps():
    # The book can repeat a position with a different ply; staging keeps the LAST row per key,
    # so the White-to-move copy must come from that same row, not from the first.
    first = TerashockEntry(sfen=_START.rsplit(" ", 1)[0] + " 10", moves=_entry(_START, 1).moves[:1])
    last = TerashockEntry(sfen=_START.rsplit(" ", 1)[0] + " 12",
                          moves=[dataclasses.replace(_entry(_START, 1).moves[0], eval=300)])
    kept = _staged([first, last])
    white = [e for e in kept.values() if cshogi.Board(e.sfen).turn == cshogi.WHITE]
    black = [e for e in kept.values() if cshogi.Board(e.sfen).turn == cshogi.BLACK]
    assert len(kept) == 2 and len(white) == 1 and len(black) == 1
    assert white[0].moves[0].eval == black[0].moves[0].eval == 300


def test_an_entry_without_moves_is_not_flipped():
    empty = TerashockEntry(sfen=_START, moves=[])
    assert len(list(T._entries_with_flips([empty]))) == 1


# ---------------------------------------------------------------------------
# Through the real importer, against a scratch PostgreSQL database.
# ---------------------------------------------------------------------------

from dlshogi.book.node_store import NodeStore  # noqa: E402
from tests.book.test_interop import _drop_book_tables, _make_config  # noqa: E402


@pytest.mark.db
@pytest.mark.asyncio
async def test_import_stores_both_sides_and_the_flipped_lookup_finds_its_moves(
    pg_scratch_database, pg_conn, tmp_path
):
    await _drop_book_tables(pg_conn)
    buf = io.StringIO()
    printer = BookDBPrinter()
    printer.write_header(buf, 2)
    for entry in (_entry(_START, 3), _entry(_MIDDLE, 3)):
        printer.write_entry(buf, entry)
    path = tmp_path / "book.db"
    path.write_text(buf.getvalue())
    cfg = dataclasses.replace(_make_config(pg_scratch_database, root_sfen=_START), terashock_book=str(path))
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    stats = await T.import_terashock(store.pool, cfg)
    assert stats.entry_count == 4

    index = T.TerashockIndex(store.pool, 256 * 1024 * 1024)
    flipped = T.flipped_entry(_entry(_START, 3))
    result = await index.lookup(T.position_key_from_sfen(cshogi.Board(flipped.sfen).sfen()))
    assert result is not None
    assert [int(m) for m in result.moves["move16"]] == [m.move16 for m in flipped.moves]
