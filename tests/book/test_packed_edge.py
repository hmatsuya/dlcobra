"""Tests for ``dlshogi.book.packed_edge`` (task 3.6).

Covers the module-level dtype layout assertions (itemsize, field offsets,
`isalignedstruct`, little-endian format), and the encode/decode round trip
including the ascending-USI order assertion and the canonical-zero
encoding of absent Terashock fields.

These are ordinary example-based tests plus one small `@given`-decorated
round-trip test; the full hypothesis-driven codec/layout property (a
dedicated Property number) is not among the 48 named design.md properties
-- Property 1 (Node_Store round trip, task 7.7) is what exercises
`PACKED_EDGE` at scale once the Node_Store exists. This file exists so the
layout assertions and the codec have their own fast, database-free
regression coverage now, per task 3.6's own description ("packed edge
layout and codec tests").
"""

from __future__ import annotations

import random

import cshogi
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dlshogi.book.packed_edge import (
    PACKED_EDGE,
    PACKED_TS_MOVE,
    decode_edges,
    encode_edges,
    patch_edges,
)
from tests.book.strategies import self_play_board_with_moves

# ---------------------------------------------------------------------------
# Dtype layout assertions (mirroring the module-level asserts, so a
# regression is caught by a named test rather than only by an import-time
# AssertionError with no test attribution)
# ---------------------------------------------------------------------------


def test_packed_edge_itemsize_is_20_bytes():
    assert PACKED_EDGE.itemsize == 20


def test_packed_edge_field_offsets():
    offsets = [PACKED_EDGE.fields[n][1] for n in PACKED_EDGE.names]
    assert offsets == [0, 2, 4, 5, 6, 8, 12]


def test_packed_edge_is_not_aligned_struct():
    """No padding: numpy's default (unaligned) struct packing is used."""
    assert PACKED_EDGE.isalignedstruct is False


def test_packed_edge_field_formats_are_little_endian():
    # Every multi-byte field carries an explicit "<" (little-endian) format;
    # the two single-byte fields need no endianness marker.
    formats = {name: PACKED_EDGE.fields[name][0].str for name in PACKED_EDGE.names}
    assert formats["move16"] == "<u2"
    assert formats["prior_q16"] == "<u2"
    assert formats["ts_depth"] in ("|u1", "u1")
    assert formats["flags"] in ("|u1", "u1")
    assert formats["ts_eval"] == "<i2"
    assert formats["visit_count"] == "<u4"
    assert formats["value_sum"] == "<f8"


def test_packed_ts_move_itemsize_is_11_bytes():
    assert PACKED_TS_MOVE.itemsize == 11


def test_packed_ts_move_field_offsets():
    offsets = [PACKED_TS_MOVE.fields[n][1] for n in PACKED_TS_MOVE.names]
    assert offsets == [0, 2, 4, 6, 7]


# ---------------------------------------------------------------------------
# Encode/decode round trip
# ---------------------------------------------------------------------------


def _edge_dict(move16, prior_q16=0, flags=0, ts_depth=0, ts_eval=0, visit_count=0, value_sum=0.0):
    return {
        "move16": move16,
        "prior_q16": prior_q16,
        "flags": flags,
        "ts_depth": ts_depth,
        "ts_eval": ts_eval,
        "visit_count": visit_count,
        "value_sum": value_sum,
    }


def test_encode_decode_round_trip_preserves_field_values():
    edges = [
        _edge_dict(cshogi.move16(0x1A2B), prior_q16=12345, visit_count=7, value_sum=1.5),
        _edge_dict(cshogi.move16(0x0142), prior_q16=1, flags=1, ts_depth=5, ts_eval=-321),
    ]
    blob = encode_edges(edges)
    decoded = decode_edges(blob)
    assert len(decoded) == 2
    by_move16 = {int(row["move16"]): row for row in decoded}
    assert by_move16[edges[0]["move16"]]["prior_q16"] == 12345
    assert by_move16[edges[0]["move16"]]["visit_count"] == 7
    assert by_move16[edges[0]["move16"]]["value_sum"] == 1.5
    assert by_move16[edges[1]["move16"]]["flags"] == 1
    assert by_move16[edges[1]["move16"]]["ts_depth"] == 5
    assert by_move16[edges[1]["move16"]]["ts_eval"] == -321


def test_decode_is_a_zero_copy_view():
    blob = encode_edges([_edge_dict(cshogi.move16(0x0142))])
    decoded = decode_edges(blob)
    # np.frombuffer over `bytes` yields a read-only array with no copy.
    assert decoded.flags["OWNDATA"] is False
    assert decoded.flags["WRITEABLE"] is False


def test_absent_terashock_fields_are_canonical_zero():
    """flags bit 0 clear => ts_eval and ts_depth are written as 0, even if
    the caller supplied nonzero values for them (Property 30's idempotence
    depends on this canonical encoding)."""
    edges = [_edge_dict(cshogi.move16(0x0142), flags=0, ts_depth=99, ts_eval=-500)]
    blob = encode_edges(edges)
    decoded = decode_edges(blob)
    assert decoded[0]["flags"] == 0
    assert decoded[0]["ts_depth"] == 0
    assert decoded[0]["ts_eval"] == 0


def test_encode_edges_of_empty_list_round_trips():
    blob = encode_edges([])
    decoded = decode_edges(blob)
    assert len(decoded) == 0


def test_encoding_the_same_logical_edge_set_is_deterministic():
    """Encoding the same edges in a different input order yields byte-identical
    output (the encoder always sorts), which is the mechanical form of
    Property 30's "independent expansions are confluent" for one node."""
    edges = [
        _edge_dict(cshogi.move16(0x1A2B), visit_count=3),
        _edge_dict(cshogi.move16(0x0142), visit_count=9),
    ]
    blob_a = encode_edges(edges)
    blob_b = encode_edges(list(reversed(edges)))
    assert blob_a == blob_b


# ---------------------------------------------------------------------------
# Ascending-USI order (the fix for the move16-vs-USI ordering divergence:
# promotion sets bit 14 of move16, which is not equivalent to lexicographic
# USI order -- see packed_edge.py's encode_edges docstring)
# ---------------------------------------------------------------------------


def test_encode_edges_orders_output_by_usi_not_by_move16():
    """A promoting move and its non-promoting counterpart must sort adjacently
    by USI ("3c1a" then "3c1a+"), not be separated by every non-promoting
    move as numeric move16 order would put them."""
    board = cshogi.Board()
    rng = random.Random(3)
    for _ in range(20):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[rng.randrange(len(moves))])
    moves = list(board.legal_moves)
    usis = [cshogi.move_to_usi(m) for m in moves]
    # This position (fixed seed) is known to contain a promotion whose
    # move16 value is numerically far from its non-promoting counterpart.
    assert any(u.endswith("+") for u in usis), "fixture position lost its promotion move"

    edges = [_edge_dict(cshogi.move16(m)) for m in moves]
    blob = encode_edges(edges)
    decoded = decode_edges(blob)
    decoded_usis = [cshogi.move_to_usi(int(m16)) for m16 in decoded["move16"]]
    assert decoded_usis == sorted(decoded_usis)
    # Would fail if the encoder still sorted by numeric move16.
    numeric_order_usis = [
        cshogi.move_to_usi(m) for m in sorted((cshogi.move16(m) for m in moves))
    ]
    assert decoded_usis != numeric_order_usis


@given(self_play_board_with_moves())
@settings(max_examples=200)
def test_encoded_edges_are_always_ascending_usi_order(position_and_moves):
    sfen, moves16 = position_and_moves
    if not moves16:
        return
    edges = [_edge_dict(m16) for m16 in moves16]
    blob = encode_edges(edges)
    decoded = decode_edges(blob)
    usis = [cshogi.move_to_usi(int(m16)) for m16 in decoded["move16"]]
    assert usis == sorted(usis)
    assert len(set(usis)) == len(usis)  # pairwise distinct


# ---------------------------------------------------------------------------
# patch_edges
# ---------------------------------------------------------------------------


def test_patch_edges_applies_deltas_to_named_moves():
    edges = [
        _edge_dict(cshogi.move16(0x1A2B), visit_count=2, value_sum=0.5),
        _edge_dict(cshogi.move16(0x0142), visit_count=5, value_sum=1.0),
    ]
    blob = encode_edges(edges)
    target_move16 = edges[0]["move16"]
    patched = patch_edges(blob, [target_move16], [3], [0.25])
    decoded = decode_edges(patched)
    row = decoded[decoded["move16"] == target_move16][0]
    assert int(row["visit_count"]) == 5
    assert row["value_sum"] == pytest.approx(0.75)


def test_patch_edges_raises_for_unknown_move16():
    edges = [_edge_dict(cshogi.move16(0x1A2B))]
    blob = encode_edges(edges)
    with pytest.raises(KeyError):
        patch_edges(blob, [0x7FFF], [1], [0.0])


def test_patch_edges_handles_promotion_pairs_correctly():
    """Regression test for the move16-vs-USI-order bug: patch_edges must
    locate the correct record even when the blob's ascending-USI order
    differs from ascending-move16 order (as it does whenever a promoting
    move and its non-promoting counterpart are both present)."""
    board = cshogi.Board()
    rng = random.Random(3)
    for _ in range(20):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[rng.randrange(len(moves))])
    moves = list(board.legal_moves)
    promo = next(m for m in moves if cshogi.move_to_usi(m).endswith("+"))
    non_promo_usi = cshogi.move_to_usi(promo)[:-1]
    non_promo = next(m for m in moves if cshogi.move_to_usi(m) == non_promo_usi)

    edges = [_edge_dict(cshogi.move16(m), visit_count=0, value_sum=0.0) for m in moves]
    blob = encode_edges(edges)

    promo16 = cshogi.move16(promo)
    non_promo16 = cshogi.move16(non_promo)
    patched = patch_edges(blob, [promo16, non_promo16], [1, 2], [0.1, 0.2])
    decoded = decode_edges(patched)
    by_move16 = {int(r["move16"]): r for r in decoded}
    assert int(by_move16[promo16]["visit_count"]) == 1
    assert by_move16[promo16]["value_sum"] == pytest.approx(0.1)
    assert int(by_move16[non_promo16]["visit_count"]) == 2
    assert by_move16[non_promo16]["value_sum"] == pytest.approx(0.2)


@given(
    self_play_board_with_moves(),
    st.data(),
)
@settings(max_examples=100)
def test_patch_edges_round_trip_is_additive(position_and_moves, data):
    sfen, moves16 = position_and_moves
    if not moves16:
        return
    edges = [_edge_dict(m16) for m16 in moves16]
    blob = encode_edges(edges)

    n = len(moves16)
    k = data.draw(st.integers(min_value=1, max_value=n))
    chosen = data.draw(
        st.lists(st.sampled_from(moves16), min_size=k, max_size=k, unique=True)
    )
    visit_deltas = [data.draw(st.integers(min_value=0, max_value=5)) for _ in chosen]
    value_deltas = [data.draw(st.floats(min_value=0, max_value=1, allow_nan=False)) for _ in chosen]

    patched = patch_edges(blob, chosen, visit_deltas, value_deltas)
    decoded = decode_edges(patched)
    by_move16 = {int(r["move16"]): r for r in decoded}
    for m16, dv, dw in zip(chosen, visit_deltas, value_deltas):
        assert int(by_move16[m16]["visit_count"]) == dv
        assert by_move16[m16]["value_sum"] == pytest.approx(dw)
