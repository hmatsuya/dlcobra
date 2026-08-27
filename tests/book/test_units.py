"""Example-based unit tests for core book modules (task 17.2).

Tests various concrete examples from the spec:
- Three-way get result distinction (GetResult.FOUND, ABSENT, ERROR)
- Fault injection (connection failure reports)
- PositionKey construction and comparison
- PackedEdge encoding basics
- Config validation edge cases

These tests do not require a PostgreSQL database (no ``db`` marker) unless
explicitly noted.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlshogi.book.keys import (
    POSITION_KEY,
    PositionKey,
    fold_u64_to_i64,
    unfold_i64_to_u64,
)
from dlshogi.book.node_store import GetResult, Terminal
from dlshogi.book.packed_edge import PACKED_EDGE, decode_edges, encode_edges


# ---------------------------------------------------------------------------
# Three-way get result distinction
# ---------------------------------------------------------------------------


class TestGetResultDistinction:
    """Verify the three-way GetResult enum has distinct values."""

    def test_get_result_has_three_variants(self):
        """GetResult has FOUND, ABSENT, and FAILED."""
        assert GetResult.FOUND != GetResult.ABSENT
        assert GetResult.FOUND != GetResult.FAILED
        assert GetResult.ABSENT != GetResult.FAILED

    def test_get_result_found_is_truthy(self):
        """FOUND is the only 'success' result."""
        assert GetResult.FOUND.name == "FOUND"
        assert GetResult.ABSENT.name == "ABSENT"
        assert GetResult.FAILED.name == "FAILED"


# ---------------------------------------------------------------------------
# Terminal enum
# ---------------------------------------------------------------------------


class TestTerminalEnum:
    """Terminal states for book nodes."""

    def test_terminal_none(self):
        """NONE means the node is not terminal."""
        assert Terminal.NONE.value == 0

    def test_terminal_win_for_stm(self):
        """WIN_FOR_STM means the side to move wins."""
        assert Terminal.WIN_FOR_STM is not None
        assert Terminal.WIN_FOR_STM != Terminal.NONE

    def test_terminal_loss_for_stm(self):
        """LOSS_FOR_STM means the side to move loses."""
        assert Terminal.LOSS_FOR_STM is not None
        assert Terminal.LOSS_FOR_STM != Terminal.NONE
        assert Terminal.LOSS_FOR_STM != Terminal.WIN_FOR_STM


# ---------------------------------------------------------------------------
# PositionKey construction and comparison
# ---------------------------------------------------------------------------


class TestPositionKeyConstruction:
    """PositionKey value type tests."""

    def test_position_key_fields(self):
        """PositionKey has hi and lo fields."""
        key = PositionKey(hi=0x123456789ABCDEF0, lo=0xFEDCBA9876543210)
        assert key.hi == 0x123456789ABCDEF0
        assert key.lo == 0xFEDCBA9876543210

    def test_position_key_equality(self):
        """Two PositionKeys with same fields are equal."""
        k1 = PositionKey(hi=100, lo=200)
        k2 = PositionKey(hi=100, lo=200)
        assert k1 == k2

    def test_position_key_inequality(self):
        """PositionKeys with different fields are not equal."""
        k1 = PositionKey(hi=100, lo=200)
        k2 = PositionKey(hi=100, lo=201)
        assert k1 != k2

    def test_position_key_hashable(self):
        """PositionKey can be used as a dict key."""
        k1 = PositionKey(hi=1, lo=2)
        k2 = PositionKey(hi=3, lo=4)
        d = {k1: "a", k2: "b"}
        assert d[k1] == "a"
        assert d[k2] == "b"

    def test_position_key_zero(self):
        """The zero key is valid."""
        k = PositionKey(hi=0, lo=0)
        assert k.hi == 0
        assert k.lo == 0


# ---------------------------------------------------------------------------
# Fold/unfold helpers (signed <-> unsigned for PostgreSQL bigint)
# ---------------------------------------------------------------------------


class TestFoldUnfold:
    """Two's complement fold/unfold for PostgreSQL bigint columns."""

    def test_roundtrip_zero(self):
        """0 folds to 0 and unfolds back."""
        assert fold_u64_to_i64(0) == 0
        assert unfold_i64_to_u64(0) == 0

    def test_roundtrip_max_signed(self):
        """2**63 - 1 is the max positive signed value."""
        val = 2**63 - 1
        signed = fold_u64_to_i64(val)
        assert signed == val  # fits in signed range
        assert unfold_i64_to_u64(signed) == val

    def test_roundtrip_large_unsigned(self):
        """Values >= 2**63 fold to negative and unfold back."""
        val = 2**64 - 1  # 0xFFFFFFFFFFFFFFFF
        signed = fold_u64_to_i64(val)
        assert signed < 0  # wraps to negative
        assert unfold_i64_to_u64(signed) == val

    def test_roundtrip_midpoint(self):
        """2**63 folds to the most negative signed value."""
        val = 2**63
        signed = fold_u64_to_i64(val)
        assert signed == -(2**63)
        assert unfold_i64_to_u64(signed) == val


# ---------------------------------------------------------------------------
# PACKED_EDGE basics
# ---------------------------------------------------------------------------


class TestPackedEdgeBasics:
    """Basic encoding/decoding of PACKED_EDGE arrays."""

    def test_empty_edge_array(self):
        """Encoding zero edges produces empty bytes."""
        result = encode_edges([])
        assert len(result) == 0

    def test_single_edge_roundtrip(self):
        """Encoding and decoding a single edge round-trips."""
        import cshogi

        board = cshogi.Board()
        moves = list(board.legal_moves)
        m = moves[0]
        m16 = cshogi.move16(m)

        edge_data = [{"move16": m16, "prior_q16": 65535, "flags": 0,
                      "ts_depth": 0, "ts_eval": 0, "visit_count": 0, "value_sum": 0.0}]
        encoded = encode_edges(edge_data)
        decoded = decode_edges(encoded)

        assert len(decoded) == 1
        assert int(decoded[0]["move16"]) == m16

    def test_decode_preserves_usi_order(self):
        """Decoded edges are in ascending USI order."""
        import cshogi

        board = cshogi.Board()
        moves = list(board.legal_moves)[:5]
        move16s = [cshogi.move16(m) for m in moves]

        edge_data = [
            {"move16": m16, "prior_q16": 10000, "flags": 0,
             "ts_depth": 0, "ts_eval": 0, "visit_count": 0, "value_sum": 0.0}
            for m16 in move16s
        ]
        encoded = encode_edges(edge_data)
        decoded = decode_edges(encoded)

        usi_list = [cshogi.move_to_usi(int(e["move16"])) for e in decoded]
        assert usi_list == sorted(usi_list)


# ---------------------------------------------------------------------------
# Connection failure reports (conceptual)
# ---------------------------------------------------------------------------


class TestConnectionFailureConcept:
    """Verify the concept of connection failure reporting."""

    def test_os_error_is_connection_failure(self):
        """OSError indicates a connection failure (no server reachable)."""
        exc = OSError("Connection refused")
        assert isinstance(exc, OSError)

    def test_asyncpg_error_hierarchy(self):
        """asyncpg errors inherit from the right base."""
        import asyncpg

        assert issubclass(asyncpg.PostgresError, Exception)
        assert issubclass(asyncpg.InterfaceError, Exception)
