"""Smoke tests for core book module imports and constants (task 17.3).

Lightweight sanity checks that verify:
- Package imports work without side effects
- Key dtype sizes are correct
- Schema file exists (if expected)
- Critical constants have the expected values

No PostgreSQL required (no ``db`` marker).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """Verify that all book modules import without error or side effect."""

    def test_import_keys(self):
        """dlshogi.book.keys imports cleanly."""
        from dlshogi.book import keys

        assert hasattr(keys, "POSITION_KEY")
        assert hasattr(keys, "PositionKey")
        assert hasattr(keys, "position_key_from_sfen")

    def test_import_packed_edge(self):
        """dlshogi.book.packed_edge imports cleanly."""
        from dlshogi.book import packed_edge

        assert hasattr(packed_edge, "PACKED_EDGE")
        assert hasattr(packed_edge, "encode_edges")
        assert hasattr(packed_edge, "decode_edges")

    def test_import_config(self):
        """dlshogi.book.config imports cleanly."""
        from dlshogi.book import config

        assert hasattr(config, "BookConfig")

    def test_import_node_store(self):
        """dlshogi.book.node_store imports cleanly."""
        from dlshogi.book import node_store

        assert hasattr(node_store, "NodeStore")
        assert hasattr(node_store, "GetResult")
        assert hasattr(node_store, "Terminal")

    def test_import_search(self):
        """dlshogi.book.search imports cleanly."""
        from dlshogi.book import search

        assert hasattr(search, "InFlightSet")
        assert hasattr(search, "select_edge")

    def test_import_propagate(self):
        """dlshogi.book.propagate imports cleanly."""
        from dlshogi.book import propagate

        assert hasattr(propagate, "propagate")
        assert hasattr(propagate, "PropagationStats")

    def test_import_export(self):
        """dlshogi.book.export imports cleanly."""
        from dlshogi.book import export

        assert hasattr(export, "ExportCounts")
        assert hasattr(export, "_compute_score")

    def test_import_book_db(self):
        """dlshogi.book.book_db imports cleanly."""
        from dlshogi.book import book_db

        assert hasattr(book_db, "BookDBPrinter")

    def test_import_repetition(self):
        """dlshogi.book.repetition imports cleanly."""
        from dlshogi.book import repetition

        assert hasattr(repetition, "RepetitionResolver")

    def test_import_evaluator(self):
        """dlshogi.book.evaluator imports cleanly."""
        from dlshogi.book import evaluator

        assert hasattr(evaluator, "Evaluator")

    def test_import_prior_mixer(self):
        """dlshogi.book.prior_mixer imports cleanly."""
        from dlshogi.book import prior_mixer

        assert hasattr(prior_mixer, "mix_priors")
        assert hasattr(prior_mixer, "win_rate")


# ---------------------------------------------------------------------------
# POSITION_KEY dtype
# ---------------------------------------------------------------------------


class TestPositionKeyDtype:
    """POSITION_KEY numpy dtype has the right shape."""

    def test_position_key_itemsize_16(self):
        """POSITION_KEY.itemsize == 16 (128 bits)."""
        from dlshogi.book.keys import POSITION_KEY

        assert POSITION_KEY.itemsize == 16

    def test_position_key_fields(self):
        """POSITION_KEY has 'hi' and 'lo' fields, both u8."""
        from dlshogi.book.keys import POSITION_KEY

        assert "hi" in POSITION_KEY.names
        assert "lo" in POSITION_KEY.names
        assert POSITION_KEY["hi"].itemsize == 8
        assert POSITION_KEY["lo"].itemsize == 8

    def test_position_key_little_endian(self):
        """POSITION_KEY fields are little-endian unsigned 64-bit."""
        from dlshogi.book.keys import POSITION_KEY

        assert POSITION_KEY["hi"].byteorder in ("<", "=")
        assert POSITION_KEY["lo"].byteorder in ("<", "=")


# ---------------------------------------------------------------------------
# PACKED_EDGE dtype
# ---------------------------------------------------------------------------


class TestPackedEdgeDtype:
    """PACKED_EDGE numpy dtype has the right shape."""

    def test_packed_edge_itemsize_20(self):
        """PACKED_EDGE.itemsize == 20 bytes."""
        from dlshogi.book.packed_edge import PACKED_EDGE

        assert PACKED_EDGE.itemsize == 20

    def test_packed_edge_field_names(self):
        """PACKED_EDGE has the expected field names."""
        from dlshogi.book.packed_edge import PACKED_EDGE

        expected_fields = {"move16", "prior_q16", "ts_depth", "flags", "ts_eval", "visit_count", "value_sum"}
        assert set(PACKED_EDGE.names) == expected_fields

    def test_packed_edge_field_offsets(self):
        """PACKED_EDGE field offsets match the spec: [0, 2, 4, 5, 6, 8, 12]."""
        from dlshogi.book.packed_edge import PACKED_EDGE

        offsets = [PACKED_EDGE.fields[name][1] for name in PACKED_EDGE.names]
        assert sorted(offsets) == [0, 2, 4, 5, 6, 8, 12]


# ---------------------------------------------------------------------------
# Schema file existence
# ---------------------------------------------------------------------------


class TestSchemaFileExistence:
    """Schema SQL file exists in the expected location."""

    def test_schema_sql_exists(self):
        """dlshogi/book/sql/schema.sql exists."""
        schema_path = Path(__file__).resolve().parents[2] / "dlshogi" / "book" / "sql" / "schema.sql"
        assert schema_path.exists(), f"Schema file not found at {schema_path}"

    def test_schema_sql_not_empty(self):
        """schema.sql is not empty."""
        schema_path = Path(__file__).resolve().parents[2] / "dlshogi" / "book" / "sql" / "schema.sql"
        content = schema_path.read_text(encoding="utf-8").strip()
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Critical constants
# ---------------------------------------------------------------------------


class TestCriticalConstants:
    """Verify critical constants match design expectations."""

    def test_node_store_schema_version(self):
        """Schema version is defined and non-empty."""
        from dlshogi.book.node_store import SCHEMA_VERSION

        assert SCHEMA_VERSION
        assert len(SCHEMA_VERSION) > 0

    def test_cshogi_book_entry_size(self):
        """cshogi.BookEntry is 16 bytes."""
        import cshogi

        assert cshogi.BookEntry.itemsize == 16
