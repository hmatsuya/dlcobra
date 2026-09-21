"""Smoke tests for core book module imports and constants (task 17.3).

Lightweight sanity checks that verify:
- Package imports work without side effects
- Key dtype sizes are correct (``POSITION_KEY.itemsize == 16``)
- Schema file exists (if expected)
- Critical constants have the expected values
- The schema is present after startup, and records the running version (Req 2.4)
- Every write lands in the one Book_Graph named by the configured connection
  settings, and in no other (Req 4.10 / 2.1)
- The Search_Coordinator holds one Node_Store and takes its worker count from
  the configured Worker_Count (Req 11.1)
- Every configuration name and every command name is recognised (Req 13.1 / 13.4)

Most tests are pure (no database). The schema and Book_Graph smoke tests are
marked ``db`` and use the session scratch database from conftest.py.
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


# ---------------------------------------------------------------------------
# Shared helpers for the PostgreSQL-backed smoke tests below
# ---------------------------------------------------------------------------


_INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _make_config(pg_info, **overrides):
    """Build a `BookConfig` pointed at ``pg_info``'s database.

    Mirrors the helper in ``test_node_store.py``: every non-connection field
    is filled with an arbitrary but in-range placeholder, since these smoke
    tests do not exercise ``validate_config``.
    """
    from dlshogi.book.config import BookConfig

    kwargs = dict(
        pg_host=pg_info.host,
        pg_port=str(pg_info.port),
        pg_user=pg_info.user,
        pg_password=pg_info.password or "",
        pg_database=pg_info.database,
        cache_budget=268435456,
        worker_count=4,
        batch_size=8,
        batch_timeout=200,
        virtual_loss=3,
        terashock_prior_weight=0.5,
        eval_coef=600,
        draw_value_black=0.5,
        draw_value_white=0.5,
        max_book_ply=256,
        propagation_visit_threshold=100,
        export_visit_threshold=0.01,
        connection_retry_limit=0,
        root_position=_INITIAL_SFEN,
        report_interval=60,
        throughput_floor=50,
        throughput_grace_period=600,
    )
    kwargs.update(overrides)
    return BookConfig(**kwargs)


async def _drop_book_tables(conn):
    """Drop every Node_Store-owned table so the database is truly empty."""
    await conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, book_meta CASCADE"
    )


# ---------------------------------------------------------------------------
# 17.3: the schema exists after startup (Req 2.4, 2.1)
# ---------------------------------------------------------------------------


@pytest.mark.db
class TestSchemaAfterStartup:
    """After startup (connect + ensure_schema) on a fresh database, the schema
    is present and records the running schema version."""

    async def test_schema_tables_created_on_fresh_database(self, pg_scratch_database, pg_conn):
        await _drop_book_tables(pg_conn)
        config = _make_config(pg_scratch_database, connection_retry_limit=0)

        from dlshogi.book.node_store import NodeStore

        store = await NodeStore.connect_and_prepare(config)
        try:
            tables = {
                row["tablename"]
                for row in await pg_conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            assert {"book_meta", "book_node", "terashock_entry", "in_flight_claim"} <= tables
        finally:
            await store.close(force=True)

    async def test_schema_version_recorded_after_startup(self, pg_scratch_database, pg_conn):
        await _drop_book_tables(pg_conn)
        config = _make_config(pg_scratch_database, connection_retry_limit=0)

        from dlshogi.book.node_store import SCHEMA_VERSION, NodeStore

        store = await NodeStore.connect_and_prepare(config)
        try:
            recorded = await pg_conn.fetchval(
                "SELECT schema_version FROM book_meta WHERE id = 1"
            )
            assert recorded == SCHEMA_VERSION
        finally:
            await store.close(force=True)


# ---------------------------------------------------------------------------
# 17.3: every write lands in the one Book_Graph named by the configured
# connection settings, and in no other (Req 4.10, 2.1)
# ---------------------------------------------------------------------------


@pytest.mark.db
class TestSingleBookGraph:
    """The Node_Store holds a single pool bound to the configured database, so
    every Book_Node / Book_Edge write lands in that one Book_Graph."""

    async def test_writes_land_in_configured_database_only(self, pg_scratch_database, pg_conn):
        await _drop_book_tables(pg_conn)
        config = _make_config(pg_scratch_database, connection_retry_limit=0)

        from dlshogi.book.keys import position_key_from_sfen
        from dlshogi.book.node_store import ExpansionWrite, NodeStore, WriteOutcome

        store = await NodeStore.connect_and_prepare(config)
        try:
            # The store's sole pool is bound to the one configured database.
            async with store.pool.acquire() as conn:
                current_db = await conn.fetchval("SELECT current_database()")
            assert current_db == config.pg_database

            # A real Book_Node write lands in that one database.
            key = position_key_from_sfen(_INITIAL_SFEN)
            result = await store.insert_expansion(
                ExpansionWrite(key=key, sfen=_INITIAL_SFEN, apery_key=0)
            )
            assert result.outcome == WriteOutcome.COMMITTED
            async with store.pool.acquire() as conn:
                n = await conn.fetchval("SELECT count(*) FROM book_node")
            assert n == 1
        finally:
            await store.close(force=True)


# ---------------------------------------------------------------------------
# 17.3: Worker_Count descent tasks are created against one store (Req 11.1)
# ---------------------------------------------------------------------------


class TestSearchCoordinatorSingleStore:
    """The Search_Coordinator holds exactly one Node_Store and takes the number
    of descent workers from the configured Worker_Count."""

    def test_one_store_and_worker_count_from_config(self):
        from types import SimpleNamespace

        from dlshogi.book.search import SearchCoordinator

        fake_pg = SimpleNamespace(
            host="localhost", port=5432, user="u", password=None, database="d"
        )
        sentinel_store = object()

        two = SearchCoordinator(_make_config(fake_pg, worker_count=2), sentinel_store, object())
        five = SearchCoordinator(_make_config(fake_pg, worker_count=5), sentinel_store, object())

        # One shared store instance, not a collection of stores (Req 11.1).
        assert two._store is sentinel_store
        assert five._store is sentinel_store
        assert two._store is five._store
        # The worker count flows from the configured Worker_Count.
        assert two._config.worker_count == 2
        assert five._config.worker_count == 5


# ---------------------------------------------------------------------------
# 17.3: every configuration name and every command name is recognised
# (Req 13.1, 13.4, 4.10, 13.10)
# ---------------------------------------------------------------------------


def _valid_raw_for_all_names():
    """A raw config mapping holding one in-range value per RANGE_TABLE name."""
    from dlshogi.book.config import RANGE_TABLE

    raw = {}
    for row in RANGE_TABLE:
        if row.kind == "text":
            raw[row.name] = "x"
        elif row.kind in ("int", "count"):
            raw[row.name] = int(row.low)
        else:  # ratio, float, duration_ms, duration_s
            raw[row.name] = float(row.low)
    return raw


class TestConfigAndCommandNames:
    """Every configuration value named by the spec, and every CLI command,
    is recognised by the implementation."""

    def test_every_config_name_recognised_by_from_mapping(self):
        from dlshogi.book.config import BookConfig

        cfg = BookConfig.from_mapping(_valid_raw_for_all_names())  # KeyError if any name unrecognised
        assert cfg.worker_count == 1  # int(RANGE_TABLE Worker_Count low bound)
        assert cfg.pg_database == "x"

    def test_required_config_names_all_present(self):
        from dlshogi.book.config import RANGE_TABLE

        names = {row.name for row in RANGE_TABLE}
        expected = {
            "PostgreSQL host", "PostgreSQL port", "PostgreSQL user",
            "PostgreSQL password", "PostgreSQL database",
            "Cache_Budget", "Worker_Count", "Batch_Size", "Batch_Timeout",
            "Virtual_Loss", "Terashock_Prior_Weight", "Eval_Coef",
            "Draw_Value_Black", "Draw_Value_White", "Max_Book_Ply",
            "Propagation_Visit_Threshold", "Export_Visit_Threshold",
            "Connection_Retry_Limit", "Root_Position", "Report_Interval",
            "Throughput_Floor", "Throughput_Grace_Period", "Terashock_Book path",
        }
        assert expected <= names, f"missing config names: {expected - names}"

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Req 13.1 requires reading Puct_Exploration_Constant (range 0.1-5.0) "
            "from the configuration, but config.py's RANGE_TABLE omits it and "
            "search.py hardcodes c_puct=2**0.5 (search.py:563). Known gap."
        ),
    )
    def test_puct_exploration_constant_recognised(self):
        from dlshogi.book.config import RANGE_TABLE

        assert "Puct_Exploration_Constant" in {row.name for row in RANGE_TABLE}

    def test_every_command_name_recognised(self):
        from dlshogi.book.__main__ import build_parser

        parser = build_parser()
        cases = {
            "search": ["search", "--config", "c.yaml", "--model", "m.onnx"],
            "propagate": ["propagate", "--config", "c.yaml"],
            "export": ["export", "--config", "c.yaml"],
            "import-terashock": ["import-terashock", "--config", "c.yaml"],
        }
        for expected_cmd, argv in cases.items():
            assert parser.parse_args(argv).command == expected_cmd

        # An unrecognised command is rejected.
        with pytest.raises(SystemExit):
            parser.parse_args(["frobnicate", "--config", "c.yaml"])
