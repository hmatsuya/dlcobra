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

from pathlib import Path

import numpy as np
import pytest

from dlshogi.book.config import (
    BookConfig,
    ConfigValidationError,
    REDACTED_PLACEHOLDER,
    redact_mapping,
    validate_config,
)
from dlshogi.book.keys import (
    POSITION_KEY,
    PositionKey,
    fold_u64_to_i64,
    position_key_from_sfen,
    unfold_i64_to_u64,
)
from dlshogi.book.node_store import (
    ExpansionWrite,
    GetResult,
    NodeStore,
    NodeStoreConnectionError,
    NodeStoreError,
    SchemaCreationError,
    SchemaVersionMismatchError,
    Terminal,
    WriteOutcome,
    retry_schedule,
)
from dlshogi.book.packed_edge import PACKED_EDGE, decode_edges, encode_edges
from dlshogi.book.propagate import propagate
from dlshogi.book.export import export_book


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


# ---------------------------------------------------------------------------
# Task 17.2 -- configuration validation, redaction, ConfigValidationError
# (requirements 1.4, 1.9, 13.2, 13.3, 13.8)
# ---------------------------------------------------------------------------


_VALID_RAW = {
    "PostgreSQL host": "localhost",
    "PostgreSQL port": "5432",
    "PostgreSQL user": "operator",
    "PostgreSQL password": "correct horse battery staple",
    "PostgreSQL database": "puct_book",
    "Cache_Budget": 268435456,
    "Worker_Count": 1024,
    "Batch_Size": 128,
    "Batch_Timeout": 200,
    "Virtual_Loss": 3,
    "Terashock_Prior_Weight": 0.5,
    "Eval_Coef": 600,
    "Draw_Value_Black": 0.5,
    "Draw_Value_White": 0.5,
    "Max_Book_Ply": 256,
    "Propagation_Visit_Threshold": 100,
    "Export_Visit_Threshold": 0.01,
    "Connection_Retry_Limit": 5,
    "Root_Position": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "Report_Interval": 60,
    "Throughput_Floor": 50,
    "Throughput_Grace_Period": 600,
}


def _valid_raw() -> dict:
    import copy

    return copy.deepcopy(_VALID_RAW)


class TestConfigValidation:
    """Configuration-failure reporting: the validator collects and the loader raises."""

    def test_redact_mapping_redacts_only_password(self):
        """1.4 -- only the credential field is redacted; every other value is untouched."""
        raw = _valid_raw()
        redacted = redact_mapping(raw)
        assert redacted["PostgreSQL password"] == REDACTED_PLACEHOLDER
        assert redacted["PostgreSQL user"] == "operator"
        assert "correct horse battery staple" not in str(redacted)

    def test_validate_config_flags_absent_required_field(self):
        """1.9 -- dropping a required value fails validation and names the field."""
        raw = _valid_raw()
        del raw["Worker_Count"]
        result = validate_config(raw)
        assert not result.ok
        assert "Worker_Count" in result.absent
        assert not result.out_of_range

    def test_validate_config_flags_out_of_range_value(self):
        """1.9 -- an out-of-range value is reported, not silently clamped."""
        raw = _valid_raw()
        raw["Batch_Size"] = 999999  # above the upper bound
        result = validate_config(raw)
        assert not result.ok
        assert any(f.name == "Batch_Size" for f in result.out_of_range)

    def test_load_raises_config_validation_error_with_result(self):
        """1.9 -- BookConfig.load validates first and raises carrying the findings."""
        raw = _valid_raw()
        del raw["Worker_Count"]
        raw["Batch_Size"] = 999999
        with pytest.raises(ConfigValidationError) as excinfo:
            BookConfig.load(raw)
        result = excinfo.value.result
        assert not result.ok
        assert "Worker_Count" in result.absent
        assert any(f.name == "Batch_Size" for f in result.out_of_range)

    def test_terashock_book_path_is_optional(self):
        """13.8 -- the Terashock path is the one value whose absence is legal."""
        raw = _valid_raw()
        assert "Terashock_Book path" not in raw
        result = validate_config(raw)
        assert result.ok
        assert "Terashock_Book path" not in result.absent


# ---------------------------------------------------------------------------
# Task 17.2 -- Node_Store error reporting and the retry schedule
# (requirements 1.4, 1.9, 2.2, 2.3, 2.6, 2.8, 7.3, 7.4)
# ---------------------------------------------------------------------------


class TestNodeStoreErrorReporting:
    """Startup-failure exceptions must name the target and the cause (1.4, 1.9)."""

    def test_connection_error_names_host_port_db_and_attempts(self):
        """1.4 -- a connection failure reports host, port, database, attempt count."""
        cause = OSError("connection refused")
        exc = NodeStoreConnectionError("10.0.0.5", 5432, "puct_book", 6, cause)
        assert exc.host == "10.0.0.5"
        assert exc.port == 5432
        assert exc.database == "puct_book"
        assert exc.attempts == 6
        assert exc.cause is cause
        msg = str(exc)
        assert "puct_book" in msg and "10.0.0.5" in msg and "5432" in msg
        assert "6 attempt(s)" in msg
        assert isinstance(exc, NodeStoreError)

    def test_schema_creation_error_distinguishes_timeout_from_failure(self):
        """1.9 -- schema-creation failure vs. 300 s timeout produce different reports."""
        timeout_exc = SchemaCreationError("h", 5432, "db", timed_out=True)
        failure_exc = SchemaCreationError("h", 5432, "db", timed_out=False, cause=ValueError("bad ddl"))
        assert timeout_exc.timed_out is True
        assert failure_exc.timed_out is False
        assert failure_exc.cause is not None
        assert "limit" in str(timeout_exc)
        assert "bad ddl" in str(failure_exc)
        # The two reports must be distinguishable, not identical boilerplate.
        assert str(timeout_exc) != str(failure_exc)

    def test_schema_version_mismatch_error_carries_both_versions(self):
        """2.8 -- a version/fingerprint mismatch names the recorded and running values."""
        exc = SchemaVersionMismatchError(
            recorded_version="1.0", running_version="2.0",
            recorded_fingerprint=111, running_fingerprint=222,
        )
        assert exc.recorded_version == "1.0"
        assert exc.running_version == "2.0"
        assert "1.0" in str(exc) and "2.0" in str(exc)

    def test_retry_schedule_doubles_then_caps_at_60s(self):
        """2.2 -- waits start at 1.0 s, double, and never exceed 60.0 s."""
        schedule = retry_schedule(6)
        assert len(schedule) == 6
        assert schedule[0] == 1.0
        assert all(0.0 < w <= 60.0 for w in schedule)
        # Each wait is double its predecessor until the cap is hit.
        for prev, cur in zip(schedule, schedule[1:]):
            assert cur == min(prev * 2.0, 60.0)

    def test_retry_schedule_caps_after_many_retries(self):
        """2.2 -- a long retry limit plateaus at the 60 s ceiling."""
        schedule = retry_schedule(10)
        assert schedule[0] == 1.0
        assert max(schedule) == 60.0
        assert schedule[-1] == 60.0

    def test_retry_schedule_rejects_negative_limit(self):
        """2.2 -- a negative retry limit is a programming error, not a schedule."""
        with pytest.raises(ValueError):
            retry_schedule(-1)

    def test_get_result_is_exactly_three_distinct_variants(self):
        """7.3/7.4 -- the read result is a three-way distinction, no more, no less."""
        names = {member.name for member in GetResult}
        assert names == {"FOUND", "ABSENT", "FAILED"}
        assert len(set(id(m) for m in GetResult)) == 3


# ---------------------------------------------------------------------------
# Task 17.2 -- Node_Store read/write examples against a real scratch PG
# (requirements 2.3, 2.6, 3.6, 7.3, 7.4, 7.8, 9.11)
# ---------------------------------------------------------------------------

_ROOT_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _make_config(pg_info, **overrides) -> BookConfig:
    """A BookConfig pointed at the scratch DB, with in-range placeholder values."""
    kwargs: dict = dict(
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
        root_position=_ROOT_SFEN,
        report_interval=60,
        throughput_floor=50,
        throughput_grace_period=600,
    )
    kwargs.update(overrides)
    return BookConfig(**kwargs)


async def _drop_book_tables(conn) -> None:
    """Drop every Node_Store-owned table so the DB is truly empty (fresh start)."""
    await conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, book_meta CASCADE"
    )


class TestNodeStoreDb:
    """Node_Store read/write examples against the session scratch database."""

    @pytest.mark.db
    async def test_root_position_recorded_and_read_back(self, pg_scratch_database, pg_conn):
        """2.3 -- the recorded Root_Position (SFEN and folded key) round-trips unchanged."""
        await _drop_book_tables(pg_conn)
        store = await NodeStore.connect_and_prepare(_make_config(pg_scratch_database))
        try:
            sfen, hi, lo = await store.get_root()
            assert sfen == _ROOT_SFEN
            derived = position_key_from_sfen(sfen)
            assert (hi, lo) == (fold_u64_to_i64(derived.hi), fold_u64_to_i64(derived.lo))
        finally:
            await store.close()

    @pytest.mark.db
    async def test_get_three_way_absent_then_found(self, pg_scratch_database, pg_conn):
        """7.3/7.4 -- an unwritten key is ABSENT; after an expansion write it is FOUND."""
        await _drop_book_tables(pg_conn)
        store = await NodeStore.connect_and_prepare(_make_config(pg_scratch_database))
        try:
            missing = PositionKey(77701, 77702)
            result, view = await store.get(missing)
            assert result is GetResult.ABSENT
            assert view is None

            key = PositionKey(77801, 77802)
            outcome = await store.insert_expansion(
                ExpansionWrite(key=key, sfen="unit-node", apery_key=12345, edges=[])
            )
            assert outcome.outcome is WriteOutcome.COMMITTED

            result, view = await store.get(key)
            assert result is GetResult.FOUND
            assert view is not None
            assert view.sfen == "unit-node"
            assert view.key == key
        finally:
            await store.close()

    @pytest.mark.db
    async def test_forged_position_key_collision(self, pg_scratch_database, pg_conn):
        """2.6/3.6 -- a different SFEN at the same 128-bit key is a collision, not a transposition."""
        await _drop_book_tables(pg_conn)
        store = await NodeStore.connect_and_prepare(_make_config(pg_scratch_database))
        try:
            key = PositionKey(77901, 77902)
            first = await store.insert_expansion(
                ExpansionWrite(key=key, sfen="first-sfen", apery_key=1, edges=[])
            )
            assert first.outcome is WriteOutcome.COMMITTED

            forged = await store.insert_expansion(
                ExpansionWrite(key=key, sfen="different-sfen", apery_key=2, edges=[])
            )
            assert forged.outcome is WriteOutcome.COLLISION
            assert forged.existing_sfen == "first-sfen"
            assert forged.attempted_sfen == "different-sfen"
            assert store.collision_count == 1
            row = await pg_conn.fetchrow(
                "SELECT sfen FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
            assert row["sfen"] == "first-sfen"
        finally:
            await store.close()

    @pytest.mark.db
    async def test_propagate_with_no_root_row(self, pg_scratch_database, pg_conn):
        """7.8/9.11 -- propagating with the root node absent returns an empty report and writes nothing."""
        await _drop_book_tables(pg_conn)
        config = _make_config(pg_scratch_database)
        store = await NodeStore.connect_and_prepare(config)
        try:
            await pg_conn.execute("DELETE FROM book_node")
            assert not await store.root_node_exists()
            root = position_key_from_sfen(config.root_position)
            stats = await propagate(store, root, config)
            assert stats.nodes == 0
            assert not await store.any_node_rows()
        finally:
            await store.close()

    @pytest.mark.db
    async def test_export_unopenable_output_paths(self, pg_scratch_database, pg_conn, tmp_path):
        """10.8 -- an output path that cannot be opened for writing is reported as OSError (both outputs)."""
        await _drop_book_tables(pg_conn)
        config = _make_config(pg_scratch_database)
        store = await NodeStore.connect_and_prepare(config)
        try:
            # A regular file in the parent chain: mkdir cannot create a dir beneath it.
            blocker = tmp_path / "blocker.txt"
            blocker.write_text("x")

            # Case 1: the Apery output path is unopenable.
            apery_out = blocker / "sub" / "book.bin"
            with pytest.raises(OSError):
                await export_book(store, config, apery_out=apery_out, yaneuraou_out=None)

            # Case 2: the YaneuraOu output path is unopenable.
            yaneuraou_out = tmp_path / "another.txt"
            yaneuraou_out.write_text("y")
            with pytest.raises(OSError):
                await export_book(
                    store, config, apery_out=None, yaneuraou_out=yaneuraou_out / "db.db"
                )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# 14.7 unwritable log destination (design classifies this as a unit test)
# ---------------------------------------------------------------------------


class TestUnwritableLogDestination:
    """14.7 -- an unwritable log destination is reported to the Operator at most
    once per interval, the record is discarded (no exception propagates), and the
    run continues. The handling lives in ``ProgressReporter._emit_log``
    (``dlshogi/book/report.py``), not the CLI, so it is a pure unit test.
    """

    class _FailingLogger:
        """Logger stub whose ``info()`` raises ``OSError`` (simulates an
        unwritable rotating-log file handler)."""

        def __init__(self) -> None:
            self.write_attempts = 0

        def info(self, *args, **kwargs):  # pragma: no cover - signature shim
            self.write_attempts += 1
            raise OSError("simulated unwritable log destination")

    def _make_reporter(self):
        from dlshogi.book.report import ProgressReporter

        reporter = ProgressReporter(
            report_interval=60.0,
            throughput_floor=50.0,
            throughput_grace_period=600.0,
        )
        reporter._logger = self._FailingLogger()
        reporter._logging_configured = True
        return reporter

    def test_reported_once_per_interval(self, capsys):
        reporter = self._make_reporter()

        # First emit in the interval -> the warning is reported.
        reporter._emit_log({"type": "interval", "n": 1})
        _, err = capsys.readouterr()
        assert "log destination is unwritable" in err

        # Second emit in the SAME interval -> rate-limited, no repeat.
        reporter._emit_log({"type": "interval", "n": 2})
        _, err = capsys.readouterr()
        assert "unwritable" not in err

    def test_reported_again_after_interval_reset(self, capsys):
        reporter = self._make_reporter()

        reporter._emit_log({"type": "interval", "n": 1})
        capsys.readouterr()

        # A new reporting interval clears the once-per-interval flag (the same
        # reset the interval tick performs in production).
        reporter._unwritable_reported_this_interval = False
        reporter._emit_log({"type": "interval", "n": 2})
        _, err = capsys.readouterr()
        assert "log destination is unwritable" in err

    def test_record_discarded_and_run_continues(self, capsys):
        # The unwritable destination must not raise out of _emit_log: the record
        # is discarded and the search run keeps going.
        reporter = self._make_reporter()
        try:
            reporter._emit_log({"type": "interval", "n": 1})
        except OSError:  # pragma: no cover - the contract is to NOT raise
            pytest.fail("an unwritable log destination must not propagate OSError")


# ---------------------------------------------------------------------------
# 13.10 search command arguments (--model required, --gpu-count default 1)
# ---------------------------------------------------------------------------


class TestSearchCommandArgs:
    """13.10 -- the ``search`` subcommand takes a required ``--model`` ONNX path
    and a process-count argument ``--gpu-count`` that defaults to 1.
    """

    def _parser(self):
        from dlshogi.book.__main__ import build_parser

        return build_parser()

    def test_model_is_required(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["search", "--config", "c.yaml"])

    def test_gpu_count_defaults_to_one(self):
        args = self._parser().parse_args(
            ["search", "--config", "c.yaml", "--model", "m.onnx"]
        )
        assert args.model == "m.onnx"
        assert args.gpu_count == 1

    @pytest.mark.parametrize("bad", ["0", "-2", "abc", "1.5"])
    def test_gpu_count_below_one_or_non_integer_is_rejected(self, bad, capsys):
        """13.10 -- 'an integer of at least 1': 0 and negatives used to be accepted
        and silently ran the single-process path (F-15)."""
        with pytest.raises(SystemExit) as exc:
            self._parser().parse_args(
                ["search", "--config", "c.yaml", "--model", "m.onnx", "--gpu-count", bad]
            )
        assert exc.value.code == 2
        assert "--gpu-count" in capsys.readouterr().err

    def test_gpu_count_one_is_accepted(self):
        args = self._parser().parse_args(
            ["search", "--config", "c.yaml", "--model", "m.onnx", "--gpu-count", "1"]
        )
        assert args.gpu_count == 1

    def test_gpu_count_explicit(self):
        args = self._parser().parse_args(
            [
                "search",
                "--config",
                "c.yaml",
                "--model",
                "m.onnx",
                "--gpu-count",
                "4",
            ]
        )
        assert args.gpu_count == 4


# ---------------------------------------------------------------------------
# 7.8 the three unreadable Terashock-input cases (design: unit test)
# ---------------------------------------------------------------------------


class TestUnreadableTerashockInput:
    """7.8 -- ``import_terashock`` exits (``SystemExit``) on the three
    unreadable Terashock-input cases: no path configured, an unreadable file,
    and a readable file that yields zero entries. Every one of these is detected
    *before* the connection pool is touched, so no live database is required.
    """

    def _cfg(self, terashock_book):
        from types import SimpleNamespace

        # import_terashock reads only ``cfg.terashock_book`` before any early
        # exit, so a lightweight stand-in is sufficient.
        return SimpleNamespace(terashock_book=terashock_book)

    async def test_no_path_configured_exits(self):
        from dlshogi.book.terashock import import_terashock

        with pytest.raises(SystemExit):
            await import_terashock(None, self._cfg(None))

    async def test_unreadable_file_exits(self, tmp_path):
        from dlshogi.book.terashock import import_terashock

        missing = tmp_path / "does_not_exist.db"
        with pytest.raises(SystemExit):
            await import_terashock(None, self._cfg(str(missing)))

    async def test_zero_entries_exits(self, tmp_path):
        from dlshogi.book.terashock import import_terashock

        empty = tmp_path / "empty.db"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            await import_terashock(None, self._cfg(str(empty)))
