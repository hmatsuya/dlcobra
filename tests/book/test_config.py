"""Unit tests for ``dlshogi.book.config`` (task 4.1).

These are ordinary example-based smoke tests, not the Property 41 / Property
42 hypothesis-driven property tests -- those are separate, optional tasks
(4.2 and 4.3) and are not implemented here. This file only checks that
`config.py`'s own surface -- the range table, the validator, the sizing
warnings, and the redaction functions -- behaves correctly on a handful of
concrete examples, since 4.2/4.3 depend on that surface's shape (in
particular `RANGE_TABLE`'s row names) matching what later tasks expect.
"""

from __future__ import annotations

import copy

import pytest

from dlshogi.book.config import (
    RANGE_TABLE,
    CREDENTIAL_FIELDS,
    REDACTED_PLACEHOLDER,
    BookConfig,
    ConfigValidationError,
    compute_warnings,
    redact_mapping,
    validate_config,
)

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
    # Terashock_Book path intentionally omitted: absence is legal (13.8).
}


def _valid_raw() -> dict:
    return copy.deepcopy(_VALID_RAW)


# ---------------------------------------------------------------------------
# Range table well-formedness
# ---------------------------------------------------------------------------


def test_range_table_rows_are_well_formed():
    """Every row has a non-empty name, a recognised kind, and low <= high."""
    seen_names = set()
    for row in RANGE_TABLE:
        assert row.name and isinstance(row.name, str)
        assert row.name not in seen_names, f"duplicate RANGE_TABLE row name {row.name!r}"
        seen_names.add(row.name)
        assert row.kind in {
            "text",
            "int",
            "count",
            "ratio",
            "float",
            "duration_ms",
            "duration_s",
        }
        assert row.low <= row.high


def test_propagation_and_export_visit_threshold_are_distinct_rows():
    """Requirement 13.7: distinct rows, distinct kinds, distinct ranges."""
    by_name = {row.name: row for row in RANGE_TABLE}
    prop = by_name["Propagation_Visit_Threshold"]
    export = by_name["Export_Visit_Threshold"]
    assert prop.kind != export.kind
    assert (prop.low, prop.high) == (0, 1000000)
    assert (export.low, export.high) == (0, 1)


def test_only_terashock_book_path_is_optional():
    """Requirement 13.8: Terashock_Book path is the only exempt-from-absent row."""
    optional = [row.name for row in RANGE_TABLE if not row.required]
    assert optional == ["Terashock_Book path"]


# ---------------------------------------------------------------------------
# Validation: the known-good and known-bad cases
# ---------------------------------------------------------------------------


def test_valid_configuration_passes():
    result = validate_config(_valid_raw())
    assert result.ok
    assert result.absent == ()
    assert result.out_of_range == ()


def test_absent_and_out_of_range_values_are_collected_together():
    """Requirement 13.2/13.3: report every problem in one pass, not fail-fast."""
    raw = _valid_raw()
    del raw["Worker_Count"]
    del raw["Cache_Budget"]
    raw["Batch_Size"] = 999999  # above the upper bound
    raw["Virtual_Loss"] = -1  # below the lower bound

    result = validate_config(raw)

    assert not result.ok
    assert set(result.absent) == {"Worker_Count", "Cache_Budget"}
    out_of_range_names = {finding.name for finding in result.out_of_range}
    assert out_of_range_names == {"Batch_Size", "Virtual_Loss"}


def test_absent_terashock_book_path_is_not_reported_absent():
    """Requirement 13.8: absence of the Terashock_Book path is legal."""
    raw = _valid_raw()
    assert "Terashock_Book path" not in raw
    result = validate_config(raw)
    assert result.ok
    assert "Terashock_Book path" not in result.absent


def test_out_of_range_terashock_book_path_is_still_reported_when_present():
    raw = _valid_raw()
    raw["Terashock_Book path"] = ""  # present but below the 1-character minimum
    result = validate_config(raw)
    assert not result.ok
    assert any(f.name == "Terashock_Book path" for f in result.out_of_range)


@pytest.mark.parametrize(
    "name,boundary_value",
    [
        ("Cache_Budget", 268435456),
        ("Cache_Budget", 1099511627776),
        ("Worker_Count", 1),
        ("Worker_Count", 1024),
        ("Batch_Size", 1),
        ("Batch_Size", 4096),
        ("Virtual_Loss", 0),
        ("Virtual_Loss", 16),
        ("Max_Book_Ply", 0),
        ("Max_Book_Ply", 1024),
        ("Propagation_Visit_Threshold", 0),
        ("Propagation_Visit_Threshold", 1000000),
        ("Connection_Retry_Limit", 0),
        ("Connection_Retry_Limit", 100),
    ],
)
def test_endpoints_of_the_permitted_range_are_accepted(name, boundary_value):
    """Property 41 will assert this generally; pin a few endpoints concretely."""
    raw = _valid_raw()
    raw[name] = boundary_value
    result = validate_config(raw)
    assert not any(f.name == name for f in result.out_of_range)


def test_root_position_out_of_length_bound_is_reported_without_crashing():
    """A malformed SFEN must be reported, never crash the validator (see
    the config.py module docstring's note on cshogi.Board aborting the
    process on some malformed input -- this module never calls it)."""
    raw = _valid_raw()
    raw["Root_Position"] = "not even close to a real sfen but short" * 10  # > 256 chars
    result = validate_config(raw)
    assert any(f.name == "Root_Position" for f in result.out_of_range)


def test_non_numeric_value_for_a_numeric_row_is_out_of_range_not_a_crash():
    raw = _valid_raw()
    raw["Worker_Count"] = "eight"
    result = validate_config(raw)
    assert any(f.name == "Worker_Count" for f in result.out_of_range)


# ---------------------------------------------------------------------------
# Sizing warnings
# ---------------------------------------------------------------------------


def test_throughput_floor_warning_scales_with_process_count():
    raw = _valid_raw()
    raw["Throughput_Floor"] = 2000
    assert any("Throughput_Floor" in w for w in compute_warnings(raw, process_count=1))
    # 2000 <= 1500 * 4, so the same floor is fine with more processes.
    assert not any("Throughput_Floor" in w for w in compute_warnings(raw, process_count=4))


def test_batch_size_and_worker_count_warnings():
    raw = _valid_raw()
    raw["Throughput_Floor"] = 10
    raw["Batch_Timeout"] = 50  # ms -> 0.05 s; limit = 2 * 10 * 0.05 = 1
    raw["Batch_Size"] = 128
    raw["Worker_Count"] = 4
    warnings = compute_warnings(raw)
    assert any("Batch_Size" in w and "Throughput_Floor" in w for w in warnings)
    assert any("Worker_Count" in w for w in warnings)


def test_warnings_do_not_affect_validity():
    """Requirement 13.7's sizing warnings are warnings, not range violations."""
    raw = _valid_raw()
    raw["Throughput_Floor"] = 99999  # triggers the process-count warning at process_count=1
    result = validate_config(raw, process_count=1)
    assert result.ok
    assert result.warnings


# ---------------------------------------------------------------------------
# Redaction (Requirement 13.6) -- field identity, not substring scanning
# ---------------------------------------------------------------------------


def test_redact_mapping_replaces_only_credential_fields():
    raw = _valid_raw()
    redacted = redact_mapping(raw)
    for name in CREDENTIAL_FIELDS:
        assert redacted[name] == REDACTED_PLACEHOLDER
    for name, value in raw.items():
        if name not in CREDENTIAL_FIELDS:
            assert redacted[name] == value


def test_redact_mapping_tolerates_password_as_substring_of_another_value():
    """A password that also occurs inside an unrelated value must not corrupt
    that unrelated value -- redaction is by field identity, not substring
    scan (see config.py's Property 42 note)."""
    raw = _valid_raw()
    raw["PostgreSQL password"] = "sw0rdfish"
    raw["Root_Position"] = "sw0rdfish is not a real sfen but contains the password"
    redacted = redact_mapping(raw)
    assert redacted["PostgreSQL password"] == REDACTED_PLACEHOLDER
    # The unrelated field keeps its exact original value, substring or not.
    assert redacted["Root_Position"] == raw["Root_Position"]
    assert "sw0rdfish" in redacted["Root_Position"]


def test_book_config_to_loggable_dict_redacts_password_only():
    cfg = BookConfig.load(_valid_raw())
    loggable = cfg.to_loggable_dict()
    assert loggable["pg_password"] == REDACTED_PLACEHOLDER
    assert loggable["pg_host"] == cfg.pg_host
    assert loggable["pg_user"] == cfg.pg_user


# ---------------------------------------------------------------------------
# BookConfig construction
# ---------------------------------------------------------------------------


def test_book_config_load_succeeds_on_valid_configuration():
    cfg = BookConfig.load(_valid_raw())
    assert cfg.worker_count == 1024
    assert cfg.terashock_book is None


def test_book_config_load_raises_config_validation_error_on_problems():
    raw = _valid_raw()
    del raw["Worker_Count"]
    with pytest.raises(ConfigValidationError) as excinfo:
        BookConfig.load(raw)
    assert "Worker_Count" in excinfo.value.result.absent


def test_book_config_carries_terashock_book_path_when_present():
    raw = _valid_raw()
    raw["Terashock_Book path"] = "/path/to/terashock.db"
    cfg = BookConfig.load(raw)
    assert cfg.terashock_book == "/path/to/terashock.db"
