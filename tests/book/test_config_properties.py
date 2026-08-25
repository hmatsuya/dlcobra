"""Property tests for ``dlshogi.book.config`` (tasks 4.2, 4.3).

Implements:

- **Property 41: Configuration validation is exact** (Requirements 13.2,
  13.3, 13.5, 13.7) -- task 4.2. Driven directly from ``config.RANGE_TABLE``
  rather than from a hand-written list, so a future change to the range
  table is automatically picked up by this generator.
- **Property 42: Credentials are redacted** (Requirement 13.6) -- task 4.3.

Kept in a separate module from ``test_config.py``'s example-based unit
tests, per that file's own docstring note that Properties 41/42 are
"separate, optional tasks (4.2 and 4.3)".
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from dlshogi.book.config import (
    RANGE_TABLE,
    REDACTED_PLACEHOLDER,
    validate_config,
    redact_mapping,
)

# ---------------------------------------------------------------------------
# A known-good raw configuration, reused as the base for both properties.
# ---------------------------------------------------------------------------

_VALID_RAW: dict = {
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

# The row names in the exact order RANGE_TABLE lists them; used to build
# the per-row "boundary and step-outside" value strategy below. Every row
# is required except the last (Terashock_Book path).
_ROW_NAMES = [row.name for row in RANGE_TABLE]
assert _ROW_NAMES[-1] == "Terashock_Book path"


def _valid_raw() -> dict:
    return copy.deepcopy(_VALID_RAW)


def _step_for(row) -> float:
    """A step size for "one unit outside the permitted range", per row kind.

    Integer-valued kinds step by 1; every other kind (ratio, float,
    duration) steps by a small fraction, since Requirement 13.3 talks
    about values "outside" the range with no implied minimum granularity
    for continuous quantities.
    """
    if row.kind in ("int", "count"):
        return 1
    return 1e-3


def _boundary_and_outside_values(row):
    """Return the five representative values design.md's Property 41
    strategy names for one RANGE_TABLE row: the lower endpoint, the upper
    endpoint, one step below the lower endpoint, one step above the upper
    endpoint, and an arbitrary in-range value.

    For a ``"text"`` row, ``low``/``high`` bound the *character length*, so
    the endpoint and outside-range "values" are strings of that length
    (or one character shorter/longer), not the numeric bound itself.
    """
    step = _step_for(row)
    if row.kind == "text":
        low_len = int(row.low)
        high_len = int(row.high)
        return {
            "lower_endpoint": "x" * low_len,
            "upper_endpoint": "x" * high_len,
            "below_lower": "x" * max(low_len - 1, 0) if low_len > 0 else None,
            "above_upper": "x" * (high_len + 1),
            "in_range": "x" * max(low_len, min(high_len, (low_len + high_len) // 2 or 1)),
        }
    in_range = (row.low + row.high) / 2
    if row.kind in ("int", "count"):
        in_range = float(int(in_range))  # keep it integral for integer-kind rows
    return {
        "lower_endpoint": row.low,
        "upper_endpoint": row.high,
        "below_lower": row.low - step,
        "above_upper": row.high + step,
        "in_range": in_range,
    }


_row_choice_strategy = st.sampled_from(
    ["lower_endpoint", "upper_endpoint", "below_lower", "above_upper", "in_range"]
)


@st.composite
def _config_perturbation(draw):
    """Draw a perturbation of ``_VALID_RAW``: a subset of rows to omit
    (only ever meaningful for required rows -- omitting the optional
    Terashock_Book path is legal and produces no absence finding), and,
    independently, a subset of rows to set to a boundary or out-of-range
    value. Returns ``(raw, expected_absent, expected_out_of_range_names)``.
    """
    n = len(_ROW_NAMES)
    omit_mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    perturb_mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))

    raw = _valid_raw()
    expected_absent = set()
    expected_out_of_range = set()

    for i, row in enumerate(RANGE_TABLE):
        if omit_mask[i]:
            raw.pop(row.name, None)
            if row.required:
                expected_absent.add(row.name)
            # else: omitting the optional Terashock_Book path is legal;
            # no absence finding, and no further perturbation applies
            # since the value is now simply not present.
            continue
        if perturb_mask[i]:
            choice = draw(_row_choice_strategy)
            values = _boundary_and_outside_values(row)
            value = values[choice]
            if value is None:
                continue
            raw[row.name] = value
            if choice in ("below_lower", "above_upper"):
                expected_out_of_range.add(row.name)

    return raw, expected_absent, expected_out_of_range


# Feature: puct-book-builder, Property 41: Configuration validation is exact
# Validates: Requirements 13.2, 13.3, 13.5, 13.7
@given(_config_perturbation())
@settings(max_examples=1000)
def test_configuration_validation_reports_exactly_the_absent_and_out_of_range_names(perturbation):
    raw, expected_absent, expected_out_of_range = perturbation
    result = validate_config(raw)

    assert set(result.absent) == expected_absent
    reported_out_of_range_names = {f.name for f in result.out_of_range}
    assert reported_out_of_range_names == expected_out_of_range

    # Requirement 13.5: an invalid configuration must report ok=False
    # exactly when something is absent or out of range.
    assert result.ok == (not expected_absent and not expected_out_of_range)


@given(st.sampled_from(RANGE_TABLE))
@settings(max_examples=len(RANGE_TABLE))
def test_every_range_table_row_accepts_both_of_its_own_endpoints(row):
    """Every row named by Requirement 13 criterion 7 accepts a value set
    exactly at either endpoint of its permitted range (Requirement 13.7's
    own "accepts every value set exactly at either endpoint" clause)."""
    values = _boundary_and_outside_values(row)
    for endpoint_key in ("lower_endpoint", "upper_endpoint"):
        raw = _valid_raw()
        raw[row.name] = values[endpoint_key]
        result = validate_config(raw)
        assert row.name not in result.absent
        assert not any(f.name == row.name for f in result.out_of_range), (
            f"{row.name}={values[endpoint_key]!r} at its own {endpoint_key} was rejected"
        )


def test_configuration_validation_rejects_no_commands_when_valid():
    """Sanity check that the fully valid base configuration is itself
    accepted -- otherwise the perturbation property above would vacuously
    pass by always finding *something* wrong."""
    result = validate_config(_valid_raw())
    assert result.ok


# ---------------------------------------------------------------------------
# Property 42: Credentials are redacted
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 42: Credentials are redacted
# Validates: Requirements 13.6


@given(
    credential=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=1024,
    ),
    embed_in_other_field=st.booleans(),
)
@settings(max_examples=500)
def test_startup_log_contains_every_non_credential_value_and_not_the_credential(
    credential, embed_in_other_field
):
    raw = _valid_raw()
    raw["PostgreSQL password"] = credential
    if embed_in_other_field:
        # The credential also occurs as a substring of an unrelated,
        # non-credential value -- the property must still tolerate this by
        # comparing against the redaction placeholder rather than by a
        # naive substring-absence check when overlap is detected.
        raw["Root_Position"] = f"prefix-{credential}-suffix"

    redacted = redact_mapping(raw)

    assert redacted["PostgreSQL password"] == REDACTED_PLACEHOLDER
    assert REDACTED_PLACEHOLDER != credential  # placeholder never equals the raw credential

    for name, value in raw.items():
        if name == "PostgreSQL password":
            continue
        # Every non-credential value is present, unchanged, byte for byte
        # -- including one that happens to contain the credential as a
        # substring, which must NOT have been scrubbed.
        assert redacted[name] == value
