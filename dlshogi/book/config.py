"""Config_Loader: BookConfig loading, validation, and credential redaction.

Implements Requirement 13 (Operator Configuration and Control): the
nineteen-name-plus-Terashock-path configuration surface of criterion 1, the
permitted-range table of criterion 7, the "collect everything, then exit"
validation ordering of criteria 2 and 3, and the credential redaction of
criterion 6. See design.md's "Language and repository placement" (this
module's stated role: "Config_Loader, Requirement 13 validation and
redaction") and the "Startup validation failures" flowchart, whose C1-C2-C3
nodes are exactly load -> validate (collecting every problem) -> redact and
log, in that order, all before any schema or Node_Store work.

**Public surface used by later tasks and by their property tests
(4.2/Property 41, 4.3/Property 42), which are not implemented here:**

- `RANGE_TABLE`: the criterion 7 range table, expressed once as a tuple of
  `RangeRow(name, kind, low, high, required)` rows. Both `validate_config`
  below and Property 41's hypothesis strategy are meant to be driven from
  this one table, per design.md's Property 41 write-up ("expressed once in
  `config.py` as a table of `(name, kind, low, high)` and consumed by both
  the validator and the test").
- `validate_config(raw, process_count=...)`: the pure validation function.
- `compute_warnings(raw, process_count=...)`: the three sizing warnings of
  the design's *Performance budget* section, which are warnings and never
  validation failures.
- `redact_mapping(raw)` / `BookConfig.to_loggable_dict()`: the criterion 6
  redaction, by field identity rather than by substring scan (see the
  module docstring note on Property 42 below).
- `BookConfig`, `BookConfig.load`, `BookConfig.from_mapping`.

**A note on "nineteen values" and the range table's row count.**
Requirement 13 criterion 1's own value list, read as a plain enumeration,
names 19 items before the exempt Terashock_Book path is even added: the
PostgreSQL connection settings (counted as *one* list item there), Cache_
Budget, Worker_Count, Batch_Size, Batch_Timeout, Virtual_Loss, Terashock_
Prior_Weight, Eval_Coef, Draw_Value_Black, Draw_Value_White, Max_Book_Ply,
Propagation_Visit_Threshold, Export_Visit_Threshold, Connection_Retry_
Limit, Root_Position, Report_Interval, Throughput_Floor, and Throughput_
Grace_Period -- that is 1 (connection settings) + 17 (scalars) = 18 items,
which is one short of "nineteen" however the list is read, unless the
Terashock_Book path itself is folded into the 19-wide count (1 + 17 + 1 =
19), which is how design.md's Property 41 strategy note reads it: "one
boolean per value named by Requirement 13 criterion 1" over a mask fixed
at width 19, with Terashock_Book's absence simply not triggering a
required-value failure (13.8) rather than being excluded from the list.

This module resolves a second, independent ambiguity the same sentence
raises: "PostgreSQL connection settings" (plural) is one item in that
count, but criterion 7 gives it as "each setting a text value of 1 to
1,024 characters" -- i.e. several independently named text values sharing
one range, not one value. Requirement 13.2 and 13.3 both ask to report the
*name* of every absent or out-of-range value, and "connection settings" is
not a name an Operator can act on the way "PostgreSQL password" is. This
module therefore lists the five connection settings (host, port, user,
password, database) as five separate `RANGE_TABLE` rows, each individually
checked and individually reported, rather than as one bundled row. That is
the deliberate judgment call the task write-up flags as "not dictated
verbatim by the spec text": `RANGE_TABLE` below has 23 rows (5 connection
settings + 17 scalars + Terashock_Book path), all still directly derived
from the criterion 7 text, and every one of the 22 required rows is
individually subject to criteria 2 and 3. A future Property 41 strategy
that wants the literal 19-wide subset mask can still get it by grouping the
five connection-setting rows under one mask boolean when deciding *which*
values to omit or perturb; nothing about that grouping needs a second
range table, because the ranges themselves are unaffected by how they are
grouped for the purpose of choosing a subset.

**PostgreSQL connection settings field naming.** ``tests/book/conftest.py``
independently reads ``PGHOST`` / ``PGPORT`` / ``PGUSER`` / ``PGPASSWORD`` /
``PGDATABASE``-style host/port/user/password/database settings and is
deliberately kept independent of this module. This module mirrors that
same five-way split as ``pg_host``, ``pg_port``, ``pg_user``,
``pg_password``, ``pg_database`` on `BookConfig`, which is also asyncpg's
own `connect(host=..., port=..., user=..., password=..., database=...)`
keyword-argument spelling (`node_store.py`, added by a later task, is the
natural consumer). Only ``pg_password`` is a credential under Requirement
13.6 / Property 42; the other four connection settings are not secret and
are never redacted.

**Root_Position legality.** Requirement 13.7 documents Root_Position's
permitted range as "an SFEN string of 1 to 256 characters that denotes a
legal Board_State". This module checks only the length bound. Full SFEN
legality is deliberately *not* attempted here: `cshogi.Board(sfen)` raises
`RuntimeError` on some malformed input and has been observed to abort the
process outright (`SIGABRT`) on other malformed input, and Requirement
13.2/13.3's validator must never crash or take down the process on
attacker- or operator-supplied configuration text. Root_Position legality
is instead verified later, against the running Position_Key binding (the
Search_Coordinator creates the root Book_Node from this same SFEN --
Requirement 4.8 -- which is where an illegal SFEN would actually surface).
A string within the 1..256 length bound therefore passes this module's
range check regardless of whether it denotes a legal position.

**Credential redaction and substring overlap (Property 42).** Property
42's strategy deliberately generates credential strings that also occur as
substrings of *other*, non-credential configuration values. Redacting by
scanning the rendered log text and stripping any substring equal to the
password would therefore also mangle that unrelated value. This module
redacts by field identity instead: `CREDENTIAL_FIELDS` (raw-mapping key
names) and `_CREDENTIAL_ATTRS` (`BookConfig` attribute names) name exactly
which fields are credentials, statically, and only those fields' values
are ever replaced; every other field's value is copied through unchanged,
byte for byte, no matter what it contains.

**No side effects at import time.** This module performs no file read, no
network access, no `argparse` parsing, and no logging configuration at
import time -- see design.md's "Why a package under dlshogi/ rather than
scripts": the modules must be importable without side effects so that
hypothesis can generate against them. `load_yaml` is the sole function
that touches the filesystem, and only when called.

**No logging framework.** Per design.md, the repository intentionally has
no logging framework in its Python code. This module does not write to a
log destination itself -- that is `report.py`'s `Progress_Reporter` job.
This module only *produces* the redacted, loggable representation
(`BookConfig.to_loggable_dict` / `redact_mapping`) and the validation
result (`ValidationResult`) for a caller (ultimately `__main__.py`'s
startup flow, task 17.1) to log and act on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# ---------------------------------------------------------------------------
# The Requirement 13 criterion 7 permitted-range table, expressed once.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeRow:
    """One row of the Requirement 13 criterion 7 permitted-range table.

    ``name`` is the value name as it appears in the raw configuration
    mapping (the same name Requirement 13.2/13.3 want reported). ``kind``
    classifies how ``low``/``high`` are interpreted, so a generic checker
    (and a future hypothesis generator) can dispatch on ``kind`` alone with
    no per-name special-casing:

    - ``"text"``: ``low``/``high`` bound the string's character length.
    - ``"int"``: an integer-valued quantity, ``low``/``high`` inclusive.
    - ``"count"``: an integer-valued quantity that is a plain count (used
      only for Propagation_Visit_Threshold, kept as a distinct kind from
      Export_Visit_Threshold's ``"ratio"`` per the task's explicit
      instruction that the two must not be able to merge by accident).
    - ``"ratio"``: a real-valued quantity in a closed interval, used for
      Export_Visit_Threshold, Terashock_Prior_Weight, Draw_Value_Black/
      White, and other [0, 1]-style values.
    - ``"float"``: any other real-valued quantity, ``low``/``high``
      inclusive.
    - ``"duration_ms"`` / ``"duration_s"``: a real-valued duration, in
      milliseconds or seconds respectively; distinguished from ``"float"``
      only so a future consumer can label units without parsing ``name``.

    ``required`` is ``True`` for every value of Requirement 13 criterion 1
    except the Terashock_Book path (criteria 13.8/13.9: its absence is
    legal and simply disables ``import-terashock`` and Terashock_Index
    construction, rather than being a startup failure).
    """

    name: str
    kind: str
    low: float
    high: float
    required: bool = True


# One row per individually-named value of Requirement 13 criteria 1 and 7.
# See the module docstring for why the five PostgreSQL connection settings
# are five rows rather than one, and why Propagation_Visit_Threshold and
# Export_Visit_Threshold are distinct rows with distinct kinds despite both
# eventually being numeric-range checks.
RANGE_TABLE: tuple[RangeRow, ...] = (
    RangeRow("PostgreSQL host", "text", 1, 1024),
    RangeRow("PostgreSQL port", "text", 1, 1024),
    RangeRow("PostgreSQL user", "text", 1, 1024),
    RangeRow("PostgreSQL password", "text", 1, 1024),
    RangeRow("PostgreSQL database", "text", 1, 1024),
    RangeRow("Cache_Budget", "int", 268435456, 1099511627776),
    RangeRow("Worker_Count", "int", 1, 1024),
    RangeRow("Batch_Size", "int", 1, 4096),
    RangeRow("Batch_Timeout", "duration_ms", 1, 1000),
    RangeRow("Virtual_Loss", "int", 0, 16),
    RangeRow("Terashock_Prior_Weight", "ratio", 0, 1),
    RangeRow("Eval_Coef", "float", 1, 10000),
    RangeRow("Draw_Value_Black", "ratio", 0, 1),
    RangeRow("Draw_Value_White", "ratio", 0, 1),
    RangeRow("Max_Book_Ply", "int", 0, 1024),
    # Absolute Book_Edge visit count (Requirement 9's Propagation_Visit_Threshold),
    # kept structurally distinct from Export_Visit_Threshold's ratio below.
    RangeRow("Propagation_Visit_Threshold", "count", 0, 1000000),
    # Book_Edge visit-count *ratio* applied at export time (Requirement 12.5);
    # a different kind from Propagation_Visit_Threshold on purpose.
    RangeRow("Export_Visit_Threshold", "ratio", 0, 1),
    RangeRow("Connection_Retry_Limit", "int", 0, 100),
    RangeRow("Root_Position", "text", 1, 256),
    RangeRow("Report_Interval", "duration_s", 1, 3600),
    RangeRow("Throughput_Floor", "float", 0.01, 100000),
    RangeRow("Throughput_Grace_Period", "duration_s", 60, 86400),
    RangeRow("Terashock_Book path", "text", 1, 4096, required=False),
)

# Raw-mapping name -> BookConfig attribute name.
_NAME_TO_ATTR: dict[str, str] = {
    "PostgreSQL host": "pg_host",
    "PostgreSQL port": "pg_port",
    "PostgreSQL user": "pg_user",
    "PostgreSQL password": "pg_password",
    "PostgreSQL database": "pg_database",
    "Cache_Budget": "cache_budget",
    "Worker_Count": "worker_count",
    "Batch_Size": "batch_size",
    "Batch_Timeout": "batch_timeout",
    "Virtual_Loss": "virtual_loss",
    "Terashock_Prior_Weight": "terashock_prior_weight",
    "Eval_Coef": "eval_coef",
    "Draw_Value_Black": "draw_value_black",
    "Draw_Value_White": "draw_value_white",
    "Max_Book_Ply": "max_book_ply",
    "Propagation_Visit_Threshold": "propagation_visit_threshold",
    "Export_Visit_Threshold": "export_visit_threshold",
    "Connection_Retry_Limit": "connection_retry_limit",
    "Root_Position": "root_position",
    "Report_Interval": "report_interval",
    "Throughput_Floor": "throughput_floor",
    "Throughput_Grace_Period": "throughput_grace_period",
    "Terashock_Book path": "terashock_book",
}

# The credential-bearing configuration value(s), Requirement 13.6 /
# Property 42. Named at both the raw-mapping level (used by
# `redact_mapping`, which operates before a BookConfig exists) and the
# BookConfig-attribute level (used by `BookConfig.to_loggable_dict`).
CREDENTIAL_FIELDS: frozenset[str] = frozenset({"PostgreSQL password"})
_CREDENTIAL_ATTRS: frozenset[str] = frozenset({"pg_password"})

REDACTED_PLACEHOLDER = "***REDACTED***"

_INTEGER_KINDS = frozenset({"int", "count"})
_NUMERIC_KINDS = frozenset({"int", "count", "ratio", "float", "duration_ms", "duration_s"})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutOfRangeFinding:
    """One out-of-range configuration value, as Requirement 13.3 wants it reported.

    Carries the value ``name``, the ``value`` exactly as supplied (so the
    report can show the Operator what was actually configured), and the
    permitted ``low``/``high`` bounds from `RANGE_TABLE`.
    """

    name: str
    value: Any
    low: float
    high: float

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.name}={self.value!r} (permitted range: [{self.low}, {self.high}])"


@dataclass(frozen=True)
class ValidationResult:
    """Everything `validate_config` found, collected rather than fail-fast.

    Requirement 13.2/13.3 both ask to report *every* absent value and
    *every* out-of-range value together, so the Operator can fix one round
    of mistakes rather than discovering them one exit at a time. ``ok`` is
    ``True`` exactly when neither list is populated; ``warnings`` never
    affects ``ok``, because the two sizing warnings of the design's
    *Performance budget* section are warnings, not range violations
    (Requirement 13.7's last bullet is descriptive of the warnings, not
    part of the permitted-range table itself).
    """

    absent: tuple[str, ...] = ()
    out_of_range: tuple[OutOfRangeFinding, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether validation passed: no absent value and no out-of-range value."""
        return not self.absent and not self.out_of_range


def _check_range(row: RangeRow, value: Any) -> Optional[OutOfRangeFinding]:
    """Return an `OutOfRangeFinding` for ``value`` against ``row``, or ``None``.

    ``row.kind == "text"`` checks the character length of a `str`; any
    non-`str` value is reported as out of range (Requirement 13.7's
    connection settings, Root_Position, and Terashock_Book path are all
    documented as text values, so a non-string supplied value is already a
    range violation under the criterion 7 text, not a distinct error
    category). Every other kind is a numeric range check; `bool` is
    rejected even though it is a `int` subclass in Python, and a
    ``"int"``/``"count"`` value that is a non-integral `float` (e.g.
    ``3.5``) is rejected too.
    """
    if row.kind == "text":
        if not isinstance(value, str):
            return OutOfRangeFinding(row.name, value, row.low, row.high)
        length = len(value)
        if row.low <= length <= row.high:
            return None
        return OutOfRangeFinding(row.name, value, row.low, row.high)

    assert row.kind in _NUMERIC_KINDS, f"unknown RangeRow kind {row.kind!r}"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return OutOfRangeFinding(row.name, value, row.low, row.high)

    numeric_value = float(value)
    if row.kind in _INTEGER_KINDS and not numeric_value.is_integer():
        return OutOfRangeFinding(row.name, value, row.low, row.high)
    if row.low <= numeric_value <= row.high:
        return None
    return OutOfRangeFinding(row.name, value, row.low, row.high)


def validate_config(raw: Mapping[str, Any], *, process_count: int = 1) -> ValidationResult:
    """Validate ``raw`` against `RANGE_TABLE` and return every finding together.

    ``raw`` is a mapping from value name (the `RangeRow.name` strings, e.g.
    ``"Worker_Count"``, ``"PostgreSQL password"``) to the value the
    configuration source supplied for it. This function performs no schema
    creation, no database connection, and no record write -- it is a pure
    function over its argument, consistent with the design's startup flow
    ordering (validate, then redact-and-log, then only later touch the
    database).

    Every `RANGE_TABLE` row is checked independently: a required value
    that is absent (missing from ``raw``, or present with value ``None``)
    is added to `ValidationResult.absent`; a present value that fails
    `_check_range` is added to `ValidationResult.out_of_range`; the
    Terashock_Book path row is the only one with ``required=False``, so
    its absence never appears in ``absent`` (Requirement 13.8), while its
    range is still checked when it *is* present.

    ``process_count`` is passed through to `compute_warnings`; see that
    function for why the validator cannot derive it from ``raw`` alone.
    """
    absent: list[str] = []
    out_of_range: list[OutOfRangeFinding] = []

    for row in RANGE_TABLE:
        present = row.name in raw and raw[row.name] is not None
        if not present:
            if row.required:
                absent.append(row.name)
            continue
        finding = _check_range(row, raw[row.name])
        if finding is not None:
            out_of_range.append(finding)

    warnings = compute_warnings(raw, process_count=process_count)
    return ValidationResult(tuple(absent), tuple(out_of_range), warnings)


def compute_warnings(raw: Mapping[str, Any], *, process_count: int = 1) -> tuple[str, ...]:
    """Return the design's two (three-formula) sizing warnings, never range violations.

    From design.md's *Performance budget* section:

    - ``Throughput_Floor > 1500 * process_count`` -- a floor configured
      against a multi-process or C++-scale rate will trip Requirement
      14.4's degraded-throughput warning permanently against this
      implementation's measured per-process ceiling of roughly 600 to
      1,250 descents/s. ``process_count`` is the number of ``search``
      processes the Operator intends to run (one per GPU, per design.md's
      "Process and component structure") -- runtime deployment
      information the validator cannot derive from the configuration
      values alone, so the caller supplies it; it defaults to 1, the
      single-process case.
    - ``Batch_Size > 2 * Throughput_Floor * Batch_Timeout`` -- sized so
      Requirement 15.7's 50%-of-Batch_Size mean batch size is achievable
      at the configured floor. Batch_Timeout is documented in
      milliseconds (Requirement 13.7) but design.md's own worked example
      ("At 1,000 descents/s and Batch_Timeout = 200 ms that is Batch_Size
      <= 400") uses Batch_Timeout in *seconds* inside this formula
      (2 * 1000 * 0.2 = 400); this function performs that ms-to-s
      conversion before comparing.
    - ``Worker_Count < 2 * Batch_Size`` -- steady-state Evaluator
      occupancy is bounded by Worker_Count, so a Worker_Count too small
      relative to Batch_Size cannot keep the batch full.

    Each warning is computed only when every value it needs is present and
    is a real number (not `bool`); a value that is absent or fails its own
    `RANGE_TABLE` check is already reported by `validate_config`, so this
    function silently skips the warnings that would need it rather than
    raising a second, redundant complaint or crashing on a bad type.
    """

    def _num(name: str) -> Optional[float]:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    throughput_floor = _num("Throughput_Floor")
    batch_size = _num("Batch_Size")
    batch_timeout_ms = _num("Batch_Timeout")
    worker_count = _num("Worker_Count")

    warnings: list[str] = []

    if throughput_floor is not None:
        ceiling = 1500 * process_count
        if throughput_floor > ceiling:
            warnings.append(
                f"Throughput_Floor ({throughput_floor}) exceeds 1500 * process_count "
                f"({process_count}) = {ceiling}; the degraded-throughput warning of "
                f"Requirement 14.4 may fire permanently against this implementation's "
                f"measured per-process descent-rate ceiling."
            )

    if throughput_floor is not None and batch_size is not None and batch_timeout_ms is not None:
        limit = 2 * throughput_floor * (batch_timeout_ms / 1000.0)
        if batch_size > limit:
            warnings.append(
                f"Batch_Size ({batch_size}) exceeds 2 * Throughput_Floor * Batch_Timeout "
                f"({limit}); the mean Evaluator batch size cannot reach 50% of Batch_Size "
                f"at this configured descent rate (Requirement 15.7)."
            )

    if worker_count is not None and batch_size is not None:
        floor = 2 * batch_size
        if worker_count < floor:
            warnings.append(
                f"Worker_Count ({worker_count}) is less than 2 * Batch_Size ({floor}); "
                f"steady-state Evaluator occupancy may fall short of Batch_Size."
            )

    return tuple(warnings)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw`` suitable for the startup configuration log.

    Every key in `CREDENTIAL_FIELDS` (currently just ``"PostgreSQL
    password"``) is replaced with `REDACTED_PLACEHOLDER`; every other key
    is copied through with its exact supplied value, unchanged. This
    redacts by field identity, not by scanning the rendered representation
    for the credential value as a substring, so a non-credential value
    that happens to contain the password as a substring is left untouched
    (see the module docstring's Property 42 note). Works on ``raw`` of any
    validity -- Requirement 13.6's logging step runs after validation
    succeeds in the normal startup flow, but this function itself makes no
    assumption that ``raw`` is valid or even complete.
    """
    redacted = dict(raw)
    for name in CREDENTIAL_FIELDS:
        if name in redacted:
            redacted[name] = REDACTED_PLACEHOLDER
    return redacted


# ---------------------------------------------------------------------------
# BookConfig
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
    """Raised by `BookConfig.load` when `validate_config` finds a problem.

    Carries the full `ValidationResult` so a caller (the startup flow in
    `__main__.py`, added by task 17.1) can format Requirement 13.2/13.3's
    exact report -- every absent name, every out-of-range name/value/range
    -- without re-deriving it, and can exit before any schema creation or
    Node_Store write, per the design's "Startup validation failures"
    flowchart.
    """

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__(self._format(result))

    @staticmethod
    def _format(result: ValidationResult) -> str:
        parts: list[str] = []
        if result.absent:
            parts.append("absent: " + ", ".join(result.absent))
        if result.out_of_range:
            parts.append(
                "out of range: " + "; ".join(str(finding) for finding in result.out_of_range)
            )
        return "; ".join(parts) if parts else "configuration is valid"


@dataclass
class BookConfig:
    """The PUCT Book Builder's operator configuration (Requirement 13).

    Holds every value of Requirement 13 criterion 1: the five PostgreSQL
    connection settings, the seventeen scalar values from Cache_Budget
    through Throughput_Grace_Period, and the optional Terashock_Book path
    (``terashock_book``, defaulting to ``None`` when absent -- Requirement
    13.8). Attribute names are the `snake_case` form of each value's
    Requirement 13 name; see `_NAME_TO_ATTR` for the exact mapping and
    design.md's ``cfg.propagation_visit_threshold`` usage in the
    propagation frame partition for the naming convention this follows.

    Construct via `BookConfig.load` (validates first, raises
    `ConfigValidationError` on any problem) or `BookConfig.from_mapping`
    (assumes ``raw`` already passed `validate_config`; used internally by
    `load` and available directly for callers -- such as a future
    property test -- that have already validated ``raw`` themselves and
    want to skip repeating the check).
    """

    pg_host: str
    pg_port: str
    pg_user: str
    pg_password: str
    pg_database: str
    cache_budget: int
    worker_count: int
    batch_size: int
    batch_timeout: float
    virtual_loss: int
    terashock_prior_weight: float
    eval_coef: float
    draw_value_black: float
    draw_value_white: float
    max_book_ply: int
    propagation_visit_threshold: int
    export_visit_threshold: float
    connection_retry_limit: int
    root_position: str
    report_interval: float
    throughput_floor: float
    throughput_grace_period: float
    terashock_book: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BookConfig":
        """Build a `BookConfig` directly from ``raw``, with no validation.

        Values are taken exactly as supplied, with no type coercion:
        `validate_config` is what enforces Requirement 13 criterion 7's
        text-vs-numeric kinds, so calling this on unvalidated input can
        raise `KeyError` for a required value later code would instead
        report as absent, or silently carry forward a value later code
        would instead report as out of range. Every caller on the startup
        path runs `validate_config` (or `BookConfig.load`, which does this
        for you) first and exits before reaching this constructor -- see
        design.md's "Startup validation failures" flowchart, node C2.
        """
        kwargs: dict[str, Any] = {}
        for row in RANGE_TABLE:
            attr = _NAME_TO_ATTR[row.name]
            if row.name in raw and raw[row.name] is not None:
                kwargs[attr] = raw[row.name]
            elif not row.required:
                kwargs[attr] = None
            else:
                raise KeyError(row.name)
        return cls(**kwargs)

    @classmethod
    def load(cls, raw: Mapping[str, Any], *, process_count: int = 1) -> "BookConfig":
        """Validate ``raw`` and build a `BookConfig`, or raise `ConfigValidationError`.

        This is `validate_config` followed by `from_mapping`, combined
        into the one call the startup flow actually wants: validate,
        and only on success construct the configuration object that the
        rest of the run consumes. Raises `ConfigValidationError` (carrying
        the full `ValidationResult`, including any warnings) when
        `validate_config` finds any absent or out-of-range value.
        """
        result = validate_config(raw, process_count=process_count)
        if not result.ok:
            raise ConfigValidationError(result)
        return cls.from_mapping(raw)

    def to_loggable_dict(self) -> dict[str, Any]:
        """Return this configuration as a dict suitable for the startup log.

        Every credential attribute in `_CREDENTIAL_ATTRS` (currently just
        ``pg_password``) is replaced with `REDACTED_PLACEHOLDER`; every
        other field is copied through with its exact value, unchanged --
        the same field-identity redaction `redact_mapping` performs on a
        raw mapping, applied here to an already-constructed `BookConfig`
        (Requirement 13.6, Property 42).
        """
        as_dict = dataclasses.asdict(self)
        for attr in _CREDENTIAL_ATTRS:
            if attr in as_dict:
                as_dict[attr] = REDACTED_PLACEHOLDER
        return as_dict


# ---------------------------------------------------------------------------
# Optional YAML convenience (thin; the tested surface is the mapping above)
# ---------------------------------------------------------------------------


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Read a YAML configuration file into a raw configuration mapping.

    Thin convenience only: no Requirement 13 validation happens here, and
    the returned mapping's keys are expected to already use the
    `RANGE_TABLE` names (e.g. ``"Worker_Count"``, ``"PostgreSQL
    password"``). This exists so an operator's config file (design.md's
    `CFG` box, "config file (YAML)") can be turned into the ``raw``
    mapping `validate_config` and `BookConfig.load` operate on, without
    requiring every caller -- including hypothesis-driven property tests,
    which generate dicts directly and never touch this function -- to go
    through a file at all.

    ``import yaml`` is deferred to inside this function rather than done at
    module level, so that importing `dlshogi.book.config` itself (and
    hypothesis generating against its pure, in-memory surface) never
    depends on PyYAML being importable; only a caller that actually wants
    to read a YAML file pays that cost.
    """
    import yaml  # local import: see docstring

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML content must be a mapping, got {type(data)!r}")
    return data


__all__ = [
    "RangeRow",
    "RANGE_TABLE",
    "CREDENTIAL_FIELDS",
    "REDACTED_PLACEHOLDER",
    "OutOfRangeFinding",
    "ValidationResult",
    "validate_config",
    "compute_warnings",
    "redact_mapping",
    "ConfigValidationError",
    "BookConfig",
    "load_yaml",
]
