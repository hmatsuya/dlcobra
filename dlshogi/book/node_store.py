"""Node_Store: connection, schema management, and reconnection.

This module implements the *connection and schema* half of the Node_Store
component (design.md's "Node_Store" section): the asyncpg pool, startup
schema creation/versioning/repair, Root_Position recording and read-back,
the connection retry schedule, and the lost-connection suspension seam.

**Scope.** The read path (`get`, `get_many`, `get_many_terminal_eval`, the
node LRU) was added by task 7.6. The write path -- `insert_expansion`
(task 8.1), the coalescing backup accumulator and `flush` (task 8.6),
`set_propagation`/`stats`/the RSS sampler (task 8.8), and the in-flight
claim mirror (task 8.10) -- is added by this module too, extending the
same class rather than replacing it, and builds on the reconnection seam
(`_run`) so every write inherits lost-connection handling for free. As of
task 9.2, `flush()` dispatches each node to one of two backup-patch
backends, chosen once at startup by detecting whether the `puct_edge`
PostgreSQL extension (task 9.1) is installed: the in-database
`puct_edge_backup` UPDATE (`_apply_backup_row_extension`) when detected,
otherwise the client-side `SELECT ... FOR UPDATE` fallback
(`_apply_backup_row_fallback`). Both paths take PostgreSQL's row-level
exclusive lock around the read-modify-write, so the no-lost-update
guarantee holds either way; see `_detect_puct_edge_extension` and the
`backend` property.

**Startup sequencing** (design.md's "Error Handling" mermaid flowchart,
nodes C4 through C17) is implemented by `NodeStore.connect` plus
`ensure_schema`, in the exact order the flowchart states:

1. connect, with the 5-second-per-attempt / 1-second-doubling-to-60-second
   retry schedule (`connect`, `_connect_with_retry`; Requirement 2.2, 2.3);
2. check whether the schema is present (`_schema_present`);
3. absent -> create the schema, record schema version, Zobrist
   fingerprint, and Root_Position, atomically and within a 300 s deadline
   (`_create_schema`; Requirement 2.4, 2.6, 2.7);
4. present -> compare the recorded schema version and Zobrist fingerprint
   against the running ones; a mismatch is a hard error with no schema
   alteration and no read or write (`_check_version_and_fingerprint`;
   Requirement 2.8); a match repairs only the absent schema elements
   (`_repair_schema`; Requirement 2.5).

The C11 ("any node rows but no root row?") through C17 checks are a later
task's concern (`__main__.py`, task 17.1): this module exposes
`get_root()` and `root_node_exists()` so that later code can build the
C11/X6 decision on top of this module without this module deciding
anything about whether search may proceed.

**No side effects at import time**, per this package's established
convention (see config.py's docstring): every connection and schema
operation happens inside an explicit `async def`, never at module import.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, TypeVar

import asyncpg
import numpy as np

from dlshogi.book import packed_edge
from dlshogi.book.keys import (
    PositionKey,
    fold_u64_to_i64,
    position_key_from_sfen,
    unfold_i64_to_u64,
    zobrist_fingerprint,
)

_LOG = logging.getLogger(__name__)

# The schema version identifier of the running PUCT_Book_Builder
# (Requirement 2.4, 2.8). design.md does not give a concrete value for
# this identifier; it is this module's own concern, incremented whenever
# dlshogi/book/sql/schema.sql's schema shape changes in a way that is not
# handled by the absent-element-only repair path.
SCHEMA_VERSION = "1"

# Path to the DDL this module applies at schema creation (task 7.1's
# output). Read lazily so that importing this module performs no I/O.
_SCHEMA_SQL_PATH = Path(__file__).parent / "sql" / "schema.sql"

# Requirement 2.6: schema creation must complete (or fail) within this
# many seconds.
_SCHEMA_CREATION_TIMEOUT_S = 300.0

# Requirement 2.2: each connection attempt (including retries) is bounded
# to this many seconds.
_CONNECT_ATTEMPT_TIMEOUT_S = 5.0

# Requirement 2.2: wait 1 second before the first retry, doubling before
# each subsequent retry, up to this maximum wait per retry.
_RETRY_INITIAL_WAIT_S = 1.0
_RETRY_MAX_WAIT_S = 60.0

# design.md's "The backup write path": "applies accumulated deltas to
# PostgreSQL every flush_interval (default 200 ms)". BookConfig has no
# dedicated field for this (see config.py's RANGE_TABLE), so this is a
# module-level constant, matching this module's existing convention for
# values design.md fixes rather than exposing as Operator configuration
# (compare _SCHEMA_CREATION_TIMEOUT_S, _CONNECT_ATTEMPT_TIMEOUT_S above).
_FLUSH_INTERVAL_S = 0.2

# Requirement 15.4: resident memory sampled at intervals of at most 10 s.
_RSS_SAMPLE_INTERVAL_S = 10.0

# Requirement 15.4's bound: Cache_Budget + evaluator-init RSS + 512 MiB.
_RSS_BOUND_OVERHEAD_BYTES = 512 * 1024 * 1024

# Judgement call (task 8.6, "on Cache_Budget pressure"): with no
# Search_Coordinator yet to drive an automatic poll, `cache_pressure_exceeded`
# is exposed as a callable a future caller can check between the 200 ms
# flusher interval ticks. Chosen as half of Cache_Budget, a deliberately
# conservative threshold since the accumulator's own footprint is meant to
# be a small fraction of Cache_Budget in practice.
_CACHE_PRESSURE_FRACTION = 0.5

# The three session roles a NodeStore instance may serve, each with its
# own `synchronous_commit` setting (design.md, "Resumability makes the
# graph reconstructible..."): "off" for the search session, trading a
# `wal_writer_delay` window of durability for commit latency, and "on" for
# propagation and export sessions, where a partially tagged `prop_epoch`
# would otherwise be possible.
Role = Literal["search", "propagate", "export"]
_SYNCHRONOUS_COMMIT_BY_ROLE: dict[str, str] = {
    "search": "off",
    "propagate": "on",
    "export": "on",
}

# The schema's expected top-level elements, used only by the
# absent-element-only repair path (`_repair_schema`). Kept intentionally
# narrow: Requirement 2.5's droppable elements are indexes, constraints,
# and non-key columns -- never whole tables holding existing rows -- so
# this list names tables (created with IF NOT EXISTS, a no-op when the
# table and its rows already exist) and the one index schema.sql defines.
#
# NOTE: these DDL strings are a deliberate, narrow duplication of
# dlshogi/book/sql/schema.sql's CREATE TABLE statements, needed because
# schema.sql's statements are not themselves idempotent (no "IF NOT
# EXISTS"). They MUST be kept in sync with schema.sql whenever that file's
# table shapes change; a schema_version bump is the signal to re-check
# them by hand.
_REPAIR_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS book_meta (
        id                     smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        schema_version         text     NOT NULL,
        zobrist_fingerprint    bigint   NOT NULL,
        root_sfen              text     NOT NULL,
        root_key_hi            bigint   NOT NULL,
        root_key_lo            bigint   NOT NULL,
        search_write_seq       bigint   NOT NULL DEFAULT 0,
        propagation_seq        bigint   NOT NULL DEFAULT 0,
        propagation_done_seq   bigint   NOT NULL DEFAULT 0,
        terashock_source_path  text,
        terashock_source_size  bigint,
        terashock_source_mtime bigint,
        terashock_entry_count  bigint,
        created_at             timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS book_node (
        key_hi           bigint  NOT NULL,
        key_lo           bigint  NOT NULL,
        sfen             text    NOT NULL,
        apery_key        bigint  NOT NULL,
        visit_count      bigint  NOT NULL DEFAULT 0 CHECK (visit_count >= 0),
        value_sum        float8  NOT NULL DEFAULT 0,
        terminal         smallint NOT NULL DEFAULT 0,
        flags            smallint NOT NULL DEFAULT 0,
        eval_win_rate    real,
        prop_value       float8,
        prop_best_move16 smallint,
        prop_epoch       integer NOT NULL DEFAULT 0,
        edge_count       smallint NOT NULL DEFAULT 0,
        edges            bytea   NOT NULL DEFAULT '',
        CONSTRAINT book_node_pkey PRIMARY KEY (key_hi, key_lo),
        CONSTRAINT book_node_edges_len CHECK (octet_length(edges) = edge_count * 20)
    ) WITH (fillfactor = 70)
    """,
    "ALTER TABLE book_node ALTER COLUMN edges SET STORAGE MAIN",
    "ALTER TABLE book_node ALTER COLUMN sfen  SET STORAGE MAIN",
    """
    CREATE TABLE IF NOT EXISTS terashock_entry (
        key_hi bigint NOT NULL,
        key_lo bigint NOT NULL,
        sfen   text   NOT NULL,
        moves  bytea  NOT NULL,
        PRIMARY KEY (key_hi, key_lo)
    ) WITH (fillfactor = 100)
    """,
    """
    CREATE TABLE IF NOT EXISTS in_flight_claim (
        key_hi     bigint NOT NULL,
        key_lo     bigint NOT NULL,
        process_id integer NOT NULL,
        worker_id  integer NOT NULL,
        claimed_at timestamptz NOT NULL,
        PRIMARY KEY (key_hi, key_lo)
    )
    """,
)

# The tables schema.sql creates; used by `_schema_present` to decide
# whether the schema exists at all (Requirement 2.4 vs 2.5/2.8 branch).
_EXPECTED_TABLES: tuple[str, ...] = (
    "book_meta",
    "book_node",
    "terashock_entry",
    "in_flight_claim",
)


class NodeStoreError(Exception):
    """Base class for Node_Store startup and operational errors."""


class NodeStoreConnectionError(NodeStoreError):
    """Raised when connecting to PostgreSQL exhausts every retry (Requirement 2.3).

    Names the configured host, port, and database, and the number of
    attempts made, so the Operator report this exception backs is never
    opaque.
    """

    def __init__(self, host: Any, port: Any, database: Any, attempts: int, cause: BaseException):
        self.host = host
        self.port = port
        self.database = database
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"could not connect to PostgreSQL database "
            f"{database!r} at {host!r}:{port!r} after {attempts} attempt(s): {cause!r}"
        )


class SchemaCreationError(NodeStoreError):
    """Raised when schema creation fails or exceeds the 300 s deadline (Requirement 2.6).

    ``timed_out`` distinguishes the two reportable causes criterion 6
    names explicitly.
    """

    def __init__(self, host: Any, port: Any, database: Any, *, timed_out: bool, cause: Optional[BaseException] = None):
        self.host = host
        self.port = port
        self.database = database
        self.timed_out = timed_out
        self.cause = cause
        reason = f"exceeded the {_SCHEMA_CREATION_TIMEOUT_S:.0f} s limit" if timed_out else f"failed: {cause!r}"
        super().__init__(
            f"schema creation for PostgreSQL database {database!r} at "
            f"{host!r}:{port!r} {reason}"
        )


class SchemaVersionMismatchError(NodeStoreError):
    """Raised when the recorded schema version or Zobrist fingerprint disagree
    with the running ones (Requirement 2.8).

    No schema element is created or altered and no ``book_node`` row is
    read or written before this is raised.
    """

    def __init__(
        self,
        *,
        recorded_version: Optional[str],
        running_version: str,
        recorded_fingerprint: Optional[int] = None,
        running_fingerprint: Optional[int] = None,
    ):
        self.recorded_version = recorded_version
        self.running_version = running_version
        self.recorded_fingerprint = recorded_fingerprint
        self.running_fingerprint = running_fingerprint
        super().__init__(
            "schema version/fingerprint mismatch: recorded schema_version="
            f"{recorded_version!r} zobrist_fingerprint={recorded_fingerprint!r}, "
            f"running schema_version={running_version!r} "
            f"zobrist_fingerprint={running_fingerprint!r}"
        )


def retry_schedule(connection_retry_limit: int) -> list[float]:
    """Return the Requirement 2.2 retry wait schedule, in seconds.

    A pure function of ``connection_retry_limit`` (the design's
    ``Connection_Retry_Limit``): the returned list has exactly
    ``connection_retry_limit`` entries, the first is 1.0, each subsequent
    entry is double its predecessor capped at 60.0, and no entry exceeds
    60.0. This is the schedule of waits *before* each of the
    ``connection_retry_limit`` retries that follow the first failed
    attempt; it says nothing about the attempts themselves, only about how
    long to wait between them, which is what makes it independently
    testable (task 7.3's Property 6) without any network or database
    access.
    """
    if connection_retry_limit < 0:
        raise ValueError(f"connection_retry_limit must be >= 0, got {connection_retry_limit}")
    schedule: list[float] = []
    wait = _RETRY_INITIAL_WAIT_S
    for _ in range(connection_retry_limit):
        schedule.append(wait)
        wait = min(wait * 2.0, _RETRY_MAX_WAIT_S)
    return schedule


_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Read path types (task 7.6): `Terminal`, `BookNodeView`, `GetResult`.
# ---------------------------------------------------------------------------


class Terminal(enum.IntEnum):
    """``book_node.terminal`` (schema.sql: "0 none, 1 win for stm, 2 loss for stm").

    An `IntEnum` so a row's raw `smallint` value round-trips through this
    type with no explicit conversion at either the read or (a later task's)
    write boundary.
    """

    NONE = 0
    WIN_FOR_STM = 1
    LOSS_FOR_STM = 2


@dataclass
class BookNodeView:
    """One Book_Node together with its full Book_Edge list (design.md's sketch).

    ``edges`` is the zero-copy `PACKED_EDGE`-dtype view `packed_edge.decode_edges`
    returns; already in ascending-USI order because the encoder wrote it that
    way (task 3.5/3.6) and the decoder performs no reorder.

    ``cyclic_flag`` is bit 0 of the row's `flags` column, decoded once here;
    the raw `flags` int itself is not exposed, matching design.md's field
    list (only `cyclic_flag`, not `flags`).
    """

    key: PositionKey
    sfen: str
    apery_key: int
    visit_count: int
    value_sum: float
    terminal: Terminal
    cyclic_flag: bool
    eval_win_rate: Optional[float]
    prop_value: Optional[float]
    prop_best_move16: Optional[int]
    edges: np.ndarray


class GetResult(enum.Enum):
    """The three distinguishable outcomes of `NodeStore.get` (Requirement 1.4)."""

    FOUND = 1
    ABSENT = 2
    FAILED = 3


_FLAG_CYCLIC = 0x01


def _row_to_view(row: asyncpg.Record) -> BookNodeView:
    """Build a `BookNodeView` from one `book_node` row (full-column select)."""
    key = PositionKey(unfold_i64_to_u64(row["key_hi"]), unfold_i64_to_u64(row["key_lo"]))
    flags = row["flags"] or 0
    return BookNodeView(
        key=key,
        sfen=row["sfen"],
        apery_key=unfold_i64_to_u64(row["apery_key"]),
        visit_count=row["visit_count"],
        value_sum=row["value_sum"],
        terminal=Terminal(row["terminal"]),
        cyclic_flag=bool(flags & _FLAG_CYCLIC),
        eval_win_rate=row["eval_win_rate"],
        prop_value=row["prop_value"],
        prop_best_move16=row["prop_best_move16"],
        edges=packed_edge.decode_edges(bytes(row["edges"])),
    )


# ---------------------------------------------------------------------------
# The node LRU (Requirement 1.5, 1.8): bounded by Cache_Budget, evicting
# rather than failing.
# ---------------------------------------------------------------------------

# Per-entry overhead estimate covering the dataclass instance, the
# PositionKey tuple key, and the OrderedDict slot itself. This is a
# deliberate approximation, not exact Python object sizing -- Property 3
# only needs *bounded, evicting, never-failing* behaviour, not a byte-exact
# accounting (design.md's own worked example is itself an estimate: "an
# 600-edge, 128-char-SFEN node is ~2.2 KB").
_LRU_ENTRY_OVERHEAD_BYTES = 128


def _estimate_view_bytes(view: BookNodeView) -> int:
    """Approximate the resident-memory footprint of one cached `BookNodeView`.

    ``len(sfen.encode()) + edges.nbytes + a fixed per-entry overhead``, per
    the task's own guidance; not exact, only bounded and monotonic in the
    inputs that actually vary in size (SFEN length, edge count).
    """
    return len(view.sfen.encode("utf-8")) + int(view.edges.nbytes) + _LRU_ENTRY_OVERHEAD_BYTES


class _NodeLRU:
    """A byte-accounted, evicting-not-failing LRU cache of `BookNodeView`.

    Keyed by `PositionKey`. `get`/`put` both touch (move-to-end) the entry
    they access, so recently used entries survive eviction preferentially
    (Requirement 1.8). Bounded by ``budget_bytes``: `put` evicts
    least-recently-used entries until the new entry fits, and if the new
    entry alone is larger than the *entire* budget, it is not cached at all
    (evicting everything else first would still leave the store over
    budget for no benefit, since the oversized entry could never coexist
    with anything else) -- this keeps `cache_bytes <= budget_bytes` an
    invariant that holds after every call, never violated even
    transiently, per Property 3's wording ("every call completes
    successfully and the tracked cache size never exceeds Cache_Budget").
    """

    def __init__(self, budget_bytes: int) -> None:
        self._budget_bytes = max(0, int(budget_bytes))
        self._entries: "OrderedDict[PositionKey, BookNodeView]" = OrderedDict()
        self._sizes: dict[PositionKey, int] = {}
        self._total_bytes = 0

    @property
    def cache_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: PositionKey) -> Optional[BookNodeView]:
        view = self._entries.get(key)
        if view is None:
            return None
        self._entries.move_to_end(key)
        return view

    def put(self, key: PositionKey, view: BookNodeView) -> None:
        size = _estimate_view_bytes(view)

        # Replacing an existing entry: drop its old size first so the
        # eviction loop below sees the correct current total.
        if key in self._entries:
            self._total_bytes -= self._sizes[key]
            del self._entries[key]
            del self._sizes[key]

        if size > self._budget_bytes:
            # A single entry that alone exceeds the whole budget is never
            # cached (see class docstring); the read that produced it still
            # succeeds, it is simply not remembered.
            return

        while self._total_bytes + size > self._budget_bytes and self._entries:
            evict_key, _ = self._entries.popitem(last=False)
            self._total_bytes -= self._sizes.pop(evict_key)

        self._entries[key] = view
        self._sizes[key] = size
        self._total_bytes += size

    def __contains__(self, key: PositionKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def invalidate(self, key: PositionKey) -> None:
        """Drop ``key`` from the cache if present (used by `set_propagation`,
        task 8.8, so a stale cached view is never served after a direct
        `prop_value`/`prop_best_move16`/`prop_epoch` UPDATE that bypasses
        the backup accumulator).
        """
        if key in self._entries:
            self._total_bytes -= self._sizes.pop(key)
            del self._entries[key]

    def shrink_by(self, amount_bytes: int) -> None:
        """Evict least-recently-used entries until at least ``amount_bytes``
        of cache have been freed, or the cache is empty (task 8.8's RSS
        sampler; Requirement 15.5). Never touches PostgreSQL: every
        written Book_Node and Book_Edge record survives this call, only
        the in-process cache shrinks.
        """
        if amount_bytes <= 0:
            return
        target = max(0, self._total_bytes - amount_bytes)
        while self._total_bytes > target and self._entries:
            evict_key, _ = self._entries.popitem(last=False)
            self._total_bytes -= self._sizes.pop(evict_key)


# ---------------------------------------------------------------------------
# Write path types (task 8.1, 8.6, 8.8, 8.10): `ExpansionWrite`,
# `WriteOutcome`, `WriteResult`, `BackupDelta`, `PropagationWrite`, `Stats`.
#
# design.md's Node_Store component sketch names these types
# (`ExpansionWrite`, `WriteResult`, `BackupDelta`, `PropagationWrite`,
# `Stats`) but does not spell out their fields; the shapes below are this
# task's own design, chosen to be exactly what `insert_expansion`,
# `backup`, and `set_propagation` need and no more.
# ---------------------------------------------------------------------------


@dataclass
class ExpansionWrite:
    """One Book_Node-plus-its-Book_Edges write, for `NodeStore.insert_expansion`.

    ``edges`` is anything `packed_edge.encode_edges` accepts (a sequence of
    mappings or attribute-holders exposing the `PACKED_EDGE` field names);
    an empty sequence is legal (a terminal node has no edges, Requirement
    8.11 / 8.6's own `terminal` field here). ``apery_key`` and ``key`` are
    the *unsigned* 64-bit values; the two's-complement fold into signed
    `bigint` happens inside `insert_expansion` itself, not here, matching
    how `BookNodeView`/`_row_to_view` keep the unsigned form at this
    module's public boundary.
    """

    key: PositionKey
    sfen: str
    apery_key: int
    terminal: Terminal = Terminal.NONE
    eval_win_rate: Optional[float] = None
    edges: Sequence[Any] = ()


class WriteOutcome(enum.Enum):
    """The four distinguishable outcomes of `NodeStore.insert_expansion`.

    ``COMMITTED``: this call's row was the one that survived the
    ``ON CONFLICT DO NOTHING`` race (Requirement 1.6).
    ``DUPLICATE``: another writer's row already existed with the *same*
    SFEN; this is an ordinary transposition-merge loss (Requirement 11.6),
    and re-applying an identical expansion also lands here (Requirement
    10.4's idempotence).
    ``COLLISION``: another writer's row already existed at the same
    Position_Key but with a *different* SFEN (Requirement 3.6) -- a true
    128-bit key collision, not a transposition.
    ``FAILED``: the statement itself could not be completed (Requirement
    1.9), e.g. a reconnect-exhaustion error re-raised by `self._run`.
    """

    COMMITTED = 1
    DUPLICATE = 2
    COLLISION = 3
    FAILED = 4


@dataclass
class WriteResult:
    """`NodeStore.insert_expansion`'s return value.

    ``existing_sfen`` / ``attempted_sfen`` are populated for `DUPLICATE`
    (``existing_sfen`` only, which equals the attempted SFEN by
    definition) and for `COLLISION` (both, since they differ -- the whole
    point of Requirement 3.6's report). ``error`` is populated only for
    `FAILED`.
    """

    outcome: WriteOutcome
    key: PositionKey
    existing_sfen: Optional[str] = None
    attempted_sfen: Optional[str] = None
    error: Optional[BaseException] = None


@dataclass
class BackupDelta:
    """One Book_Node's pending backup increments (`NodeStore.backup`, task 8.6).

    ``edge_deltas`` maps ``move16 -> (visit_delta, value_delta)``. A
    descent (or a later task's caller) constructs one `BackupDelta` per
    Book_Node on its descent path and passes the whole path's list to one
    `backup()` call; `backup()` merges each of these into the persistent
    per-key accumulator entry of the same shape, so this same dataclass
    doubles as the accumulator's own per-key storage.
    """

    key: PositionKey
    node_visit_delta: int = 0
    node_value_delta: float = 0.0
    flags_or: int = 0
    edge_deltas: dict[int, tuple[int, float]] = field(default_factory=dict)


@dataclass
class PropagationWrite:
    """One Book_Node's `prop_value`/`prop_best_move16`/`prop_epoch` write.

    Unlike `BackupDelta`, these are absolute values, not deltas
    (Requirement 9's propagated value replaces, it does not accumulate),
    so `set_propagation` issues a direct UPDATE rather than routing
    through the visit/value accumulator (see design.md's "Value
    propagation..." section; the accumulator's increment semantics do not
    fit an absolute write). ``prop_epoch`` is the current propagation pass
    id (`book_meta.propagation_seq`, obtained by the caller via
    `NodeStore.next_propagation_seq()`), which is what makes the memo
    lookup in a later pass a plain equality test.
    """

    key: PositionKey
    prop_value: float
    prop_best_move16: Optional[int]
    prop_epoch: int


@dataclass
class Stats:
    """`NodeStore.stats()`'s return value (task 8.8).

    Latencies are seconds; ``*_count`` is the number of samples the mean
    and p95 were computed from (0 means "no data yet", in which case the
    mean/p95 fields read 0.0 rather than NaN -- see `_LatencyHistogram`).
    """

    read_latency_mean_s: float
    read_latency_p95_s: float
    read_count: int
    write_latency_mean_s: float
    write_latency_p95_s: float
    write_count: int
    cache_bytes: int
    cache_budget_bytes: int
    duplicate_count: int
    collision_count: int
    cache_release_events: int


class _LatencyHistogram:
    """A minimal read/write latency tracker: mean and p95 over recorded samples.

    design.md's Progress_Reporter section describes "per-interval bucketed
    histograms as numpy arrays, reset each interval"; `report.py` itself
    (task 16.1) is not implemented yet, so this is a simpler in-process
    tracker sufficient for `NodeStore.stats()` alone: an unbounded list of
    per-call latencies in seconds, from which mean/p95 are computed on
    demand via numpy. `reset()` is exposed so a future `report.py` can
    apply the same "reset each Report_Interval" policy without this class
    changing.
    """

    def __init__(self) -> None:
        self._samples: list[float] = []

    def record(self, seconds: float) -> None:
        self._samples.append(seconds)

    def reset(self) -> None:
        self._samples = []

    @property
    def count(self) -> int:
        return len(self._samples)

    def mean(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    def p95(self) -> float:
        return float(np.percentile(self._samples, 95)) if self._samples else 0.0


def _default_rss_bytes() -> int:
    """Read this process's current resident set size, in bytes.

    Prefers ``/proc/self/status``'s ``VmRSS`` (Linux, matching this
    repository's deployment target), which is *current* RSS. Falls back to
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (KB on Linux) only when
    ``/proc`` is unavailable; that fallback reports *peak*, not current,
    RSS, so it is a last resort, not the primary path. No `psutil`
    dependency is added, per this task's explicit instruction; `psutil` is
    not in `Pipfile` and adding it was not authorized.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:  # noqa: BLE001 - RSS accounting must never crash the caller
        return 0


def _apply_pending_to_view(
    view: BookNodeView, pending: Optional[BackupDelta]
) -> BookNodeView:
    """Overlay a pending (unflushed) `BackupDelta` onto a `BookNodeView`.

    Requirement 4.5's visit-count invariant must hold on *read*, before any
    flush; this is what makes that true for `get`/`get_many`. Returns
    ``view`` itself, unmodified, when ``pending`` is `None` or carries no
    actual delta (the common case -- most reads find nothing pending),
    so callers that never touch the write path pay no allocation cost.
    Otherwise returns a new `BookNodeView` (via `dataclasses.replace`) with
    a freshly copied, patched ``edges`` array; the cached/decoded array
    itself is never mutated in place, since it may be shared by other
    readers (the node LRU, or a concurrent `get` of the same key).
    """
    if pending is None or (
        pending.node_visit_delta == 0
        and pending.node_value_delta == 0.0
        and pending.flags_or == 0
        and not pending.edge_deltas
    ):
        return view

    edges = view.edges
    if pending.edge_deltas:
        edges = edges.copy()
        index_by_move16 = {int(m16): i for i, m16 in enumerate(edges["move16"])}
        for move16, (visit_delta, value_delta) in pending.edge_deltas.items():
            i = index_by_move16.get(int(move16))
            if i is not None:
                edges[i]["visit_count"] = edges[i]["visit_count"] + visit_delta
                edges[i]["value_sum"] = edges[i]["value_sum"] + value_delta

    return dataclasses.replace(
        view,
        visit_count=view.visit_count + pending.node_visit_delta,
        value_sum=view.value_sum + pending.node_value_delta,
        cyclic_flag=view.cyclic_flag or bool(pending.flags_or & _FLAG_CYCLIC),
        edges=edges,
    )


class NodeStore:
    """Node_Store: PostgreSQL-backed Book_Graph storage.

    Construct with `NodeStore(config, role=...)` and call `await
    store.connect()` followed by `await store.ensure_schema()` to run the
    full startup sequence, or `await NodeStore.connect_and_prepare(config,
    role=...)` to do both in one call. Neither constructor performs any
    I/O; `connect` and `ensure_schema` are the only methods that touch the
    network before the read/write path (task 7.6/8.x) methods are called.

    ``role`` selects the `synchronous_commit` setting for every connection
    this instance's pool opens (design.md's `synchronous_commit`
    paragraph): ``"search"`` -> ``off``, ``"propagate"`` / ``"export"`` ->
    ``on``. One `NodeStore` instance serves one role, matching "one
    process per GPU" / the single-process propagate and export commands.
    """

    def __init__(self, config: Any, *, role: Role = "search") -> None:
        self._config = config
        self._role = role
        self._pool: Optional[asyncpg.Pool] = None
        # Requirement 2.9: normally *set* (not suspended, proceed); cleared
        # while suspended so `await suspended.wait()` blocks callers until
        # a reconnect succeeds.
        self.suspended = asyncio.Event()
        self.suspended.set()
        self._reconnect_lock = asyncio.Lock()
        # Root_Position, populated by ensure_schema()/_load_root() so
        # get_root() needs no round trip once startup has run.
        self._root_sfen: Optional[str] = None
        self._root_key_hi: Optional[int] = None
        self._root_key_lo: Optional[int] = None
        # Node LRU (task 7.6, Requirement 1.5/1.8): bounded by Cache_Budget,
        # evicting rather than failing. Populated by `get`/`get_many`, never
        # by `get_many_terminal_eval` (that method's whole point is to skip
        # the cache for below-threshold propagation children).
        self._cache = _NodeLRU(int(self._config.cache_budget))
        # Pending backup accumulator (task 8.6): keyed by PositionKey,
        # holding uncommitted node/edge visit and value deltas. `get`
        # overlays this on top of whatever base row it finds (cache or
        # PostgreSQL), so uncommitted increments are visible on read
        # (Requirement 4.5). Populated by `backup()`, drained by `flush()`.
        self._pending_backup: dict[PositionKey, "BackupDelta"] = {}

        # -- task 9.2: backend selection for the backup patch path ---------
        # Defaults to False (the fallback path) until _detect_puct_edge_
        # extension has run; ensure_schema() runs it once, at startup.
        self._puct_edge_extension_available: bool = False

        # -- task 8.1: expansion write counters (Requirement 11.6, 3.6) ----
        self.duplicate_count = 0
        self.collision_count = 0

        # -- task 8.6: flusher lifecycle -----------------------------------
        self._flusher_task: Optional[asyncio.Task] = None
        self._flusher_stop_event: Optional[asyncio.Event] = None

        # -- task 8.8: stats, RSS sampler ----------------------------------
        self._read_latency = _LatencyHistogram()
        self._write_latency = _LatencyHistogram()
        self._evaluator_baseline_rss_bytes = 0
        self._rss_reader: Callable[[], int] = _default_rss_bytes
        self.cache_release_events = 0
        self._on_cache_release: Optional[Callable[[], None]] = None
        self._rss_task: Optional[asyncio.Task] = None
        self._rss_stop_event: Optional[asyncio.Event] = None

    # -- connection settings, from BookConfig ---------------------------

    @property
    def _connect_kwargs(self) -> dict[str, Any]:
        server_settings = {"synchronous_commit": _SYNCHRONOUS_COMMIT_BY_ROLE[self._role]}
        kwargs: dict[str, Any] = {
            "host": self._config.pg_host,
            "port": self._config.pg_port,
            "user": self._config.pg_user,
            "database": self._config.pg_database,
            "server_settings": server_settings,
        }
        if self._config.pg_password:
            kwargs["password"] = self._config.pg_password
        return kwargs

    @property
    def role(self) -> Role:
        return self._role

    @property
    def effective_synchronous_commit(self) -> str:
        """The `synchronous_commit` value this instance's connections use."""
        return _SYNCHRONOUS_COMMIT_BY_ROLE[self._role]

    @property
    def backend(self) -> Literal["puct_edge_extension", "client_side_fallback"]:
        """Which backup-patch backend `flush()` currently dispatches to.

        Reflects `self._puct_edge_extension_available`, which is `False`
        (i.e. this reads `"client_side_fallback"`) until
        `_detect_puct_edge_extension` has run -- normally as part of
        `ensure_schema()`/`connect_and_prepare()`. Exposed as a read-only
        property (task 9.2) so tests and callers can assert on the
        resolved backend without reaching into a private attribute.
        """
        return (
            "puct_edge_extension"
            if self._puct_edge_extension_available
            else "client_side_fallback"
        )

    # -- connect, with the Requirement 2.2/2.3 retry schedule -----------

    async def connect(self) -> None:
        """Create the asyncpg pool, retrying on the Requirement 2.2 schedule.

        Raises `NodeStoreConnectionError` (Requirement 2.3) once
        ``Connection_Retry_Limit`` retries are exhausted, i.e. after
        ``Connection_Retry_Limit + 1`` total failed attempts. Logs the
        resolved `synchronous_commit` setting once this succeeds
        (design.md: "The Progress_Reporter logs the effective setting at
        startup so the trade is visible" -- this module logs it via the
        stdlib `logging` module directly, since `report.py`'s
        Progress_Reporter does not yet exist as a real implementation).
        """
        retry_limit = int(self._config.connection_retry_limit)
        schedule = retry_schedule(retry_limit)
        max_size = min(int(self._config.worker_count), 64)

        attempts = 0
        last_exc: Optional[BaseException] = None
        for wait_before in (None, *schedule):
            if wait_before is not None:
                await asyncio.sleep(wait_before)
            attempts += 1
            try:
                await self._probe_connect()
                break
            except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                continue
        else:
            raise NodeStoreConnectionError(
                self._config.pg_host,
                self._config.pg_port,
                self._config.pg_database,
                attempts,
                last_exc if last_exc is not None else RuntimeError("no attempt made"),
            )

        self._pool = await asyncpg.create_pool(
            min_size=1,
            max_size=max_size,
            **self._connect_kwargs,
        )
        _LOG.info(
            "Node_Store connected: host=%s port=%s database=%s role=%s "
            "synchronous_commit=%s max_size=%d",
            self._config.pg_host,
            self._config.pg_port,
            self._config.pg_database,
            self._role,
            self.effective_synchronous_commit,
            max_size,
        )

    async def _probe_connect(self) -> None:
        """One single-connection probe, bounded to the 5 s per-attempt limit.

        Used only to validate that a connection can be established before
        `create_pool` is called; the probe connection is closed
        immediately, whether it succeeds or not.
        """
        conn = await asyncio.wait_for(
            asyncpg.connect(**self._connect_kwargs), timeout=_CONNECT_ATTEMPT_TIMEOUT_S
        )
        await conn.close()

    async def close(self) -> None:
        """Close the pool. Safe to call even if `connect` was never called."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise NodeStoreError("Node_Store.connect() has not been called")
        return self._pool

    # -- schema management (Requirement 2.4, 2.5, 2.6, 2.7, 2.8) ---------

    async def ensure_schema(self) -> None:
        """Run the full C6-C9 schema decision (design.md's startup flowchart).

        - schema absent -> `_create_schema` (create + record version/
          fingerprint/root, atomically, within the 300 s deadline);
        - schema present -> `_check_version_and_fingerprint`, then
          `_repair_schema` (create only what is absent).

        Always loads `get_root()`'s cached values at the end, so later
        code (this module's `root_node_exists()`, and `__main__.py`'s C11
        check) can rely on them without an extra round trip.

        Each branch below acquires its own connection rather than sharing
        one held across the whole method: `_create_schema` acquires
        internally, and holding a connection here too would deadlock the
        pool when `max_size == 1` (`Worker_Count == 1`).
        """
        async with self.pool.acquire() as conn:
            present = await self._schema_present(conn)
        if not present:
            await self._create_schema()
        else:
            async with self.pool.acquire() as conn:
                await self._check_version_and_fingerprint(conn)
                await self._repair_schema(conn)
        await self._load_root()
        await self._detect_puct_edge_extension()

    async def _detect_puct_edge_extension(self) -> None:
        """Task 9.2: detect the `puct_edge` extension once, at startup.

        Sets `self._puct_edge_extension_available` (read via the
        `backend` property) and logs the resolved backup-patch backend
        at INFO once, clearly, so which path is in use is visible in the
        startup log.

        Checks `pg_extension` for an extension named ``puct_edge``,
        rather than probing `pg_proc`/`pg_catalog` for a function named
        `puct_edge_backup`: the question this method answers is "was
        `CREATE EXTENSION puct_edge;` run", not "does some function with
        this name happen to exist" -- the control/SQL files task 9.1
        added (`puct_edge.control`, `puct_edge--1.0.sql`) are designed for
        the normal `CREATE EXTENSION` flow, and `pg_extension` is the
        direct, minimal-surface way to ask that question.

        Any failure of the detection query itself (e.g. `pg_extension`
        unreadable for a benign permissions reason) is treated as
        "extension not available" -- narrowly caught as
        `asyncpg.PostgresError` -- rather than raised, so a detection
        hiccup falls back safely instead of aborting startup. A genuine
        connection-lost exception during detection is not caught here:
        it propagates so the caller sees it through the same path every
        other statement in this module does (this method does not go
        through `_run`/the suspension seam, since it only runs once
        during startup, before any descent could be waiting on
        `suspended`).
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'puct_edge'"
                )
            self._puct_edge_extension_available = row is not None
        except asyncpg.PostgresError as exc:
            _LOG.warning(
                "Node_Store: puct_edge extension detection query failed (%s); "
                "falling back to the client-side patch path",
                exc,
            )
            self._puct_edge_extension_available = False

        if self._puct_edge_extension_available:
            _LOG.info("Node_Store backup path: puct_edge extension (in-database)")
        else:
            _LOG.info(
                "Node_Store backup path: client-side SELECT ... FOR UPDATE fallback"
            )

    async def _schema_present(self, conn: asyncpg.Connection) -> bool:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "AND tablename = ANY($1::text[])",
            list(_EXPECTED_TABLES),
        )
        return len(rows) > 0

    async def _create_schema(self) -> None:
        """Create the schema from scratch (Requirement 2.4, 2.6, 2.7).

        Applies `dlshogi/book/sql/schema.sql` verbatim, then records the
        running schema version, Zobrist fingerprint, and the configured
        Root_Position, all inside one transaction and bounded to the 300 s
        deadline. A failure or timeout rolls the transaction back, so no
        partially created schema element is left behind (a plain
        multi-statement transaction's rollback satisfies that
        requirement directly), and re-raises as `SchemaCreationError`.
        """
        sql_text = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        root_sfen = str(self._config.root_position)
        root_key = position_key_from_sfen(root_sfen)
        root_key_hi = fold_u64_to_i64(root_key.hi)
        root_key_lo = fold_u64_to_i64(root_key.lo)
        fingerprint = fold_u64_to_i64(zobrist_fingerprint())

        async def _do_create() -> None:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(sql_text)
                    await conn.execute(
                        "INSERT INTO book_meta "
                        "(id, schema_version, zobrist_fingerprint, root_sfen, "
                        " root_key_hi, root_key_lo) "
                        "VALUES (1, $1, $2, $3, $4, $5)",
                        SCHEMA_VERSION,
                        fingerprint,
                        root_sfen,
                        root_key_hi,
                        root_key_lo,
                    )

        try:
            await asyncio.wait_for(_do_create(), timeout=_SCHEMA_CREATION_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise SchemaCreationError(
                self._config.pg_host, self._config.pg_port, self._config.pg_database,
                timed_out=True,
            ) from exc
        except asyncpg.PostgresError as exc:
            raise SchemaCreationError(
                self._config.pg_host, self._config.pg_port, self._config.pg_database,
                timed_out=False, cause=exc,
            ) from exc

    async def _check_version_and_fingerprint(self, conn: asyncpg.Connection) -> None:
        """Requirement 2.8: a mismatch is a hard, inert error.

        Reads only `book_meta`; creates and alters nothing, whether this
        raises or not. Called before `_repair_schema` so a mismatch is
        detected before any repair DDL runs.
        """
        row = await conn.fetchrow(
            "SELECT schema_version, zobrist_fingerprint FROM book_meta WHERE id = 1"
        )
        running_fingerprint = fold_u64_to_i64(zobrist_fingerprint())
        if row is None:
            raise SchemaVersionMismatchError(
                recorded_version=None,
                running_version=SCHEMA_VERSION,
                recorded_fingerprint=None,
                running_fingerprint=running_fingerprint,
            )
        recorded_version = row["schema_version"]
        recorded_fingerprint = row["zobrist_fingerprint"]
        if recorded_version != SCHEMA_VERSION or recorded_fingerprint != running_fingerprint:
            raise SchemaVersionMismatchError(
                recorded_version=recorded_version,
                running_version=SCHEMA_VERSION,
                recorded_fingerprint=recorded_fingerprint,
                running_fingerprint=running_fingerprint,
            )

    async def _repair_schema(self, conn: asyncpg.Connection) -> None:
        """Requirement 2.5: create only the absent schema elements.

        Only called once `_check_version_and_fingerprint` has confirmed
        the recorded version and fingerprint match the running ones.
        Every statement in `_REPAIR_DDL` is idempotent (`CREATE TABLE IF
        NOT EXISTS`, or an `ALTER COLUMN ... SET STORAGE` that is a no-op
        when already set), so existing rows and already-present elements
        are left untouched.
        """
        async with conn.transaction():
            for statement in _REPAIR_DDL:
                await conn.execute(statement)

    # -- Root_Position recording and read-back (Requirement 2.7) --------

    async def _load_root(self) -> None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT root_sfen, root_key_hi, root_key_lo FROM book_meta WHERE id = 1"
            )
        if row is not None:
            self._root_sfen = row["root_sfen"]
            self._root_key_hi = row["root_key_hi"]
            self._root_key_lo = row["root_key_lo"]

    async def get_root(self) -> tuple[str, int, int]:
        """Return the recorded ``(root_sfen, root_key_hi, root_key_lo)``.

        Always the value recorded at schema creation (Requirement 2.7:
        "return that recorded Root_Position unchanged on every subsequent
        start"), regardless of what the *current* run's configuration
        specifies for Root_Position; comparing the two against each other
        is the caller's job (`__main__.py`, a later task), not this
        method's.
        """
        if self._root_sfen is None:
            await self._load_root()
        if self._root_sfen is None:
            raise NodeStoreError("book_meta has no recorded Root_Position")
        assert self._root_key_hi is not None and self._root_key_lo is not None
        return self._root_sfen, self._root_key_hi, self._root_key_lo

    async def root_node_exists(self) -> bool:
        """Whether a ``book_node`` row matches the recorded root key.

        A small helper for the design's C11 startup check ("any node rows
        but no root row?"): this method only answers "does the root row
        exist", not the C11/X6 decision itself, which weighs this answer
        against whether *any* `book_node` row exists at all. That
        weighing belongs to a later task's caller (`__main__.py`).
        """
        _, root_key_hi, root_key_lo = await self.get_root()
        async with self.pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT 1 FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                root_key_hi,
                root_key_lo,
            )
        return row is not None

    async def any_node_rows(self) -> bool:
        """Whether ``book_node`` has any row at all (the other half of C11)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchval("SELECT 1 FROM book_node LIMIT 1")
        return row is not None

    # -- startup convenience ---------------------------------------------

    @classmethod
    async def connect_and_prepare(cls, config: Any, *, role: Role = "search") -> "NodeStore":
        """`connect()` followed by `ensure_schema()`, in one call."""
        store = cls(config, role=role)
        await store.connect()
        await store.ensure_schema()
        return store

    # -- the lost-connection suspension seam (Requirement 2.9) ----------
    #
    # `_run` is the one catch point every statement-issuing method (this
    # module's own, and every method task 7.6/8.x adds to this same
    # class) is meant to go through. Contract:
    #
    #   - `fn(pool)` is called with the live `asyncpg.Pool`; it should
    #     acquire its own connection (`async with pool.acquire() as
    #     conn:`) for the duration of a single statement, per design.md's
    #     "Descent tasks acquire a connection for the duration of a
    #     single statement, not for the duration of a descent."
    #   - On success, the result is returned unchanged.
    #   - On `asyncpg.exceptions.ConnectionDoesNotExistError` or
    #     `ConnectionResetError`, `suspended` is cleared (so every other
    #     caller that awaits `suspended.wait()` blocks), a reconnect loop
    #     runs using the same `retry_schedule`/attempt-timeout logic as
    #     `connect`, and:
    #       - if the reconnect succeeds within `Connection_Retry_Limit`
    #         retries, `suspended` is set again and the *original*
    #         exception is re-raised to the caller (the caller's own
    #         in-progress write is a failed write under Requirement 1.9,
    #         per Requirement 2.9 -- `_run` does not retry the caller's
    #         statement itself, only the connection);
    #       - if the reconnect is exhausted, `NodeStoreConnectionError` is
    #         raised and `suspended` is left cleared, since the caller
    #         who catches it is expected to exit the process (mirroring
    #         the startup case, Requirement 2.3).
    #   - Any other exception propagates unchanged; `suspended` is left
    #     untouched.
    #
    # Callers that only need "wait until not suspended, then issue one
    # statement" (the common case for future read/write methods) should
    # await `suspended.wait()` before calling `_run`, since `_run` itself
    # does not block new callers that arrive while a reconnect is already
    # in progress -- it lets `asyncio.Lock` do that serialization instead.

    _CONNECTION_LOST_EXCEPTIONS = (asyncpg.exceptions.ConnectionDoesNotExistError, ConnectionResetError)

    async def _run(self, fn: Callable[[asyncpg.Pool], Awaitable[_T]]) -> _T:
        """Run ``fn(self.pool)``, catching the two connection-lost exceptions.

        See the section comment above for the full contract. This is the
        one seam this module and every later read/write method built on
        top of it are meant to funnel statement execution through.
        """
        try:
            return await fn(self.pool)
        except self._CONNECTION_LOST_EXCEPTIONS as exc:
            await self._handle_connection_lost(exc)
            raise

    async def _handle_connection_lost(self, exc: BaseException) -> None:
        """Suspend, reconnect on the Requirement 2.2 schedule, then resume.

        Multiple concurrent callers may observe a lost connection at
        once; `_reconnect_lock` ensures only the first actually runs the
        reconnect loop, and the rest simply wait for it to finish (whether
        it succeeds or raises) before re-raising their own original
        exception.
        """
        async with self._reconnect_lock:
            if self.suspended.is_set():
                # Another caller already reconnected while we were
                # waiting for the lock; nothing left to do.
                return
            _LOG.warning(
                "Node_Store lost its PostgreSQL connection (%r); suspending "
                "reads and writes and attempting to reconnect", exc,
            )
            await self.close()
            try:
                await self.connect()
            except NodeStoreConnectionError:
                _LOG.error(
                    "Node_Store could not reconnect after exhausting the "
                    "retry schedule; leaving reads and writes suspended"
                )
                raise
            _LOG.info("Node_Store reconnected; resuming reads and writes")
            self.suspended.set()

    # -- read path (task 7.6) --------------------------------------------
    #
    # `get`/`get_many`/`get_many_terminal_eval` all route their PostgreSQL
    # access through `self._run`, per the seam contract documented above:
    # each acquires a connection for the duration of one statement only,
    # never held across the LRU bookkeeping or across multiple awaits.

    @property
    def cache_bytes(self) -> int:
        """Resident bytes the node LRU is currently tracking (Property 3)."""
        return self._cache.cache_bytes

    async def get(self, key: PositionKey) -> tuple[GetResult, Optional[BookNodeView]]:
        """Return the Book_Node at ``key``, or `GetResult.ABSENT`/`FAILED`.

        Consults, in order (design.md's Node_Store section):

        1. the node LRU -- a hit returns `(GetResult.FOUND, view)` directly,
           with no PostgreSQL round trip;
        2. the pending backup accumulator -- any uncommitted delta for
           ``key`` is overlaid onto whatever base view was found (LRU hit
           or PostgreSQL row), via `_apply_pending_to_view` (Requirement
           4.5);
        3. PostgreSQL -- a single-row primary-key `SELECT`. No row found
           means `(GetResult.ABSENT, None)`; this path issues no INSERT and
           no UPDATE, satisfying Requirement 1.4 by construction. A row
           found is decoded into a `BookNodeView`, inserted into the LRU,
           and returned as `(GetResult.FOUND, view)`.

        A reconnect-exhaustion or other statement-execution error re-raised
        by `self._run` is caught here and reported as
        `(GetResult.FAILED, None)` rather than propagated, since `get`'s
        signature carries an explicit `FAILED` case for exactly this
        outcome (unlike `get_many`/`get_many_terminal_eval`, whose return
        types have no equivalent slot -- see those methods' docstrings).
        """
        start = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None:
            view = _apply_pending_to_view(cached, self._pending_backup.get(key))
            self._read_latency.record(time.monotonic() - start)
            return GetResult.FOUND, view

        # Requirement 4.5: uncommitted deltas merged by `backup()` but not
        # yet flushed must be visible on read. Consulted here, between the
        # LRU check above and the PostgreSQL query below.
        pending = self._pending_backup.get(key)

        key_hi = fold_u64_to_i64(key.hi)
        key_lo = fold_u64_to_i64(key.lo)

        async def _do_get(pool: asyncpg.Pool) -> Optional[asyncpg.Record]:
            async with pool.acquire() as conn:
                return await conn.fetchrow(
                    "SELECT key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
                    "terminal, flags, eval_win_rate, prop_value, prop_best_move16, edges "
                    "FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                    key_hi,
                    key_lo,
                )

        try:
            row = await self._run(_do_get)
        except Exception as exc:  # noqa: BLE001 - see docstring: FAILED is a real return value
            _LOG.error("Node_Store.get(%r) failed: %r", key, exc)
            return GetResult.FAILED, None

        if row is None:
            self._read_latency.record(time.monotonic() - start)
            return GetResult.ABSENT, None

        view = _row_to_view(row)
        self._cache.put(key, view)
        view = _apply_pending_to_view(view, pending)
        self._read_latency.record(time.monotonic() - start)
        return GetResult.FOUND, view

    async def get_many(self, keys: Sequence[PositionKey]) -> list[Optional[BookNodeView]]:
        """Batched `get`, one statement over every key in ``keys``.

        Uses an ``unnest``-based two-parallel-array idiom rather than
        ``= ANY($1::key_pair[])`` (design.md's own sketch names a
        ``key_pair`` composite type, which schema.sql does not define and
        this task must not add): ``WHERE (key_hi, key_lo) IN (SELECT * FROM
        unnest($1::bigint[], $2::bigint[]))``, passing the folded key_hi and
        key_lo lists as two parallel `bigint[]` arrays. Needs no schema
        change and no custom type.

        Returns a list positionally parallel to ``keys`` (`None` where
        absent), built by looking each input key up in a
        ``(key_hi, key_lo) -> BookNodeView`` dict assembled from the query
        result -- the query itself makes no ordering promise. Every FOUND
        view is inserted into the node LRU, same as `get`.

        Unlike `get`, a caught statement-execution error is *not* converted
        to a sentinel here: `get_many`'s return type (`list[BookNodeView |
        None]`, design.md's sketch) has no FAILED-equivalent slot, so once
        `self._run`'s own reconnection seam has had its chance to reconnect
        (and re-raises the original exception either way), that exception
        is allowed to propagate to the caller rather than being swallowed
        into some ad hoc "all None" result that would be indistinguishable
        from "every key genuinely absent".
        """
        if not keys:
            return []

        key_his = [fold_u64_to_i64(k.hi) for k in keys]
        key_los = [fold_u64_to_i64(k.lo) for k in keys]

        async def _do_get_many(pool: asyncpg.Pool) -> list[asyncpg.Record]:
            async with pool.acquire() as conn:
                return await conn.fetch(
                    "SELECT key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
                    "terminal, flags, eval_win_rate, prop_value, prop_best_move16, edges "
                    "FROM book_node WHERE (key_hi, key_lo) IN "
                    "(SELECT * FROM unnest($1::bigint[], $2::bigint[]))",
                    key_his,
                    key_los,
                )

        rows = await self._run(_do_get_many)

        by_folded_key: dict[tuple[int, int], BookNodeView] = {}
        for row in rows:
            view = _row_to_view(row)
            by_folded_key[(row["key_hi"], row["key_lo"])] = view
            self._cache.put(view.key, view)

        result: list[Optional[BookNodeView]] = []
        for key, key_hi, key_lo in zip(keys, key_his, key_los):
            result.append(by_folded_key.get((key_hi, key_lo)))
        return result

    async def get_many_terminal_eval(
        self, keys: Sequence[PositionKey]
    ) -> list[Optional[tuple[Terminal, Optional[float]]]]:
        """The narrow `(terminal, eval_win_rate)` projection, batched.

        Design.md's stated optimization for below-threshold propagation
        children: "a below-threshold child transfers ~24 bytes instead of
        ~1.8 kB, and its edge blob never enters the node LRU". This method
        therefore never touches `self._cache`, in either direction --  no
        lookup, no population -- so a call here has zero effect on
        `cache_bytes`.

        Same batched `unnest` idiom as `get_many`; same propagate-rather-
        than-swallow error handling for the same reason (no FAILED-
        equivalent slot in this method's return type either).
        """
        if not keys:
            return []

        key_his = [fold_u64_to_i64(k.hi) for k in keys]
        key_los = [fold_u64_to_i64(k.lo) for k in keys]

        async def _do_get_many_terminal_eval(pool: asyncpg.Pool) -> list[asyncpg.Record]:
            async with pool.acquire() as conn:
                return await conn.fetch(
                    "SELECT key_hi, key_lo, terminal, eval_win_rate FROM book_node "
                    "WHERE (key_hi, key_lo) IN "
                    "(SELECT * FROM unnest($1::bigint[], $2::bigint[]))",
                    key_his,
                    key_los,
                )

        rows = await self._run(_do_get_many_terminal_eval)

        by_folded_key: dict[tuple[int, int], tuple[Terminal, Optional[float]]] = {
            (row["key_hi"], row["key_lo"]): (Terminal(row["terminal"]), row["eval_win_rate"])
            for row in rows
        }

        result: list[Optional[tuple[Terminal, Optional[float]]]] = []
        for key_hi, key_lo in zip(key_his, key_los):
            result.append(by_folded_key.get((key_hi, key_lo)))
        return result

    # -- write path (task 8.1): the atomic expansion write ---------------

    async def insert_expansion(self, w: ExpansionWrite) -> WriteResult:
        """The atomic `INSERT ... ON CONFLICT (key_hi, key_lo) DO NOTHING`.

        One row carries the node and every one of its edges, so "node plus
        every edge, or nothing" (Requirement 1.6, 10.3) is a property of
        this single row insert -- there is no second statement that could
        half-apply. An empty ``RETURNING`` means another task or another
        process's row already exists at this Position_Key; this method
        then reads that retained row's SFEN back and distinguishes:

        - the retained SFEN matches ``w.sfen`` -> `WriteOutcome.DUPLICATE`
          (Requirement 11.6: an ordinary transposition-merge loss, or a
          re-application of the identical expansion, Requirement 10.4's
          idempotence); the duplicate counter is incremented, the retained
          row's evaluation fields are left untouched (this method issues
          no UPDATE on this path at all), and neither caller is made to
          fail;
        - the retained SFEN differs from ``w.sfen`` -> `WriteOutcome.COLLISION`
          (Requirement 3.6): no Book_Edge referencing that row is created
          by this method's caller (this method creates none either way --
          edge creation is the caller's business, this method only writes
          the row), the retained row's fields are left unchanged, and both
          SFEN strings are reported via the returned `WriteResult`.

        A statement-execution error (reconnect exhaustion or otherwise,
        re-raised by `self._run`) is reported as `WriteOutcome.FAILED`
        rather than propagated, matching Requirement 1.9's write-failure
        result naming the Position_Key; no partial row is left behind,
        since the `INSERT` either commits whole or not at all.
        """
        start = time.monotonic()
        key_hi = fold_u64_to_i64(w.key.hi)
        key_lo = fold_u64_to_i64(w.key.lo)
        apery_key = fold_u64_to_i64(w.apery_key)
        edges_blob = packed_edge.encode_edges(w.edges)
        edge_count = len(w.edges)

        async def _do_insert(pool: asyncpg.Pool) -> Optional[asyncpg.Record]:
            async with pool.acquire() as conn:
                return await conn.fetchrow(
                    "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, terminal, "
                    "eval_win_rate, edge_count, edges) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                    "ON CONFLICT (key_hi, key_lo) DO NOTHING "
                    "RETURNING key_hi",
                    key_hi,
                    key_lo,
                    w.sfen,
                    apery_key,
                    w.terminal.value,
                    w.eval_win_rate,
                    edge_count,
                    edges_blob,
                )

        try:
            row = await self._run(_do_insert)
        except Exception as exc:  # noqa: BLE001 - FAILED is a real return value (Req 1.9)
            _LOG.error("Node_Store.insert_expansion(%r) failed: %r", w.key, exc)
            self._write_latency.record(time.monotonic() - start)
            return WriteResult(outcome=WriteOutcome.FAILED, key=w.key, error=exc)

        if row is not None:
            # This call's row is the one that survived the race.
            self._write_latency.record(time.monotonic() - start)
            return WriteResult(outcome=WriteOutcome.COMMITTED, key=w.key)

        # Empty RETURNING: another writer's row already exists. Read it
        # back to distinguish an ordinary duplicate from a Position_Key
        # collision (Requirement 3.6) -- ON CONFLICT DO NOTHING alone
        # cannot tell the two apart.
        async def _do_read_existing(pool: asyncpg.Pool) -> Optional[asyncpg.Record]:
            async with pool.acquire() as conn:
                return await conn.fetchrow(
                    "SELECT sfen FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                    key_hi,
                    key_lo,
                )

        try:
            existing = await self._run(_do_read_existing)
        except Exception as exc:  # noqa: BLE001 - see above
            _LOG.error(
                "Node_Store.insert_expansion(%r) failed reading the retained row: %r",
                w.key,
                exc,
            )
            self._write_latency.record(time.monotonic() - start)
            return WriteResult(outcome=WriteOutcome.FAILED, key=w.key, error=exc)

        self._write_latency.record(time.monotonic() - start)
        if existing is None:
            # Vanishingly unlikely (the row we just lost the race to was
            # deleted between the INSERT and this SELECT); treat as a
            # duplicate rather than crash the caller.
            self.duplicate_count += 1
            return WriteResult(outcome=WriteOutcome.DUPLICATE, key=w.key, existing_sfen=w.sfen)

        existing_sfen = existing["sfen"]
        if existing_sfen == w.sfen:
            self.duplicate_count += 1
            return WriteResult(
                outcome=WriteOutcome.DUPLICATE, key=w.key, existing_sfen=existing_sfen
            )

        self.collision_count += 1
        _LOG.warning(
            "Node_Store.insert_expansion: Position_Key collision at %r: "
            "existing sfen=%r, attempted sfen=%r",
            w.key,
            existing_sfen,
            w.sfen,
        )
        return WriteResult(
            outcome=WriteOutcome.COLLISION,
            key=w.key,
            existing_sfen=existing_sfen,
            attempted_sfen=w.sfen,
        )

    # -- write path (task 8.6): the coalescing backup accumulator --------

    def backup(self, deltas: Sequence[BackupDelta]) -> None:
        """Merge ``deltas`` into the in-process accumulator.

        Deliberately a plain ``def``, not ``async def`` (design.md: "no
        `await` sits between a descent's completion and its deltas
        becoming visible to `get`"). Increments are commutative and
        associative, so merging is exactly equivalent to applying them one
        at a time in any order (Requirement 11.5's confluence, Requirement
        11.8's no-lost-update invariant over the coalesced result).

        Each `BackupDelta` in ``deltas`` is merged into
        ``self._pending_backup[delta.key]``, creating that entry if
        absent: node visit/value deltas add, ``flags_or`` ORs, and
        per-``move16`` edge deltas add component-wise (creating the
        per-edge entry if this is the first delta seen for that edge).
        """
        for delta in deltas:
            existing = self._pending_backup.get(delta.key)
            if existing is None:
                # Copy the incoming delta's edge_deltas dict rather than
                # aliasing it, so a caller that reuses/mutates its own
                # BackupDelta after calling backup() cannot corrupt the
                # accumulator.
                self._pending_backup[delta.key] = BackupDelta(
                    key=delta.key,
                    node_visit_delta=delta.node_visit_delta,
                    node_value_delta=delta.node_value_delta,
                    flags_or=delta.flags_or,
                    edge_deltas=dict(delta.edge_deltas),
                )
                continue
            existing.node_visit_delta += delta.node_visit_delta
            existing.node_value_delta += delta.node_value_delta
            existing.flags_or |= delta.flags_or
            for move16, (visit_delta, value_delta) in delta.edge_deltas.items():
                prev_v, prev_w = existing.edge_deltas.get(move16, (0, 0.0))
                existing.edge_deltas[move16] = (prev_v + visit_delta, prev_w + value_delta)

    def _estimated_pending_bytes(self) -> int:
        """A rough byte estimate of the pending accumulator, for the
        Cache_Budget-pressure flush trigger below. Not used for anything
        that needs to be exact.
        """
        total = 0
        for delta in self._pending_backup.values():
            total += 64 + 24 * len(delta.edge_deltas)
        return total

    def cache_pressure_exceeded(self) -> bool:
        """Whether the pending accumulator's estimated size suggests an
        out-of-band flush is warranted (design.md's "on Cache_Budget
        pressure" flush trigger).

        There is no Search_Coordinator yet (task 13) to poll this on a
        schedule, so it is exposed as a plain predicate a future caller
        can check; the 200 ms interval flusher (`start_flusher`) already
        covers the steady-state case on its own. See the
        `_CACHE_PRESSURE_FRACTION` module constant for the threshold and
        the judgement call it documents.
        """
        budget = int(self._config.cache_budget)
        return self._estimated_pending_bytes() > _CACHE_PRESSURE_FRACTION * budget

    async def flush(self) -> None:
        """Apply every pending accumulator delta to PostgreSQL, then clear it.

        Snapshots and replaces ``self._pending_backup`` in one synchronous
        step (no `await` on that line), so deltas merged by a concurrent
        `backup()` call *during* this flush accumulate into the new,
        now-current dict rather than being lost from the one being
        written (design.md: "that swap is the one line of the flusher
        that must not contain an `await`").

        Applies the snapshot as one `UPDATE` per node inside one
        transaction, dispatching each node to one of two backend methods
        chosen once at startup (`self._puct_edge_extension_available`,
        set by `_detect_puct_edge_extension`): `_apply_backup_row_extension`
        (the in-database `puct_edge_backup` UPDATE) when the `puct_edge`
        C extension of task 9.1 is detected, otherwise
        `_apply_backup_row_fallback` (`SELECT edges ... FOR UPDATE`,
        `packed_edge.patch_edges`, `UPDATE ... SET edges = $n`). Both
        methods take PostgreSQL's row-level exclusive lock around the
        read-modify-write, so Requirement 11.8's no-lost-update guarantee
        holds under either backend.

        A node present in the snapshot but absent from `book_node` (its
        expansion write has not yet committed, or was lost) is skipped
        rather than raising: the deltas for that key remain lost only in
        the sense that a descent that walked through an unexpanded node is
        not possible by construction (the node was created by
        `insert_expansion` before any descent could select an edge under
        it), so this is a defensive no-op, not an expected path.
        """
        snapshot = self._pending_backup
        self._pending_backup = {}

        if not snapshot:
            return

        start = time.monotonic()

        apply_row = (
            self._apply_backup_row_extension
            if self._puct_edge_extension_available
            else self._apply_backup_row_fallback
        )

        async def _do_flush(pool: asyncpg.Pool) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for delta in snapshot.values():
                        await apply_row(conn, delta)

        try:
            await self._run(_do_flush)
        except Exception:
            # A failed flush re-merges the snapshot back into the live
            # accumulator (rather than silently dropping it), so the next
            # flush attempt retries the same deltas; get() continues to
            # see them via the pending-accumulator overlay in the
            # meantime. This is consistent with Requirement 2.9: a
            # connection loss suspends operations and resumes them, it
            # does not discard already-backed-up-in-memory deltas.
            self.backup(list(snapshot.values()))
            self._write_latency.record(time.monotonic() - start)
            raise
        self._write_latency.record(time.monotonic() - start)

    async def _apply_backup_row_fallback(
        self, conn: asyncpg.Connection, delta: BackupDelta
    ) -> None:
        """Apply one node's accumulated deltas, via the client-side fallback patch.

        `SELECT edges FROM book_node WHERE key_hi = $1 AND key_lo = $2 FOR
        UPDATE`, patch client-side with `packed_edge.patch_edges`, then one
        `UPDATE ... SET visit_count = visit_count + $, value_sum =
        value_sum + $, flags = flags | $, edges = $n`, all within the
        caller's transaction. The `FOR UPDATE` row lock is what gives the
        same no-lost-update guarantee the in-database `puct_edge_backup`
        path (`_apply_backup_row_extension`) gives, at the cost of one
        extra round trip per node per flush (design.md's "The backup
        write path", paragraph 2's fallback note).

        Used by `flush()` whenever `self._puct_edge_extension_available`
        is `False`, i.e. whenever the `puct_edge` C extension of task 9.1
        was not detected at startup (`_detect_puct_edge_extension`).
        """
        key_hi = fold_u64_to_i64(delta.key.hi)
        key_lo = fold_u64_to_i64(delta.key.lo)

        row = await conn.fetchrow(
            "SELECT edges FROM book_node WHERE key_hi = $1 AND key_lo = $2 FOR UPDATE",
            key_hi,
            key_lo,
        )
        if row is None:
            _LOG.warning(
                "Node_Store.flush: no book_node row for %r; dropping %d pending "
                "delta(s) for an unexpanded node",
                delta.key,
                1 + len(delta.edge_deltas),
            )
            return

        edges_blob = bytes(row["edges"])
        if delta.edge_deltas:
            move16s = list(delta.edge_deltas.keys())
            visit_deltas = [delta.edge_deltas[m][0] for m in move16s]
            value_deltas = [delta.edge_deltas[m][1] for m in move16s]
            edges_blob = packed_edge.patch_edges(edges_blob, move16s, visit_deltas, value_deltas)

        await conn.execute(
            "UPDATE book_node SET visit_count = visit_count + $3, "
            "value_sum = value_sum + $4, flags = flags | $5, edges = $6 "
            "WHERE key_hi = $1 AND key_lo = $2",
            key_hi,
            key_lo,
            delta.node_visit_delta,
            delta.node_value_delta,
            delta.flags_or,
            edges_blob,
        )
        # The row this flush just wrote may be cached; invalidate so the
        # next get() re-reads the committed row rather than serving a
        # stale cached copy now that the pending overlay for this key has
        # been cleared (the snapshot dict backup() built this delta from
        # was already swapped out before flush() started applying it).
        self._cache.invalidate(delta.key)

    async def _apply_backup_row_extension(
        self, conn: asyncpg.Connection, delta: BackupDelta
    ) -> None:
        """Apply one node's accumulated deltas via the in-database `puct_edge_backup`.

        Issues design.md's "The backup write path" single-statement form:

            UPDATE book_node
               SET visit_count = visit_count + $3,
                   value_sum   = value_sum   + $4,
                   flags       = flags | $5,
                   edges       = puct_edge_backup(edges, $6::int2[], $7::int4[], $8::float8[])
             WHERE key_hi = $1 AND key_lo = $2;

        Unlike `_apply_backup_row_fallback`, this never reads `edges`
        first: the patch happens inside the UPDATE expression itself, so
        there is no `SELECT ... FOR UPDATE` round trip and no client-side
        byte patching at all. PostgreSQL's row-level exclusive lock taken
        by the UPDATE is what gives the same no-lost-update guarantee the
        fallback gets from `FOR UPDATE`.

        When `delta.edge_deltas` is empty, `edges` is left as `edges`
        (unchanged) rather than calling `puct_edge_backup` with
        zero-length arrays -- simpler, and avoids a function call whose
        only effect would be a copy, for the node-only-delta case that
        `set_propagation`-free backups without edge deltas never actually
        produce today but that a defensive caller could.

        Used by `flush()` whenever `self._puct_edge_extension_available`
        is `True`. Like the fallback, matches "no book_node row" (an
        unexpanded node) by an empty `UPDATE ... WHERE` match count and
        drops the delta with a warning rather than raising, since
        `asyncpg`'s `execute()` on an UPDATE does not raise for a
        no-op match; the row count in the returned status string is
        checked instead.
        """
        key_hi = fold_u64_to_i64(delta.key.hi)
        key_lo = fold_u64_to_i64(delta.key.lo)

        if delta.edge_deltas:
            move16s = list(delta.edge_deltas.keys())
            visit_deltas = [delta.edge_deltas[m][0] for m in move16s]
            value_deltas = [delta.edge_deltas[m][1] for m in move16s]
            status = await conn.execute(
                "UPDATE book_node SET visit_count = visit_count + $3, "
                "value_sum = value_sum + $4, flags = flags | $5, "
                "edges = puct_edge_backup(edges, $6::int2[], $7::int4[], $8::float8[]) "
                "WHERE key_hi = $1 AND key_lo = $2",
                key_hi,
                key_lo,
                delta.node_visit_delta,
                delta.node_value_delta,
                delta.flags_or,
                move16s,
                visit_deltas,
                value_deltas,
            )
        else:
            status = await conn.execute(
                "UPDATE book_node SET visit_count = visit_count + $3, "
                "value_sum = value_sum + $4, flags = flags | $5 "
                "WHERE key_hi = $1 AND key_lo = $2",
                key_hi,
                key_lo,
                delta.node_visit_delta,
                delta.node_value_delta,
                delta.flags_or,
            )

        if status.rsplit(" ", 1)[-1] == "0":
            _LOG.warning(
                "Node_Store.flush: no book_node row for %r; dropping %d pending "
                "delta(s) for an unexpanded node",
                delta.key,
                1 + len(delta.edge_deltas),
            )
            return

        # Same invalidation as the fallback path (see its docstring):
        # a stale cached view must not be served after this flush.
        self._cache.invalidate(delta.key)

    # -- flusher lifecycle (task 8.6): 200 ms interval, and on stop -------

    def start_flusher(self) -> None:
        """Start the background flusher coroutine (idempotent).

        Wakes every `_FLUSH_INTERVAL_S` (200 ms, design.md's
        `flush_interval` default) and calls `flush()`. There is no
        Search_Coordinator yet (task 13) to start this automatically, so a
        caller (a test, or a later task's supervisor) must call this
        explicitly; `NodeStore` does not start it on `connect()` or
        `ensure_schema()`, since a `propagate`/`export`-role instance has
        no use for it at all.
        """
        if self._flusher_task is not None and not self._flusher_task.done():
            return
        self._flusher_stop_event = asyncio.Event()
        self._flusher_task = asyncio.ensure_future(
            self._flusher_loop(self._flusher_stop_event)
        )

    async def stop_flusher(self) -> None:
        """Stop the background flusher, flushing any remaining deltas first.

        Requirement 10.5's "flush pending writes ... and exit": this
        method's own final `flush()` call is what a caller's stop-request
        handling is meant to await before considering the accumulator
        drained.
        """
        if self._flusher_stop_event is not None:
            self._flusher_stop_event.set()
        if self._flusher_task is not None:
            await self._flusher_task
            self._flusher_task = None
        await self.flush()

    async def _flusher_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_FLUSH_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
            try:
                await self.flush()
            except Exception:  # noqa: BLE001 - the flusher must keep running
                _LOG.exception("Node_Store background flusher: flush() failed")

    # -- write path (task 8.8): propagation writes ------------------------

    async def set_propagation(self, w: Sequence[PropagationWrite]) -> None:
        """Write `prop_value`/`prop_best_move16`/`prop_epoch` for every entry of ``w``.

        These are absolute values, not deltas (unlike `backup`'s
        increment semantics), so this issues a direct `UPDATE` per node --
        batched as one `executemany` inside one transaction -- rather than
        routing through the visit/value accumulator (see this module's
        `PropagationWrite` docstring). Propagation writes run under the
        `role="propagate"` session's `synchronous_commit = on`, set at the
        connection level by `NodeStore.__init__`/`connect`, not by this
        method.
        """
        if not w:
            return

        start = time.monotonic()
        args = [
            (
                fold_u64_to_i64(entry.key.hi),
                fold_u64_to_i64(entry.key.lo),
                entry.prop_value,
                entry.prop_best_move16,
                entry.prop_epoch,
            )
            for entry in w
        ]

        async def _do_set_propagation(pool: asyncpg.Pool) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        "UPDATE book_node SET prop_value = $3, prop_best_move16 = $4, "
                        "prop_epoch = $5 WHERE key_hi = $1 AND key_lo = $2",
                        args,
                    )

        await self._run(_do_set_propagation)
        for entry in w:
            self._cache.invalidate(entry.key)
        self._write_latency.record(time.monotonic() - start)

    async def next_propagation_seq(self) -> int:
        """Advance and return `book_meta.propagation_seq` (the next pass id).

        A propagation pass calls this once at its own start to obtain the
        `prop_epoch` value it will write via `set_propagation`
        (design.md's "Value propagation..." section: "Memo in the database
        via `prop_epoch = pass_id` from `book_meta.propagation_seq`").
        """

        async def _do_bump(pool: asyncpg.Pool) -> int:
            async with pool.acquire() as conn:
                return await conn.fetchval(
                    "UPDATE book_meta SET propagation_seq = propagation_seq + 1 "
                    "WHERE id = 1 RETURNING propagation_seq"
                )

        return await self._run(_do_bump)

    async def mark_propagation_done(self, pass_id: int) -> None:
        """Record that propagation pass ``pass_id`` has completed.

        Sets `book_meta.propagation_done_seq = pass_id`, which is what
        lets a later export (Requirement 12.9's staleness check,
        `prop_epoch <> propagation_done_seq`) tell a fully-propagated
        graph from one with a partial or superseded pass.
        """

        async def _do_mark(pool: asyncpg.Pool) -> None:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE book_meta SET propagation_done_seq = $1 WHERE id = 1", pass_id
                )

        await self._run(_do_mark)

    async def bump_search_write_seq(self) -> int:
        """Advance and return `book_meta.search_write_seq`.

        Incremented by the search side whenever it commits a write that
        would make a previously completed propagation pass stale (an
        expansion or a backup); a later export's staleness warning
        (Requirement 12.9's "or `search_write_seq > propagation_done_seq`")
        compares this against `propagation_done_seq`. No caller in this
        task's scope invokes this yet (task 13's Search_Coordinator will);
        it is provided here purely as the `book_meta` maintenance helper
        design.md's schema implies.
        """

        async def _do_bump(pool: asyncpg.Pool) -> int:
            async with pool.acquire() as conn:
                return await conn.fetchval(
                    "UPDATE book_meta SET search_write_seq = search_write_seq + 1 "
                    "WHERE id = 1 RETURNING search_write_seq"
                )

        return await self._run(_do_bump)

    # -- write path (task 8.8): stats --------------------------------------

    def stats(self) -> Stats:
        """Read/write latency histograms and cache byte accounting."""
        return Stats(
            read_latency_mean_s=self._read_latency.mean(),
            read_latency_p95_s=self._read_latency.p95(),
            read_count=self._read_latency.count,
            write_latency_mean_s=self._write_latency.mean(),
            write_latency_p95_s=self._write_latency.p95(),
            write_count=self._write_latency.count,
            cache_bytes=self.cache_bytes,
            cache_budget_bytes=int(self._config.cache_budget),
            duplicate_count=self.duplicate_count,
            collision_count=self.collision_count,
            cache_release_events=self.cache_release_events,
        )

    # -- write path (task 8.8): RSS sampler and cache release -------------

    def set_evaluator_baseline_rss(self, rss_bytes: int) -> None:
        """Record the resident memory at completion of Evaluator initialization.

        Requirement 15.4's bound is `Cache_Budget + evaluator RSS + 512
        MiB`. No Evaluator exists yet (task 10), so this value defaults to
        0 and is meant to be set once, right after a future Evaluator
        finishes initializing, by whatever code drives that
        initialization (task 13's Search_Coordinator, or a test).
        """
        self._evaluator_baseline_rss_bytes = int(rss_bytes)

    def set_rss_reader(self, reader: Callable[[], int]) -> None:
        """Replace the RSS-reading callable (a test seam).

        Defaults to `_default_rss_bytes` (real `/proc/self/status` /
        `resource.getrusage` reading). A test injects a callable
        returning a fixed value here to exercise the eviction path
        without needing the process's actual RSS to cross the bound.
        """
        self._rss_reader = reader

    def set_on_cache_release(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a callback invoked once per cache-release event.

        `report.py`'s real Progress_Reporter does not exist yet (task
        16.1); `cache_release_events` (a plain counter) is always
        incremented, and this callback is an additional, optional hook a
        caller (a test, or a future Progress_Reporter) can observe events
        through without polling the counter.
        """
        self._on_cache_release = callback

    def rss_bound_bytes(self) -> int:
        """Requirement 15.4's bound: Cache_Budget + evaluator RSS + 512 MiB."""
        return (
            int(self._config.cache_budget)
            + self._evaluator_baseline_rss_bytes
            + _RSS_BOUND_OVERHEAD_BYTES
        )

    def check_rss_and_release(self) -> bool:
        """Take one RSS sample; evict from the node LRU if it exceeds the bound.

        Requirement 15.5: releases cached Book_Node/Book_Edge records
        (never anything already written to PostgreSQL) until the *cache*
        no longer accounts for the excess, reports a cache-release event,
        and returns whether a release happened. One sample only shrinks
        the in-process cache; it does not itself guarantee the *next* RSS
        sample will be within bound (RSS includes far more than this
        cache), which is why the RSS sampler loop below keeps sampling
        rather than treating one release as sufficient -- matching
        Requirement 15.5's own wording, "until a subsequent sample is at
        or below that bound".
        """
        rss = self._rss_reader()
        bound = self.rss_bound_bytes()
        if rss <= bound:
            return False
        excess = rss - bound
        self._cache.shrink_by(excess)
        self.cache_release_events += 1
        if self._on_cache_release is not None:
            self._on_cache_release()
        return True

    def start_rss_sampler(self, interval_s: float = _RSS_SAMPLE_INTERVAL_S) -> None:
        """Start the background RSS sampler (idempotent).

        Samples at most every `_RSS_SAMPLE_INTERVAL_S` (10 s, Requirement
        15.4's "at intervals of at most 10 seconds"); a caller may pass a
        shorter ``interval_s`` for testing. Like `start_flusher`, there is
        no Search_Coordinator yet to start this automatically, so a caller
        must call this explicitly.
        """
        if self._rss_task is not None and not self._rss_task.done():
            return
        self._rss_stop_event = asyncio.Event()
        self._rss_task = asyncio.ensure_future(
            self._rss_sampler_loop(self._rss_stop_event, interval_s)
        )

    async def stop_rss_sampler(self) -> None:
        """Stop the background RSS sampler."""
        if self._rss_stop_event is not None:
            self._rss_stop_event.set()
        if self._rss_task is not None:
            await self._rss_task
            self._rss_task = None

    async def _rss_sampler_loop(self, stop_event: asyncio.Event, interval_s: float) -> None:
        while not stop_event.is_set():
            try:
                self.check_rss_and_release()
            except Exception:  # noqa: BLE001 - the sampler must keep running
                _LOG.exception("Node_Store RSS sampler: check_rss_and_release() failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass

    # -- write path (task 8.10): in-flight claim mirror -------------------

    async def upsert_in_flight_claims(
        self, claims: Sequence[tuple[PositionKey, int, int, datetime]]
    ) -> None:
        """Upsert rows into `in_flight_claim` for the given claims.

        ``claims`` is a sequence of ``(key, process_id, worker_id,
        claimed_at)`` tuples -- the design's "diagnostic mirror" of the
        per-process In_Flight_Set (design.md's In_Flight_Set section):
        "once per Report_Interval it upserts into a logged
        `in_flight_claim` table the claims that have been held longer than
        one Report_Interval". Selecting *which* claims qualify (held
        longer than one Report_Interval) and running that once-per-
        Report_Interval loop is the Search_Coordinator's job (task 13);
        this method is only the storage primitive it calls.
        """
        if not claims:
            return

        args = [
            (fold_u64_to_i64(key.hi), fold_u64_to_i64(key.lo), process_id, worker_id, claimed_at)
            for key, process_id, worker_id, claimed_at in claims
        ]

        async def _do_upsert(pool: asyncpg.Pool) -> None:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        "INSERT INTO in_flight_claim (key_hi, key_lo, process_id, worker_id, "
                        "claimed_at) VALUES ($1, $2, $3, $4, $5) "
                        "ON CONFLICT (key_hi, key_lo) DO UPDATE SET "
                        "process_id = EXCLUDED.process_id, worker_id = EXCLUDED.worker_id, "
                        "claimed_at = EXCLUDED.claimed_at",
                        args,
                    )

        await self._run(_do_upsert)

    async def delete_in_flight_claims(self, keys: Sequence[PositionKey]) -> None:
        """Delete `in_flight_claim` rows for claims that have been released.

        The other half of the diagnostic mirror: once a claim is released
        (the In_Flight_Set entry it mirrors is gone), its mirror row is no
        longer meaningful and is deleted.
        """
        if not keys:
            return

        key_his = [fold_u64_to_i64(k.hi) for k in keys]
        key_los = [fold_u64_to_i64(k.lo) for k in keys]

        async def _do_delete(pool: asyncpg.Pool) -> None:
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM in_flight_claim WHERE (key_hi, key_lo) IN "
                    "(SELECT * FROM unnest($1::bigint[], $2::bigint[]))",
                    key_his,
                    key_los,
                )

        await self._run(_do_delete)

    async def clear_in_flight_claims_at_startup(self) -> int:
        """`SELECT count(*)` then `TRUNCATE in_flight_claim` (Requirement 10.2).

        Reads no `book_node` row and touches no visit count or value sum:
        the count and the truncate both target `in_flight_claim` alone.
        Returns the pre-clear row count, which is what Requirement 10.2's
        "report the number of cleared entries" needs.

        This is deliberately *not* called from `connect_and_prepare` or
        `ensure_schema`: Requirement 10.2's clearing is conceptually a
        Search_Coordinator startup step (it is meaningful only in relation
        to the in-process In_Flight_Set that a `search`-role run
        maintains), not schema management, and a `propagate`/`export`-role
        instance has no In_Flight_Set to reconcile against at all. A
        caller on the search startup path (`__main__.py`, task 17.1) is
        expected to call this explicitly, once, after `ensure_schema()`
        and before the first Selection_Descent.
        """

        async def _do_clear(pool: asyncpg.Pool) -> int:
            async with pool.acquire() as conn:
                count = await conn.fetchval("SELECT count(*) FROM in_flight_claim")
                await conn.execute("TRUNCATE in_flight_claim")
                return count

        return await self._run(_do_clear)


__all__ = [
    "SCHEMA_VERSION",
    "Role",
    "NodeStore",
    "NodeStoreError",
    "NodeStoreConnectionError",
    "SchemaCreationError",
    "SchemaVersionMismatchError",
    "retry_schedule",
    "BookNodeView",
    "GetResult",
    "Terminal",
    "ExpansionWrite",
    "WriteOutcome",
    "WriteResult",
    "BackupDelta",
    "PropagationWrite",
    "Stats",
]
