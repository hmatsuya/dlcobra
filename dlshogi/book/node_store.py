"""Node_Store: connection, schema management, and reconnection.

This module implements the *connection and schema* half of the Node_Store
component (design.md's "Node_Store" section): the asyncpg pool, startup
schema creation/versioning/repair, Root_Position recording and read-back,
the connection retry schedule, and the lost-connection suspension seam.

**Scope.** The read path (`get`, `get_many`, `get_many_terminal_eval`, the
node LRU) is added by task 7.6; the write path (`insert_expansion`, the
backup accumulator, `flush`, `set_propagation`) is added by task 8.x. Both
later tasks extend *this* class rather than replacing it, and both are
meant to build on the reconnection seam (`_run`) this module exposes so
they inherit lost-connection handling for free. This module's own scope --
pool creation, schema management, versioning, retry, reconnection
suspension, `synchronous_commit` -- is complete on its own and does not
depend on either later task.

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
import enum
import logging
from collections import OrderedDict
from dataclasses import dataclass
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
        # Pending backup accumulator (task 8.6, not yet implemented): once
        # added, `get` will consult this dict (keyed by PositionKey, holding
        # uncommitted visit/value deltas) between the LRU check and the
        # PostgreSQL query, so uncommitted increments are visible on read
        # (design.md's Node_Store section; Requirement 4.5). Declared here,
        # empty, purely as task 8.6's extension point.
        self._pending_backup: dict[PositionKey, Any] = {}

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
        2. the pending backup accumulator -- not yet implemented (task 8.6
           adds it); see the `# TODO(task 8.6)` marker below for exactly
           where that lookup belongs;
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
        cached = self._cache.get(key)
        if cached is not None:
            return GetResult.FOUND, cached

        # TODO(task 8.6): consult self._pending_backup here, between the
        # LRU check above and the PostgreSQL query below, once the backup
        # accumulator exists, so uncommitted increments are visible on
        # read (Requirement 4.5).

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
            return GetResult.ABSENT, None

        view = _row_to_view(row)
        self._cache.put(key, view)
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

    # -- write path stubs, filled in by task 8.x --------------------------
    #
    # The following methods are intentionally unimplemented here. Their
    # signatures are given (from design.md's Node_Store component sketch)
    # so the intended public class shape is visible early; task 8.x fills
    # in the write path on this same class.

    async def insert_expansion(self, w):  # noqa: ANN001 - ExpansionWrite, task 8.1
        """TODO(task 8.1): the atomic `INSERT ... ON CONFLICT DO NOTHING`."""
        raise NotImplementedError("Node_Store.insert_expansion is implemented by task 8.1")

    def backup(self, deltas):  # noqa: ANN001 - Sequence[BackupDelta], task 8.6
        """TODO(task 8.6): merge deltas into the in-process accumulator."""
        raise NotImplementedError("Node_Store.backup is implemented by task 8.6")

    async def flush(self) -> None:
        """TODO(task 8.6): flush the accumulator to PostgreSQL."""
        raise NotImplementedError("Node_Store.flush is implemented by task 8.6")

    async def set_propagation(self, w):  # noqa: ANN001 - Sequence[PropagationWrite], task 8.8
        """TODO(task 8.8): write `prop_value`/`prop_best_move16`/`prop_epoch`."""
        raise NotImplementedError("Node_Store.set_propagation is implemented by task 8.8")

    def stats(self):
        """TODO(task 8.8): read/write latency histograms and cache accounting."""
        raise NotImplementedError("Node_Store.stats is implemented by task 8.8")


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
]
