"""Multi-process integration test (task 19.3).

Two *real OS processes* run concurrent expansions and counter backups
against one shared PostgreSQL database over overlapping key regions, and a
separate check measures the Terashock_Index lookup latency over an index of
at least 10^7 entries. Between them these validate:

- **Requirement 11.1** -- multiple Search_Workers expanding one shared
  Book_Graph served by one Node_Store instance. Here "worker" is realised
  as a whole OS process, which is the strongest form of the requirement:
  design.md's *In_Flight_Set* section makes the In_Flight_Set per process,
  so the only thing that keeps two processes from creating two rows for the
  same Position_Key is the database, not any shared in-memory set.
- **Requirement 11.6** -- when two workers try to create a Book_Node with
  the same Position_Key, exactly one row is retained (the first committed),
  its evaluation fields are not a mixture, the loser gets a
  duplicate-detected result and continues, and the duplicate count is
  reported. We assert exactly one row per Position_Key and that the total
  duplicate count reported across the two processes is consistent with the
  observed overlap of their key regions.
- **Requirement 11.8** (the "no lost counter updates" half of 11.1) -- when
  two processes update the visit count / value sum of the same Book_Node
  and the same Book_Edge concurrently, no increment is lost: the final
  stored counters equal the exact sum of every increment both processes
  issued. This is guaranteed by PostgreSQL's row lock through the
  ``SELECT ... FOR UPDATE`` client-side fallback patch path (task 8.6) or
  the ``puct_edge`` extension path (task 9.2); this test exercises whichever
  one ``NodeStore`` selects at startup.
- **Requirement 7.1** -- the Terashock_Index returns the Terashock_Entry for
  a Position_Key within 1 ms at the 95th percentile for a Terashock_Book of
  at least 10,000,000 entries.

Why separate OS processes rather than asyncio tasks. This mirrors
design.md's Property 34 parametrisation (task 8.5): "One parametrisation of
this test runs the two writers in *separate OS processes* rather than
separate tasks, because that is the configuration the per-GPU process model
actually produces and the one Requirement 11.6 has to cover for the
per-process In_Flight_Set to be safe." Two asyncio tasks in one process
share one In_Flight_Set and one connection pool; two processes share
nothing but the database, so the duplicate resolution and the row lock are
tested where they actually have to hold.

Hermeticity. Every child process is spawned from a ``multiprocessing``
*spawn* context (a fresh interpreter, a fresh cppshogi/Zobrist init, its own
event loop, its own asyncpg pool, its own In_Flight_Set) using this
interpreter -- the venv interpreter -- as the executable. Every test reaps
and, on timeout, kills every child in a ``finally`` block, and manages its
own scratch state by truncating the tables it uses at the start of the test
rather than relying on cross-test ordering. Connection settings are passed
to the children as primitives through the process arguments, so a child
never has to import this test module's fixtures.

All tests here carry the ``db`` marker and are optional (task 19.3 is a
``*`` task); they skip gracefully when no PostgreSQL server is reachable via
``pg_scratch_database``.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from dlshogi.book.config import BookConfig
from dlshogi.book.keys import (
    PositionKey,
    fold_u64_to_i64,
    position_key_from_sfen,
)
from dlshogi.book.node_store import (
    BackupDelta,
    ExpansionWrite,
    NodeStore,
    Terminal,
    WriteOutcome,
)
from dlshogi.book.terashock import TerashockIndex

pytestmark = pytest.mark.db

_INITIAL_POSITION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

# A spawn context: a genuinely separate OS process with a fresh interpreter,
# not a fork that would inherit this process's asyncio loop, cppshogi state,
# and open PostgreSQL sockets. sys.executable under the venv is the venv
# interpreter, which is the one that can import cppshogi/asyncpg at all.
_MP = mp.get_context("spawn")

# A hard per-process wall-clock budget. A child that overruns it is killed
# in the test's finally block so a hung child never wedges the suite.
_CHILD_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# Connection settings passed to children as plain primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConnParams:
    """The subset of connection settings a child needs, as picklable primitives.

    Deliberately a small plain dataclass rather than the conftest
    ``PgConnectInfo`` or a ``BookConfig``: it is passed across the spawn
    boundary by pickle, so it must not carry anything a fresh interpreter
    cannot reconstruct, and a child must be able to build its own
    ``BookConfig`` from it without importing this test module's fixtures.
    """

    host: str
    port: int
    user: str
    password: Optional[str]
    database: str

    @classmethod
    def from_pg_info(cls, pg_info) -> "_ConnParams":
        return cls(
            host=pg_info.host,
            port=pg_info.port,
            user=pg_info.user,
            password=pg_info.password,
            database=pg_info.database,
        )


def _build_config(conn: _ConnParams, **overrides) -> BookConfig:
    """Build a `BookConfig` from ``conn`` with in-range placeholder values.

    Mirrors ``tests/book/test_node_store.py``'s ``_make_config``: every
    non-connection field is an arbitrary but in-range value, since the
    multi-process test does not exercise ``validate_config``. Defined at
    module scope (not inside a fixture) so a spawned child can call it too.
    """
    kwargs: dict = dict(
        pg_host=conn.host,
        pg_port=str(conn.port),
        pg_user=conn.user,
        pg_password=conn.password or "",
        pg_database=conn.database,
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
        root_position=_INITIAL_POSITION_SFEN,
        report_interval=60,
        throughput_floor=50,
        throughput_grace_period=600,
    )
    kwargs.update(overrides)
    return BookConfig(**kwargs)


# ---------------------------------------------------------------------------
# Synthetic position generation (deterministic, board-free)
# ---------------------------------------------------------------------------
#
# The multi-process write test needs many *distinct* Position_Keys with a
# controllable overlap between the two processes, plus a valid SFEN and a
# short packed-edge list per key. It does NOT need the keys to be the real
# Position_Keys of real Board_States: insert_expansion stores whatever key,
# sfen, apery_key and packed edges it is handed, and Requirement 11.6 is a
# statement about equal *keys*, not about board legality. So the keys are
# generated deterministically and the sfen is a synthetic marker unique to
# the key. Requirement 11.6's "retain the first, do not mix" is exercised by
# giving the two processes *different* eval_win_rate values for the same key.


def _synthetic_sfen(i: int) -> str:
    """A synthetic, per-key-unique SFEN marker string."""
    return f"synthetic-position-{i}"


def _synthetic_key(i: int) -> PositionKey:
    """A deterministic distinct Position_Key for index ``i``.

    Uses a fixed multiplicative hash so both processes agree on the key for
    a given index without any shared state, and so distinct indices give
    distinct keys within the ranges this test uses.
    """
    hi = (i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    lo = ((i + 1) * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
    return PositionKey(hi, lo)


def _one_edge(move16: int) -> dict:
    """One PACKED_EDGE-shaped edge dict with zeroed counters."""
    return {
        "move16": move16,
        "prior_q16": 0,
        "ts_depth": 0,
        "flags": 0,
        "ts_eval": 0,
        "visit_count": 0,
        "value_sum": 0.0,
    }


# ---------------------------------------------------------------------------
# The child-process entry points (must be top-level for spawn to pickle them)
# ---------------------------------------------------------------------------


def _writer_child(
    conn_dict: dict,
    start_index: int,
    count: int,
    eval_win_rate: float,
    result_queue,
) -> None:
    """Child process body: expand ``count`` keys starting at ``start_index``.

    Runs its own event loop, its own ``NodeStore`` (its own asyncpg pool and
    its own In_Flight_Set), and issues one ``insert_expansion`` per key in
    ``[start_index, start_index + count)``. It reports back, via
    ``result_queue``, how many of its own expansions COMMITTED and how many
    came back DUPLICATE, and its ``NodeStore.duplicate_count``. Two children
    with overlapping ``[start_index, start_index + count)`` ranges race on
    the overlap; the database keeps exactly one row per key and the loser of
    each race sees a DUPLICATE (Requirement 11.6).
    """
    conn = _ConnParams(**conn_dict)

    async def _run() -> tuple[int, int, int]:
        config = _build_config(conn, connection_retry_limit=0)
        store = await NodeStore.connect_and_prepare(config)
        committed = 0
        duplicate = 0
        try:
            for i in range(start_index, start_index + count):
                key = _synthetic_key(i)
                # Each process writes a distinct eval_win_rate, so
                # Requirement 11.6's "evaluation fields are not a mixture"
                # is checkable afterwards: the surviving row's eval_win_rate
                # must equal exactly one process's value.
                write = ExpansionWrite(
                    key=key,
                    sfen=_synthetic_sfen(i),
                    apery_key=i,
                    terminal=Terminal.NONE,
                    eval_win_rate=eval_win_rate,
                    edges=[_one_edge(move16=1)],
                )
                result = await store.insert_expansion(write)
                if result.outcome is WriteOutcome.COMMITTED:
                    committed += 1
                elif result.outcome is WriteOutcome.DUPLICATE:
                    duplicate += 1
                elif result.outcome is WriteOutcome.FAILED:
                    raise RuntimeError(f"expansion FAILED for index {i}: {result.error!r}")
                # COLLISION cannot happen here: the two processes write the
                # same sfen for the same index, so a lost race is always a
                # DUPLICATE, never a COLLISION.
            return committed, duplicate, store.duplicate_count
        finally:
            await store.close()

    try:
        committed, duplicate, dup_counter = asyncio.run(_run())
        result_queue.put(
            {
                "pid": os.getpid(),
                "committed": committed,
                "duplicate": duplicate,
                "duplicate_count": dup_counter,
                "error": None,
            }
        )
    except BaseException as exc:  # report any failure back to the parent
        result_queue.put(
            {
                "pid": os.getpid(),
                "committed": None,
                "duplicate": None,
                "duplicate_count": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _counter_child(
    conn_dict: dict,
    key_hi: int,
    key_lo: int,
    move16: int,
    increments: int,
    barrier_deadline: float,
    result_queue,
) -> None:
    """Child process body: apply ``increments`` counter backups to one key.

    Both counter children target the *same* Book_Node and the *same*
    Book_Edge, each issuing ``increments`` visit/value backups through the
    coalescing accumulator and flushing. Because both processes hit one row,
    the row-level lock in the flush write path (``SELECT ... FOR UPDATE`` in
    the fallback, or the ``puct_edge`` UPDATE in the extension path) is what
    stands between a correct final counter and a lost update. Each child
    reports back how many increments it issued so the parent can assert the
    stored total equals the exact sum (Requirement 11.8).

    ``barrier_deadline`` is a wall-clock instant both children spin-wait
    until before starting, so their flushes actually overlap in time rather
    than running back to back; a crude barrier is enough and needs no shared
    primitive across the spawn boundary.
    """
    conn = _ConnParams(**conn_dict)
    key = PositionKey(key_hi, key_lo)

    async def _run() -> int:
        config = _build_config(conn, connection_retry_limit=0)
        store = await NodeStore.connect_and_prepare(config)
        try:
            # Spin until the shared start instant so both children contend.
            while time.time() < barrier_deadline:
                await asyncio.sleep(0.001)
            for _ in range(increments):
                store.backup(
                    [
                        BackupDelta(
                            key=key,
                            node_visit_delta=1,
                            node_value_delta=1.0,
                            flags_or=0,
                            edge_deltas={move16: (1, 1.0)},
                        )
                    ]
                )
                # Flush every increment individually so the two processes
                # interleave many small read-modify-writes on the one row,
                # which is the contention Requirement 11.8 is about. A
                # single coalesced flush per process would still be correct
                # but would barely exercise the row lock.
                await store.flush()
            return increments
        finally:
            await store.close()

    try:
        applied = asyncio.run(_run())
        result_queue.put({"pid": os.getpid(), "applied": applied, "error": None})
    except BaseException as exc:
        result_queue.put(
            {"pid": os.getpid(), "applied": None, "error": f"{type(exc).__name__}: {exc}"}
        )


# ---------------------------------------------------------------------------
# Child lifecycle helper: start, join with timeout, always reap/kill
# ---------------------------------------------------------------------------


def _run_children_collecting(target, arg_tuples, timeout_s: float = _CHILD_TIMEOUT_S):
    """Spawn one child per entry in ``arg_tuples``, join, and collect results.

    Each child is passed its own ``result_queue`` appended to its args. The
    function guarantees, in a ``finally`` block, that every child is joined
    and any still-alive child is terminated then killed, so no test can leak
    a running child. It returns the list of result dicts the children put on
    their queues (order not guaranteed to match ``arg_tuples``).
    """
    procs = []
    queues = []
    try:
        for args in arg_tuples:
            q = _MP.Queue()
            p = _MP.Process(target=target, args=(*args, q))
            queues.append(q)
            procs.append(p)
        for p in procs:
            p.start()

        # Drain each queue with a join, honoring the overall timeout.
        results = []
        deadline = time.time() + timeout_s
        for p, q in zip(procs, queues):
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                raise TimeoutError(
                    f"child pid {p.pid} did not finish within {timeout_s}s"
                )
            # A child that finished always put exactly one result dict.
            try:
                results.append(q.get_nowait())
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"child pid {p.pid} exited (code {p.exitcode}) without a result"
                ) from exc
        return results
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            if p.is_alive():
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5.0)
        for q in queues:
            q.close()


# ---------------------------------------------------------------------------
# scratch-state helpers (each test manages its own state)
# ---------------------------------------------------------------------------


async def _prepare_clean_schema(pg_info, pg_conn) -> None:
    """Establish a correct, populated schema and empty the data tables.

    The session fixture applies ``schema.sql`` once, which creates the
    tables but does *not* insert the ``book_meta`` row (that is written by
    ``ensure_schema``/``connect_and_prepare``, which records the schema
    version, Zobrist fingerprint, and Root_Position). A child process that
    connects before ``book_meta`` is populated would hit a spurious
    ``SchemaVersionMismatchError``. So a "parent" ``NodeStore`` prepares the
    schema first -- mirroring design.md's rule that only the parent creates
    or repairs the schema and each ``search`` child then repeats the
    read-only half of the checks -- and then the data tables are truncated
    (but ``book_meta`` is left intact) so the test starts from an empty
    graph with a valid meta row the children can validate against.
    """
    # Drop the Node_Store-owned tables first so the parent takes the
    # from-scratch schema-creation path, which is the only path that
    # inserts the book_meta row (schema version, Zobrist fingerprint,
    # Root_Position). The session fixture's one-time schema.sql apply
    # creates the tables but not that row, so a child connecting over a
    # present-but-unpopulated book_meta would raise a spurious
    # SchemaVersionMismatchError; recreating from scratch avoids it.
    # This mirrors tests/book/test_node_store.py's _drop_book_tables +
    # connect_and_prepare pattern.
    await pg_conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, "
        "book_meta CASCADE"
    )
    parent_config = _build_config(
        _ConnParams.from_pg_info(pg_info), connection_retry_limit=0
    )
    parent = await NodeStore.connect_and_prepare(parent_config)
    await parent.close()


# ---------------------------------------------------------------------------
# Test 1: two writer processes, overlapping regions (Req 11.1, 11.6)
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_two_processes_overlapping_regions_one_row_per_key(
    pg_scratch_database, pg_conn
):
    """Two OS processes expand overlapping key regions against one database.

    Region A = indices [0, 600); region B = indices [400, 1000); the
    overlap is exactly the 200 indices [400, 600). Afterwards:

    - exactly one ``book_node`` row exists per distinct Position_Key across
      the union [0, 1000), i.e. 1000 rows, with no duplicate key
      (Requirement 11.6: exactly one row retained);
    - the surviving row for every key carries an ``eval_win_rate`` equal to
      exactly one of the two processes' values, never a blend
      (Requirement 11.6: evaluation fields are not a mixture);
    - the total DUPLICATE outcomes the two processes report equals the
      overlap size (each of the 200 shared keys is committed by exactly one
      process and seen as a duplicate by exactly the other), and the sum of
      the two processes' ``NodeStore.duplicate_count`` counters matches it
      too (Requirement 11.6: the duplicate count is reported).
    """
    # A parent NodeStore prepares/validates the schema and populates
    # book_meta before any child connects (design.md: only the parent
    # creates or repairs the schema), then the data tables are cleared.
    await _prepare_clean_schema(pg_scratch_database, pg_conn)

    conn_params = _ConnParams.from_pg_info(pg_scratch_database)
    conn_dict = conn_params.__dict__

    region_a = (0, 600)   # start_index, count
    region_b = (400, 600)
    overlap = 200         # |[400,600) ∩ [0,600)| and the shared tail of B
    union_size = 1000     # |[0, 1000)|

    eval_a = 0.25
    eval_b = 0.75

    # Run in a thread executor so this async test's own loop is not the one
    # blocking on join(); multiprocessing join is synchronous.
    results = await asyncio.get_running_loop().run_in_executor(
        None,
        _run_children_collecting,
        _writer_child,
        [
            (conn_dict, region_a[0], region_a[1], eval_a),
            (conn_dict, region_b[0], region_b[1], eval_b),
        ],
    )

    for r in results:
        assert r["error"] is None, f"writer child failed: {r['error']}"

    total_committed = sum(r["committed"] for r in results)
    total_duplicate = sum(r["duplicate"] for r in results)
    total_dup_counter = sum(r["duplicate_count"] for r in results)

    # Exactly one row per distinct key: 1000 keys, 1000 rows, no dup key.
    row_count = await pg_conn.fetchval("SELECT count(*) FROM book_node")
    distinct_keys = await pg_conn.fetchval(
        "SELECT count(*) FROM (SELECT DISTINCT key_hi, key_lo FROM book_node) s"
    )
    assert row_count == union_size
    assert distinct_keys == union_size

    # Every key committed exactly once across both processes; the rest were
    # seen as duplicates. committed + duplicate over both processes counts
    # every attempt, and attempts = |A| + |B| = 600 + 600 = 1200.
    assert total_committed == union_size
    assert total_duplicate == overlap
    assert total_committed + total_duplicate == region_a[1] + region_b[1]
    # The reported duplicate counters agree with the observed overlap.
    assert total_dup_counter == overlap

    # No blended evaluation field: every surviving row's eval_win_rate is
    # one of the two exact values the two processes wrote.
    distinct_evals = {
        round(float(row["eval_win_rate"]), 9)
        for row in await pg_conn.fetch(
            "SELECT DISTINCT eval_win_rate FROM book_node"
        )
    }
    assert distinct_evals <= {eval_a, eval_b}

    # And the row is byte-consistent with a single writer: for the overlap
    # keys the eval must be exactly eval_a or eval_b (the winner's), never
    # something in between.
    for i in (400, 450, 599):  # a few keys inside the overlap
        key = _synthetic_key(i)
        row = await pg_conn.fetchrow(
            "SELECT eval_win_rate, sfen FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row is not None
        assert float(row["eval_win_rate"]) in (eval_a, eval_b)
        assert row["sfen"] == _synthetic_sfen(i)


# ---------------------------------------------------------------------------
# Test 2: two processes hammering one counter (Req 11.1, 11.8)
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_two_processes_counter_updates_lose_nothing(
    pg_scratch_database, pg_conn
):
    """Two OS processes update the same node and edge counters; nothing is lost.

    A single Book_Node with a single Book_Edge is pre-created, then two
    processes each apply ``increments`` visit/value backups to it,
    flushing after each so the read-modify-write on the one packed ``edges``
    blob is done many times and the two processes' flushes interleave. The
    final stored node ``visit_count`` and the edge's ``visit_count`` must
    equal exactly ``2 * increments`` -- the sum of every increment both
    processes issued -- which holds only if PostgreSQL's row lock serialized
    the read-modify-writes (Requirement 11.8: no-lost-update).
    """
    await _prepare_clean_schema(pg_scratch_database, pg_conn)

    conn_params = _ConnParams.from_pg_info(pg_scratch_database)
    conn_dict = conn_params.__dict__

    move16 = 1
    key = _synthetic_key(123456)

    # Pre-create the target node with one zero-counter edge.
    setup_config = _build_config(conn_params, connection_retry_limit=0)
    setup_store = await NodeStore.connect_and_prepare(setup_config)
    try:
        result = await setup_store.insert_expansion(
            ExpansionWrite(
                key=key,
                sfen=_synthetic_sfen(123456),
                apery_key=123456,
                terminal=Terminal.NONE,
                eval_win_rate=0.5,
                edges=[_one_edge(move16=move16)],
            )
        )
        assert result.outcome is WriteOutcome.COMMITTED
    finally:
        await setup_store.close()

    increments = 150
    barrier_deadline = time.time() + 2.0  # both children start ~together

    results = await asyncio.get_running_loop().run_in_executor(
        None,
        _run_children_collecting,
        _counter_child,
        [
            (conn_dict, key.hi, key.lo, move16, increments, barrier_deadline),
            (conn_dict, key.hi, key.lo, move16, increments, barrier_deadline),
        ],
    )

    for r in results:
        assert r["error"] is None, f"counter child failed: {r['error']}"
    total_applied = sum(r["applied"] for r in results)
    assert total_applied == 2 * increments

    # The final node visit_count and value_sum must equal the exact sum of
    # every increment; a lost update would leave them below 2 * increments.
    row = await pg_conn.fetchrow(
        "SELECT visit_count, value_sum, edges FROM book_node "
        "WHERE key_hi = $1 AND key_lo = $2",
        fold_u64_to_i64(key.hi),
        fold_u64_to_i64(key.lo),
    )
    assert row is not None
    assert row["visit_count"] == 2 * increments
    assert float(row["value_sum"]) == pytest.approx(2.0 * increments, rel=1e-9)

    # The per-edge counter must also be intact: decode the packed edge blob
    # and read back the one edge's visit_count and value_sum.
    from dlshogi.book.packed_edge import decode_edges

    edges = decode_edges(bytes(row["edges"]))
    assert len(edges) == 1
    assert int(edges[0]["move16"]) == move16
    assert int(edges[0]["visit_count"]) == 2 * increments
    assert float(edges[0]["value_sum"]) == pytest.approx(2.0 * increments, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 3: Terashock_Index 1 ms p95 over >= 10^7 entries (Req 7.1)
# ---------------------------------------------------------------------------


def _terashock_row_iter(n: int):
    """Yield ``n`` small ``terashock_entry`` rows for copy_records_to_table.

    Each row is ``(key_hi, key_lo, sfen, moves)``. To keep the load cheap on
    disk and fast to build (design.md's Terashock_Index section budgets
    ~2 GB heap + 340 MB index for 10^7 entries, and the load note asks to
    keep row size small), the SFEN is a short synthetic marker and the
    packed ``moves`` blob is empty. The lookup path decodes ``moves`` with
    ``np.frombuffer`` and does not require it to be non-empty, so an empty
    blob exercises the same b-tree probe + heap fetch the requirement bounds
    without inflating every row.

    Keys are generated with the same multiplicative hash the write test
    uses, so distinct indices give distinct primary keys.
    """
    for i in range(n):
        pk = _synthetic_key(i)
        yield (
            fold_u64_to_i64(pk.hi),
            fold_u64_to_i64(pk.lo),
            f"ts-{i}",
            b"",
        )


@pytest.mark.db
@pytest.mark.slow
async def test_terashock_index_p95_lookup_under_1ms_over_10m_entries(
    pg_scratch_database, pg_conn
):
    """The Terashock_Index returns an entry within 1 ms p95 over >= 10^7 rows.

    Loads 10,000,000 ``terashock_entry`` rows with ``copy_records_to_table``
    -- the cheap bulk loader design.md's Terashock_Index section prescribes
    -- then measures ``TerashockIndex.lookup`` latency over 10,000 sampled,
    pairwise-distinct keys, so the in-process LRU never serves a repeat and
    every measured lookup is a real PostgreSQL b-tree probe plus one heap
    page fetch. Requirement 7.1's bound is the 95th percentile at or under
    1 ms.

    Load approach (documented per the task). The rows are streamed from a
    generator into one ``copy_records_to_table`` call in chunks so the
    driver never materialises all 10^7 tuples at once; the SFEN is a short
    marker and the packed ``moves`` blob is empty, keeping each row tiny
    (design.md budgets ~2 GB heap + 340 MB index at this scale, well within
    the ~226 GB free). After the copy, ``ANALYZE terashock_entry`` refreshes
    the planner statistics, and a warmup pass over the sample keys pulls the
    upper b-tree levels into ``shared_buffers`` (the three internal index
    levels stay resident permanently; only leaf + heap are cold), which is
    the steady state the 1 ms bound describes.

    This test is additionally marked ``slow``: loading 10^7 rows takes on the
    order of a minute. It skips gracefully if the load cannot complete within
    its budget, reporting what it managed, rather than failing on a slow or
    space-constrained host.
    """
    n_entries = 10_000_000
    n_samples = 10_000

    # Ensure the schema (and thus terashock_entry) exists and is empty,
    # regardless of what a prior test in the session left behind.
    await _prepare_clean_schema(pg_scratch_database, pg_conn)

    # --- Bulk load via copy_records_to_table, in chunks. ---
    load_start = time.monotonic()
    chunk = 500_000
    loaded = 0
    gen = _terashock_row_iter(n_entries)
    try:
        while loaded < n_entries:
            batch = []
            for _ in range(min(chunk, n_entries - loaded)):
                batch.append(next(gen))
            await pg_conn.copy_records_to_table(
                "terashock_entry",
                records=batch,
                columns=["key_hi", "key_lo", "sfen", "moves"],
            )
            loaded += len(batch)
    except Exception as exc:  # pragma: no cover - host/space dependent
        pytest.skip(f"could not bulk-load {n_entries} terashock rows: {exc}")

    load_elapsed = time.monotonic() - load_start

    row_count = await pg_conn.fetchval("SELECT count(*) FROM terashock_entry")
    assert row_count >= 10_000_000, f"only {row_count} rows loaded"

    # Refresh planner stats so the lookup uses the index, not a seq scan.
    await pg_conn.execute("ANALYZE terashock_entry")

    # --- Measure lookup latency over distinct sample keys. ---
    # Sample indices spread across the whole key space so probes hit many
    # different leaf pages rather than one hot page.
    step = max(1, n_entries // n_samples)
    sample_indices = [(j * step) % n_entries for j in range(n_samples)]
    sample_keys = [_synthetic_key(i) for i in sample_indices]

    # A large LRU here would let repeated probes of the same leaf be served
    # from process memory and understate the PostgreSQL latency; the sample
    # keys are pairwise distinct, so no LRU hit occurs regardless, but we
    # keep the budget small to be explicit about that.
    index = TerashockIndex(pg_conn_pool_shim(pg_conn), cache_budget_bytes=1 << 20)

    # Warmup: pull the upper b-tree levels into shared_buffers. Warmup
    # lookups are not measured.
    for key in sample_keys[: min(2000, n_samples)]:
        await index.lookup(key)
    index.clear_cache()

    latencies_ms = np.empty(n_samples, dtype=np.float64)
    for j, key in enumerate(sample_keys):
        t0 = time.perf_counter()
        result = await index.lookup(key)
        latencies_ms[j] = (time.perf_counter() - t0) * 1000.0
        # Every sampled key exists, so the lookup must find it.
        assert result is not None
        # Distinct keys -> never an LRU hit, so every measurement is a
        # real database probe.
    assert index.hits == 0

    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    mean_ms = float(latencies_ms.mean())

    # Requirement 7.1: 1 ms at the 95th percentile.
    assert p95_ms <= 1.0, (
        f"Terashock_Index p95 lookup {p95_ms:.3f} ms exceeds the 1 ms bound "
        f"(mean {mean_ms:.3f} ms, p99 {p99_ms:.3f} ms) over {row_count} rows; "
        f"load took {load_elapsed:.1f}s"
    )


class _PoolShim:
    """Adapt a single asyncpg ``Connection`` to the ``pool.acquire()`` API.

    ``TerashockIndex`` acquires a connection from a pool via
    ``async with pool.acquire() as conn``. The db-marked tests here already
    hold one ``pg_conn`` connection from the fixture; wrapping it in this
    shim lets ``TerashockIndex`` run its lookups on that same connection
    without standing up a second pool, which is all the p95 measurement
    needs (it is single-threaded and does one lookup at a time).
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def pg_conn_pool_shim(conn) -> _PoolShim:
    """Return a `_PoolShim` wrapping ``conn`` (a plain factory for readability)."""
    return _PoolShim(conn)
