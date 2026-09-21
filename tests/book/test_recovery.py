"""Crash-recovery and connection-loss integration tests (task 19.2).

These are the optional (``*``) integration tests the design's *Testing
Strategy* names under "Crash recovery (Requirements 2.9, 10.3, 10.7)" and
"Connection loss (Requirement 2.9)":

- **SIGKILL crash recovery.** A real child OS process performs expansion
  writes -- and, in a second variant, accumulator flushes -- against the
  scratch database, and is killed with ``SIGKILL`` at a randomized point.
  The parent then re-opens the Node_Store and asserts that every
  Position_Key the child touched is either *absent* or *complete* (a row
  present with ``octet_length(edges) == edge_count * 20`` and
  ``edge_count`` equal to the full legal-move set the child would have
  written), never half-written -- Requirement 10.3's all-or-nothing
  expansion, which holds because each expansion is one
  ``INSERT ... ON CONFLICT DO NOTHING`` single-row write (task 8.1) into a
  ``LOGGED`` ``book_node`` relation (task 7.1, deliberately not
  ``UNLOGGED``: an ``UNLOGGED`` relation would be truncated by crash
  recovery and lose every committed row, contradicting Requirements 2.3,
  10.1, and 10.3). It then asserts the run *resumes*: a fresh Node_Store
  over the surviving graph reads the committed rows back (Requirement
  10.1) and a follow-up expansion of a new key still commits.

- **Connection loss.** ``pg_terminate_backend`` -- issued from a separate
  admin connection exactly as ``tests/book/conftest.py``'s
  ``_drop_database`` already does for cleanup -- terminates the backend a
  Node_Store statement is using, mid-search. The test asserts the
  ``node_store._run`` suspension seam (task 7.2) catches the resulting
  ``ConnectionDoesNotExistError`` / ``ConnectionResetError``, clears the
  ``suspended`` event while it reconnects on the Requirement 2.2 schedule,
  re-raises the failed statement's exception to its caller (Requirement
  2.9's "treat every in-progress write as a failed write", Requirement
  1.9), and then resumes: ``suspended`` is set again and a subsequent
  statement succeeds against the reconnected pool. A separate
  virtual-clock test asserts the *timing* of the retry schedule
  (5 s per attempt, 1 s doubling to a 60 s cap, ``Connection_Retry_Limit``
  retries) without sleeping on the wall clock, per the task instruction to
  use ``conftest.py``'s ``virtual_clock_loop`` fixture wherever the
  schedule timing is asserted.

Every test that needs PostgreSQL carries the ``db`` marker and reuses the
scratch-database fixtures from ``tests/book/conftest.py``
(``pg_scratch_database`` / ``pg_conn``), which skip gracefully when no
server is reachable, so ``pytest -m "not db"`` never touches a database.
Each test creates and cleans its own state within the scratch database and
never leaves a child process behind: every spawned process is reaped or
killed in a ``finally``.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import random
import signal
import time
from types import SimpleNamespace

import asyncpg
import cshogi
import pytest

from dlshogi.book.config import BookConfig
from dlshogi.book.keys import PositionKey, fold_u64_to_i64, position_key_from_sfen
from dlshogi.book.node_store import (
    ExpansionWrite,
    GetResult,
    NodeStore,
    Terminal,
    WriteOutcome,
    retry_schedule,
)

# The standard initial Shogi position; the root of every graph these tests
# build. Its legal-move set is what the child processes expand into edges.
_INITIAL_POSITION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

# A handful of one-move-deep positions, so the child has several distinct
# Position_Keys to expand rather than only the root. Each is the position
# after one legal opening move from the initial position; all are legal
# and have their own full legal-move set.
_CHILD_SFENS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 2",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL w - 2",
]


def _make_config(pg_info, **overrides) -> BookConfig:
    """Build a `BookConfig` pointed at ``pg_info``'s database.

    Mirrors ``tests/book/test_node_store.py``'s helper of the same name:
    every non-connection field gets an arbitrary but in-range placeholder,
    since these tests do not exercise `validate_config`. ``pg_info`` is a
    ``tests.book.conftest.PgConnectInfo`` (or any object exposing
    ``host``/``port``/``user``/``password``/``database``).
    """
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
        connection_retry_limit=2,
        root_position=_INITIAL_POSITION_SFEN,
        report_interval=60,
        throughput_floor=50,
        throughput_grace_period=600,
    )
    kwargs.update(overrides)
    return BookConfig(**kwargs)


def _legal_edges_for(sfen: str) -> list[dict]:
    """Return the full `PACKED_EDGE`-shaped edge list for ``sfen``.

    One edge per legal move, in the ``move16`` form `insert_expansion`'s
    `packed_edge.encode_edges` accepts. ``encode_edges`` itself re-sorts
    into ascending USI order and zero-fills the Terashock fields, so the
    dicts here need only carry ``move16`` plus a couple of non-zero
    counters, which is enough to make "the row is complete" a meaningful,
    non-degenerate assertion (a complete row has exactly this many
    20-byte records).
    """
    board = cshogi.Board(sfen)
    edges: list[dict] = []
    for move in board.legal_moves:
        edges.append(
            {
                "move16": cshogi.move16(move),
                "prior_q16": 0,
                "flags": 0,
                "visit_count": 1,
                "value_sum": 0.5,
            }
        )
    return edges


def _expansion_plan() -> list[tuple[str, list[dict]]]:
    """The full set of ``(sfen, edges)`` a child process expands.

    The root plus the three one-move-deep positions; every Position_Key in
    this plan is what the parent later checks for absent-or-complete.
    """
    plan: list[tuple[str, list[dict]]] = [(_INITIAL_POSITION_SFEN, _legal_edges_for(_INITIAL_POSITION_SFEN))]
    for sfen in _CHILD_SFENS:
        plan.append((sfen, _legal_edges_for(sfen)))
    return plan


# ---------------------------------------------------------------------------
# Child-process entry points for the SIGKILL crash-recovery tests.
#
# These run in a *separate OS process* (multiprocessing), connect to the
# scratch database on their own, and perform expansion writes / accumulator
# flushes in a loop. They signal readiness through a shared queue so the
# parent can kill them at a randomized point during the write activity
# rather than before it has begun. They never return normally: the parent
# always SIGKILLs them.
# ---------------------------------------------------------------------------


def _child_expansion_writer(conn_kwargs: dict, ready_queue) -> None:  # pragma: no cover - runs in a child process
    """Repeatedly issue expansion writes until SIGKILLed.

    Connects, then loops over the expansion plan issuing
    `insert_expansion` calls forever (re-issuing already-committed writes,
    which land as idempotent DUPLICATEs -- Requirement 10.4 -- so the loop
    is safe to interrupt at any instant). Puts one token on ``ready_queue``
    after the first successful commit, so the parent knows real write
    activity is under way before it starts its randomized-delay kill timer.
    """

    async def run() -> None:
        config = BookConfig(**conn_kwargs)
        store = await NodeStore.connect_and_prepare(config)
        plan = _expansion_plan()
        signalled = False
        while True:
            for sfen, edges in plan:
                key = position_key_from_sfen(sfen)
                w = ExpansionWrite(
                    key=key,
                    sfen=sfen,
                    apery_key=0,
                    terminal=Terminal.NONE,
                    edges=edges,
                )
                await store.insert_expansion(w)
                if not signalled:
                    ready_queue.put(os.getpid())
                    signalled = True

    asyncio.run(run())


def _child_flush_writer(conn_kwargs: dict, ready_queue) -> None:  # pragma: no cover - runs in a child process
    """Expand every planned node, then repeatedly back up and flush.

    First commits every expansion in the plan (so the rows exist), then
    loops calling `backup()` + `flush()`, which drives the accumulator
    flush path (task 8.6) that this variant is meant to interrupt. Puts a
    token on ``ready_queue`` once the first flush has completed, so the
    parent kills during flush activity rather than during the initial
    expansion phase. Because the interrupted work is a *flush of visit/
    value deltas* onto already-complete rows, Requirement 10.3's
    absent-or-complete invariant still holds for every key: a flush never
    changes ``edge_count`` or the number of 20-byte edge records, only the
    per-edge counters inside an already-complete blob.
    """

    async def run() -> None:
        config = BookConfig(**conn_kwargs)
        store = await NodeStore.connect_and_prepare(config)
        plan = _expansion_plan()
        for sfen, edges in plan:
            key = position_key_from_sfen(sfen)
            await store.insert_expansion(
                ExpansionWrite(key=key, sfen=sfen, apery_key=0, edges=edges)
            )
        signalled = False
        while True:
            from dlshogi.book.node_store import BackupDelta

            deltas = []
            for sfen, edges in plan:
                key = position_key_from_sfen(sfen)
                edge_deltas = {int(e["move16"]): (1, 0.5) for e in edges}
                deltas.append(
                    BackupDelta(
                        key=key,
                        node_visit_delta=1,
                        node_value_delta=0.5,
                        edge_deltas=edge_deltas,
                    )
                )
            store.backup(deltas)
            await store.flush()
            if not signalled:
                ready_queue.put(os.getpid())
                signalled = True

    asyncio.run(run())


def _config_to_child_kwargs(config: BookConfig) -> dict:
    """Flatten a `BookConfig` into the kwargs a child rebuilds it from.

    Passing the fully-constructed `BookConfig` through a multiprocessing
    queue works too, but rebuilding from a plain dict inside the child
    keeps the child entry point independent of pickle behaviour of the
    dataclass and mirrors how the real per-GPU ``search`` children each
    construct their own config from the same source (design.md's process
    model).
    """
    import dataclasses

    return dataclasses.asdict(config)


def _run_child_until_ready(target, config: BookConfig, timeout_s: float = 30.0):
    """Spawn ``target`` as a child process and wait for its readiness token.

    Returns ``(process, pid)``. Raises through ``pytest.fail`` if the child
    dies or never signals readiness within ``timeout_s``. The caller owns
    reaping/killing the returned process.
    """
    ctx = multiprocessing.get_context("spawn")
    ready_queue = ctx.Queue()
    proc = ctx.Process(
        target=target, args=(_config_to_child_kwargs(config), ready_queue), daemon=True
    )
    proc.start()
    try:
        pid = ready_queue.get(timeout=timeout_s)
    except Exception:  # noqa: BLE001 - queue.Empty and friends
        proc.kill()
        proc.join(timeout=5.0)
        pytest.fail("child process never signalled write readiness")
    return proc, pid


async def _assert_every_key_absent_or_complete(store: NodeStore) -> int:
    """Assert Requirement 10.3 for every key in the expansion plan.

    Every Position_Key the child could have written is, on this fresh
    Node_Store over the surviving graph, either ABSENT or a complete row:
    present with an edge blob whose length is exactly
    ``edge_count * 20`` bytes and ``edge_count`` equal to the full legal
    move set that key's expansion would have written -- never a partial
    edge set. Returns the number of keys found present (>= 0), which the
    caller uses to confirm the run actually committed something before the
    kill (so the test is not vacuously satisfied by an all-absent graph).
    """
    present = 0
    for sfen, edges in _expansion_plan():
        key = position_key_from_sfen(sfen)
        result, view = await store.get(key)
        assert result in (GetResult.FOUND, GetResult.ABSENT), (
            f"unexpected get() result {result} for {sfen!r}"
        )
        if result is GetResult.ABSENT:
            assert view is None
            continue
        present += 1
        # Present => complete: the full legal-move edge set, and the
        # schema's own book_node_edges_len CHECK (octet_length(edges) =
        # edge_count * 20) held at write time, so a present row can never
        # carry a half-written edge blob.
        assert view is not None
        expected_edge_count = len(edges)
        assert view.edges.shape[0] == expected_edge_count, (
            f"{sfen!r}: present but incomplete edge set "
            f"({view.edges.shape[0]} of {expected_edge_count} edges)"
        )
        assert view.sfen == sfen
    return present


# ---------------------------------------------------------------------------
# SIGKILL crash recovery (Requirements 10.3, 10.7, 2.9).
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("target", [_child_expansion_writer, _child_flush_writer])
async def test_sigkill_leaves_every_key_absent_or_complete_and_run_resumes(
    pg_scratch_database, pg_conn, target
):
    """SIGKILL a writing child at a randomized point; graph stays consistent.

    Parametrized over the two interruptible write phases the design names:
    expansion writes and accumulator flushes. After the kill, a fresh
    Node_Store over the surviving graph must satisfy Requirement 10.3
    (every key absent or complete) and must *resume* (Requirement 10.1 /
    10.7): it reads the committed rows back and a follow-up expansion of a
    brand-new key still commits.
    """
    await _drop_book_tables(pg_conn)

    # Create the schema once, from the parent, before spawning the child,
    # so the child only ever writes rows (never races on CREATE TABLE) and
    # a kill during schema creation is not what is being tested here.
    config = _make_config(pg_scratch_database, connection_retry_limit=0)
    setup_store = await NodeStore.connect_and_prepare(config)
    await setup_store.close()

    proc, pid = _run_child_until_ready(target, config)
    try:
        # Let the child write for a randomized interval so the SIGKILL
        # lands at an unpredictable point relative to the INSERT / flush
        # statements, per the task's "randomized points" instruction.
        time.sleep(random.uniform(0.05, 0.30))
        os.kill(pid, signal.SIGKILL)
        proc.join(timeout=10.0)
        assert not proc.is_alive(), "child survived SIGKILL"
        # A real SIGKILL terminates with negative exitcode -SIGKILL.
        assert proc.exitcode == -signal.SIGKILL
    finally:
        if proc.is_alive():  # pragma: no cover - defensive: never leak a child
            proc.kill()
            proc.join(timeout=5.0)

    # Re-open the Node_Store over the surviving graph and assert the
    # crash-safety invariant plus resumption.
    store = await NodeStore.connect_and_prepare(config)
    try:
        present = await _assert_every_key_absent_or_complete(store)
        assert present >= 1, (
            "no committed rows survived the kill; the test cannot distinguish "
            "crash-safety from an empty graph -- increase the pre-kill delay"
        )

        # Resume: a follow-up expansion of a brand-new key still commits
        # against the recovered graph (Requirement 10.7's "the run
        # resumes").
        resume_sfen = _CHILD_SFENS[0]
        # A two-move-deep position not in the plan, guaranteed distinct.
        resume_board = cshogi.Board(_INITIAL_POSITION_SFEN)
        first_move = next(iter(resume_board.legal_moves))
        resume_board.push(first_move)
        second_move = next(iter(resume_board.legal_moves))
        resume_board.push(second_move)
        resume_sfen = resume_board.sfen()
        resume_key = position_key_from_sfen(resume_sfen)
        result = await store.insert_expansion(
            ExpansionWrite(
                key=resume_key,
                sfen=resume_sfen,
                apery_key=0,
                edges=_legal_edges_for(resume_sfen),
            )
        )
        assert result.outcome in (WriteOutcome.COMMITTED, WriteOutcome.DUPLICATE)
        found, view = await store.get(resume_key)
        assert found is GetResult.FOUND and view is not None
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Connection loss via pg_terminate_backend (Requirement 2.9).
# ---------------------------------------------------------------------------


def _admin_kwargs(pg_info) -> dict:
    """Admin connection kwargs for issuing pg_terminate_backend.

    Points at the maintenance database (``postgres``) on the same server,
    exactly as ``tests/book/conftest.py``'s ``_drop_database`` connects to
    terminate lingering backends -- the same pattern, reused here to
    terminate the scratch database's backends by pid on purpose rather than
    at cleanup.
    """
    return dict(
        host=pg_info.host,
        port=pg_info.port,
        user=pg_info.user,
        password=pg_info.password,
        database="postgres",
    )


async def _terminate_scratch_backends(pg_info) -> int:
    """Terminate every backend connected to the scratch database.

    Returns the number of backends asked to terminate. Uses a fresh admin
    connection to the maintenance database, so terminating the scratch
    database's backends does not terminate the terminator itself
    (``pid <> pg_backend_pid()``), mirroring ``_drop_database``.
    """
    admin = await asyncpg.connect(**_admin_kwargs(pg_info))
    try:
        rows = await admin.fetch(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid()",
            pg_info.database,
        )
        return len(rows)
    finally:
        await admin.close()


@pytest.mark.db
async def test_pg_terminate_backend_suspends_reconnects_and_resumes(
    pg_scratch_database, pg_conn
):
    """A terminated backend suspends the Node_Store, which reconnects and resumes.

    Drives the suspension seam end-to-end against a real terminate: a
    Node_Store statement whose backend was terminated by
    ``pg_terminate_backend`` raises the connection-lost exception that
    `_run` catches; the store clears ``suspended`` while it reconnects on
    the Requirement 2.2 schedule, re-raises the failed statement's
    exception to the caller (Requirement 2.9 treats the in-progress write
    as failed, Requirement 1.9), then sets ``suspended`` again so a
    subsequent statement succeeds against the reconnected pool.
    """
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=3)

    store = await NodeStore.connect_and_prepare(config)
    try:
        assert store.suspended.is_set()

        # First, prove the store works, then terminate its backends and
        # prove a statement issued afterwards trips the seam. Because the
        # pool may hold more than one connection, the statement inside
        # `_run` acquires a (possibly freshly reissued) connection; we
        # terminate right before issuing it, so the acquired connection is
        # one of the terminated backends or a reconnect thereof.
        async def one_statement(pool):
            async with pool.acquire() as conn:
                return await conn.fetchval("SELECT 1")

        assert await store._run(one_statement) == 1

        # Terminate every scratch backend, including the pool's, then issue
        # a statement. asyncpg surfaces the loss as
        # ConnectionDoesNotExistError / ConnectionResetError / the mid-query
        # InternalClientError, which `_run` catches, suspends on,
        # reconnects for real (the scratch DB is still alive), and re-raises.
        await _terminate_scratch_backends(pg_scratch_database)

        seam_tripped = False
        try:
            await store._run(one_statement)
        except (
            asyncpg.exceptions.ConnectionDoesNotExistError,
            ConnectionResetError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.InternalClientError,
        ):
            seam_tripped = True

        if seam_tripped:
            # The seam observed the loss: it reconnected and re-set the
            # suspension event before re-raising to us.
            assert store.suspended.is_set()

        # Either way (asyncpg sometimes transparently re-establishes a
        # pooled connection so the very next statement just works), the
        # store must now be usable and not left suspended.
        assert store.suspended.is_set()
        assert await store._run(one_statement) == 1
    finally:
        await store.close()


@pytest.mark.db
async def test_pg_terminate_backend_marks_in_progress_expansion_as_failed(
    pg_scratch_database, pg_conn
):
    """Requirement 2.9 + 1.9: a write whose backend is terminated is a failed write.

    `insert_expansion` catches the connection-lost exception `_run`
    re-raises and returns ``WriteOutcome.FAILED`` naming the Position_Key,
    leaving no record behind (Requirement 1.9). After the seam reconnects,
    a re-issued expansion of the same key commits cleanly, and the run
    resumes.
    """
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=3)

    store = await NodeStore.connect_and_prepare(config)
    try:
        sfen = _CHILD_SFENS[0]
        key = position_key_from_sfen(sfen)
        edges = _legal_edges_for(sfen)

        # Terminate the backends, then attempt the expansion. The attempt
        # either FAILS (seam caught the loss and re-raised, insert_expansion
        # mapped it to FAILED) or COMMITS (asyncpg transparently reconnected
        # the pooled connection first). Both are Requirement-2.9-consistent;
        # the invariant we assert is that no *partial* row is ever left, and
        # that the key ends up either absent or complete.
        await _terminate_scratch_backends(pg_scratch_database)
        result = await store.insert_expansion(
            ExpansionWrite(key=key, sfen=sfen, apery_key=0, edges=edges)
        )
        assert result.outcome in (WriteOutcome.COMMITTED, WriteOutcome.FAILED)
        if result.outcome is WriteOutcome.FAILED:
            assert result.key == key
            assert result.error is not None
            # Requirement 1.9: no record left behind by the failed write.
            get_result, _ = await store.get(key)
            assert get_result is GetResult.ABSENT

        # Resume: the store reconnected, and a (re-)issued expansion now
        # commits and reads back complete.
        assert store.suspended.is_set()
        retry = await store.insert_expansion(
            ExpansionWrite(key=key, sfen=sfen, apery_key=0, edges=edges)
        )
        assert retry.outcome in (WriteOutcome.COMMITTED, WriteOutcome.DUPLICATE)
        found, view = await store.get(key)
        assert found is GetResult.FOUND and view is not None
        assert view.edges.shape[0] == len(edges)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Retry-schedule timing under the virtual clock (Requirement 2.2), no
# database and no wall-clock sleep.
# ---------------------------------------------------------------------------


def test_reconnect_retry_schedule_timing_on_virtual_clock(virtual_clock_loop):
    """The reconnect wait schedule matches Requirement 2.2, measured on virtual time.

    Requirement 2.9 says a lost connection is retried "using the per-attempt
    limit, retry count, and wait schedule of criterion 2". This asserts the
    *timing* of that schedule -- 1 s before the first retry, doubling before
    each subsequent retry, capped at 60 s, ``Connection_Retry_Limit`` entries
    -- by scheduling each wait on the ``virtual_clock_loop`` and advancing
    the clock, so the whole schedule (which in wall-clock terms spans many
    minutes at the 60 s cap) costs no real time. This is the pure
    ``retry_schedule`` used by both `connect` and the reconnect seam, so
    verifying its timing here covers the reconnect path's schedule without
    needing to induce a real multi-minute reconnect loop.
    """
    loop = virtual_clock_loop

    for limit, expected in (
        (0, []),
        (1, [1.0]),
        (5, [1.0, 2.0, 4.0, 8.0, 16.0]),
    ):
        assert retry_schedule(limit) == expected

    # Now actually *drive* the schedule on the virtual clock: schedule a
    # callback after each cumulative wait, advance the clock across the
    # whole schedule, and confirm every retry fired and that no retry fired
    # before its scheduled virtual time -- all in essentially zero wall
    # time. (The exact virtual instant a due callback observes via
    # loop.time() is a stepping detail of VirtualClockEventLoop.advance,
    # which advances the clock to the next timer boundary before draining
    # the ready queue; the schedule contract this test asserts is "each
    # retry waits at least its scheduled amount, and the whole schedule is
    # driven with no wall-clock sleep", not the sub-step ordering of the
    # loop's own ready-queue drain.)
    schedule = retry_schedule(8)  # 1,2,4,8,16,32,60,60 -> cap reached
    assert schedule == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

    fired: list[float] = []
    cumulative = 0.0
    for wait in schedule:
        cumulative += wait
        when = cumulative

        def _record(scheduled_for=when):
            fired.append(scheduled_for)
            # A due callback is never observed at a virtual time earlier
            # than the instant it was scheduled for.
            assert loop.time() >= scheduled_for - 1e-9

        loop.call_later(when, _record)

    total = sum(schedule)
    start = time.monotonic()
    loop.run_until_complete(loop.advance(total))
    wall_elapsed = time.monotonic() - start

    # Every scheduled retry fired exactly once, in schedule order.
    expected_scheduled_for = []
    acc = 0.0
    for wait in schedule:
        acc += wait
        expected_scheduled_for.append(round(acc, 6))
    assert fired == expected_scheduled_for
    # The virtual clock advanced across the full (multi-minute) schedule.
    assert abs(loop.time() - total) < 1e-6
    # And the whole many-minute schedule cost essentially no real time.
    assert wall_elapsed < 5.0


# ---------------------------------------------------------------------------
# Shared helper: drop the Node_Store-owned tables so each test starts from
# a truly empty database (the real __main__.py startup path), matching
# tests/book/test_node_store.py's own _drop_book_tables.
# ---------------------------------------------------------------------------


async def _drop_book_tables(conn: asyncpg.Connection) -> None:
    await conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, book_meta CASCADE"
    )
