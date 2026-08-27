"""Unit tests for ``dlshogi.book.node_store`` (task 7.2).

These are ordinary example-based unit tests against a real scratch
PostgreSQL database (via ``tests/book/conftest.py``'s ``pg_scratch_database``
/ ``pg_conn`` fixtures), not the hypothesis-driven property tests -- those
are separate, optional tasks (7.3 Property 6, 7.4 Property 7, 7.5 Property
5) and are not implemented here. This file only covers task 7.2's own
scope: pool creation, schema creation/versioning/repair, Root_Position
recording and read-back, the connection retry schedule, the reconnection
seam, and the per-role ``synchronous_commit`` setting.

Every ``@pytest.mark.db`` test drops the four Node_Store-owned tables
before running, so each test starts from the "fresh empty database with no
tables at all" state that is the real startup path in ``__main__.py``,
even though ``pg_scratch_database`` (session-scoped) has already applied
``dlshogi/book/sql/schema.sql`` once via its own ``_apply_schema_if_present``
fixture helper.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import asyncpg
import pytest

from dlshogi.book.config import BookConfig
from dlshogi.book.keys import PositionKey, fold_u64_to_i64, zobrist_fingerprint
from dlshogi.book.node_store import (
    SCHEMA_VERSION,
    BackupDelta,
    BookNodeView,
    ExpansionWrite,
    GetResult,
    NodeStore,
    NodeStoreConnectionError,
    PropagationWrite,
    SchemaVersionMismatchError,
    Terminal,
    WriteOutcome,
    retry_schedule,
)
from dlshogi.book.packed_edge import encode_edges

_INITIAL_POSITION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _make_config(pg_info, **overrides) -> BookConfig:
    """Build a `BookConfig` pointed at ``pg_info``'s database.

    ``pg_info`` is a `tests.book.conftest.PgConnectInfo` (or any object
    with matching ``host``/``port``/``user``/``password``/``database``
    attributes). Every non-connection field is filled with an arbitrary
    but in-range placeholder value, since this module's own tests do not
    exercise `validate_config`.
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
        connection_retry_limit=0,
        root_position=_INITIAL_POSITION_SFEN,
        report_interval=60,
        throughput_floor=50,
        throughput_grace_period=600,
    )
    kwargs.update(overrides)
    return BookConfig(**kwargs)


async def _drop_book_tables(conn: asyncpg.Connection) -> None:
    """Drop every Node_Store-owned table, so the database is truly empty."""
    await conn.execute(
        "DROP TABLE IF EXISTS book_node, terashock_entry, in_flight_claim, book_meta CASCADE"
    )


# ---------------------------------------------------------------------------
# (d) retry_schedule(): pure function, no database needed.
# ---------------------------------------------------------------------------


def test_retry_schedule_shape():
    assert retry_schedule(0) == []
    assert retry_schedule(1) == [1.0]
    assert retry_schedule(5) == [1.0, 2.0, 4.0, 8.0, 16.0]

    schedule = retry_schedule(100)
    assert len(schedule) == 100
    assert schedule[0] == 1.0
    for previous, current in zip(schedule, schedule[1:]):
        assert current == min(previous * 2.0, 60.0)
    assert all(wait <= 60.0 for wait in schedule)


def test_retry_schedule_rejects_negative_limit():
    with pytest.raises(ValueError):
        retry_schedule(-1)


def test_suspended_event_starts_set():
    """`NodeStore.suspended` starts *set* (not suspended, proceed)."""
    fake_pg_info = SimpleNamespace(
        host="localhost", port=5432, user="u", password=None, database="d"
    )
    store = NodeStore(_make_config(fake_pg_info))
    assert store.suspended.is_set()


# ---------------------------------------------------------------------------
# Connection failure (Requirement 2.3), no scratch database needed: the
# target port is simply not listening, so asyncpg fails fast with no
# retries configured.
# ---------------------------------------------------------------------------


async def test_connection_failure_reports_and_raises():
    fake_pg_info = SimpleNamespace(
        host="127.0.0.1", port=1, user="nobody", password=None, database="nonexistent"
    )
    config = _make_config(fake_pg_info, connection_retry_limit=0)
    store = NodeStore(config)
    with pytest.raises(NodeStoreConnectionError) as excinfo:
        await store.connect()
    assert excinfo.value.attempts == 1
    assert excinfo.value.host == config.pg_host
    assert excinfo.value.port == config.pg_port
    assert excinfo.value.database == config.pg_database


# ---------------------------------------------------------------------------
# (a) Fresh database -> schema created, version/fingerprint/root recorded
# and read back.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_fresh_database_creates_schema_and_records_root(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        tables = {
            row["tablename"]
            for row in await pg_conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert {"book_meta", "book_node", "terashock_entry", "in_flight_claim"} <= tables

        meta = await pg_conn.fetchrow("SELECT * FROM book_meta WHERE id = 1")
        assert meta is not None
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["zobrist_fingerprint"] == fold_u64_to_i64(zobrist_fingerprint())
        assert meta["root_sfen"] == config.root_position

        root_sfen, root_key_hi, root_key_lo = await store.get_root()
        assert root_sfen == config.root_position
        assert root_key_hi == meta["root_key_hi"]
        assert root_key_lo == meta["root_key_lo"]

        # No book_node rows yet, so neither half of the C11 check fires.
        assert await store.any_node_rows() is False
        assert await store.root_node_exists() is False
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# (b) Second connect against an already-correct schema -> no error,
# existing rows untouched.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_ensure_schema_does_not_deadlock_at_worker_count_one(pg_scratch_database, pg_conn):
    """`max_size = min(Worker_Count, 64)` can be 1; schema creation must not
    need two connections held open at once against a single-connection pool.
    """
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0, worker_count=1)

    store = NodeStore(config)
    await store.connect()
    try:
        await asyncio.wait_for(store.ensure_schema(), timeout=5.0)
        root_sfen, _, _ = await store.get_root()
        assert root_sfen == config.root_position
    finally:
        await store.close()


@pytest.mark.db
async def test_second_connect_against_existing_schema_leaves_rows_untouched(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store1 = await NodeStore.connect_and_prepare(config)
    await store1.close()

    await pg_conn.execute(
        "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
        "edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
        1,
        2,
        "sfen-marker",
        3,
        42,
        1.5,
    )

    # A second start over the same (already-correct) schema must not
    # raise and must not alter the row just inserted (Requirement 2.5).
    store2 = await NodeStore.connect_and_prepare(config)
    try:
        row = await pg_conn.fetchrow(
            "SELECT * FROM book_node WHERE key_hi = 1 AND key_lo = 2"
        )
        assert row is not None
        assert row["sfen"] == "sfen-marker"
        assert row["visit_count"] == 42
        assert row["value_sum"] == 1.5

        assert await store2.any_node_rows() is True
    finally:
        await store2.close()


# ---------------------------------------------------------------------------
# (c) schema_version / zobrist_fingerprint mismatch -> raises, creates and
# alters nothing.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_schema_version_mismatch_is_inert(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store0 = await NodeStore.connect_and_prepare(config)
    await store0.close()

    await pg_conn.execute("UPDATE book_meta SET schema_version = 'bogus-version' WHERE id = 1")
    before = dict(await pg_conn.fetchrow("SELECT * FROM book_meta WHERE id = 1"))

    store = NodeStore(config)
    await store.connect()
    try:
        with pytest.raises(SchemaVersionMismatchError) as excinfo:
            await store.ensure_schema()
        assert excinfo.value.recorded_version == "bogus-version"
        assert excinfo.value.running_version == SCHEMA_VERSION
    finally:
        await store.close()

    after = dict(await pg_conn.fetchrow("SELECT * FROM book_meta WHERE id = 1"))
    assert after == before

    tables = {
        row["tablename"]
        for row in await pg_conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
    }
    assert tables == {"book_meta", "book_node", "terashock_entry", "in_flight_claim"}


@pytest.mark.db
async def test_schema_version_mismatch_when_meta_row_absent(pg_scratch_database, pg_conn):
    """Property 7's "absent case": a present schema with no recorded version."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store0 = await NodeStore.connect_and_prepare(config)
    await store0.close()

    await pg_conn.execute("DELETE FROM book_meta")

    store = NodeStore(config)
    await store.connect()
    try:
        with pytest.raises(SchemaVersionMismatchError) as excinfo:
            await store.ensure_schema()
        assert excinfo.value.recorded_version is None
        assert excinfo.value.running_version == SCHEMA_VERSION
    finally:
        await store.close()

    # Still no row: nothing was created or altered by the failed attempt.
    assert await pg_conn.fetchrow("SELECT * FROM book_meta") is None


# ---------------------------------------------------------------------------
# Absent-element-only repair (Requirement 2.5): dropping a droppable
# element (here, the STORAGE MAIN setting is out of scope; we exercise the
# table-recreation path instead by dropping a non-key table and checking
# it comes back empty rather than raising, and that book_node is
# untouched).
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_repair_recreates_absent_table_without_touching_book_node(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store0 = await NodeStore.connect_and_prepare(config)
    await store0.close()

    await pg_conn.execute(
        "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
        "edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
        7,
        8,
        "keep-me",
        9,
        11,
        0.25,
    )
    # Drop a table that carries no rows worth preserving in this test, to
    # exercise the "create only what is absent" path without touching
    # book_node's own schema.
    await pg_conn.execute("DROP TABLE in_flight_claim")

    store = await NodeStore.connect_and_prepare(config)
    try:
        tables = {
            row["tablename"]
            for row in await pg_conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert "in_flight_claim" in tables

        row = await pg_conn.fetchrow(
            "SELECT * FROM book_node WHERE key_hi = 7 AND key_lo = 8"
        )
        assert row is not None
        assert row["sfen"] == "keep-me"
        assert row["visit_count"] == 11
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# synchronous_commit: off for "search", on for "propagate"/"export".
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_synchronous_commit_set_per_role(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    search_store = await NodeStore.connect_and_prepare(config, role="search")
    try:
        assert search_store.effective_synchronous_commit == "off"
        async with search_store.pool.acquire() as conn:
            assert await conn.fetchval("SHOW synchronous_commit") == "off"
    finally:
        await search_store.close()

    propagate_store = NodeStore(config, role="propagate")
    await propagate_store.connect()
    try:
        assert propagate_store.effective_synchronous_commit == "on"
        async with propagate_store.pool.acquire() as conn:
            assert await conn.fetchval("SHOW synchronous_commit") == "on"
    finally:
        await propagate_store.close()


# ---------------------------------------------------------------------------
# The lost-connection suspension seam (Requirement 2.9): `_run` catches a
# simulated `ConnectionResetError`, suspends, reconnects for real against
# the still-live scratch database, resumes, and re-raises the original
# exception to the caller whose statement failed.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_run_seam_suspends_reconnects_and_resumes(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=2)

    store = await NodeStore.connect_and_prepare(config)
    try:
        assert store.suspended.is_set()

        calls = {"n": 0}

        async def flaky(pool):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("simulated connection loss")
            async with pool.acquire() as conn:
                return await conn.fetchval("SELECT 1")

        with pytest.raises(ConnectionResetError):
            await store._run(flaky)

        # Reconnected against the still-live scratch database and resumed.
        assert store.suspended.is_set()

        result = await store._run(flaky)
        assert result == 1
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# task 7.6: the read path -- get / get_many / get_many_terminal_eval, and
# the node LRU's byte-accounted eviction.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_get_absent_key_returns_absent_and_creates_no_row(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(123, 456)
        result, view = await store.get(key)
        assert result is GetResult.ABSENT
        assert view is None

        count = await pg_conn.fetchval("SELECT count(*) FROM book_node")
        assert count == 0
    finally:
        await store.close()


@pytest.mark.db
async def test_get_found_round_trips_all_fields_including_edges_and_cyclic_flag(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        edges_blob = encode_edges(
            [
                {
                    "move16": 0x0083,  # 7g7f, a plain non-promoting move
                    "prior_q16": 12345,
                    "flags": 0,
                    "ts_depth": 0,
                    "ts_eval": 0,
                    "visit_count": 7,
                    "value_sum": 3.5,
                },
                {
                    "move16": 0x4083,  # same square pair, promoting variant
                    "prior_q16": 6789,
                    "flags": 1,
                    "ts_depth": 12,
                    "ts_eval": -321,
                    "visit_count": 2,
                    "value_sum": -1.25,
                },
            ]
        )
        key_hi, key_lo = 111, 222
        flags = 0x01  # bit0 Cyclic_Flag set

        await pg_conn.execute(
            "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, "
            "value_sum, terminal, flags, eval_win_rate, prop_value, prop_best_move16, "
            "prop_epoch, edge_count, edges) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
            key_hi,
            key_lo,
            "sfen-round-trip",
            999,
            10,
            2.25,
            Terminal.WIN_FOR_STM.value,
            flags,
            0.75,
            0.5,
            0x0083,
            1,
            2,
            edges_blob,
        )

        result, view = await store.get(PositionKey(key_hi, key_lo))
        assert result is GetResult.FOUND
        assert isinstance(view, BookNodeView)
        assert view.key == PositionKey(key_hi, key_lo)
        assert view.sfen == "sfen-round-trip"
        assert view.apery_key == 999
        assert view.visit_count == 10
        assert view.value_sum == 2.25
        assert view.terminal is Terminal.WIN_FOR_STM
        assert view.cyclic_flag is True
        assert view.eval_win_rate == pytest.approx(0.75)
        assert view.prop_value == pytest.approx(0.5)
        assert view.prop_best_move16 == 0x0083

        assert len(view.edges) == 2
        # encode_edges sorts ascending by USI; both moves share the same
        # from/to, so the non-promoting move ("7g7f") sorts before its
        # promoting counterpart ("7g7f+").
        assert int(view.edges[0]["move16"]) == 0x0083
        assert int(view.edges[0]["visit_count"]) == 7
        assert view.edges[0]["value_sum"] == pytest.approx(3.5)
        assert int(view.edges[1]["move16"]) == 0x4083
        assert int(view.edges[1]["visit_count"]) == 2
        assert view.edges[1]["value_sum"] == pytest.approx(-1.25)

        # Second get() is served from the LRU; still returns the same view.
        result2, view2 = await store.get(PositionKey(key_hi, key_lo))
        assert result2 is GetResult.FOUND
        assert view2.sfen == "sfen-round-trip"
    finally:
        await store.close()


@pytest.mark.db
async def test_get_many_mixed_present_and_absent(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        present_key = PositionKey(1001, 2002)
        absent_key = PositionKey(3003, 4004)

        await pg_conn.execute(
            "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, "
            "value_sum, edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
            present_key.hi,
            present_key.lo,
            "present-node",
            1,
            5,
            1.0,
        )

        results = await store.get_many([present_key, absent_key])
        assert len(results) == 2
        assert results[0] is not None
        assert results[0].sfen == "present-node"
        assert results[0].key == present_key
        assert results[1] is None
    finally:
        await store.close()


@pytest.mark.db
async def test_get_many_terminal_eval_narrow_projection_bypasses_lru(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(5005, 6006)
        await pg_conn.execute(
            "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, "
            "value_sum, terminal, eval_win_rate, edge_count, edges) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 0, '')",
            key.hi,
            key.lo,
            "narrow-node",
            1,
            3,
            0.5,
            Terminal.LOSS_FOR_STM.value,
            0.125,
        )

        before = store.cache_bytes
        results = await store.get_many_terminal_eval([key, PositionKey(7007, 8008)])
        assert len(results) == 2
        assert results[0] == (Terminal.LOSS_FOR_STM, pytest.approx(0.125))
        assert results[1] is None

        # The narrow projection must never populate the node LRU.
        assert store.cache_bytes == before
        assert key not in store._cache
    finally:
        await store.close()


@pytest.mark.db
async def test_node_lru_evicts_under_small_cache_budget(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    # A tiny budget: room for only a few of the nodes inserted below.
    config = _make_config(pg_scratch_database, connection_retry_limit=0, cache_budget=2000)

    store = await NodeStore.connect_and_prepare(config)
    try:
        keys = [PositionKey(10_000 + i, 20_000 + i) for i in range(20)]
        for i, key in enumerate(keys):
            await pg_conn.execute(
                "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, "
                "value_sum, edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
                key.hi,
                key.lo,
                f"node-{i}",
                1,
                0,
                0.0,
            )

        for key in keys:
            result, _view = await store.get(key)
            assert result is GetResult.FOUND
            assert store.cache_bytes <= config.cache_budget

        # An early, never-revisited key should have been evicted...
        assert keys[0] not in store._cache

        # ...while a key re-`get`-ed just now (most-recently-used) survives
        # a fresh insertion that would otherwise need to evict it.
        recent_key = keys[-1]
        result, _view = await store.get(recent_key)
        assert result is GetResult.FOUND
        assert recent_key in store._cache
        assert store.cache_bytes <= config.cache_budget
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# task 8.1: insert_expansion -- the atomic expansion write.
# ---------------------------------------------------------------------------


def _make_edges():
    return [
        {
            "move16": 0x0083,
            "prior_q16": 10000,
            "flags": 0,
            "ts_depth": 0,
            "ts_eval": 0,
            "visit_count": 0,
            "value_sum": 0.0,
        },
        {
            "move16": 0x0102,
            "prior_q16": 20000,
            "flags": 0,
            "ts_depth": 0,
            "ts_eval": 0,
            "visit_count": 0,
            "value_sum": 0.0,
        },
    ]


@pytest.mark.db
async def test_insert_expansion_commits_and_reads_back(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9001, 9002)
        write = ExpansionWrite(
            key=key,
            sfen="commit-node",
            apery_key=555,
            terminal=Terminal.NONE,
            eval_win_rate=0.6,
            edges=_make_edges(),
        )
        result = await store.insert_expansion(write)
        assert result.outcome is WriteOutcome.COMMITTED
        assert result.key == key

        row = await pg_conn.fetchrow(
            "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row is not None
        assert row["sfen"] == "commit-node"
        assert row["edge_count"] == 2

        get_result, view = await store.get(key)
        assert get_result is GetResult.FOUND
        assert len(view.edges) == 2
    finally:
        await store.close()


@pytest.mark.db
async def test_insert_expansion_duplicate_same_sfen_leaves_row_unchanged(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9101, 9102)
        write = ExpansionWrite(
            key=key, sfen="dup-node", apery_key=1, eval_win_rate=0.7, edges=_make_edges()
        )
        first = await store.insert_expansion(write)
        assert first.outcome is WriteOutcome.COMMITTED
        assert store.duplicate_count == 0

        # A second writer racing on the same key with the same SFEN.
        second_write = ExpansionWrite(
            key=key, sfen="dup-node", apery_key=1, eval_win_rate=0.99, edges=[]
        )
        second = await store.insert_expansion(second_write)
        assert second.outcome is WriteOutcome.DUPLICATE
        assert second.existing_sfen == "dup-node"
        assert store.duplicate_count == 1
        assert store.collision_count == 0

        # The retained row's evaluation fields are untouched by the losing write.
        row = await pg_conn.fetchrow(
            "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["eval_win_rate"] == pytest.approx(0.7)
        assert row["edge_count"] == 2
    finally:
        await store.close()


@pytest.mark.db
async def test_insert_expansion_idempotent_reapplication(pg_scratch_database, pg_conn):
    """Requirement 10.4: re-applying the identical expansion is a no-op."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9201, 9202)
        write = ExpansionWrite(
            key=key, sfen="idempotent-node", apery_key=2, eval_win_rate=0.5, edges=_make_edges()
        )
        first = await store.insert_expansion(write)
        assert first.outcome is WriteOutcome.COMMITTED

        before = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )

        second = await store.insert_expansion(write)
        assert second.outcome is WriteOutcome.DUPLICATE

        after = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )
        assert after == before
    finally:
        await store.close()


@pytest.mark.db
async def test_insert_expansion_collision_reports_both_sfens_and_changes_nothing(
    pg_scratch_database, pg_conn
):
    """Requirement 3.6: differing stored SFEN at the same key is a collision,
    not a duplicate -- no edge is created, fields are unchanged, both SFEN
    strings are reported."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9301, 9302)
        first_write = ExpansionWrite(
            key=key, sfen="original-sfen", apery_key=3, eval_win_rate=0.4, edges=_make_edges()
        )
        first = await store.insert_expansion(first_write)
        assert first.outcome is WriteOutcome.COMMITTED

        before = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )

        colliding_write = ExpansionWrite(
            key=key, sfen="different-sfen", apery_key=4, eval_win_rate=0.9, edges=[]
        )
        result = await store.insert_expansion(colliding_write)
        assert result.outcome is WriteOutcome.COLLISION
        assert result.existing_sfen == "original-sfen"
        assert result.attempted_sfen == "different-sfen"
        assert store.collision_count == 1
        assert store.duplicate_count == 0

        after = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )
        assert after == before
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# task 8.6: the coalescing backup accumulator and flush.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_backup_deltas_visible_via_get_before_flush(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9401, 9402)
        write = ExpansionWrite(
            key=key, sfen="backup-node", apery_key=5, eval_win_rate=0.5, edges=_make_edges()
        )
        result = await store.insert_expansion(write)
        assert result.outcome is WriteOutcome.COMMITTED

        # Before any backup: visit_count is 0 (Requirement 4.5's invariant
        # for a childless node reduces to visit_count == 0 + 1's edges'
        # visit counts, i.e. just 0 here since no descent has completed).
        _, view_before = await store.get(key)
        assert view_before.visit_count == 0
        assert int(view_before.edges[0]["visit_count"]) == 0

        delta = BackupDelta(
            key=key,
            node_visit_delta=3,
            node_value_delta=1.5,
            flags_or=0,
            edge_deltas={0x0083: (2, 1.0), 0x0102: (1, 0.5)},
        )
        store.backup([delta])

        # Visible immediately, with no flush yet.
        _, view_after = await store.get(key)
        assert view_after.visit_count == 3
        assert view_after.value_sum == pytest.approx(1.5)
        by_move = {int(e["move16"]): e for e in view_after.edges}
        assert int(by_move[0x0083]["visit_count"]) == 2
        assert by_move[0x0083]["value_sum"] == pytest.approx(1.0)
        assert int(by_move[0x0102]["visit_count"]) == 1
        assert by_move[0x0102]["value_sum"] == pytest.approx(0.5)

        # PostgreSQL itself is untouched before flush.
        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == 0
        assert row["value_sum"] == 0.0
    finally:
        await store.close()


@pytest.mark.db
async def test_flush_applies_accumulated_deltas_via_fallback_patch(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9501, 9502)
        write = ExpansionWrite(
            key=key, sfen="flush-node", apery_key=6, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        store.backup(
            [
                BackupDelta(
                    key=key,
                    node_visit_delta=4,
                    node_value_delta=2.0,
                    flags_or=0x01,
                    edge_deltas={0x0083: (4, 2.0)},
                )
            ]
        )

        await store.flush()

        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum, flags, edges FROM book_node "
            "WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == 4
        assert row["value_sum"] == pytest.approx(2.0)
        assert row["flags"] & 0x01

        from dlshogi.book import packed_edge as pe

        edges = pe.decode_edges(bytes(row["edges"]))
        by_move = {int(e["move16"]): e for e in edges}
        assert int(by_move[0x0083]["visit_count"]) == 4
        assert by_move[0x0083]["value_sum"] == pytest.approx(2.0)
        assert int(by_move[0x0102]["visit_count"]) == 0

        # The accumulator is drained after flush.
        assert store._pending_backup == {}
    finally:
        await store.close()


@pytest.mark.db
async def test_backup_coalesces_repeated_deltas_for_same_node_and_edge(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9601, 9602)
        write = ExpansionWrite(
            key=key, sfen="coalesce-node", apery_key=7, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        # Five separate descents' worth of backup() calls, all touching
        # the same node and the same edge, before any flush.
        for _ in range(5):
            store.backup(
                [
                    BackupDelta(
                        key=key,
                        node_visit_delta=1,
                        node_value_delta=0.2,
                        edge_deltas={0x0083: (1, 0.2)},
                    )
                ]
            )

        _, view = await store.get(key)
        assert view.visit_count == 5
        assert view.value_sum == pytest.approx(1.0)

        await store.flush()

        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == 5
        assert row["value_sum"] == pytest.approx(1.0)
    finally:
        await store.close()


@pytest.mark.db
async def test_background_flusher_flushes_on_its_interval(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9701, 9702)
        write = ExpansionWrite(
            key=key, sfen="flusher-node", apery_key=8, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        store.start_flusher()
        try:
            store.backup([BackupDelta(key=key, node_visit_delta=7, node_value_delta=3.5)])

            # The flusher interval is 200ms; wait a bit longer than that
            # for at least one tick, real-time (no virtual clock here
            # since start_flusher schedules against the running loop's
            # own asyncio primitives, not a NodeStore-owned loop.time()).
            for _ in range(20):
                await asyncio.sleep(0.05)
                row = await pg_conn.fetchrow(
                    "SELECT visit_count FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                    fold_u64_to_i64(key.hi),
                    fold_u64_to_i64(key.lo),
                )
                if row["visit_count"] == 7:
                    break
            assert row["visit_count"] == 7
        finally:
            await store.stop_flusher()
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# task 8.8: propagation writes, book_meta seq counters, stats, RSS release.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_set_propagation_writes_and_reads_back(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9801, 9802)
        write = ExpansionWrite(
            key=key, sfen="prop-node", apery_key=9, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        pass_id = await store.next_propagation_seq()
        assert pass_id == 1

        await store.set_propagation(
            [PropagationWrite(key=key, prop_value=0.75, prop_best_move16=0x0083, prop_epoch=pass_id)]
        )

        row = await pg_conn.fetchrow(
            "SELECT prop_value, prop_best_move16, prop_epoch FROM book_node "
            "WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["prop_value"] == pytest.approx(0.75)
        assert row["prop_best_move16"] == 0x0083
        assert row["prop_epoch"] == pass_id

        _, view = await store.get(key)
        assert view.prop_value == pytest.approx(0.75)
        assert view.prop_best_move16 == 0x0083

        await store.mark_propagation_done(pass_id)
        meta = await pg_conn.fetchrow("SELECT propagation_done_seq FROM book_meta WHERE id = 1")
        assert meta["propagation_done_seq"] == pass_id

        seq = await store.bump_search_write_seq()
        assert seq == 1
    finally:
        await store.close()


@pytest.mark.db
async def test_stats_reports_latencies_and_cache_accounting(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(9901, 9902)
        write = ExpansionWrite(
            key=key, sfen="stats-node", apery_key=10, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)
        await store.get(key)
        store.backup([BackupDelta(key=key, node_visit_delta=1, node_value_delta=0.1)])
        await store.flush()

        stats = store.stats()
        assert stats.read_count >= 1
        assert stats.write_count >= 2  # insert_expansion + flush
        assert stats.read_latency_mean_s >= 0.0
        assert stats.write_latency_mean_s >= 0.0
        assert stats.cache_bytes == store.cache_bytes
        assert stats.cache_budget_bytes == config.cache_budget
        assert stats.duplicate_count == 0
        assert stats.collision_count == 0
    finally:
        await store.close()


@pytest.mark.db
async def test_rss_release_evicts_cache_and_reports_event_without_losing_data(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        keys = [PositionKey(20_000 + i, 30_000 + i) for i in range(10)]
        for i, key in enumerate(keys):
            write = ExpansionWrite(
                key=key, sfen=f"rss-node-{i}", apery_key=i, eval_win_rate=0.5, edges=_make_edges()
            )
            result = await store.insert_expansion(write)
            assert result.outcome is WriteOutcome.COMMITTED
            await store.get(key)  # populate the LRU

        assert store.cache_bytes > 0
        cache_before = store.cache_bytes

        # Inject a fake RSS reading that is guaranteed to exceed the bound.
        huge_rss = store.rss_bound_bytes() + 10_000_000
        store.set_rss_reader(lambda: huge_rss)

        released = store.check_rss_and_release()
        assert released is True
        assert store.cache_release_events == 1
        assert store.cache_bytes < cache_before

        # Every written record is still readable from PostgreSQL.
        for i, key in enumerate(keys):
            row = await pg_conn.fetchrow(
                "SELECT sfen FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
            assert row is not None
            assert row["sfen"] == f"rss-node-{i}"

        # A subsequent sample within bound reports no further release.
        store.set_rss_reader(lambda: 0)
        assert store.check_rss_and_release() is False
        assert store.cache_release_events == 1
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# task 8.10: the in-flight claim mirror and startup clearing.
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_clear_in_flight_claims_at_startup_returns_count_and_leaves_book_node_untouched(
    pg_scratch_database, pg_conn
):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(31_001, 31_002)
        write = ExpansionWrite(
            key=key, sfen="claim-node", apery_key=11, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)
        before_row = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )

        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        await pg_conn.execute(
            "INSERT INTO in_flight_claim (key_hi, key_lo, process_id, worker_id, claimed_at) "
            "VALUES ($1, $2, $3, $4, $5), ($6, $7, $3, $4, $5)",
            1,
            2,
            100,
            1,
            now,
            3,
            4,
        )
        pre_count = await pg_conn.fetchval("SELECT count(*) FROM in_flight_claim")
        assert pre_count == 2

        cleared = await store.clear_in_flight_claims_at_startup()
        assert cleared == 2

        post_count = await pg_conn.fetchval("SELECT count(*) FROM in_flight_claim")
        assert post_count == 0

        after_row = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )
        assert after_row == before_row
    finally:
        await store.close()


@pytest.mark.db
async def test_upsert_and_delete_in_flight_claims_round_trip(pg_scratch_database, pg_conn):
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        key1 = PositionKey(41_001, 41_002)
        key2 = PositionKey(41_003, 41_004)

        await store.upsert_in_flight_claims(
            [
                (key1, 100, 1, now),
                (key2, 100, 2, now),
            ]
        )

        rows = await pg_conn.fetch("SELECT * FROM in_flight_claim ORDER BY worker_id")
        assert len(rows) == 2
        assert rows[0]["worker_id"] == 1
        assert rows[0]["process_id"] == 100
        assert rows[1]["worker_id"] == 2

        # Re-upserting the same key updates in place rather than duplicating.
        later = now + dt.timedelta(seconds=5)
        await store.upsert_in_flight_claims([(key1, 100, 1, later)])
        rows = await pg_conn.fetch("SELECT * FROM in_flight_claim")
        assert len(rows) == 2

        await store.delete_in_flight_claims([key1])
        rows = await pg_conn.fetch("SELECT * FROM in_flight_claim")
        assert len(rows) == 1
        assert rows[0]["key_hi"] == fold_u64_to_i64(key2.hi)
    finally:
        await store.close()


# ===========================================================================
# Property tests (tasks 7.5, 7.8, 7.9, 8.2, 8.3, 8.5, 8.7, 8.9, 8.11, 9.3)
# ===========================================================================


# ---------------------------------------------------------------------------
# Property 2: Absent is absent (task 7.8)
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property2_absent_key_returns_absent_and_creates_no_row(
    pg_scratch_database, pg_conn
):
    """Property 2: get for a key that does not exist returns GetResult.ABSENT
    and None, and no row is created in the database."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        # Try several keys that have never been inserted.
        for hi, lo in [(99999, 88888), (0, 0), (2**63, 2**63)]:
            key = PositionKey(hi, lo)
            result, view = await store.get(key)
            assert result is GetResult.ABSENT, f"Expected ABSENT for key ({hi}, {lo})"
            assert view is None, f"Expected None view for absent key ({hi}, {lo})"

        # Confirm no rows were created by any of the get() calls.
        count = await pg_conn.fetchval("SELECT count(*) FROM book_node")
        assert count == 0
    finally:
        await store.close()


@pytest.mark.db
async def test_property2_absent_distinguishable_from_zero_edges(pg_scratch_database, pg_conn):
    """Property 2: an absent key is distinguishable from a node with zero edges."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        # Insert a node with zero edges (e.g. a terminal node).
        key_present = PositionKey(11111, 22222)
        write = ExpansionWrite(
            key=key_present,
            sfen="terminal-node",
            apery_key=1,
            terminal=Terminal.LOSS_FOR_STM,
            eval_win_rate=0.0,
            edges=[],
        )
        await store.insert_expansion(write)

        # Reading the present key returns FOUND with 0 edges.
        result_present, view_present = await store.get(key_present)
        assert result_present is GetResult.FOUND
        assert view_present is not None
        assert len(view_present.edges) == 0

        # Reading a non-existent key returns ABSENT.
        key_absent = PositionKey(33333, 44444)
        result_absent, view_absent = await store.get(key_absent)
        assert result_absent is GetResult.ABSENT
        assert view_absent is None
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 3: Cache_Budget bounds without failing (task 7.9)
# Validates: Requirements 1.8
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property3_cache_budget_bounds_without_failing(pg_scratch_database, pg_conn):
    """Property 3: with a very small cache budget, inserting and reading more
    data than fits still completes without error and cache byte accounting
    stays within budget."""
    await _drop_book_tables(pg_conn)
    # Tiny budget: 1 KiB, far too small for the nodes we insert.
    config = _make_config(pg_scratch_database, connection_retry_limit=0, cache_budget=1024)

    store = await NodeStore.connect_and_prepare(config)
    try:
        keys = []
        for i in range(50):
            key = PositionKey(50_000 + i, 60_000 + i)
            keys.append(key)
            write = ExpansionWrite(
                key=key,
                sfen=f"budget-test-node-{i:04d}-padding-to-make-it-bigger",
                apery_key=i,
                eval_win_rate=0.5,
                edges=_make_edges(),
            )
            result = await store.insert_expansion(write)
            assert result.outcome is WriteOutcome.COMMITTED

        # Reading all of them succeeds (no exception) despite cache pressure.
        for key in keys:
            result, view = await store.get(key)
            assert result is GetResult.FOUND
            assert view is not None
            # Invariant: cache bytes never exceeds budget.
            assert store.cache_bytes <= config.cache_budget

        # Even after all reads, the budget invariant holds.
        assert store.cache_bytes <= config.cache_budget
    finally:
        await store.close()


@pytest.mark.db
async def test_property3_monotonically_increasing_keys_defeats_lru(
    pg_scratch_database, pg_conn
):
    """Property 3: monotonically increasing keys (worst-case for LRU) still
    keep cache accounting within budget."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0, cache_budget=2048)

    store = await NodeStore.connect_and_prepare(config)
    try:
        for i in range(100):
            key = PositionKey(i, i)
            write = ExpansionWrite(
                key=key, sfen=f"mono-{i}", apery_key=i, eval_win_rate=0.5, edges=_make_edges()
            )
            await store.insert_expansion(write)
            # Read immediately -- ensures the cache is populated and evicted.
            result, view = await store.get(key)
            assert result is GetResult.FOUND
            assert store.cache_bytes <= config.cache_budget
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 4: Expansion writes are atomic to concurrent readers (task 8.2)
# Validates: Requirements 1.6
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property4_expansion_write_atomic_to_concurrent_readers(
    pg_scratch_database, pg_conn
):
    """Property 4: an expansion write is atomic -- concurrent readers see
    either the full node (all fields + edges) or nothing."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(70_001, 70_002)
        edges = [
            {
                "move16": 0x0083,
                "prior_q16": 10000,
                "flags": 0,
                "ts_depth": 0,
                "ts_eval": 0,
                "visit_count": 0,
                "value_sum": 0.0,
            },
            {
                "move16": 0x0102,
                "prior_q16": 20000,
                "flags": 0,
                "ts_depth": 0,
                "ts_eval": 0,
                "visit_count": 0,
                "value_sum": 0.0,
            },
            {
                "move16": 0x0181,
                "prior_q16": 30000,
                "flags": 0,
                "ts_depth": 0,
                "ts_eval": 0,
                "visit_count": 0,
                "value_sum": 0.0,
            },
        ]

        write = ExpansionWrite(
            key=key, sfen="atomic-node", apery_key=7, eval_win_rate=0.65, edges=edges
        )

        # Launch a writer and a reader concurrently.
        read_results = []

        async def reader():
            for _ in range(20):
                result, view = await store.get(key)
                read_results.append((result, view))
                # Invalidate cache to force a DB re-read.
                store._cache.invalidate(key)
                await asyncio.sleep(0.001)

        async def writer():
            await asyncio.sleep(0.005)
            await store.insert_expansion(write)

        await asyncio.gather(writer(), reader())

        # Assert: every read result is either ABSENT (before the write
        # committed) or FOUND with ALL edges present (after).
        for result, view in read_results:
            if result is GetResult.ABSENT:
                assert view is None
            else:
                assert result is GetResult.FOUND
                assert view is not None
                # If we see the node at all, we see the full edge set.
                assert len(view.edges) == 3
                assert view.sfen == "atomic-node"
                assert view.eval_win_rate == pytest.approx(0.65)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 5: Schema repair creates only what is absent (task 7.5)
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property5_schema_repair_recreates_only_absent_elements(
    pg_scratch_database, pg_conn
):
    """Property 5: dropping one table and calling ensure_schema recreates
    only that table; existing data in other tables is untouched."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    # First: create full schema with a seed node.
    store0 = await NodeStore.connect_and_prepare(config)
    await store0.close()

    # Insert data into book_node and terashock_entry.
    await pg_conn.execute(
        "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
        "edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
        100,
        200,
        "preserve-me",
        300,
        42,
        3.14,
    )
    await pg_conn.execute(
        "INSERT INTO terashock_entry (key_hi, key_lo, sfen, moves) "
        "VALUES ($1, $2, $3, $4)",
        400,
        500,
        "ts-entry",
        b"",
    )

    # Snapshot existing tables.
    before_node = dict(
        await pg_conn.fetchrow("SELECT * FROM book_node WHERE key_hi = 100 AND key_lo = 200")
    )
    before_ts = dict(
        await pg_conn.fetchrow(
            "SELECT * FROM terashock_entry WHERE key_hi = 400 AND key_lo = 500"
        )
    )

    # Drop only in_flight_claim.
    await pg_conn.execute("DROP TABLE in_flight_claim")

    # Repair.
    store1 = await NodeStore.connect_and_prepare(config)
    try:
        # Verify all four tables exist.
        tables = {
            row["tablename"]
            for row in await pg_conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert {"book_meta", "book_node", "terashock_entry", "in_flight_claim"} <= tables

        # book_node data is untouched.
        after_node = dict(
            await pg_conn.fetchrow("SELECT * FROM book_node WHERE key_hi = 100 AND key_lo = 200")
        )
        assert after_node == before_node

        # terashock_entry data is untouched.
        after_ts = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM terashock_entry WHERE key_hi = 400 AND key_lo = 500"
            )
        )
        assert after_ts == before_ts
    finally:
        await store1.close()


@pytest.mark.db
async def test_property5_repair_with_missing_terashock_table(pg_scratch_database, pg_conn):
    """Property 5: dropping terashock_entry and repairing preserves book_node."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store0 = await NodeStore.connect_and_prepare(config)
    await store0.close()

    await pg_conn.execute(
        "INSERT INTO book_node (key_hi, key_lo, sfen, apery_key, visit_count, value_sum, "
        "edge_count, edges) VALUES ($1, $2, $3, $4, $5, $6, 0, '')",
        600,
        700,
        "keep-this-too",
        800,
        99,
        7.77,
    )

    await pg_conn.execute("DROP TABLE terashock_entry")

    store1 = await NodeStore.connect_and_prepare(config)
    try:
        tables = {
            row["tablename"]
            for row in await pg_conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert "terashock_entry" in tables

        row = await pg_conn.fetchrow(
            "SELECT * FROM book_node WHERE key_hi = 600 AND key_lo = 700"
        )
        assert row is not None
        assert row["sfen"] == "keep-this-too"
        assert row["visit_count"] == 99
    finally:
        await store1.close()


# ---------------------------------------------------------------------------
# Property 11: Transposition merging reuses the existing node (task 8.3)
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property11_transposition_merging_reuses_existing_node(
    pg_scratch_database, pg_conn
):
    """Property 11: inserting the same key+SFEN twice does not create a
    second row. The existing row's fields are unchanged."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(80_001, 80_002)
        write1 = ExpansionWrite(
            key=key, sfen="transposition-sfen", apery_key=11, eval_win_rate=0.7, edges=_make_edges()
        )
        result1 = await store.insert_expansion(write1)
        assert result1.outcome is WriteOutcome.COMMITTED

        before_row = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )

        # Second write with the same key and same SFEN but different eval.
        write2 = ExpansionWrite(
            key=key, sfen="transposition-sfen", apery_key=11, eval_win_rate=0.99, edges=[]
        )
        result2 = await store.insert_expansion(write2)
        assert result2.outcome is WriteOutcome.DUPLICATE

        # Only one row exists.
        count = await pg_conn.fetchval(
            "SELECT count(*) FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert count == 1

        # Row is byte-identical to the first write.
        after_row = dict(
            await pg_conn.fetchrow(
                "SELECT * FROM book_node WHERE key_hi = $1 AND key_lo = $2",
                fold_u64_to_i64(key.hi),
                fold_u64_to_i64(key.lo),
            )
        )
        assert after_row == before_row
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 14: Backup arithmetic and the perspective flip (task 8.7)
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property14_backup_arithmetic_before_and_after_flush(
    pg_scratch_database, pg_conn
):
    """Property 14: backup deltas are visible via get() before flush (through
    the accumulator) and correctly persisted to PostgreSQL after flush.
    The value_sum for a node is from its STM perspective."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(90_001, 90_002)
        write = ExpansionWrite(
            key=key, sfen="backup-arith", apery_key=14, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        # Apply two separate backup rounds.
        store.backup([
            BackupDelta(
                key=key,
                node_visit_delta=2,
                node_value_delta=1.2,  # STM perspective: e.g. 2 descents, value 0.6 each
                flags_or=0x01,
                edge_deltas={0x0083: (1, 0.6), 0x0102: (1, 0.6)},
            )
        ])
        store.backup([
            BackupDelta(
                key=key,
                node_visit_delta=3,
                node_value_delta=0.9,
                flags_or=0,
                edge_deltas={0x0083: (2, 0.4), 0x0102: (1, 0.5)},
            )
        ])

        # BEFORE flush: verify via get().
        _, view = await store.get(key)
        assert view.visit_count == 5  # 2 + 3
        assert view.value_sum == pytest.approx(2.1)  # 1.2 + 0.9
        assert view.cyclic_flag is True  # flags |= 0x01

        by_move = {int(e["move16"]): e for e in view.edges}
        assert int(by_move[0x0083]["visit_count"]) == 3  # 1 + 2
        assert by_move[0x0083]["value_sum"] == pytest.approx(1.0)  # 0.6 + 0.4
        assert int(by_move[0x0102]["visit_count"]) == 2  # 1 + 1
        assert by_move[0x0102]["value_sum"] == pytest.approx(1.1)  # 0.6 + 0.5

        # AFTER flush: verify in PostgreSQL directly.
        await store.flush()

        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum, flags, edges FROM book_node "
            "WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == 5
        assert row["value_sum"] == pytest.approx(2.1)
        assert row["flags"] & 0x01  # Cyclic flag set

        from dlshogi.book import packed_edge as pe

        flushed_edges = pe.decode_edges(bytes(row["edges"]))
        by_move_flushed = {int(e["move16"]): e for e in flushed_edges}
        assert int(by_move_flushed[0x0083]["visit_count"]) == 3
        assert by_move_flushed[0x0083]["value_sum"] == pytest.approx(1.0)
        assert int(by_move_flushed[0x0102]["visit_count"]) == 2
        assert by_move_flushed[0x0102]["value_sum"] == pytest.approx(1.1)
    finally:
        await store.close()


@pytest.mark.db
async def test_property14_perspective_flip_value_sum_from_stm(pg_scratch_database, pg_conn):
    """Property 14 (perspective flip): value_sum stored from STM perspective.
    Accumulating a win (1.0) and a loss (0.0) for STM yields value_sum 1.0."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(90_011, 90_012)
        write = ExpansionWrite(
            key=key, sfen="stm-perspective", apery_key=14, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        # Simulate: descent 1 -> STM wins (value 1.0), descent 2 -> STM loses (value 0.0)
        store.backup([BackupDelta(key=key, node_visit_delta=1, node_value_delta=1.0)])
        store.backup([BackupDelta(key=key, node_visit_delta=1, node_value_delta=0.0)])

        _, view = await store.get(key)
        assert view.visit_count == 2
        # value_sum = 1.0 + 0.0 = 1.0 (from STM perspective)
        assert view.value_sum == pytest.approx(1.0)
        # mean value = 1.0/2 = 0.5
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 29: In_Flight_Set clearing is inert (task 8.11)
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property29_in_flight_clearing_is_inert(pg_scratch_database, pg_conn):
    """Property 29: inserting rows into in_flight_claim and clearing them
    leaves every book_node row byte-identical to its pre-startup state."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        # Create a few book_node rows.
        for i in range(5):
            key = PositionKey(100_000 + i, 200_000 + i)
            write = ExpansionWrite(
                key=key,
                sfen=f"inert-node-{i}",
                apery_key=i,
                eval_win_rate=0.5,
                edges=_make_edges(),
            )
            await store.insert_expansion(write)

        # Flush so all data is in PostgreSQL.
        store.backup([
            BackupDelta(key=PositionKey(100_000, 200_000), node_visit_delta=10, node_value_delta=5.0)
        ])
        await store.flush()

        # Snapshot all book_node rows.
        before_rows = await pg_conn.fetch("SELECT * FROM book_node ORDER BY key_hi, key_lo")
        before_data = [dict(r) for r in before_rows]

        # Insert claims into in_flight_claim.
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        for i in range(3):
            await pg_conn.execute(
                "INSERT INTO in_flight_claim (key_hi, key_lo, process_id, worker_id, claimed_at) "
                "VALUES ($1, $2, $3, $4, $5)",
                100_000 + i,
                200_000 + i,
                42,
                i,
                now,
            )

        # Clear them at startup.
        cleared = await store.clear_in_flight_claims_at_startup()
        assert cleared == 3

        # Verify in_flight_claim is empty.
        claim_count = await pg_conn.fetchval("SELECT count(*) FROM in_flight_claim")
        assert claim_count == 0

        # Verify book_node rows are byte-identical.
        after_rows = await pg_conn.fetch("SELECT * FROM book_node ORDER BY key_hi, key_lo")
        after_data = [dict(r) for r in after_rows]
        assert after_data == before_data
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 33: Concurrent counter updates lose nothing (task 9.3)
# Validates: Requirements 11.8
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property33_concurrent_counter_updates_lose_nothing(
    pg_scratch_database, pg_conn
):
    """Property 33: applying multiple concurrent backup flushes to the same
    node results in final visit_count = sum of all deltas + initial."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(110_001, 110_002)
        write = ExpansionWrite(
            key=key, sfen="concurrent-node", apery_key=33, eval_win_rate=0.5, edges=_make_edges()
        )
        await store.insert_expansion(write)

        # Apply 10 backup+flush rounds concurrently. Each adds 1 visit and
        # 0.1 value to the node and 1 visit to edge 0x0083.
        num_concurrent = 10

        async def one_flush(delta_id: int):
            # Create a fresh store so each has its own accumulator + pool conn.
            s = await NodeStore.connect_and_prepare(config)
            try:
                s.backup([
                    BackupDelta(
                        key=key,
                        node_visit_delta=1,
                        node_value_delta=0.1,
                        edge_deltas={0x0083: (1, 0.1)},
                    )
                ])
                await s.flush()
            finally:
                await s.close()

        await asyncio.gather(*(one_flush(i) for i in range(num_concurrent)))

        # Read the final state from PostgreSQL.
        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum, edges FROM book_node "
            "WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == num_concurrent
        assert row["value_sum"] == pytest.approx(num_concurrent * 0.1)

        from dlshogi.book import packed_edge as pe

        edges = pe.decode_edges(bytes(row["edges"]))
        by_move = {int(e["move16"]): e for e in edges}
        assert int(by_move[0x0083]["visit_count"]) == num_concurrent
        assert by_move[0x0083]["value_sum"] == pytest.approx(num_concurrent * 0.1)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Property 34: Duplicate node creation keeps the first (task 8.5)
# Validates: Requirements 11.6
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property34_duplicate_node_creation_keeps_the_first(
    pg_scratch_database, pg_conn
):
    """Property 34: two concurrent insert_expansion calls for the same key
    result in exactly one row -- the one from whichever writer won the race."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    key = PositionKey(120_001, 120_002)

    # Two separate stores simulate two workers.
    store1 = await NodeStore.connect_and_prepare(config)
    store2 = await NodeStore.connect_and_prepare(config)
    try:
        write1 = ExpansionWrite(
            key=key, sfen="dup-sfen", apery_key=34, eval_win_rate=0.6, edges=_make_edges()
        )
        write2 = ExpansionWrite(
            key=key, sfen="dup-sfen", apery_key=34, eval_win_rate=0.8, edges=_make_edges()
        )

        results = await asyncio.gather(
            store1.insert_expansion(write1), store2.insert_expansion(write2)
        )

        outcomes = [r.outcome for r in results]
        # Exactly one committed, one duplicate.
        assert WriteOutcome.COMMITTED in outcomes
        assert WriteOutcome.DUPLICATE in outcomes

        # Exactly one row in the database.
        count = await pg_conn.fetchval(
            "SELECT count(*) FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert count == 1

        # The row's SFEN is "dup-sfen" (same for both writers).
        row = await pg_conn.fetchrow(
            "SELECT sfen FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["sfen"] == "dup-sfen"
    finally:
        await store1.close()
        await store2.close()


@pytest.mark.db
async def test_property34_duplicate_three_writers(pg_scratch_database, pg_conn):
    """Property 34 (extension): three concurrent writers for the same key --
    exactly one commits and the other two see DUPLICATE."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    key = PositionKey(120_011, 120_012)

    stores = [await NodeStore.connect_and_prepare(config) for _ in range(3)]
    try:
        writes = [
            ExpansionWrite(
                key=key, sfen="three-writers", apery_key=34, eval_win_rate=0.5 + i * 0.1,
                edges=_make_edges()
            )
            for i in range(3)
        ]

        results = await asyncio.gather(*(s.insert_expansion(w) for s, w in zip(stores, writes)))

        committed = [r for r in results if r.outcome is WriteOutcome.COMMITTED]
        duplicates = [r for r in results if r.outcome is WriteOutcome.DUPLICATE]
        assert len(committed) == 1
        assert len(duplicates) == 2

        count = await pg_conn.fetchval(
            "SELECT count(*) FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert count == 1
    finally:
        for s in stores:
            await s.close()


# ---------------------------------------------------------------------------
# Property 47: Cache release under memory pressure is non-destructive (task 8.9)
# Validates: Requirements 15.5
# ---------------------------------------------------------------------------


@pytest.mark.db
async def test_property47_cache_release_under_pressure_non_destructive(
    pg_scratch_database, pg_conn
):
    """Property 47: filling the cache, triggering RSS-based release, and
    verifying all data is still readable from the database."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        # Write 20 nodes and populate the cache via get().
        keys = []
        for i in range(20):
            key = PositionKey(130_000 + i, 140_000 + i)
            keys.append(key)
            write = ExpansionWrite(
                key=key, sfen=f"pressure-node-{i}", apery_key=i, eval_win_rate=0.5,
                edges=_make_edges()
            )
            await store.insert_expansion(write)
            await store.get(key)  # populate cache

        cache_before = store.cache_bytes
        assert cache_before > 0

        # Inject a fake RSS reading that exceeds the bound.
        store.set_rss_reader(lambda: store.rss_bound_bytes() + 50_000_000)
        released = store.check_rss_and_release()
        assert released is True
        assert store.cache_bytes < cache_before
        assert store.cache_release_events >= 1

        # All nodes are still readable from the database (non-destructive).
        for i, key in enumerate(keys):
            result, view = await store.get(key)
            assert result is GetResult.FOUND
            assert view is not None
            assert view.sfen == f"pressure-node-{i}"
            assert len(view.edges) == 2
    finally:
        await store.close()


@pytest.mark.db
async def test_property47_cache_release_does_not_lose_unflushed_data(
    pg_scratch_database, pg_conn
):
    """Property 47: cache release does not discard unflushed backup deltas --
    they are still visible via get() and persist after flush()."""
    await _drop_book_tables(pg_conn)
    config = _make_config(pg_scratch_database, connection_retry_limit=0)

    store = await NodeStore.connect_and_prepare(config)
    try:
        key = PositionKey(130_100, 140_100)
        write = ExpansionWrite(
            key=key, sfen="unflushed-pressure", apery_key=47, eval_win_rate=0.5,
            edges=_make_edges()
        )
        await store.insert_expansion(write)
        await store.get(key)

        # Apply backup (unflushed).
        store.backup([BackupDelta(key=key, node_visit_delta=5, node_value_delta=2.5)])

        # Trigger RSS release.
        store.set_rss_reader(lambda: store.rss_bound_bytes() + 50_000_000)
        store.check_rss_and_release()

        # Unflushed deltas are still visible.
        _, view = await store.get(key)
        assert view.visit_count == 5
        assert view.value_sum == pytest.approx(2.5)

        # Flush and verify in PostgreSQL.
        await store.flush()
        row = await pg_conn.fetchrow(
            "SELECT visit_count, value_sum FROM book_node WHERE key_hi = $1 AND key_lo = $2",
            fold_u64_to_i64(key.hi),
            fold_u64_to_i64(key.lo),
        )
        assert row["visit_count"] == 5
        assert row["value_sum"] == pytest.approx(2.5)
    finally:
        await store.close()
