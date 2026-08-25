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
