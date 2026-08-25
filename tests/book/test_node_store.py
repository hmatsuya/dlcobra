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
    BookNodeView,
    GetResult,
    NodeStore,
    NodeStoreConnectionError,
    SchemaVersionMismatchError,
    Terminal,
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
