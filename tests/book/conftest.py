"""Shared pytest fixtures for the ``tests/book`` property and unit test suite.

Three fixtures are provided here:

- ``pg_scratch_database`` (session-scoped): creates a scratch PostgreSQL
  database named ``puct_test_<pid>`` and drops it at session teardown.
  Skips gracefully, via ``pytest.skip``, when no PostgreSQL server is
  reachable, so ``pytest -m "not db"`` never needs a database and a run
  without PostgreSQL available does not error out for the ``db``-marked
  tests either -- they are reported as skipped.
- ``pg_conn`` / ``truncate_and_repopulate`` (function-scoped): a per-test
  connection to the scratch database and an async callable that truncates
  every table currently in the scratch database's ``public`` schema and
  optionally repopulates it. Properties that touch PostgreSQL truncate and
  repopulate once per hypothesis example (not once per test function), so
  the callable returned by ``truncate_and_repopulate`` is meant to be
  awaited from inside the ``@given``-decorated test body, once per example.
- ``virtual_clock_loop`` (function-scoped): an event loop whose ``time()``
  the test advances explicitly via ``loop.run_until_complete(loop.advance(...))``,
  so properties whose statements involve durations (Batch_Timeout, the
  300-second claim reaper, Report_Interval, ...) cost nothing to test and
  never sleep on the wall clock.

The connection settings used here are read directly from the standard
``PGHOST`` / ``PGPORT`` / ``PGUSER`` / ``PGPASSWORD`` / ``PGDATABASE``
environment variables (the same ones ``libpq`` and ``asyncpg`` read),
falling back to the local Unix-socket defaults. This harness is
deliberately independent of ``dlshogi.book.config.BookConfig``, which does
not exist yet as a real implementation (it is added by task 4.1): the test
harness must not depend on the very component the later property tests
exercise.
"""

from __future__ import annotations

import asyncio
import getpass
import heapq
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import asyncpg
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------

# Repository root, three levels up from tests/book/conftest.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_SQL_PATH = _REPO_ROOT / "dlshogi" / "book" / "sql" / "schema.sql"


@dataclass(frozen=True)
class PgConnectInfo:
    """Connection settings for one PostgreSQL database."""

    host: str
    port: int
    user: str
    password: Optional[str]
    database: str

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "database": self.database,
        }
        if self.password:
            kwargs["password"] = self.password
        return kwargs


def _admin_connect_info() -> PgConnectInfo:
    """Connection settings for the maintenance database (usually 'postgres').

    Used only to CREATE DATABASE / DROP DATABASE the scratch database; every
    other operation targets the scratch database itself.
    """
    return PgConnectInfo(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", getpass.getuser()),
        password=os.environ.get("PGPASSWORD"),
        database=os.environ.get("PGDATABASE", "postgres"),
    )


# ---------------------------------------------------------------------------
# Session-scoped scratch database
# ---------------------------------------------------------------------------


async def _create_database(admin_info: PgConnectInfo, db_name: str) -> None:
    conn = await asyncpg.connect(**admin_info.as_kwargs())
    try:
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()


async def _drop_database(admin_info: PgConnectInfo, db_name: str) -> None:
    conn = await asyncpg.connect(**admin_info.as_kwargs())
    try:
        # Terminate any lingering backends before dropping, so a connection
        # left open by a failed test does not turn the drop into a hang.
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid()",
            db_name,
        )
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await conn.close()


async def _apply_schema_if_present(scratch_info: PgConnectInfo) -> None:
    """Apply dlshogi/book/sql/schema.sql to the scratch database, once.

    Silently does nothing when the file does not yet exist or contains no
    SQL statements (task 7.1, which writes the real DDL, has not run yet in
    a fresh checkout of this task). This keeps the fixture usable both
    before and after schema.sql exists.
    """
    if not _SCHEMA_SQL_PATH.exists():
        return
    sql_text = _SCHEMA_SQL_PATH.read_text(encoding="utf-8").strip()
    if not sql_text:
        return
    conn = await asyncpg.connect(**scratch_info.as_kwargs())
    try:
        await conn.execute(sql_text)
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def pg_scratch_database():
    """Create ``puct_test_<pid>``, apply schema.sql if present, drop at teardown.

    Skips every test that requests this fixture (directly or through
    ``pg_conn`` / ``truncate_and_repopulate``) when no PostgreSQL server is
    reachable with the configured connection settings.
    """
    admin_info = _admin_connect_info()
    db_name = f"puct_test_{os.getpid()}"

    try:
        asyncio.run(_create_database(admin_info, db_name))
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL is not available for db-marked tests: {exc}")
        return

    scratch_info = replace(admin_info, database=db_name)

    try:
        asyncio.run(_apply_schema_if_present(scratch_info))
    except asyncpg.PostgresError as exc:
        asyncio.run(_drop_database(admin_info, db_name))
        pytest.fail(f"failed to apply dlshogi/book/sql/schema.sql: {exc}")
        return

    yield scratch_info

    asyncio.run(_drop_database(admin_info, db_name))


# ---------------------------------------------------------------------------
# Per-test connection, and the per-example truncate-and-repopulate helper
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pg_conn(pg_scratch_database: PgConnectInfo):
    """One asyncpg connection to the scratch database, closed after the test."""
    conn = await asyncpg.connect(**pg_scratch_database.as_kwargs())
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def truncate_and_repopulate(
    pg_conn: asyncpg.Connection,
) -> Callable[[Optional[Callable[[asyncpg.Connection], Awaitable[None]]]], Awaitable[None]]:
    """Return an async ``reset(populate=None)`` callable.

    ``await reset()`` truncates every table in the scratch database's
    ``public`` schema (``RESTART IDENTITY CASCADE``, so serial columns and
    foreign-key-linked tables reset cleanly too). ``await reset(populate)``
    additionally calls ``await populate(pg_conn)`` afterwards to repopulate
    example-specific seed rows.

    Property tests that touch PostgreSQL call this once per hypothesis
    example from inside their ``@given``-decorated test body, rather than
    relying on a fixture teardown/setup cycle that hypothesis would not
    trigger between examples.
    """

    async def reset(
        populate: Optional[Callable[[asyncpg.Connection], Awaitable[None]]] = None,
    ) -> None:
        rows = await pg_conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        if rows:
            table_list = ", ".join(f'"{row["tablename"]}"' for row in rows)
            await pg_conn.execute(
                f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"
            )
        if populate is not None:
            await populate(pg_conn)

    return reset


# ---------------------------------------------------------------------------
# Virtual-clock event loop
# ---------------------------------------------------------------------------


class VirtualClockEventLoop(asyncio.SelectorEventLoop):
    """An event loop whose ``time()`` is a manually advanced virtual clock.

    ``asyncio.BaseEventLoop.time()`` normally returns ``time.monotonic()``.
    This subclass returns an internal counter instead, so any code that
    schedules work relative to ``loop.time()`` -- ``loop.call_later``,
    ``asyncio.sleep``, ``asyncio.wait_for`` -- can be driven deterministically
    by a test through ``advance()``, with no wall-clock sleeping. This is
    what lets a test exercise a 1000 ms Batch_Timeout, a 300 s claim reaper,
    or an 86,400 s Throughput_Grace_Period in well under a second.

    Usage from a synchronous test or a ``hypothesis.stateful`` rule:

        loop = VirtualClockEventLoop()
        loop.run_until_complete(loop.advance(0.5))   # advance 500 ms
    """

    def __init__(self, *, selector=None) -> None:
        super().__init__(selector=selector)
        self._virtual_time = 0.0

    def time(self) -> float:
        return self._virtual_time

    async def advance(self, seconds: float) -> None:
        """Move the virtual clock forward by ``seconds``, firing due timers.

        Advances in steps rather than one jump, so a timer that itself
        schedules a further ``call_later`` (a chain of retries, for
        instance) also becomes due within the same ``advance()`` call if it
        falls within the requested window.
        """
        if seconds < 0:
            raise ValueError("virtual clock cannot move backwards")
        target = self._virtual_time + seconds
        while True:
            while self._scheduled and self._scheduled[0]._cancelled:
                heapq.heappop(self._scheduled)
            if not self._scheduled or self._scheduled[0]._when > target:
                self._virtual_time = target
                break
            self._virtual_time = self._scheduled[0]._when
            # Yield to the loop so _run_once() moves the now-due handle(s)
            # from _scheduled to _ready and runs them.
            await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.fixture
def virtual_clock_loop():
    """A fresh ``VirtualClockEventLoop``, closed after the test.

    Deliberately a plain (synchronous) fixture rather than a
    ``pytest_asyncio`` fixture: tests that need the virtual clock drive this
    loop explicitly with ``loop.run_until_complete(...)``, including for the
    ``advance()`` calls themselves, instead of running under the per-test
    real-time loop that ``asyncio_mode = auto`` would otherwise supply.
    """
    loop = VirtualClockEventLoop()
    try:
        yield loop
    finally:
        loop.close()
