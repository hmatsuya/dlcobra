"""Regression tests for review finding F-14 (dev-loop 20261020-dlcobra-ce5a).

``flush()`` found no ``book_node`` row for the backup delta of every descent
that ended at Max_Book_Ply (such leaves are deliberately not persisted, F-16)
and logged a WARNING for each one: 900 lines in a 1500-descent run, burying
the one case that would be a real fault (a delta with edge deltas, i.e. a node
the descent selected an edge from, so it must have had a row).
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from dlshogi.book.keys import position_key_from_sfen
from dlshogi.book.node_store import BackupDelta, NodeStore
from dlshogi.book.packed_edge import decode_edges
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _drop_book_tables,
    _make_config,
    _run_small_search,
)

_LOGGER = "dlshogi.book.node_store"
_KEY = position_key_from_sfen(_INITIAL_SFEN)


def _rowless_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER and r.levelno >= logging.WARNING and "no book_node row" in r.getMessage()
    ]


class TestNoteRowlessDelta:
    """The accounting rule, without a database."""

    def _store(self, pg_info):
        return NodeStore(_make_config(pg_info, root_sfen=_INITIAL_SFEN), role="search")

    def test_leaf_only_delta_is_counted_quietly(self, pg_scratch_database, caplog):
        store = self._store(pg_scratch_database)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            store._note_rowless_delta(BackupDelta(key=_KEY, node_visit_delta=1, node_value_delta=0.5))
        assert store.rowless_leaf_deltas == 1
        assert store.rowless_deltas_dropped == 0
        assert _rowless_warnings(caplog) == []

    def test_delta_with_edges_is_a_warning(self, pg_scratch_database, caplog):
        store = self._store(pg_scratch_database)
        delta = BackupDelta(key=_KEY, node_visit_delta=1, node_value_delta=0.5, edge_deltas={7: (1, 0.5)})
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            store._note_rowless_delta(delta)
        assert store.rowless_deltas_dropped == 1
        assert store.rowless_leaf_deltas == 0
        assert len(_rowless_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_extension_backend_routes_through_the_same_rule(self, pg_scratch_database, caplog):
        """The puct_edge backend learns 'no row' from the UPDATE status string."""

        class _NoRowConn:
            async def execute(self, *_a, **_k):
                return "UPDATE 0"

        store = self._store(pg_scratch_database)
        leaf = BackupDelta(key=_KEY, node_visit_delta=1, node_value_delta=0.5)
        path = BackupDelta(key=_KEY, node_visit_delta=1, node_value_delta=0.5, edge_deltas={7: (1, 0.5)})
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await store._apply_backup_row_extension(_NoRowConn(), leaf)
            await store._apply_backup_row_extension(_NoRowConn(), path)
        assert (store.rowless_leaf_deltas, store.rowless_deltas_dropped) == (1, 1)
        assert len(_rowless_warnings(caplog)) == 1


@pytest.mark.db
@pytest.mark.asyncio
async def test_ply_limited_search_loses_no_visits_and_logs_no_warning(
    pg_scratch_database, pg_conn, caplog
):
    """Max_Book_Ply=1 from the start position: every descent ends at an unpersisted leaf."""
    await _drop_book_tables(pg_conn)
    cfg = dataclasses.replace(
        _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN), max_book_ply=1
    )
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    try:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            coordinator = await _run_small_search(store, cfg, max_descents=200)
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT visit_count, edges FROM book_node")
        edge_visits = int(decode_edges(bytes(row["edges"]))["visit_count"].sum())

        # The leaf visits are recorded on the root's edges; nothing is lost.
        assert row["visit_count"] == coordinator.descents_completed
        assert row["visit_count"] == edge_visits + 1  # R4.5
        assert store.rowless_leaf_deltas > 0
        assert store.rowless_deltas_dropped == 0
        assert _rowless_warnings(caplog) == []
    finally:
        await store.close()
