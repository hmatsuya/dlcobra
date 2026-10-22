"""Regression test for review finding L-11 (dev-loop 20261020-dlcobra-ce5a).

``NodeStore.bump_search_write_seq`` had no caller, so ``book_meta.search_write_seq``
stayed 0 and export's R12.8 warning ("search wrote after the last propagation
pass") could never fire. The counter also has to be comparable with
``propagation_done_seq``, which counts propagation passes, so search sets it to
``propagation_seq + 1`` rather than incrementing it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from dlshogi.book.export import export_book
from dlshogi.book.node_store import NodeStore
from dlshogi.book.keys import position_key_from_sfen
from dlshogi.book.propagate import propagate
from tests.book.test_run_starts_flusher import _running

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _seqs(pg_conn):
    row = await pg_conn.fetchrow(
        "SELECT search_write_seq AS w, propagation_done_seq AS d FROM book_meta WHERE id = 1"
    )
    return row["w"], row["d"]


async def _export_warns(cfg, tmp_path, caplog) -> bool:
    store = await NodeStore.connect_and_prepare(cfg, role="export")
    try:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="dlshogi.book.export"):
            await export_book(store, cfg, tmp_path / "b.bin", None)
    finally:
        await store.close()
    return any("Stale-propagation" in r.getMessage() for r in caplog.records)


async def test_export_warns_exactly_while_search_is_newer_than_propagation(
    pg_scratch_database, pg_conn, tmp_path, caplog
):
    cfg, store, evaluator, coordinator = await _running(pg_scratch_database, pg_conn)
    run = asyncio.ensure_future(coordinator.run())
    await asyncio.sleep(1)
    coordinator._request_stop()
    await asyncio.wait_for(run, 60)
    await evaluator.stop()

    try:
        written, done = await _seqs(pg_conn)
        assert written > done, "run() did not mark the graph as written by search"
        assert await _export_warns(cfg, tmp_path, caplog)

        # Two passes: a plain increment would line up after one and then break.
        for _ in range(2):
            prop_store = await NodeStore.connect_and_prepare(cfg, role="propagate")
            try:
                await propagate(prop_store, position_key_from_sfen(cfg.root_position), cfg)
            finally:
                await prop_store.close()
        written, done = await _seqs(pg_conn)
        assert done == 2 and written <= done, (written, done)
        assert not await _export_warns(cfg, tmp_path, caplog)

        # More search after the pass: stale again; a second call is idempotent.
        first = await store.bump_search_write_seq()
        assert await store.bump_search_write_seq() == first == done + 1
        assert await _export_warns(cfg, tmp_path, caplog)
    finally:
        await store.close()
