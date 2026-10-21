"""Regression tests for review finding F-11 (dev-loop 20261020-dlcobra-ce5a).

Requirement 10.8: a configured Root_Position that differs from the recorded one
was never detected.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dlshogi.book.__main__ import _check_root_mismatch
from dlshogi.book.node_store import NodeStore
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _drop_book_tables,
    _make_config,
    _run_small_search,
)

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

# Initial position after 7g7f: a different, legal Root_Position.
_OTHER_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"


async def _searched_store(pg_scratch_database, pg_conn, descents: int):
    await _drop_book_tables(pg_conn)
    cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    await _run_small_search(store, cfg, max_descents=descents)
    return cfg, store


async def test_root_mismatch_detected(pg_scratch_database, pg_conn):
    """F-11: R10.8 -- a configured root that differs from the recorded one exits."""
    cfg, store = await _searched_store(pg_scratch_database, pg_conn, 5)
    try:
        await _check_root_mismatch(store, cfg)  # matching root: passes

        other = replace(cfg, root_position=_OTHER_SFEN)
        with pytest.raises(SystemExit) as exc:
            await _check_root_mismatch(store, other)
        assert exc.value.code == 1
    finally:
        await store.close()


async def test_root_check_passes_on_empty_book(pg_scratch_database, pg_conn):
    """F-11: no node rows yet is a first start -- any root is accepted."""
    await _drop_book_tables(pg_conn)
    cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    try:
        await _check_root_mismatch(store, replace(cfg, root_position=_OTHER_SFEN))
    finally:
        await store.close()
