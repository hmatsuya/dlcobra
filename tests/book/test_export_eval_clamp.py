"""Regression test for review finding F-10 (dev-loop 20261020-dlcobra-ce5a).

The YaneuraOu ``.db`` export packed the int32 Apery score into a signed 16-bit
field, so ``Eval_Coef`` >= ~950 with a child ``prop_value`` of 0 made
``struct.pack`` raise and killed the whole export.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cshogi
import numpy as np
import pytest

from dlshogi.book.export import _compute_score, export_book
from dlshogi.book.keys import position_key_from_sfen
from dlshogi.book.node_store import NodeStore
from dlshogi.book.propagate import propagate
from tests.book.reference.book_db import ref_parse_book_db
from tests.book.test_interop import (
    _INITIAL_SFEN,
    _drop_book_tables,
    _make_config,
    _run_small_search,
)

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _searched_store(pg_scratch_database, pg_conn, descents: int):
    await _drop_book_tables(pg_conn)
    cfg = _make_config(pg_scratch_database, root_sfen=_INITIAL_SFEN)
    store = await NodeStore.connect_and_prepare(cfg, role="search")
    await _run_small_search(store, cfg, max_descents=descents)
    return cfg, store


async def test_yaneuraou_export_clamps_eval_and_apery_keeps_int32(
    pg_scratch_database, pg_conn, tmp_path: Path
):
    """F-10: Eval_Coef=1000 with every child prop_value=0 must not crash."""
    cfg, store = await _searched_store(pg_scratch_database, pg_conn, 60)
    try:
        cfg = replace(cfg, eval_coef=1000.0)
        await propagate(store, position_key_from_sfen(cfg.root_position), cfg)
        # Force the stored propagated values to the extreme: child lost -> score
        # ~ +34 539, which does not fit a signed 16-bit field.
        async with store.pool.acquire() as conn:
            await conn.execute("UPDATE book_node SET prop_value = 0.0")

        apery = tmp_path / "book.bin"
        db = tmp_path / "book.db"
        counts = await export_book(store, cfg, apery, db)  # raised struct.error before the fix

        assert counts.yaneuraou_entries > 0
        evals = [
            m.eval
            for e in ref_parse_book_db(db.read_text()).entries
            for m in e.moves
        ]
        assert evals and max(evals) == 32000 and min(evals) >= -32000

        scores = np.fromfile(apery, dtype=cshogi.BookEntry)["score"]
        assert scores.size and scores.max() == _compute_score(0.0, 1000.0)  # Apery keeps the int32 clamp (R12.6); .db is clamped to 32000
    finally:
        await store.close()
