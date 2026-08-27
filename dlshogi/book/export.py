"""Book_Exporter: Apery binary and YaneuraOu ``.db`` text export.

Implements Requirement 12 (Book Export for Competition Use): a two-phase
external sort that streams the Book_Graph out of PostgreSQL without holding
it in RAM, producing an Apery binary book file and/or a YaneuraOu ``.db``
text file. See design.md's "Export without holding the graph in RAM" for
the full derivation.

**Phase 1 (run generation):** An asyncpg server-side cursor streams
``book_node`` in heap order with no predicate, so both sides to move are
exported (Requirement 12.12). For each node the exporter reconstructs the
``Board`` from ``sfen``, decodes the packed edges, applies per-edge
exclusion filters (Requirements 12.5, 12.9), asserts legality (Requirement
12.7), and emits records into a preallocated 64 MiB numpy buffer. When
the buffer fills it is sorted in place and written as a run file.

**Phase 2 (k-way merge):** Run files are opened as ``np.memmap`` and
merged with ``heapq.merge`` over the total order of Requirements 12.2 and
12.3 (Apery) or byte-wise SFEN order (YaneuraOu). Output is written to
a temporary path and ``os.replace()``-ed on success (Requirement 12.10).

**No side effects at import time**, per this package's convention.
"""

from __future__ import annotations

import heapq
import io
import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence, Tuple

import asyncpg
import cshogi
import numpy as np

from dlshogi.book.book_db import BookDBPrinter, TerashockEntry, TerashockMove
from dlshogi.book.keys import fold_u64_to_i64, unfold_i64_to_u64
from dlshogi.book.packed_edge import PACKED_EDGE, decode_edges

_LOG = logging.getLogger(__name__)

# 64 MiB buffer for Apery records: BookEntry is 16 bytes.
_APERY_BUF_RECORDS = (64 * 1024 * 1024) // cshogi.BookEntry.itemsize  # 4,194,304

# 64 MiB buffer for YaneuraOu variable-length records (byte budget).
_YANEURAOU_BUF_BYTES = 64 * 1024 * 1024

# Asyncpg cursor fetch size (rows per batch from the server).
_CURSOR_FETCH_SIZE = 1000

# INT32 bounds for score clamping.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# Terashock flag bit 0.
_FLAG_TERASHOCK_PRESENT = 0x01


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class ExportCounts:
    """Export result counts (Requirement 12.11)."""

    apery_records: int = 0
    yaneuraou_entries: int = 0
    excluded_edges: int = 0
    apery_key_multi_node: int = 0


# ---------------------------------------------------------------------------
# Score computation (Requirement 12.6)
# ---------------------------------------------------------------------------


def _compute_score(child_prop_value: float, eval_coef: float) -> int:
    """Compute the Apery score from a child's propagated value.

    ``child_prop_value`` is the child's ``prop_value`` from the database,
    expressed from the child's side-to-move perspective. The parent's
    perspective is ``v = 1 - child_prop_value``. The score formula is:
    ``clip(round(-log(1/v - 1) * Eval_Coef), INT32_MIN, INT32_MAX)``.
    """
    v = 1.0 - child_prop_value
    # Clamp v to avoid log(0) or division by zero.
    v = max(1e-15, min(1.0 - 1e-15, v))
    logit = -math.log(1.0 / v - 1.0)
    raw_score = round(logit * eval_coef)
    return max(_INT32_MIN, min(_INT32_MAX, raw_score))


# ---------------------------------------------------------------------------
# Apery sort key (Requirements 12.2, 12.3)
# ---------------------------------------------------------------------------


def _apery_sort_key(rec: np.void) -> Tuple[int, int, int, int]:
    """Return the total-order sort key for one BookEntry record.

    Order: ascending key (unsigned u8), descending score, descending count,
    ascending fromToPro.
    """
    return (
        int(rec["key"]),
        -int(rec["score"]),
        -int(rec["count"]),
        int(rec["fromToPro"]),
    )


def _sort_apery_buffer(arr: np.ndarray, n: int) -> None:
    """Sort the first ``n`` entries of ``arr`` in Apery order, in place.

    Uses np.lexsort with keys applied last-key-first. score (<i4) and
    count (<u2) are widened to int64 before negation to avoid overflow.
    """
    sub = arr[:n]
    order = np.lexsort(
        (
            sub["fromToPro"],  # ascending (last tie-break)
            -sub["count"].astype(np.int64),  # descending
            -sub["score"].astype(np.int64),  # descending
            sub["key"],  # primary, already <u8 so unsigned comparison
        )
    )
    arr[:n] = sub[order]


# ---------------------------------------------------------------------------
# YaneuraOu record encoding (variable-length, length-prefixed)
# ---------------------------------------------------------------------------

# Record format: 4 bytes sfen_len (u32 LE) + sfen_bytes + 4 bytes block_len (u32 LE) + block_bytes
# The sfen_bytes serve as the sort key (bytes comparison).
# The block_bytes contain the serialised move lines for that entry.


def _encode_yaneuraou_record(sfen: str, moves: List[TerashockMove]) -> bytes:
    """Encode one YaneuraOu entry as a length-prefixed binary record."""
    sfen_bytes = sfen.encode("utf-8")
    # Build the move block: each move serialised as a fixed struct.
    # Format per move: move16(u16) + reply16(u16) + eval(i16) + depth(u8) + count(u32) = 11 bytes
    move_data = bytearray()
    for m in moves:
        reply = m.reply16 if m.reply16 is not None else 0
        move_data.extend(struct.pack("<HHhBI", m.move16, reply, m.eval, m.depth, m.count))
    block_bytes = bytes(move_data)
    return (
        struct.pack("<I", len(sfen_bytes))
        + sfen_bytes
        + struct.pack("<I", len(block_bytes))
        + block_bytes
    )


def _decode_yaneuraou_record(data: bytes, offset: int) -> Tuple[bytes, bytes, int]:
    """Decode one YaneuraOu record, returning (sfen_bytes, block_bytes, next_offset)."""
    sfen_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    sfen_bytes = data[offset : offset + sfen_len]
    offset += sfen_len
    block_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    block_bytes = data[offset : offset + block_len]
    offset += block_len
    return sfen_bytes, block_bytes, offset


def _decode_yaneuraou_moves(block_bytes: bytes) -> List[TerashockMove]:
    """Decode the move block back into TerashockMove objects."""
    moves = []
    offset = 0
    while offset < len(block_bytes):
        move16, reply16, eval_val, depth, count = struct.unpack_from("<HHhBI", block_bytes, offset)
        offset += 11
        moves.append(
            TerashockMove(
                move16=move16,
                reply16=reply16 if reply16 != 0 else None,
                eval=eval_val,
                depth=depth,
                count=count,
            )
        )
    return moves


# ---------------------------------------------------------------------------
# Phase 1: Run generation (streaming from PostgreSQL)
# ---------------------------------------------------------------------------


class _ExportLegalityError(Exception):
    """Raised when a move fails the legality assertion (Requirement 12.7)."""

    pass


async def _phase1_generate_runs(
    pool: asyncpg.Pool,
    eval_coef: float,
    export_visit_threshold: float,
    propagation_done_seq: int,
    apery_out: Optional[Path],
    yaneuraou_out: Optional[Path],
    tmp_dir: Path,
    counts: ExportCounts,
) -> Tuple[List[Path], List[Path]]:
    """Stream book_node and generate sorted run files for both formats.

    Returns (apery_run_files, yaneuraou_run_files).
    """
    apery_runs: List[Path] = []
    yaneuraou_runs: List[Path] = []

    # Pre-allocate Apery buffer.
    apery_buf: Optional[np.ndarray] = None
    apery_buf_pos = 0
    if apery_out is not None:
        apery_buf = np.empty(_APERY_BUF_RECORDS, dtype=cshogi.BookEntry)

    # YaneuraOu buffer: list of (sfen_bytes, record_bytes) tuples.
    yaneuraou_buf: List[Tuple[bytes, bytes]] = []
    yaneuraou_buf_size = 0
    yaneuraou_entry_count = 0

    board = cshogi.Board()

    async with pool.acquire() as conn:
        # Use a transaction for the server-side cursor.
        async with conn.transaction():
            # Server-side cursor streaming book_node in heap order (no ORDER BY).
            cursor = await conn.cursor(
                "SELECT key_hi, key_lo, sfen, apery_key, visit_count, edges FROM book_node"
            )

            while True:
                rows = await cursor.fetch(_CURSOR_FETCH_SIZE)
                if not rows:
                    break

                for row in rows:
                    node_key_hi = row["key_hi"]
                    node_key_lo = row["key_lo"]
                    sfen = row["sfen"]
                    apery_key_signed = row["apery_key"]
                    node_visit_count = row["visit_count"]
                    edges_blob = row["edges"]

                    if not edges_blob:
                        continue

                    edges = decode_edges(edges_blob)
                    if len(edges) == 0:
                        continue

                    # Reconstruct the board from sfen.
                    try:
                        board.set_sfen(sfen)
                    except Exception as exc:
                        raise _ExportLegalityError(
                            f"Cannot set board from sfen '{sfen}': {exc}"
                        ) from exc

                    apery_key = unfold_i64_to_u64(apery_key_signed)

                    # -- Exclusion: parent visit count == 0 (Requirement 12.9)
                    if node_visit_count == 0:
                        counts.excluded_edges += len(edges)
                        continue

                    # Collect child keys for batched child lookup.
                    move16_list = edges["move16"].tolist()

                    # Batched child lookup: fetch prop_value, prop_epoch,
                    # terminal, eval_win_rate for all child positions.
                    child_info = await _batch_child_lookup(
                        pool, sfen, move16_list, propagation_done_seq
                    )

                    # Entries for YaneuraOu export (collected per-node).
                    yaneuraou_moves_for_node: List[TerashockMove] = []

                    for i, edge in enumerate(edges):
                        move16_val = int(edge["move16"])
                        visit_count = int(edge["visit_count"])

                        # -- Exclusion: visit-count ratio below threshold (Req 12.5)
                        ratio = visit_count / node_visit_count
                        if ratio < export_visit_threshold:
                            counts.excluded_edges += 1
                            continue

                        # -- Child info lookup.
                        ci = child_info.get(move16_val)
                        if ci is None:
                            # Child not found in DB: exclude (no prop_value).
                            counts.excluded_edges += 1
                            continue

                        child_prop_value = ci["prop_value"]
                        child_prop_epoch = ci["prop_epoch"]

                        # -- Exclusion: no prop_value or stale (Req 12.9)
                        if child_prop_value is None:
                            counts.excluded_edges += 1
                            continue
                        if child_prop_epoch != propagation_done_seq:
                            counts.excluded_edges += 1
                            continue

                        # -- Legality assertion (Requirement 12.7)
                        full_move = board.move_from_move16(move16_val)
                        if full_move not in board.legal_moves:
                            raise _ExportLegalityError(
                                f"Move {move16_val} (USI: {cshogi.move_to_usi(move16_val)}) "
                                f"is not legal at sfen '{sfen}'. The graph is corrupt."
                            )

                        # -- Score computation (Requirement 12.6)
                        score = _compute_score(child_prop_value, eval_coef)

                        # -- Emit Apery record (Requirement 12.1)
                        if apery_buf is not None:
                            apery_buf[apery_buf_pos]["key"] = apery_key
                            apery_buf[apery_buf_pos]["fromToPro"] = move16_val
                            apery_buf[apery_buf_pos]["count"] = min(visit_count, 65535)
                            apery_buf[apery_buf_pos]["score"] = score
                            apery_buf_pos += 1
                            counts.apery_records += 1

                            # Flush buffer when full.
                            if apery_buf_pos >= _APERY_BUF_RECORDS:
                                _sort_apery_buffer(apery_buf, apery_buf_pos)
                                run_path = tmp_dir / f"apery_run_{len(apery_runs):04d}.bin"
                                apery_buf[:apery_buf_pos].tofile(str(run_path))
                                apery_runs.append(run_path)
                                apery_buf_pos = 0

                        # -- Emit YaneuraOu move.
                        if yaneuraou_out is not None:
                            # Build TerashockMove with Terashock fields from the edge.
                            ts_depth = int(edge["ts_depth"])
                            ts_eval = int(edge["ts_eval"])
                            flags = int(edge["flags"])
                            has_terashock = bool(flags & _FLAG_TERASHOCK_PRESENT)

                            yaneuraou_moves_for_node.append(
                                TerashockMove(
                                    move16=move16_val,
                                    reply16=None,  # No reply in export context
                                    eval=score,  # Use the computed score as eval
                                    depth=ts_depth if has_terashock else 0,
                                    count=visit_count,
                                )
                            )

                    # After processing all edges of a node, emit YaneuraOu entry.
                    if yaneuraou_out is not None and yaneuraou_moves_for_node:
                        # Order moves descending by eval (Requirement 12.4).
                        yaneuraou_moves_for_node.sort(key=lambda m: -m.eval)
                        record = _encode_yaneuraou_record(sfen, yaneuraou_moves_for_node)
                        sfen_key = sfen.encode("utf-8")
                        yaneuraou_buf.append((sfen_key, record))
                        yaneuraou_buf_size += len(record)
                        yaneuraou_entry_count += 1
                        counts.yaneuraou_entries += 1

                        # Flush buffer when it exceeds budget.
                        if yaneuraou_buf_size >= _YANEURAOU_BUF_BYTES:
                            run_path = _flush_yaneuraou_run(
                                yaneuraou_buf, tmp_dir, len(yaneuraou_runs)
                            )
                            yaneuraou_runs.append(run_path)
                            yaneuraou_buf = []
                            yaneuraou_buf_size = 0

    # Flush remaining Apery buffer.
    if apery_buf is not None and apery_buf_pos > 0:
        _sort_apery_buffer(apery_buf, apery_buf_pos)
        run_path = tmp_dir / f"apery_run_{len(apery_runs):04d}.bin"
        apery_buf[:apery_buf_pos].tofile(str(run_path))
        apery_runs.append(run_path)

    # Flush remaining YaneuraOu buffer.
    if yaneuraou_buf:
        run_path = _flush_yaneuraou_run(yaneuraou_buf, tmp_dir, len(yaneuraou_runs))
        yaneuraou_runs.append(run_path)

    return apery_runs, yaneuraou_runs


def _flush_yaneuraou_run(
    buf: List[Tuple[bytes, bytes]], tmp_dir: Path, run_index: int
) -> Path:
    """Sort a YaneuraOu buffer by SFEN key (bytes order) and write a run file."""
    buf.sort(key=lambda item: item[0])
    run_path = tmp_dir / f"yaneuraou_run_{run_index:04d}.bin"
    with open(run_path, "wb") as f:
        for _sfen_key, record in buf:
            f.write(record)
    return run_path


async def _batch_child_lookup(
    pool: asyncpg.Pool,
    sfen: str,
    move16_list: List[int],
    propagation_done_seq: int,
) -> dict:
    """Batch-lookup child nodes for all edges of a given parent.

    Returns a dict mapping move16 -> {prop_value, prop_epoch, terminal, eval_win_rate}.
    Uses position_keys_after to derive child keys from the parent sfen and moves.
    """
    from dlshogi.book.keys import position_keys_after, fold_u64_to_i64

    if not move16_list:
        return {}

    moves16_arr = np.array(move16_list, dtype="<u2")
    try:
        child_keys = position_keys_after(sfen, moves16_arr)
    except Exception:
        # If position_keys_after fails (e.g. invalid sfen), return empty.
        return {}

    # Build query parameters: lists of key_hi, key_lo pairs.
    key_his = [fold_u64_to_i64(int(child_keys[i]["hi"])) for i in range(len(child_keys))]
    key_los = [fold_u64_to_i64(int(child_keys[i]["lo"])) for i in range(len(child_keys))]

    # Query all child nodes in one batch.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key_hi, key_lo, prop_value, prop_epoch, terminal, eval_win_rate "
            "FROM book_node WHERE (key_hi, key_lo) IN "
            "(SELECT unnest($1::bigint[]), unnest($2::bigint[]))",
            key_his,
            key_los,
        )

    # Build a lookup from (key_hi, key_lo) -> row data.
    row_map = {}
    for r in rows:
        row_map[(r["key_hi"], r["key_lo"])] = r

    # Map move16 -> child info.
    result = {}
    for i, m16 in enumerate(move16_list):
        kh = key_his[i]
        kl = key_los[i]
        r = row_map.get((kh, kl))
        if r is not None:
            result[m16] = {
                "prop_value": r["prop_value"],
                "prop_epoch": r["prop_epoch"],
                "terminal": r["terminal"],
                "eval_win_rate": r["eval_win_rate"],
            }

    return result


# ---------------------------------------------------------------------------
# Phase 2: k-way merge and output (Apery)
# ---------------------------------------------------------------------------


def _apery_run_generator(
    mmap: np.ndarray,
) -> Generator[Tuple[Tuple[int, int, int, int], int, np.void], None, None]:
    """Yield (sort_key, global_index, record) from a memory-mapped run."""
    for i in range(len(mmap)):
        rec = mmap[i]
        key = (
            int(rec["key"]),
            -int(rec["score"]),
            -int(rec["count"]),
            int(rec["fromToPro"]),
        )
        yield (key, i, rec)


def _phase2_merge_apery(run_files: List[Path], output_path: Path) -> None:
    """Merge Apery run files and write the final binary book file."""
    if not run_files:
        # Write an empty file.
        with open(output_path, "wb"):
            pass
        return

    # Open all run files as memory-mapped arrays.
    mmaps = [np.memmap(str(p), dtype=cshogi.BookEntry, mode="r") for p in run_files]

    # Create generators for heapq.merge.
    # Each generator yields (sort_key_tuple, run_index, record_index_within_run).
    # We need a unique tiebreaker across runs to ensure stable merge.
    def _gen(mmap: np.ndarray, run_idx: int):
        for i in range(len(mmap)):
            rec = mmap[i]
            key = (
                int(rec["key"]),
                -int(rec["score"]),
                -int(rec["count"]),
                int(rec["fromToPro"]),
                run_idx,
                i,
            )
            yield key, run_idx, i

    generators = [_gen(m, idx) for idx, m in enumerate(mmaps)]

    # Write output in chunks.
    out_buf = np.empty(min(_APERY_BUF_RECORDS, 65536), dtype=cshogi.BookEntry)
    out_pos = 0

    # Write to temp file then replace.
    dest_dir = output_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), suffix=".tmp")
    os.close(tmp_fd)

    try:
        with open(tmp_path, "wb") as f:
            for sort_key, run_idx, rec_idx in heapq.merge(*generators):
                out_buf[out_pos] = mmaps[run_idx][rec_idx]
                out_pos += 1
                if out_pos >= len(out_buf):
                    out_buf[:out_pos].tofile(f)
                    out_pos = 0

            if out_pos > 0:
                out_buf[:out_pos].tofile(f)

        os.replace(tmp_path, str(output_path))
    except Exception:
        # Clean up temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    finally:
        # Close mmaps.
        for m in mmaps:
            del m


# ---------------------------------------------------------------------------
# Phase 2: k-way merge and output (YaneuraOu)
# ---------------------------------------------------------------------------


def _yaneuraou_run_generator(
    run_path: Path,
) -> Generator[Tuple[bytes, bytes], None, None]:
    """Yield (sfen_bytes, full_record_bytes) from a YaneuraOu run file."""
    data = run_path.read_bytes()
    offset = 0
    while offset < len(data):
        sfen_bytes, block_bytes, next_offset = _decode_yaneuraou_record(data, offset)
        # Yield the sfen_bytes as key and the full record slice.
        record = data[offset:next_offset]
        yield sfen_bytes, record
        offset = next_offset


def _phase2_merge_yaneuraou(
    run_files: List[Path], output_path: Path, total_entry_count: int
) -> None:
    """Merge YaneuraOu run files and write the final .db text file."""
    dest_dir = output_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), suffix=".tmp")
    os.close(tmp_fd)

    printer = BookDBPrinter()

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            # Write header first (entry count is known).
            printer.write_header(f, total_entry_count)

            if not run_files:
                pass
            elif len(run_files) == 1:
                # Single run: just iterate through it.
                for sfen_bytes, record in _yaneuraou_run_generator(run_files[0]):
                    _write_yaneuraou_entry_from_record(printer, f, record)
            else:
                # k-way merge over all run files.
                def _keyed_gen(run_path: Path, run_idx: int):
                    for sfen_bytes, record in _yaneuraou_run_generator(run_path):
                        yield (sfen_bytes, run_idx, record)

                generators = [_keyed_gen(p, i) for i, p in enumerate(run_files)]
                for _sfen_key, _run_idx, record in heapq.merge(*generators):
                    _write_yaneuraou_entry_from_record(printer, f, record)

        os.replace(tmp_path, str(output_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_yaneuraou_entry_from_record(printer: BookDBPrinter, f, record: bytes) -> None:
    """Decode a binary record and write the TerashockEntry through the printer."""
    offset = 0
    sfen_len = struct.unpack_from("<I", record, offset)[0]
    offset += 4
    sfen = record[offset : offset + sfen_len].decode("utf-8")
    offset += sfen_len
    block_len = struct.unpack_from("<I", record, offset)[0]
    offset += 4
    block_bytes = record[offset : offset + block_len]

    moves = _decode_yaneuraou_moves(block_bytes)
    entry = TerashockEntry(sfen=sfen, moves=moves)
    printer.write_entry(f, entry)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def export_book(
    store: Any,
    cfg: Any,
    apery_out: Optional[Path],
    yaneuraou_out: Optional[Path],
) -> ExportCounts:
    """Export the Book_Graph to Apery binary and/or YaneuraOu .db format.

    Parameters
    ----------
    store : NodeStore
        A connected Node_Store instance whose ``pool`` property yields the
        asyncpg pool.
    cfg : BookConfig
        The operator configuration; uses ``eval_coef`` and
        ``export_visit_threshold``.
    apery_out : Path or None
        Destination path for the Apery binary book file. ``None`` to skip.
    yaneuraou_out : Path or None
        Destination path for the YaneuraOu ``.db`` text file. ``None`` to skip.

    Returns
    -------
    ExportCounts
        The counts of records written and edges excluded.

    Raises
    ------
    _ExportLegalityError
        If a move fails the legality assertion (the graph is corrupt).
    OSError
        If an output path cannot be opened for writing.
    """
    if apery_out is None and yaneuraou_out is None:
        return ExportCounts()

    pool = store.pool
    counts = ExportCounts()

    # -- Read propagation_done_seq and search_write_seq from book_meta.
    async with pool.acquire() as conn:
        meta_row = await conn.fetchrow(
            "SELECT search_write_seq, propagation_done_seq FROM book_meta WHERE id = 1"
        )

    if meta_row is None:
        _LOG.warning("book_meta row not found; export proceeding with propagation_done_seq=0")
        propagation_done_seq = 0
        search_write_seq = 0
    else:
        propagation_done_seq = meta_row["propagation_done_seq"]
        search_write_seq = meta_row["search_write_seq"]

    # -- Stale-propagation warning (Requirement 12.8).
    if propagation_done_seq == 0 or search_write_seq > propagation_done_seq:
        _LOG.warning(
            "Stale-propagation warning: search_write_seq=%d, propagation_done_seq=%d. "
            "The export will proceed, but propagated values may not reflect all search writes.",
            search_write_seq,
            propagation_done_seq,
        )

    # -- Create temporary directory for run files.
    # Use the destination directory's parent for locality with the output.
    if apery_out is not None:
        tmp_base = apery_out.parent
    elif yaneuraou_out is not None:
        tmp_base = yaneuraou_out.parent
    else:
        tmp_base = Path(".")
    tmp_base.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(dir=str(tmp_base), prefix="book_export_"))

    try:
        # -- Phase 1: generate sorted run files.
        apery_runs, yaneuraou_runs = await _phase1_generate_runs(
            pool=pool,
            eval_coef=cfg.eval_coef,
            export_visit_threshold=cfg.export_visit_threshold,
            propagation_done_seq=propagation_done_seq,
            apery_out=apery_out,
            yaneuraou_out=yaneuraou_out,
            tmp_dir=tmp_dir,
            counts=counts,
        )

        # -- Phase 2: k-way merge and write final output.
        if apery_out is not None:
            _phase2_merge_apery(apery_runs, apery_out)

        if yaneuraou_out is not None:
            _phase2_merge_yaneuraou(
                yaneuraou_runs, yaneuraou_out, counts.yaneuraou_entries
            )

    finally:
        # Clean up temporary run files.
        import shutil

        try:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        except Exception:
            pass

    _LOG.info(
        "Export complete: apery_records=%d, yaneuraou_entries=%d, excluded_edges=%d",
        counts.apery_records,
        counts.yaneuraou_entries,
        counts.excluded_edges,
    )

    return counts


__all__ = [
    "ExportCounts",
    "export_book",
]
