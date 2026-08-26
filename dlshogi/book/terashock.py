"""Terashock_Index: Position_Key lookup and the import-terashock command.

Implements Requirement 7.1 (Terashock_Index: position-key lookup with
last-wins duplicate handling, 1 ms p95 for 10^7 entries) and Requirement
7.8 (unreadable or empty Terashock_Book -> report and exit before writing
to the Node_Store). See design.md's "Terashock_Index" section for the full
derivation.

**Import path.** ``import_terashock`` parses the configured ``.db`` file
with ``BookDBParser`` (task 5), computes the Position_Key for each parsed
entry's SFEN, packs its Terashock_Move list into ``PACKED_TS_MOVE`` bytes,
and bulk-loads via ``asyncpg``'s ``copy_records_to_table`` into an unlogged
staging table. Because ``COPY`` cannot express ``ON CONFLICT``, the import
then issues one ``INSERT ... SELECT ... ON CONFLICT (key_hi, key_lo) DO
UPDATE`` so the *last* parsed entry wins and the conflict count becomes the
duplicate-SFEN count (Requirement 7.1). Source identity (path, size, mtime,
entry count, parser version) is recorded in ``book_meta`` and compared at
startup; a mismatch or absence triggers the import before any command is
accepted (Requirement 7.1).

**Lookup path.** ``TerashockIndex.lookup`` queries ``terashock_entry`` by
primary key ``(key_hi, key_lo)`` and returns a ``TerashockLookupResult``
(the decoded ``PACKED_TS_MOVE`` array and the SFEN). A small in-process
LRU, bounded by a portion of ``Cache_Budget``, absorbs repeated lookups;
note that a lookup happens only on the *first* expansion of a node, so the
query rate equals the expansion rate, not the descent rate (design.md's
"In PostgreSQL, not in memory" section).

**Progress_Reporter integration.** ``report.py`` (task 16.1) does not
exist yet, so ``import_terashock`` uses ``logging`` for progress/error
reporting and returns an ``ImportStats`` dataclass for the caller to
inspect. The eventual ``Progress_Reporter`` need only read these same
stats.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import asyncpg
import numpy as np

from dlshogi.book.book_db import BookDBParser, TerashockEntry, TerashockMove
from dlshogi.book.config import BookConfig
from dlshogi.book.keys import (
    PositionKey,
    fold_u64_to_i64,
    position_key_from_sfen,
    unfold_i64_to_u64,
)
from dlshogi.book.packed_edge import PACKED_TS_MOVE

_LOG = logging.getLogger(__name__)

# Parser version constant: recorded in book_meta to detect when the parser
# logic changes and a reimport is needed (even if the .db file itself has
# not changed on disk). Bump this whenever BookDBParser semantics change.
PARSER_VERSION = 1

# Fraction of Cache_Budget allocated to the Terashock_Index LRU.
# The LRU stores decoded TerashockLookupResult objects; each entry is
# approximately sizeof(PACKED_TS_MOVE) * avg_moves + SFEN overhead.
# 5% of Cache_Budget is conservative -- the bulk of Cache_Budget goes
# to the descent task's node cache and the Evaluator's staging buffers.
_LRU_BUDGET_FRACTION = 0.05

# Estimated per-entry overhead in the LRU (Python object overhead + dict
# entry + SFEN string + PositionKey tuple). Conservative estimate for
# byte accounting.
_LRU_ENTRY_OVERHEAD_BYTES = 256


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerashockLookupResult:
    """One Terashock_Entry looked up by Position_Key.

    ``moves`` is a numpy array of dtype ``PACKED_TS_MOVE``, carrying the
    packed Terashock_Move records in file order (matching the order they
    were written by ``import_terashock``). ``sfen`` is the entry's SFEN
    string as stored in ``terashock_entry``.
    """

    sfen: str
    moves: np.ndarray  # dtype=PACKED_TS_MOVE


@dataclass
class ImportStats:
    """Statistics returned by ``import_terashock`` (Requirement 7.1).

    ``entry_count`` is the number of unique Position_Key entries written to
    ``terashock_entry``. ``duplicate_count`` is the number of parsed entries
    whose Position_Key collided with an earlier entry (last wins).
    ``rejected_line_count`` is the number of lines the parser rejected.
    """

    entry_count: int
    duplicate_count: int
    rejected_line_count: int


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------


def pack_terashock_moves(moves: list[TerashockMove]) -> bytes:
    """Pack a list of ``TerashockMove`` into ``PACKED_TS_MOVE`` bytes.

    ``reply16`` of ``None`` (the literal ``none`` opponent reply,
    Requirement 6.3) is encoded as 0 in the packed format.
    """
    n = len(moves)
    arr = np.zeros(n, dtype=PACKED_TS_MOVE)
    for i, m in enumerate(moves):
        arr[i]["move16"] = m.move16
        arr[i]["reply16"] = m.reply16 if m.reply16 is not None else 0
        arr[i]["eval"] = m.eval
        arr[i]["depth"] = m.depth
        arr[i]["count"] = m.count
    return arr.tobytes()


def unpack_terashock_moves(blob: bytes) -> np.ndarray:
    """Decode a ``PACKED_TS_MOVE`` byte blob into a numpy structured array.

    Returns a zero-copy view (read-only) when possible.
    """
    if len(blob) == 0:
        return np.empty(0, dtype=PACKED_TS_MOVE)
    return np.frombuffer(blob, dtype=PACKED_TS_MOVE)


# ---------------------------------------------------------------------------
# Source identity comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceIdentity:
    """The five-field source identity recorded in book_meta."""

    path: str
    size: int
    mtime: int
    entry_count: int
    parser_version: int = PARSER_VERSION


def _get_file_identity(book_path: Path) -> Optional[_SourceIdentity]:
    """Read the configured file's identity from the filesystem.

    Returns ``None`` if the file does not exist or cannot be stat'd.
    """
    try:
        st = book_path.stat()
        return _SourceIdentity(
            path=str(book_path),
            size=st.st_size,
            mtime=int(st.st_mtime),
            entry_count=-1,  # unknown until parsed
            parser_version=PARSER_VERSION,
        )
    except OSError:
        return None


async def check_terashock_source(pool: asyncpg.Pool, cfg: BookConfig) -> bool:
    """Compare the recorded source identity with the configured file.

    Returns ``True`` if the import is needed (mismatch or absence), or
    ``False`` if the recorded identity matches and no reimport is required.
    If no ``terashock_book`` is configured, returns ``False`` (no import
    needed because there is nothing to import).
    """
    if cfg.terashock_book is None:
        return False

    book_path = Path(cfg.terashock_book)
    file_id = _get_file_identity(book_path)
    if file_id is None:
        # File doesn't exist or can't be stat'd -- import will fail with
        # Requirement 7.8's error, but we signal "needed" so the caller
        # attempts it and gets the proper error report.
        return True

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT terashock_source_path, terashock_source_size, "
            "       terashock_source_mtime, terashock_entry_count "
            "FROM book_meta WHERE id = 1"
        )

    if row is None:
        return True

    recorded_path = row["terashock_source_path"]
    recorded_size = row["terashock_source_size"]
    recorded_mtime = row["terashock_source_mtime"]
    recorded_count = row["terashock_entry_count"]

    if recorded_path is None:
        # No previous import recorded.
        return True

    # Compare each field; any mismatch triggers reimport.
    if (
        recorded_path != str(book_path)
        or recorded_size != file_id.size
        or recorded_mtime != file_id.mtime
    ):
        _LOG.info(
            "Terashock source identity mismatch: recorded (%s, size=%s, mtime=%s) "
            "vs current (%s, size=%s, mtime=%s); reimport needed",
            recorded_path, recorded_size, recorded_mtime,
            str(book_path), file_id.size, file_id.mtime,
        )
        return True

    # All fields match -- no reimport needed.
    _LOG.debug(
        "Terashock source identity matches: %s (size=%d, mtime=%d, entries=%s)",
        recorded_path, recorded_size, recorded_mtime, recorded_count,
    )
    return False


# ---------------------------------------------------------------------------
# import-terashock
# ---------------------------------------------------------------------------


async def import_terashock(
    pool: asyncpg.Pool,
    cfg: BookConfig,
) -> ImportStats:
    """Parse and bulk-load the configured Terashock_Book into ``terashock_entry``.

    Implements the import path of design.md's Terashock_Index section:
    1. Parse the .db file with BookDBParser.
    2. Requirement 7.8: unreadable or zero entries -> report and exit.
    3. Compute Position_Key for each entry, pack moves as PACKED_TS_MOVE.
    4. Bulk-load via copy_records_to_table into an unlogged staging table.
    5. INSERT ... SELECT ... ON CONFLICT DO UPDATE (last wins).
    6. Record source identity in book_meta.
    7. Return ImportStats.

    Raises ``SystemExit`` (via ``sys.exit(1)``) if the file is unreadable
    or yields zero entries (Requirement 7.8: "report and exit before
    writing to the Node_Store").
    """
    if cfg.terashock_book is None:
        _LOG.error(
            "import_terashock called but no Terashock_Book path is configured"
        )
        sys.exit(1)

    book_path = Path(cfg.terashock_book)

    # -- Step 1: Read and parse the .db file --
    try:
        text = book_path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.error(
            "Terashock_Book is unreadable: %s (%s) (Requirement 7.8)",
            book_path, exc,
        )
        sys.exit(1)

    rejected_lines: list[tuple[int, str]] = []

    def _on_rejected(line_number: int, content: str) -> None:
        rejected_lines.append((line_number, content))

    parser = BookDBParser(on_rejected_line=_on_rejected)
    book = parser.parse_text(text)

    # -- Step 2: Requirement 7.8 check --
    if len(book.entries) == 0:
        _LOG.error(
            "Terashock_Book contains no Terashock_Entry records: %s "
            "(Requirement 7.8: exit before writing to the Node_Store)",
            book_path,
        )
        sys.exit(1)

    if rejected_lines:
        _LOG.warning(
            "Terashock_Book parser rejected %d line(s) from %s",
            len(rejected_lines), book_path,
        )

    _LOG.info(
        "Terashock_Book parsed: %d entries from %s (rejected %d lines)",
        len(book.entries), book_path, len(rejected_lines),
    )

    # -- Step 3: Prepare records for bulk load --
    records: list[tuple[int, int, str, bytes]] = []
    for entry in book.entries:
        try:
            pk = position_key_from_sfen(entry.sfen)
        except Exception:
            # position_key_from_sfen may fail on malformed SFENs that
            # cppshogi cannot parse. Skip silently -- these entries
            # effectively have no Position_Key and cannot be indexed.
            _LOG.debug("Skipping entry with unparseable SFEN: %s", entry.sfen[:80])
            continue
        key_hi = fold_u64_to_i64(pk.hi)
        key_lo = fold_u64_to_i64(pk.lo)
        moves_blob = pack_terashock_moves(entry.moves)
        records.append((key_hi, key_lo, entry.sfen, moves_blob))

    if len(records) == 0:
        _LOG.error(
            "Terashock_Book yielded zero indexable entries (all SFENs "
            "failed Position_Key computation): %s (Requirement 7.8)",
            book_path,
        )
        sys.exit(1)

    _LOG.info("Prepared %d records for bulk load", len(records))

    # -- Step 4: Bulk-load into unlogged staging table --
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create the unlogged staging table (dropped at end of txn).
            await conn.execute("""
                CREATE UNLOGGED TABLE IF NOT EXISTS _terashock_staging (
                    seq        bigint NOT NULL,
                    key_hi     bigint NOT NULL,
                    key_lo     bigint NOT NULL,
                    sfen       text   NOT NULL,
                    moves      bytea  NOT NULL
                )
            """)
            await conn.execute("TRUNCATE _terashock_staging")

            # copy_records_to_table for bulk load.
            await conn.copy_records_to_table(
                "_terashock_staging",
                records=[(i, *r) for i, r in enumerate(records)],
                columns=["seq", "key_hi", "key_lo", "sfen", "moves"],
            )

            _LOG.info("Staging table loaded with %d rows", len(records))

            # -- Step 5: INSERT ... SELECT ... ON CONFLICT DO UPDATE --
            # The "last parsed entry wins" semantic: we use DISTINCT ON
            # ordered by seq DESC so that for duplicate (key_hi, key_lo),
            # the row with the highest seq (last in file) is chosen.
            # First truncate the target table so we do a clean reimport.
            await conn.execute("TRUNCATE terashock_entry")

            inserted = await conn.fetchval("""
                INSERT INTO terashock_entry (key_hi, key_lo, sfen, moves)
                SELECT DISTINCT ON (key_hi, key_lo)
                       key_hi, key_lo, sfen, moves
                FROM _terashock_staging
                ORDER BY key_hi, key_lo, seq DESC
                ON CONFLICT (key_hi, key_lo) DO UPDATE
                    SET sfen  = EXCLUDED.sfen,
                        moves = EXCLUDED.moves
                RETURNING 1
            """)

            # Count unique entries and duplicates.
            unique_count = await conn.fetchval(
                "SELECT count(*) FROM terashock_entry"
            )
            duplicate_count = len(records) - unique_count

            _LOG.info(
                "terashock_entry: %d unique entries, %d duplicates (last wins)",
                unique_count, duplicate_count,
            )

            # Clean up staging table.
            await conn.execute("DROP TABLE IF EXISTS _terashock_staging")

            # -- Step 6: Record source identity in book_meta --
            st = book_path.stat()
            await conn.execute(
                """
                UPDATE book_meta SET
                    terashock_source_path  = $1,
                    terashock_source_size  = $2,
                    terashock_source_mtime = $3,
                    terashock_entry_count  = $4
                WHERE id = 1
                """,
                str(book_path),
                st.st_size,
                int(st.st_mtime),
                unique_count,
            )

    stats = ImportStats(
        entry_count=unique_count,
        duplicate_count=duplicate_count,
        rejected_line_count=len(rejected_lines),
    )

    _LOG.info(
        "import-terashock complete: %d entries, %d duplicates, %d rejected lines",
        stats.entry_count, stats.duplicate_count, stats.rejected_line_count,
    )

    return stats


# ---------------------------------------------------------------------------
# TerashockIndex (LRU-cached lookup)
# ---------------------------------------------------------------------------


class TerashockIndex:
    """In-process Position_Key -> TerashockLookupResult index with LRU cache.

    The LRU is an ``OrderedDict``-based cache bounded by a byte budget
    derived from ``cache_budget_bytes`` (a portion of the configured
    Cache_Budget, per design.md's "A small in-process LRU inside
    Cache_Budget"). A lookup happens only on the *first* expansion of a
    node, so the query rate equals the expansion rate, not the descent
    rate.
    """

    def __init__(self, pool: asyncpg.Pool, cache_budget_bytes: int) -> None:
        self._pool = pool
        # LRU budget: a fraction of the total cache budget.
        self._max_bytes = max(
            int(cache_budget_bytes * _LRU_BUDGET_FRACTION),
            _LRU_ENTRY_OVERHEAD_BYTES * 16,  # minimum 16 entries worth
        )
        self._cache: OrderedDict[PositionKey, TerashockLookupResult] = OrderedDict()
        self._cache_bytes = 0

        # Counters for future Progress_Reporter (task 16.1).
        self.hits = 0
        self.misses = 0
        self.queries = 0

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the LRU cache."""
        return len(self._cache)

    @property
    def cache_bytes(self) -> int:
        """Approximate bytes currently used by the LRU cache."""
        return self._cache_bytes

    async def lookup(self, key: PositionKey) -> Optional[TerashockLookupResult]:
        """Look up a Position_Key in the Terashock_Index.

        Returns the ``TerashockLookupResult`` for the given key, or
        ``None`` if no Terashock_Entry exists for that Position_Key. The
        result is cached in the in-process LRU for subsequent lookups.
        """
        self.queries += 1

        # Check LRU cache first.
        result = self._cache.get(key)
        if result is not None:
            # Move to end (most recently used).
            self._cache.move_to_end(key)
            self.hits += 1
            return result

        # Cache miss -- query PostgreSQL.
        self.misses += 1
        key_hi = fold_u64_to_i64(key.hi)
        key_lo = fold_u64_to_i64(key.lo)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT sfen, moves FROM terashock_entry "
                "WHERE key_hi = $1 AND key_lo = $2",
                key_hi, key_lo,
            )

        if row is None:
            # No entry exists; cache the negative result as None sentinel.
            # We do NOT cache negatives in the OrderedDict to avoid
            # unbounded growth from positions without Terashock entries.
            return None

        sfen: str = row["sfen"]
        moves_blob: bytes = row["moves"]
        moves_arr = unpack_terashock_moves(moves_blob)

        result = TerashockLookupResult(sfen=sfen, moves=moves_arr)

        # Insert into LRU cache with byte accounting.
        entry_bytes = _estimate_entry_bytes(result)
        self._evict_until_fits(entry_bytes)
        self._cache[key] = result
        self._cache_bytes += entry_bytes

        return result

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict LRU entries until ``needed_bytes`` fits within budget."""
        while self._cache and (self._cache_bytes + needed_bytes > self._max_bytes):
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= _estimate_entry_bytes(evicted)
        # Ensure non-negative.
        if self._cache_bytes < 0:
            self._cache_bytes = 0

    def clear_cache(self) -> None:
        """Clear the entire LRU cache."""
        self._cache.clear()
        self._cache_bytes = 0


def _estimate_entry_bytes(result: TerashockLookupResult) -> int:
    """Estimate the memory footprint of one LRU cache entry."""
    # PACKED_TS_MOVE array bytes + SFEN string + Python object overhead.
    moves_bytes = result.moves.nbytes if result.moves is not None else 0
    sfen_bytes = len(result.sfen.encode("utf-8")) if result.sfen else 0
    return _LRU_ENTRY_OVERHEAD_BYTES + moves_bytes + sfen_bytes


__all__ = [
    "PARSER_VERSION",
    "TerashockIndex",
    "TerashockLookupResult",
    "ImportStats",
    "check_terashock_source",
    "import_terashock",
    "pack_terashock_moves",
    "unpack_terashock_moves",
]
