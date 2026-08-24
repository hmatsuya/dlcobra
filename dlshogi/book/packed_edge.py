"""PACKED_EDGE numpy dtype and the packed edge-list codec.

A Book_Node's whole Book_Edge list is packed into a single ``bytea``
column so that "node plus every edge" is one primary-key row read (see
design.md's "The storage layout decision" and "Packed edge record"). This
module defines the two little-endian, no-padding record layouts
(``PACKED_EDGE`` for Book_Edges, ``PACKED_TS_MOVE`` for Terashock_Moves)
and the codec around ``PACKED_EDGE``: ``encode_edges``/``decode_edges`` for
the zero-copy round trip, and ``patch_edges`` for the client-side fallback
backup patch described in design.md's "The backup write path" section
("A fallback for operators who cannot install an extension").
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 20 bytes, no padding. Field order and offsets match design.md's "Packed
# edge record" table exactly.
PACKED_EDGE = np.dtype(
    [
        ("move16", "<u2"),  # offset  0: cshogi.move16(move) == Move::proFromAndTo
        ("prior_q16", "<u2"),  # offset  2: prior probability as round(p * 65535)
        ("ts_depth", "u1"),  # offset  4: Terashock search depth 0..127
        ("flags", "u1"),  # offset  5: bit0 = Terashock eval/depth present
        ("ts_eval", "<i2"),  # offset  6: Terashock evaluation value -32000..32000
        ("visit_count", "<u4"),  # offset  8: Requirement 4.4
        ("value_sum", "<f8"),  # offset 12: Requirement 4.4, IEEE 754 binary64
    ]
)
assert PACKED_EDGE.itemsize == 20
assert [PACKED_EDGE.fields[n][1] for n in PACKED_EDGE.names] == [0, 2, 4, 5, 6, 8, 12]

# 11 bytes, no padding. Terashock_Move records, used for the ``moves``
# column of ``terashock_entry`` (design.md's Terashock_Index section).
PACKED_TS_MOVE = np.dtype(
    [
        ("move16", "<u2"),  # offset 0
        ("reply16", "<u2"),  # offset 2, 0 == the literal "none" (Requirement 6.3)
        ("eval", "<i2"),  # offset 4
        ("depth", "u1"),  # offset 6
        ("count", "<u4"),  # offset 7
    ]
)
assert PACKED_TS_MOVE.itemsize == 11
assert [PACKED_TS_MOVE.fields[n][1] for n in PACKED_TS_MOVE.names] == [0, 2, 4, 6, 7]

# Bit 0 of ``flags``: set iff ts_eval/ts_depth carry a real Terashock
# evaluation (Requirements 7.2, 7.9). Absent Terashock fields are written
# as canonical zeros regardless of this bit, so byte-for-byte comparison
# of two encodings of the same logical edge set agrees (Property 30).
_FLAG_TERASHOCK_PRESENT = 0x01


def encode_edges(edges: Sequence) -> bytes:
    """Pack an edge list into its canonical ``PACKED_EDGE`` byte encoding.

    ``edges`` is any sequence of objects (or mappings) exposing the
    ``PACKED_EDGE`` field names -- ``move16``, ``prior_q16``, ``ts_depth``,
    ``flags``, ``ts_eval``, ``visit_count``, ``value_sum``. The result is
    sorted ascending by ``move16``, which is equivalent to ascending USI
    order because ``move16`` (Apery's ``proFromAndTo``) determines the USI
    string with no position context (Requirement 1.3; see design.md's
    "Packed edge record"). An edge with no Terashock evaluation (``flags``
    bit 0 clear, or no ``flags``/``ts_eval``/``ts_depth`` supplied at all)
    is written with ``ts_eval = 0`` and ``ts_depth = 0``, the canonical
    zero encoding Property 30's idempotence check relies on.
    """
    n = len(edges)
    arr = np.zeros(n, dtype=PACKED_EDGE)
    for i, e in enumerate(edges):
        move16 = _field(e, "move16")
        flags = _field(e, "flags", 0)
        has_terashock = bool(flags & _FLAG_TERASHOCK_PRESENT)
        arr[i]["move16"] = move16
        arr[i]["prior_q16"] = _field(e, "prior_q16", 0)
        arr[i]["flags"] = _FLAG_TERASHOCK_PRESENT if has_terashock else 0
        arr[i]["ts_depth"] = _field(e, "ts_depth", 0) if has_terashock else 0
        arr[i]["ts_eval"] = _field(e, "ts_eval", 0) if has_terashock else 0
        arr[i]["visit_count"] = _field(e, "visit_count", 0)
        arr[i]["value_sum"] = _field(e, "value_sum", 0.0)
    order = np.argsort(arr["move16"], kind="stable")
    arr = arr[order]
    return arr.tobytes()


def decode_edges(blob: bytes) -> np.ndarray:
    """Return a zero-copy ``PACKED_EDGE`` view of a packed edge-list blob.

    ``np.frombuffer`` yields a read-only view with no copy and no per-row
    Python object, which is the whole point of the packed layout (see
    design.md's "Search_Coordinator and descent tasks" scoring block and
    the "Node_Store" component description).
    """
    return np.frombuffer(blob, dtype=PACKED_EDGE)


def patch_edges(
    blob: bytes,
    move16: Sequence[int],
    visit_deltas: Sequence[int],
    value_deltas: Sequence[float],
) -> bytes:
    """Apply visit-count and value-sum deltas to a packed edge-list blob.

    This is the client-side half of the fallback backup path documented in
    design.md's "The backup write path" section (the paragraph beginning
    "A fallback for operators who cannot install an extension"), used when
    the ``puct_edge`` PostgreSQL C extension of task 9 is unavailable. The
    caller is expected to have taken the row's lock (``SELECT ... FOR
    UPDATE``) before reading ``blob`` and to write the returned bytes back
    in the same transaction, which gives the same no-lost-update guarantee
    as the in-database ``puct_edge_backup`` function at one extra round
    trip per node per flush.

    Mirrors the ``puct_edge_backup(bytea, int2[], int4[], float8[])`` SQL
    signature of design.md: ``move16`` names the edges to patch (parallel
    to the SQL function's ``int2[]``), ``visit_deltas`` and
    ``value_deltas`` are the corresponding per-edge increments. Every
    named ``move16`` must already be present in ``blob``; the patch never
    creates a new edge record.
    """
    arr = np.frombuffer(blob, dtype=PACKED_EDGE).copy()
    keys = arr["move16"]
    idx = np.searchsorted(keys, move16)
    for i, m, dv, dw in zip(idx, move16, visit_deltas, value_deltas):
        if i >= len(keys) or keys[i] != m:
            raise KeyError(f"move16 {m} not present in packed edge list")
        arr[i]["visit_count"] = arr[i]["visit_count"] + dv
        arr[i]["value_sum"] = arr[i]["value_sum"] + dw
    return arr.tobytes()


def _field(edge, name: str, default=None):
    """Read ``name`` off ``edge``, which may be a mapping or an attribute holder."""
    if isinstance(edge, dict) or isinstance(edge, np.void):
        try:
            return edge[name]
        except (KeyError, ValueError):
            if default is not None:
                return default
            raise
    value = getattr(edge, name, default)
    if value is None:
        raise AttributeError(f"edge has no field {name!r} and no default was given")
    return value
