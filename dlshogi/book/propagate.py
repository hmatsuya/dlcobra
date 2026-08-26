"""Value_Propagator: negamax value propagation over the Book_Graph.

Implements the forward walk from Root_Position over an explicit frame stack,
computing propagated values by Requirement 9's precedence order and writing
`prop_value`, `prop_best_move16`, `prop_epoch` for every reachable Book_Node
in the above-threshold subgraph.

The algorithm is single-writer: one coroutine in one process, regardless of
GPU count. Parallelising across subtrees was rejected because transpositions
make subtrees overlap (see design.md's "Value propagation under a packed edge
list" section). The batched prefetch of step 1 recovers the I/O concurrency
that parallel workers would have provided, without nondeterminism.

**No side effects at import time**, per this package's established convention.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dlshogi.book.config import BookConfig
from dlshogi.book.keys import PositionKey, position_keys_after, side_to_move_is_white
from dlshogi.book.node_store import (
    BookNodeView,
    GetResult,
    NodeStore,
    PropagationWrite,
    Terminal,
)

_LOG = logging.getLogger(__name__)

# Hard guard on stack depth when Max_Book_Ply == 0 (depth limit disabled).
_HARD_FRAME_GUARD = 1024


@dataclass
class PropagationStats:
    """Propagation pass outcome reported to the Progress_Reporter (Req 9.7)."""

    nodes: int  # Book_Nodes pushed as frames and written
    draw_revisits: int  # Criterion 5 on-path revisits
    unevaluated_leaves: int  # Criterion 10: edgeless, no eval_win_rate
    path_cutoffs: int  # Frames not pushed because of depth guard
    below_threshold_edges: int  # Edges resolved via criteria 12/13
    root_value: float  # The propagated value of the Root_Position


# ---------------------------------------------------------------------------
# Frame: one entry on the explicit DFS stack.
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    """One frame on the explicit DFS stack.

    ``contributions`` collects Child_Contribution values as edges are
    resolved. ``edge_idx`` tracks which edge we are currently processing.
    """

    key: PositionKey
    view: BookNodeView
    # Child keys computed once via position_keys_after
    child_keys: np.ndarray
    # Parallel boolean mask: True where edge is below threshold
    below_mask: np.ndarray
    # Prefetched above-threshold children (BookNodeView | None)
    full_children: list[Optional[BookNodeView]]
    # Prefetched below-threshold children ((Terminal, Optional[float]) | None)
    narrow_children: list[Optional[tuple[Terminal, Optional[float]]]]
    # Indices into the *original* edge array for each group
    above_indices: np.ndarray
    below_indices: np.ndarray
    # Contributions collected so far, parallel to edges
    contributions: list[Optional[float]]
    # Current edge index being processed
    edge_idx: int


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def propagate(
    store: NodeStore, root: PositionKey, cfg: BookConfig
) -> PropagationStats:
    """Run one Propagation_Pass over the Book_Graph rooted at ``root``.

    Returns a `PropagationStats` summarising the pass.  If the root node is
    absent, logs an error and returns without modifying any row
    (Requirement 9.11).
    """
    # --- Step 1: verify root exists ---
    result, root_view = await store.get(root)
    if result != GetResult.FOUND or root_view is None:
        _LOG.error(
            "propagate: Root_Position %r is absent from the Node_Store; "
            "exiting without modifying any row.",
            root,
        )
        return PropagationStats(
            nodes=0,
            draw_revisits=0,
            unevaluated_leaves=0,
            path_cutoffs=0,
            below_threshold_edges=0,
            root_value=0.0,
        )

    # --- Step 2: allocate pass_id ---
    pass_id: int = await store.next_propagation_seq()

    # --- Determine stack depth bound ---
    max_depth = max(cfg.max_book_ply, 1) if cfg.max_book_ply > 0 else _HARD_FRAME_GUARD

    # --- State ---
    stats = PropagationStats(
        nodes=0,
        draw_revisits=0,
        unevaluated_leaves=0,
        path_cutoffs=0,
        below_threshold_edges=0,
        root_value=0.0,
    )

    # Local memo: {PositionKey: prop_value} for nodes computed in this pass.
    # Replaces the need to read prop_epoch from the database (single-writer).
    memo: dict[PositionKey, float] = {}

    # Path set: keys currently on the frame stack (cycle detection, Req 9.5).
    path_keys: set[PositionKey] = set()

    # Explicit frame stack.
    stack: list[_Frame] = []

    # --- Helper: draw value for a given key's side to move ---
    def _draw_value_for(key: PositionKey) -> float:
        if side_to_move_is_white(key):
            return cfg.draw_value_white
        return cfg.draw_value_black

    # --- Helper: resolve a terminal child's contribution ---
    def _terminal_contribution(terminal: Terminal) -> float:
        """Return 1.0 for win-for-STM, 0.0 for loss-for-STM."""
        if terminal == Terminal.WIN_FOR_STM:
            return 1.0
        return 0.0

    # --- Helper: build a frame for a node ---
    async def _build_frame(key: PositionKey, view: BookNodeView) -> _Frame:
        """Prefetch children and construct a frame for the given node."""
        edges = view.edges
        n_edges = len(edges)

        # Compute all child keys in one batched call
        child_keys = position_keys_after(view.sfen, edges["move16"])

        # Partition by threshold
        threshold = cfg.propagation_visit_threshold
        if threshold > 0:
            below_mask = edges["visit_count"] < threshold
        else:
            below_mask = np.zeros(n_edges, dtype=bool)

        above_indices = np.where(~below_mask)[0]
        below_indices = np.where(below_mask)[0]

        # Build key lists for each group
        above_keys = [
            PositionKey(int(child_keys[i]["hi"]), int(child_keys[i]["lo"]))
            for i in above_indices
        ]
        below_keys = [
            PositionKey(int(child_keys[i]["hi"]), int(child_keys[i]["lo"]))
            for i in below_indices
        ]

        # Prefetch both groups concurrently
        full_task = store.get_many(above_keys)
        narrow_task = store.get_many_terminal_eval(below_keys)
        full_children, narrow_children = await asyncio.gather(full_task, narrow_task)

        return _Frame(
            key=key,
            view=view,
            child_keys=child_keys,
            below_mask=below_mask,
            full_children=full_children,
            narrow_children=narrow_children,
            above_indices=above_indices,
            below_indices=below_indices,
            contributions=[None] * n_edges,
            edge_idx=0,
        )

    # --- Push root frame ---
    root_frame = await _build_frame(root, root_view)
    stack.append(root_frame)
    path_keys.add(root)

    # --- Main loop: process frames until stack is empty ---
    while stack:
        frame = stack[-1]
        edges = frame.view.edges
        n_edges = len(edges)

        # If all edges have been processed, finalize this frame
        if frame.edge_idx >= n_edges:
            # Compute propagated value for this node
            best_move16: Optional[int] = None

            if frame.view.terminal != Terminal.NONE:
                # Terminal node (criterion 4): 1 for win, 0 for loss
                value = _terminal_contribution(frame.view.terminal)
            elif n_edges == 0:
                # Edgeless leaf (criteria 9, 10)
                if frame.view.eval_win_rate is not None:
                    value = frame.view.eval_win_rate
                else:
                    value = _draw_value_for(frame.key)
                    stats.unevaluated_leaves += 1
            else:
                # value = 1 - min(Child_Contributions)
                min_contrib = min(c for c in frame.contributions if c is not None)
                value = 1.0 - min_contrib

                # Best move: first edge within 1e-6 of minimum in ascending
                # USI order (edges are already sorted ascending by USI).
                for idx, c in enumerate(frame.contributions):
                    if c is not None and abs(c - min_contrib) < 1e-6:
                        best_move16 = int(edges[idx]["move16"])
                        break

            # Write propagation result to database
            await store.set_propagation([
                PropagationWrite(
                    key=frame.key,
                    prop_value=value,
                    prop_best_move16=best_move16,
                    prop_epoch=pass_id,
                )
            ])

            # Record in memo
            memo[frame.key] = value
            stats.nodes += 1

            # Pop frame
            stack.pop()
            path_keys.discard(frame.key)

            # If there's a parent frame waiting, provide contribution
            if stack:
                parent = stack[-1]
                # The child contribution from the parent's perspective is
                # already handled: the parent stored which edge_idx was
                # pending. The contribution is the child's value (which the
                # parent will negate via 1 - value at finalization, but
                # actually the contribution IS the child's value from the
                # child's perspective, and the parent does 1 - min(...)).
                # Actually, Child_Contribution is already from the child's
                # perspective — the parent does `1 - min(contributions)`.
                # So we store the child's propagated value directly.
                parent.contributions[parent.edge_idx - 1] = value

            continue

        # Process current edge
        i = frame.edge_idx
        frame.edge_idx += 1

        # Get child key for this edge
        child_key = PositionKey(
            int(frame.child_keys[i]["hi"]), int(frame.child_keys[i]["lo"])
        )

        # --- Precedence order (Requirement 9 criterion 15) ---

        # Determine if this edge is below-threshold
        is_below = bool(frame.below_mask[i])

        if is_below:
            # This edge is below threshold — use the narrow prefetch
            # Find which index in below_indices this corresponds to
            below_pos = int(np.searchsorted(frame.below_indices, i))
            child_info = frame.narrow_children[below_pos]

            # Branch 1: Terminal child (criterion 4)
            if child_info is not None:
                terminal, eval_wr = child_info
                if terminal != Terminal.NONE:
                    frame.contributions[i] = _terminal_contribution(terminal)
                    continue

            # Branch 2: Child on current path (criterion 5)
            if child_key in path_keys:
                frame.contributions[i] = _draw_value_for(child_key)
                stats.draw_revisits += 1
                continue

            # Branch 3: Below-threshold rules (criteria 12/13)
            stats.below_threshold_edges += 1
            edge_visit_count = int(edges[i]["visit_count"])
            if edge_visit_count >= 1:
                # Criterion 12: 1 - clip(value_sum / visit_count, 0, 1)
                edge_value_sum = float(edges[i]["value_sum"])
                mean = edge_value_sum / edge_visit_count
                clamped = max(0.0, min(1.0, mean))
                frame.contributions[i] = 1.0 - clamped
            else:
                # Criterion 13: visit_count == 0
                # Use child's eval_win_rate if available, else Draw_Value_*
                if child_info is not None:
                    _, eval_wr = child_info
                    if eval_wr is not None:
                        frame.contributions[i] = eval_wr
                    else:
                        frame.contributions[i] = _draw_value_for(child_key)
                else:
                    # Child row absent
                    frame.contributions[i] = _draw_value_for(child_key)
            continue

        else:
            # This edge is above-threshold — use the full prefetch
            # Find which index in above_indices this corresponds to
            above_pos = int(np.searchsorted(frame.above_indices, i))
            child_view = frame.full_children[above_pos]

            # Branch 1: Terminal child (criterion 4)
            if child_view is not None and child_view.terminal != Terminal.NONE:
                frame.contributions[i] = _terminal_contribution(child_view.terminal)
                continue

            # Branch 2: Child on current path (criterion 5)
            if child_key in path_keys:
                frame.contributions[i] = _draw_value_for(child_key)
                stats.draw_revisits += 1
                continue

            # Branch 3: Below-threshold should not apply here (edge is above)
            # (This branch is unreachable for above-threshold edges)

            # Branch 4: Otherwise — the child's own propagated value
            if child_view is None:
                # Child row absent — treat as unevaluated leaf
                frame.contributions[i] = _draw_value_for(child_key)
                stats.unevaluated_leaves += 1
                continue

            # Check for memo hit: if we've already computed this node in
            # this pass AND the child's Cyclic_Flag is clear
            if child_key in memo and not child_view.cyclic_flag:
                frame.contributions[i] = memo[child_key]
                continue

            # Edgeless leaf (criteria 9, 10)
            if len(child_view.edges) == 0:
                if child_view.eval_win_rate is not None:
                    leaf_value = child_view.eval_win_rate
                else:
                    leaf_value = _draw_value_for(child_key)
                    stats.unevaluated_leaves += 1

                # Write propagation for edgeless leaf
                await store.set_propagation([
                    PropagationWrite(
                        key=child_key,
                        prop_value=leaf_value,
                        prop_best_move16=None,
                        prop_epoch=pass_id,
                    )
                ])
                memo[child_key] = leaf_value
                stats.nodes += 1
                frame.contributions[i] = leaf_value
                continue

            # Need to descend: push child frame (if depth allows)
            if len(stack) >= max_depth:
                stats.path_cutoffs += 1
                # Use the child's existing prop_value if available,
                # else draw value
                if child_view.prop_value is not None:
                    frame.contributions[i] = child_view.prop_value
                else:
                    frame.contributions[i] = _draw_value_for(child_key)
                continue

            # Push child frame — but first, we need to "pause" the current
            # frame at this edge. When the child pops, it will set
            # contributions[edge_idx - 1] which is contributions[i].
            child_frame = await _build_frame(child_key, child_view)
            stack.append(child_frame)
            path_keys.add(child_key)
            # Do NOT continue — the while loop will pick up the new top frame

    # --- Completion ---
    stats.root_value = memo.get(root, 0.0)

    # Mark propagation done
    await store.mark_propagation_done(pass_id)

    _LOG.info(
        "propagate: pass_id=%d complete — nodes=%d, draw_revisits=%d, "
        "unevaluated_leaves=%d, path_cutoffs=%d, below_threshold_edges=%d, "
        "root_value=%.6f",
        pass_id,
        stats.nodes,
        stats.draw_revisits,
        stats.unevaluated_leaves,
        stats.path_cutoffs,
        stats.below_threshold_edges,
        stats.root_value,
    )

    return stats
