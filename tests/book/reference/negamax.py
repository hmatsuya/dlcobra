"""Reference negamax implementation for cross-checking the propagator.

Written directly from Requirement 9's text: the four-way precedence order
(terminal, on-path revisit, below-threshold, then recurse) and the
``value = 1 - min(Child_Contributions)`` aggregation.

This module imports nothing from ``dlshogi.book`` and has no side effects at
import time. It is intentionally simple and recursive (no memo, no batching)
so that it serves as an independent oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Graph model (in-memory, test-only)
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    """One node in the reference graph."""

    # Edges: list of (move16, child_key, visit_count, value_sum)
    edges: List[Tuple[int, str, int, float]] = field(default_factory=list)
    # Terminal status: "win", "loss", or None
    terminal: Optional[str] = None
    # eval_win_rate when edgeless or below threshold
    eval_win_rate: Optional[float] = None


# Graph: dict mapping node_key (str) -> NodeInfo
Graph = Dict[str, NodeInfo]


# ---------------------------------------------------------------------------
# Reference negamax
# ---------------------------------------------------------------------------


def negamax(
    graph: Graph,
    node_key: str,
    threshold: int,
    draw_value_black: float,
    draw_value_white: float,
    *,
    is_white: bool = False,
    path: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Reference negamax from Requirement 9 text.

    Parameters
    ----------
    graph : Graph
        Dict of node_key -> NodeInfo. Keys are opaque strings.
    node_key : str
        The root key to start propagation from.
    threshold : int
        Propagation_Visit_Threshold. Edges with visit_count < threshold are
        resolved via the below-threshold criteria (12/13) instead of recursing.
    draw_value_black : float
        Draw value for black-to-move nodes.
    draw_value_white : float
        Draw value for white-to-move nodes.
    is_white : bool
        Whether the root node is white-to-move (for draw value selection).
    path : set of str or None
        On-path keys for cycle detection (Criterion 5).

    Returns
    -------
    Dict[str, float]
        Mapping from every reachable node_key to its propagated value.
    """
    results: Dict[str, float] = {}
    if path is None:
        path = set()

    def _draw_value(key: str, white: bool) -> float:
        return draw_value_white if white else draw_value_black

    def _recurse(key: str, white: bool) -> float:
        """Compute propagated value for one node (the recursive core)."""
        if key in results:
            return results[key]

        node = graph.get(key)
        if node is None:
            # Absent node: treat as unevaluated leaf
            val = _draw_value(key, white)
            results[key] = val
            return val

        # Criterion 4: terminal node
        if node.terminal == "win":
            results[key] = 1.0
            return 1.0
        elif node.terminal == "loss":
            results[key] = 0.0
            return 0.0

        # Edgeless leaf (criteria 9, 10)
        if not node.edges:
            if node.eval_win_rate is not None:
                val = node.eval_win_rate
            else:
                val = _draw_value(key, white)
            results[key] = val
            return val

        # Collect child contributions
        path.add(key)
        contributions: List[float] = []

        for _move16, child_key, visit_count, value_sum in node.edges:
            child_white = not white  # child has opposite side to move

            # Criterion 5: on-path revisit
            if child_key in path:
                contributions.append(_draw_value(child_key, child_white))
                continue

            # Below-threshold (criteria 12/13)
            if threshold > 0 and visit_count < threshold:
                if visit_count >= 1:
                    # Criterion 12: 1 - clip(value_sum / visit_count, 0, 1)
                    mean = value_sum / visit_count
                    clamped = max(0.0, min(1.0, mean))
                    contributions.append(1.0 - clamped)
                else:
                    # Criterion 13: use child's eval_win_rate or draw value
                    child_node = graph.get(child_key)
                    if child_node is not None and child_node.eval_win_rate is not None:
                        contributions.append(child_node.eval_win_rate)
                    else:
                        contributions.append(_draw_value(child_key, child_white))
                continue

            # Recurse into above-threshold child
            child_value = _recurse(child_key, child_white)
            contributions.append(child_value)

        path.discard(key)

        # value = 1 - min(Child_Contributions)
        if contributions:
            value = 1.0 - min(contributions)
        else:
            # All edges excluded by some means; fall back to draw value
            value = _draw_value(key, white)

        results[key] = value
        return value

    _recurse(node_key, is_white)
    return results
