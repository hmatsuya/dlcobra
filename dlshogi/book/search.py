"""Search_Coordinator: PUCT selection, expansion, and value backup.

Implements the Search_Coordinator component from the PUCT Book Builder
design: the In_Flight_Set (task 13.1), PUCT selection (task 13.3), the
descent task (task 13.6), and the supervisor/resume/shutdown (task 13.12).

**Code invariant (Requirement 11.2):** ``InFlightSet.test_and_add`` is a
plain synchronous ``def``, not ``async def``. It contains no ``await``, no
``asyncio.sleep``, and no call to any coroutine between the membership test
and the insertion. Callers must not interleave anything between obtaining
``True`` and entering the ``try`` block whose ``finally`` releases the
claim.  This invariant is enforced structurally: the function is annotated
``def``, so inserting an ``await`` would be a syntax error.

**No side effects at import time**, per this package's established
convention.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import cshogi
import numpy as np

from dlshogi.book.config import BookConfig
from dlshogi.book.evaluator import Evaluator, EvalResult, EvaluatorBatchError
from dlshogi.book.keys import (
    PositionKey,
    apery_book_key,
    position_key,
    position_key_from_sfen,
    position_keys_after,
    side_to_move_is_white,
)
from dlshogi.book.node_store import (
    BackupDelta,
    BookNodeView,
    ExpansionWrite,
    GetResult,
    NodeStore,
    Terminal,
    WriteOutcome,
    WriteResult,
)
from dlshogi.book.packed_edge import PACKED_EDGE, encode_edges
from dlshogi.book.prior_mixer import mix_priors, terashock_q0, win_rate
from dlshogi.book.repetition import (
    RepetitionClass,
    RepetitionResolver,
    RepetitionResult,
    TerminalResult,
    check_terminal,
)
from dlshogi.book.terashock import TerashockIndex, TerashockLookupResult

_LOG = logging.getLogger(__name__)

# Flag bit-0 in PACKED_EDGE["flags"]: Terashock evaluation present.
_FLAG_TERASHOCK_PRESENT = 0x01

# Flag bit-0 in BackupDelta.flags_or: cyclic flag.
_FLAG_CYCLIC = 0x01

# Per-descent timeout (Requirement 15.6).
_DESCENT_TIMEOUT_S = 10.0

# Claim reaper cycle (design.md): wakes once per second.
_REAPER_INTERVAL_S = 1.0

# Claim reaper threshold: claims older than 300 s are stale and discarded.
_REAPER_STALE_THRESHOLD_S = 300.0

# Requirement 11.3: claims older than 1000 ms are considered stale for
# test_and_add purposes (not enforced in this module's test_and_add itself,
# but documented here).
_CLAIM_STALE_MS = 1000.0

# Shutdown grace period (Requirement 10.7): 60 s.
_SHUTDOWN_GRACE_S = 60.0


# ---------------------------------------------------------------------------
# In_Flight_Set (task 13.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """A single In_Flight_Set claim."""

    worker_id: int
    claimed_at: float  # time.monotonic() timestamp


class InFlightSet:
    """In-process, authoritative In_Flight_Set.

    A plain ``dict[PositionKey, Claim]`` with no lock, because a
    single-threaded asyncio event loop cannot preempt between the test
    and the add.  See the module docstring for the stated code invariant.
    """

    def __init__(self) -> None:
        self._claims: dict[PositionKey, Claim] = {}
        # Counters for Progress_Reporter (task 16.1).
        self.claim_count = 0
        self.release_count = 0
        self.reaper_discard_count = 0

    @property
    def size(self) -> int:
        """Number of currently held claims."""
        return len(self._claims)

    def test_and_add(self, key: PositionKey, worker_id: int) -> bool:
        """Atomically test membership and add a claim.

        Returns ``True`` if the claim was successfully added (the key was
        not already claimed). Returns ``False`` if the key is already
        claimed by another worker.

        **Code invariant (Requirement 11.2):** This is a plain ``def``.
        It contains no ``await``, no ``asyncio.sleep()``, and no call to
        any coroutine between the membership test and the insertion.
        """
        if key in self._claims:
            return False
        self._claims[key] = Claim(worker_id=worker_id, claimed_at=time.monotonic())
        self.claim_count += 1
        return True

    def discard(self, key: PositionKey) -> None:
        """Release a claim, if it exists."""
        if key in self._claims:
            del self._claims[key]
            self.release_count += 1

    def contains(self, key: PositionKey) -> bool:
        """Whether ``key`` is currently claimed."""
        return key in self._claims

    def get_claims(self) -> dict[PositionKey, Claim]:
        """Snapshot of current claims (for diagnostics)."""
        return dict(self._claims)

    def clear(self) -> int:
        """Clear all claims and return how many were cleared (Requirement 10.2)."""
        n = len(self._claims)
        self._claims.clear()
        return n


class ClaimReaper:
    """Background coroutine that discards stale In_Flight_Set claims.

    Wakes once per second (design.md: "a background coroutine waking once
    per second"), reports and discards claims older than 300 s
    (Requirement 11.3's reaper).
    """

    def __init__(self, in_flight: InFlightSet) -> None:
        self._in_flight = in_flight
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the reaper coroutine (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.ensure_future(self._reap_loop())

    async def stop(self) -> None:
        """Stop the reaper coroutine."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _reap_loop(self) -> None:
        """Periodically discard stale claims."""
        try:
            while True:
                await asyncio.sleep(_REAPER_INTERVAL_S)
                now = time.monotonic()
                stale_keys = [
                    key
                    for key, claim in self._in_flight._claims.items()
                    if (now - claim.claimed_at) > _REAPER_STALE_THRESHOLD_S
                ]
                for key in stale_keys:
                    _LOG.warning(
                        "Reaper: discarding stale claim %r (age %.1f s)",
                        key,
                        now - self._in_flight._claims[key].claimed_at,
                    )
                    self._in_flight.discard(key)
                    self._in_flight.reaper_discard_count += 1
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# PUCT Selection (task 13.3)
# ---------------------------------------------------------------------------


def select_edge(
    edges: np.ndarray,
    node_visit_count: int,
    c_puct: float,
    virtual_loss: int,
    in_flight_mask: np.ndarray,
    excluded_mask: np.ndarray,
    eval_coef: float,
) -> Optional[int]:
    """Vectorized PUCT edge selection (Requirements 4.2, 4.7, 11.4).

    Parameters
    ----------
    edges : np.ndarray
        Decoded PACKED_EDGE array (dtype=PACKED_EDGE) for the node.
    node_visit_count : int
        The parent node's visit_count.
    c_puct : float
        The C_PUCT exploration constant (design.md uses sqrt(2) by convention;
        this parameter allows the caller to pass any value).
    virtual_loss : int
        The configured Virtual_Loss value (0..16).
    in_flight_mask : np.ndarray
        Boolean mask: True where the edge's child is currently in the
        In_Flight_Set (contributes virtual loss).
    excluded_mask : np.ndarray
        Boolean mask: True where the edge is completely excluded for this
        descent (in-flight children; Requirement 4.7). Gets -inf score.
    eval_coef : float
        The configured Eval_Coef for computing q0 from Terashock evals.

    Returns
    -------
    int or None
        Index into ``edges`` of the selected edge, or None if all edges
        are excluded.
    """
    n_edges = len(edges)
    if n_edges == 0:
        return None

    # Check if all edges are excluded
    if excluded_mask.all():
        return None

    # Extract fields
    n = edges["visit_count"].astype(np.float64)
    w = edges["value_sum"]
    p = edges["prior_q16"].astype(np.float64) * (1.0 / 65535.0)

    # Virtual loss: applied only where in_flight_mask is True
    vl = virtual_loss * in_flight_mask.astype(np.float64)
    n_eff = n + vl
    n_par = float(node_visit_count) + vl.sum()

    # q0: 0.5 for edges with no Terashock evaluation,
    # win_rate(ts_eval, eval_coef) otherwise.
    flags = edges["flags"]
    ts_eval = edges["ts_eval"].astype(np.float64)
    has_terashock = (flags & _FLAG_TERASHOCK_PRESENT).astype(np.bool_)
    q0 = np.where(has_terashock, win_rate(ts_eval, eval_coef), 0.5)

    # Q-value: w/n_eff where n_eff > 0, else q0
    q = np.where(n_eff > 0, w / np.maximum(n_eff, 1.0), q0)

    # PUCT score
    score = q + c_puct * p * (np.sqrt(n_par) / (1.0 + n_eff))

    # Excluded edges get -inf
    score = np.where(excluded_mask, -np.inf, score)

    # Tie-break: first index within 1e-6 of max (USI-sorted array)
    max_score = score.max()
    if not np.isfinite(max_score):
        # All scores are -inf (all excluded)
        return None

    candidates = np.flatnonzero(score >= max_score - 1e-6)
    return int(candidates[0])


# ---------------------------------------------------------------------------
# Descent path tracking
# ---------------------------------------------------------------------------


@dataclass
class _PathNode:
    """One node on the descent path, used for value backup."""

    key: PositionKey
    side_to_move_is_white: int  # 0 = Black, 1 = White (from key.hi & 1)
    edge_move16: Optional[int] = None  # The move16 that brought us here (None for root)


# ---------------------------------------------------------------------------
# Descent task (task 13.6) and SearchCoordinator (task 13.12)
# ---------------------------------------------------------------------------


class SearchCoordinator:
    """The Search_Coordinator: supervisor over Worker_Count descent tasks.

    Construct with a validated ``BookConfig``, a connected ``NodeStore``,
    a started ``Evaluator``, and an optional ``TerashockIndex``. Then
    ``await coordinator.run()`` to start the search loop.
    """

    def __init__(
        self,
        config: BookConfig,
        node_store: NodeStore,
        evaluator: Evaluator,
        terashock_index: Optional[TerashockIndex] = None,
    ) -> None:
        self._config = config
        self._store = node_store
        self._evaluator = evaluator
        self._terashock = terashock_index

        self.in_flight = InFlightSet()
        self._reaper = ClaimReaper(self.in_flight)

        # Shutdown coordination
        self._stop_requested = False
        self._shutdown_event = asyncio.Event()

        # Counters
        self.descents_completed = 0
        self.descents_abandoned = 0
        self.descents_timed_out = 0
        self.descents_failed = 0
        self.expansions_committed = 0
        self.expansions_duplicate = 0
        self.expansions_collision = 0
        self.expansions_failed = 0
        self.forced_shutdown = False

    # -- public entry point ------------------------------------------------

    async def run(self) -> None:
        """Run the search loop until stopped (Requirements 4.13, 10.5-10.8, 11.1).

        Installs SIGINT/SIGTERM handlers, starts the claim reaper, creates
        exactly Worker_Count descent tasks, and replaces each completed task.
        On stop, lets in-flight descents finish within 60 s, flushes the
        accumulator, and exits.
        """
        loop = asyncio.get_running_loop()

        # Install signal handlers (Requirement 10.6)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        # Start claim reaper
        self._reaper.start()

        # Clear any leftover in-flight claims (Requirement 10.2)
        cleared = self.in_flight.clear()
        if cleared > 0:
            _LOG.info("Startup: cleared %d stale in-flight claims", cleared)

        # Verify root position matches
        await self._check_root_position()

        worker_count = self._config.worker_count
        tasks: dict[asyncio.Task, int] = {}  # task -> worker_id
        next_worker_id = 0

        try:
            # Launch initial Worker_Count tasks
            for _ in range(worker_count):
                if self._stop_requested:
                    break
                task = asyncio.ensure_future(
                    self._run_descent_task(next_worker_id)
                )
                tasks[task] = next_worker_id
                next_worker_id += 1

            # Supervisor loop: replace completed tasks
            while tasks and not self._stop_requested:
                done, _ = await asyncio.wait(
                    tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    worker_id = tasks.pop(task)
                    # Retrieve result/exception
                    exc = task.exception() if not task.cancelled() else None
                    if exc is not None:
                        _LOG.error(
                            "Worker %d terminated with exception: %r",
                            worker_id,
                            exc,
                        )
                        self.descents_failed += 1

                    # Replace the task if not stopping
                    if not self._stop_requested:
                        new_task = asyncio.ensure_future(
                            self._run_descent_task(next_worker_id)
                        )
                        tasks[new_task] = next_worker_id
                        next_worker_id += 1

            # Stop requested: wait for in-flight descents to finish
            if tasks:
                _LOG.info(
                    "Stop requested: waiting for %d in-flight descent(s)...",
                    len(tasks),
                )
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks.keys(), return_exceptions=True),
                        timeout=_SHUTDOWN_GRACE_S,
                    )
                except asyncio.TimeoutError:
                    _LOG.warning(
                        "Shutdown grace period exceeded; cancelling %d task(s)",
                        len(tasks),
                    )
                    self.forced_shutdown = True
                    for task in tasks:
                        task.cancel()
                    # Await cancelled tasks
                    await asyncio.gather(*tasks.keys(), return_exceptions=True)

        finally:
            # Stop the reaper
            await self._reaper.stop()

            # Flush the accumulator (Requirement 10.7)
            try:
                await self._store.flush()
            except Exception:  # noqa: BLE001
                _LOG.error("Failed to flush accumulator during shutdown", exc_info=True)

            # Remove signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)

        if self.forced_shutdown:
            _LOG.warning("Forced shutdown completed (grace period exceeded)")
        else:
            _LOG.info(
                "Search stopped: %d descents completed, %d abandoned, %d timed out",
                self.descents_completed,
                self.descents_abandoned,
                self.descents_timed_out,
            )

    # -- signal handler ----------------------------------------------------

    def _request_stop(self) -> None:
        """Set the stop flag (called from signal handler)."""
        if not self._stop_requested:
            _LOG.info("Stop requested (signal received)")
            self._stop_requested = True
            self._shutdown_event.set()

    # -- root position check -----------------------------------------------

    async def _check_root_position(self) -> None:
        """Verify that the configured Root_Position matches the stored one.

        Raises if there's a mismatch (Requirement 10.1).
        """
        root_sfen, root_key_hi, root_key_lo = await self._store.get_root()
        configured_key = position_key_from_sfen(self._config.root_position)
        from dlshogi.book.keys import fold_u64_to_i64

        expected_hi = fold_u64_to_i64(configured_key.hi)
        expected_lo = fold_u64_to_i64(configured_key.lo)
        if expected_hi != root_key_hi or expected_lo != root_key_lo:
            raise RuntimeError(
                f"Root_Position mismatch: configured SFEN "
                f"{self._config.root_position!r} produces key "
                f"({configured_key.hi:#x}, {configured_key.lo:#x}), "
                f"but the database records ({root_key_hi:#x}, {root_key_lo:#x}). "
                f"Use the same Root_Position as the original schema creation."
            )

    # -- per-task descent wrapper ------------------------------------------

    async def _run_descent_task(self, worker_id: int) -> None:
        """One descent task with timeout and cleanup (Requirements 11.9, 15.6).

        The per-task ``claimed`` list is released in ``finally``, so
        cancellation, timeout, and arbitrary exceptions all take the same
        cleanup path.
        """
        claimed: List[PositionKey] = []
        try:
            await asyncio.wait_for(
                self._descent(worker_id, claimed),
                timeout=_DESCENT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            _LOG.debug("Worker %d: descent timed out", worker_id)
            self.descents_timed_out += 1
        except asyncio.CancelledError:
            # Cooperative shutdown (Requirement 10.6): re-raise after cleanup
            raise
        except EvaluatorBatchError:
            # Requirement 5.6: batch failure, descent discarded
            _LOG.debug("Worker %d: evaluator batch failed", worker_id)
            self.descents_failed += 1
        except Exception:
            _LOG.error(
                "Worker %d: unexpected exception in descent",
                worker_id,
                exc_info=True,
            )
            self.descents_failed += 1
        finally:
            # Release all claims this task holds (Requirement 11.9)
            for key in claimed:
                self.in_flight.discard(key)

    # -- the descent coroutine ---------------------------------------------

    async def _descent(self, worker_id: int, claimed: List[PositionKey]) -> None:
        """One complete Selection_Descent (task 13.6).

        Traverses the Book_Graph from the root, selecting edges via PUCT,
        expanding leaf nodes, and backing up the leaf value through the
        descent path.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DESCENT_TIMEOUT_S

        config = self._config
        root_key = position_key_from_sfen(config.root_position)

        # Set up a cshogi Board at the root position
        board = cshogi.Board(config.root_position)

        # Repetition resolver for this descent
        resolver = RepetitionResolver(
            draw_value_black=config.draw_value_black,
            draw_value_white=config.draw_value_white,
        )

        # Descent path for value backup
        path: List[_PathNode] = []

        # C_PUCT constant (design.md uses sqrt(2) as the standard value)
        c_puct = 2.0 ** 0.5

        # Ensure root node exists
        result, root_view = await self._store.get(root_key)
        if result == GetResult.ABSENT:
            # Create root node (visit_count=0, value_sum=0)
            root_write = ExpansionWrite(
                key=root_key,
                sfen=config.root_position,
                apery_key=apery_book_key(config.root_position),
                terminal=Terminal.NONE,
                eval_win_rate=None,
                edges=(),
            )
            write_result = await self._store.insert_expansion(root_write)
            if write_result.outcome == WriteOutcome.FAILED:
                _LOG.error("Failed to create root node")
                return
            # Re-read root
            result, root_view = await self._store.get(root_key)
            if result != GetResult.FOUND:
                _LOG.error("Root node not found after creation")
                return
        elif result == GetResult.FAILED:
            _LOG.error("Failed to read root node")
            return

        assert root_view is not None
        current_key = root_key
        current_view = root_view
        depth = 0

        # Push root onto repetition resolver
        resolver.push(root_key, board.is_check())

        # Track the root on the path (no edge brought us here)
        path.append(_PathNode(
            key=root_key,
            side_to_move_is_white=side_to_move_is_white(root_key),
            edge_move16=None,
        ))

        leaf_value: Optional[float] = None
        cyclic_indices: tuple[int, ...] = ()

        while True:
            # Per-node deadline check for CPU-bound stalls
            if loop.time() > deadline:
                # Abandon: timeout reached
                return

            # Stop check
            if self._stop_requested:
                return

            # Termination rule 1: Max_Book_Ply depth reached
            if config.max_book_ply > 0 and depth >= config.max_book_ply:
                # Leaf: value from terminal check or evaluator
                terminal_result = check_terminal(board)
                if terminal_result.terminal != Terminal.NONE:
                    leaf_value = terminal_result.value
                else:
                    # Check if this is a resume case (already evaluated)
                    if self._is_already_evaluated(current_view):
                        leaf_value = current_view.eval_win_rate if current_view.eval_win_rate is not None else 0.5
                    else:
                        eval_result = await self._evaluator.enqueue(board)
                        leaf_value = eval_result.win_rate
                break

            # Termination rule 2: Terminal state
            terminal_result = check_terminal(board)
            if terminal_result.terminal != Terminal.NONE:
                leaf_value = terminal_result.value
                break

            # Check repetition
            rep_result = resolver.classify(board)
            if rep_result.classification != RepetitionClass.NON_REPETITION:
                leaf_value = rep_result.value
                cyclic_indices = rep_result.cyclic_indices
                break

            # Termination rule 3: No edges -> expand
            if len(current_view.edges) == 0 and current_view.terminal == Terminal.NONE:
                # Check if this is a resume case
                if self._is_already_evaluated(current_view):
                    # Already evaluated on a previous run; use stored win rate
                    leaf_value = current_view.eval_win_rate if current_view.eval_win_rate is not None else 0.5
                    break

                # Expand: claim the node in the In_Flight_Set
                if not self.in_flight.test_and_add(current_key, worker_id):
                    # Already claimed by another worker; abandon
                    self.descents_abandoned += 1
                    return
                claimed.append(current_key)

                try:
                    # Generate legal moves
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        # Terminal: no legal moves (shouldn't reach here, but guard)
                        leaf_value = 0.0
                        break

                    # Evaluator
                    eval_result = await self._evaluator.enqueue(board)

                    # Terashock lookup
                    ts_lookup: Optional[TerashockLookupResult] = None
                    if self._terashock is not None:
                        ts_lookup = await self._terashock.lookup(current_key)

                    # Build edge data
                    edges_data = self._build_edges(
                        board, legal_moves, eval_result, ts_lookup
                    )

                    # Write expansion
                    expansion = ExpansionWrite(
                        key=current_key,
                        sfen=board.sfen(),
                        apery_key=apery_book_key(board.sfen()),
                        terminal=Terminal.NONE,
                        eval_win_rate=eval_result.win_rate,
                        edges=edges_data,
                    )
                    write_result = await self._store.insert_expansion(expansion)

                    if write_result.outcome == WriteOutcome.COMMITTED:
                        self.expansions_committed += 1
                    elif write_result.outcome == WriteOutcome.DUPLICATE:
                        self.expansions_duplicate += 1
                    elif write_result.outcome == WriteOutcome.COLLISION:
                        self.expansions_collision += 1
                    elif write_result.outcome == WriteOutcome.FAILED:
                        self.expansions_failed += 1
                        return

                    # Leaf value is the evaluator's win rate
                    leaf_value = eval_result.win_rate
                finally:
                    # Release claim
                    self.in_flight.discard(current_key)
                    if current_key in claimed:
                        claimed.remove(current_key)

                break

            # Termination rule 4: Select an edge
            edges = current_view.edges
            if len(edges) == 0:
                # No edges and terminal -- already handled above; this is
                # a node that had its edges but all are excluded.
                leaf_value = current_view.eval_win_rate if current_view.eval_win_rate is not None else 0.5
                break

            # Determine excluded edges (children in In_Flight_Set)
            child_keys = self._get_child_keys(current_view, edges)
            excluded_mask = np.array(
                [self.in_flight.contains(ck) for ck in child_keys],
                dtype=np.bool_,
            )
            # in_flight_mask is the same as excluded_mask for virtual loss
            in_flight_mask = excluded_mask.copy()

            selected_idx = select_edge(
                edges=edges,
                node_visit_count=current_view.visit_count,
                c_puct=c_puct,
                virtual_loss=config.virtual_loss,
                in_flight_mask=in_flight_mask,
                excluded_mask=excluded_mask,
                eval_coef=config.eval_coef,
            )

            if selected_idx is None:
                # All edges excluded (Requirement 4.9): abandon
                self.descents_abandoned += 1
                return

            # Move to the selected child
            selected_edge = edges[selected_idx]
            move16 = int(selected_edge["move16"])
            move = cshogi.move16_to_move(move16, board)
            board.push(move)
            depth += 1

            # Compute child key
            child_key = child_keys[selected_idx]

            # Push onto repetition resolver
            resolver.push(child_key, board.is_check())

            # Record on path
            path.append(_PathNode(
                key=child_key,
                side_to_move_is_white=side_to_move_is_white(child_key),
                edge_move16=move16,
            ))

            # Read child node
            child_result, child_view = await self._store.get(child_key)
            if child_result == GetResult.FAILED:
                _LOG.error("Worker %d: failed to read child node", worker_id)
                return
            elif child_result == GetResult.ABSENT:
                # Child doesn't exist yet -- this will be an expansion on the
                # next iteration (edges will be empty on the view we build)
                child_view = BookNodeView(
                    key=child_key,
                    sfen=board.sfen(),
                    apery_key=apery_book_key(board.sfen()),
                    visit_count=0,
                    value_sum=0.0,
                    terminal=Terminal.NONE,
                    cyclic_flag=False,
                    eval_win_rate=None,
                    prop_value=None,
                    prop_best_move16=None,
                    edges=np.empty(0, dtype=PACKED_EDGE),
                )

            current_key = child_key
            current_view = child_view

        # -- Value backup --------------------------------------------------

        if leaf_value is None:
            # Should not happen, but guard against it
            return

        self.descents_completed += 1

        # Compute the leaf's side to move
        leaf_stm_is_white = side_to_move_is_white(current_key)

        # Build backup deltas for each node on the path
        deltas: List[BackupDelta] = []

        for i, path_node in enumerate(path):
            # Perspective conversion: if the leaf's stm differs from this
            # node's stm, the node gets 1 - leaf_value
            node_stm_is_white = path_node.side_to_move_is_white
            if node_stm_is_white == leaf_stm_is_white:
                node_value = leaf_value
            else:
                node_value = 1.0 - leaf_value

            # Determine flags_or for cyclic marking
            flags_or = 0
            if cyclic_indices and i in cyclic_indices:
                flags_or = _FLAG_CYCLIC

            # Edge delta: the edge that brought us INTO the next node
            # is recorded on the PARENT node. So for path_node at index i,
            # the edge delta uses the move16 from path[i+1] (if it exists).
            edge_deltas: dict[int, tuple[int, float]] = {}
            if i + 1 < len(path):
                next_move16 = path[i + 1].edge_move16
                if next_move16 is not None:
                    # Edge value is from the parent's perspective (same as node_value)
                    edge_deltas[next_move16] = (1, node_value)

            delta = BackupDelta(
                key=path_node.key,
                node_visit_delta=1,
                node_value_delta=node_value,
                flags_or=flags_or,
                edge_deltas=edge_deltas,
            )
            deltas.append(delta)

        # Apply backup (plain def, not async)
        if deltas:
            self._store.backup(deltas)

    # -- helpers -----------------------------------------------------------

    def _is_already_evaluated(self, view: BookNodeView) -> bool:
        """Resume check: a node with edges or a terminal state is already evaluated.

        Requirement 10.1: the Evaluator is not invoked for such nodes.
        """
        if view.terminal != Terminal.NONE:
            return True
        if len(view.edges) > 0:
            return True
        if view.eval_win_rate is not None:
            return True
        return False

    def _build_edges(
        self,
        board: cshogi.Board,
        legal_moves: list[int],
        eval_result: EvalResult,
        ts_lookup: Optional[TerashockLookupResult],
    ) -> list[dict[str, Any]]:
        """Build the edge list for an expansion (Requirements 4.3, 7.2, 7.4, 7.6).

        Returns a list of dicts suitable for ``encode_edges``, sorted
        ascending by move USI string.
        """
        n = len(legal_moves)
        moves16 = np.array(
            [cshogi.move16(m) for m in legal_moves], dtype=np.uint16
        )

        # Terashock matching: find which legal moves have a Terashock entry
        ts_evals = np.zeros(n, dtype=np.int16)
        ts_depths = np.zeros(n, dtype=np.uint8)
        ts_mask = np.zeros(n, dtype=np.bool_)

        if ts_lookup is not None and len(ts_lookup.moves) > 0:
            # Build a map from move16 -> (eval, depth) from Terashock moves
            ts_move_map: dict[int, tuple[int, int]] = {}
            for ts_move in ts_lookup.moves:
                ts_move_map[int(ts_move["move16"])] = (
                    int(ts_move["eval"]),
                    int(ts_move["depth"]),
                )

            # Match against legal moves (Requirement 7.4: illegal Terashock
            # moves are dropped by virtue of not appearing in legal_moves)
            for i, m16 in enumerate(moves16):
                if int(m16) in ts_move_map:
                    ts_evals[i], ts_depths[i] = ts_move_map[int(m16)]
                    ts_mask[i] = True

        # Mix priors (Requirement 7.3)
        prior_q16 = mix_priors(
            policy=eval_result.policy,
            ts_evals=ts_evals,
            ts_mask=ts_mask,
            terashock_prior_weight=self._config.terashock_prior_weight,
            eval_coef=self._config.eval_coef,
        )

        # Build edges and sort by USI string
        edge_list: list[tuple[str, dict[str, Any]]] = []
        for i, move in enumerate(legal_moves):
            m16 = int(moves16[i])
            usi_str = cshogi.move_to_usi(move)
            flags = _FLAG_TERASHOCK_PRESENT if ts_mask[i] else 0
            edge_list.append((
                usi_str,
                {
                    "move16": m16,
                    "prior_q16": int(prior_q16[i]),
                    "ts_depth": int(ts_depths[i]),
                    "flags": flags,
                    "ts_eval": int(ts_evals[i]),
                    "visit_count": 0,
                    "value_sum": 0.0,
                },
            ))

        # Sort ascending by USI string
        edge_list.sort(key=lambda x: x[0])
        return [e[1] for e in edge_list]

    def _get_child_keys(
        self, view: BookNodeView, edges: np.ndarray
    ) -> list[PositionKey]:
        """Compute the Position_Key of each edge's child.

        Uses ``position_keys_after`` for efficient batch computation.
        """
        moves16 = edges["move16"]
        keys_arr = position_keys_after(view.sfen, moves16)
        return [
            PositionKey(int(keys_arr[i]["hi"]), int(keys_arr[i]["lo"]))
            for i in range(len(keys_arr))
        ]
