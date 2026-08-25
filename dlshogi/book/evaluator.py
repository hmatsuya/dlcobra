"""Evaluator: batched neural-network evaluation of expanded nodes.

Implements the *mechanism* half of design.md's Evaluator component
(Requirement 5): one collector coroutine per process that batches
`enqueue()` calls from concurrent descent tasks over an `asyncio.Queue`,
dispatches to an injected inference session at `Batch_Size` or
`Batch_Timeout` (whichever comes first), decodes the policy head into a
per-legal-move probability distribution via `cshogi.dlshogi.make_move_label`,
and resolves every request's `asyncio.Future` with an `EvalResult` or fails
the whole batch on any invocation error, short result, or non-finite
output.

**Scope of this module (task 10.1, plus the task 10.5 adapter).** The
neural network itself is *injected* as ``session``, an object exposing a
synchronous ``run(features1_batch, features2_batch) -> (policy_logits_batch,
values_batch)`` method (the `InferenceSession` protocol below) -- design.md:
"The session is injected, so a stub returning drawn arrays satisfies the
whole property suite with no GPU." `OnnxRuntimeInferenceSession` below is
task 10.5's real onnxruntime adapter (TensorRT execution provider,
`io_binding`, following `dlshogi/utils/usi_policy_only.py`), implementing
this same `InferenceSession` protocol; nothing in the `Evaluator` class
itself changes to accommodate it -- it is constructed and passed to
`Evaluator(session=..., ...)` exactly as a test stub would be. Properties
17, 18, and 19 (tasks 10.2-10.4) are not implemented here.

**Batching mechanism.** `Evaluator.enqueue(board)` computes this
Board_State's features with `cshogi.dlshogi.make_input_features` into a
small per-request array (so a descent task's synchronous write into it can
never race with any other request's write), wraps them into an
`EvalRequest` together with the legal-move list and side to move, puts the
request on an unbounded `asyncio.Queue`, and awaits the request's future.
A single background collector task (`start()`/`stop()`, mirroring
`node_store.py`'s flusher lifecycle) pops requests from that queue into the
Evaluator's own preallocated per-batch staging buffer -- two numpy arrays
of shape ``(Batch_Size, FEATURES1_NUM, 9, 9)`` / ``(Batch_Size,
FEATURES2_NUM, 9, 9)`` -- and dispatches to the injected session once the
buffer reaches `Batch_Size` (Requirement 5.2) or once `Batch_Timeout` has
elapsed since the first request of the currently-forming batch arrived
(Requirement 5.3), whichever comes first. The deadline is computed once,
right after the first item of a new batch is popped from an otherwise
empty queue, and `remaining = deadline - loop.time()` is recomputed on
every iteration of the inner accumulation loop, exactly as design.md's
Evaluator section describes ("The earliest-arrival timestamp is captured
when the first request enters an empty buffer, so the timeout is measured
from accumulation, not from the last arrival").

**Why `run_in_executor`.** The injected session's `run` method is a
synchronous, blocking call (onnxruntime's `run_with_iobinding` is
synchronous C code even when using the TensorRT/CUDA execution
providers), so the collector dispatches it via
``loop.run_in_executor(None, session.run, f1, f2)`` rather than calling it
directly on the event loop thread. This keeps the rest of the process
(other descent tasks' `NodeStore` reads/writes, the backup flusher, ...)
running concurrently with one in-flight inference call, consistent with
design.md's "Why Python, and not C++" argument that this workload is
overwhelmingly I/O-wait, not CPU-bound. A synchronous test stub works
identically through the executor; it simply runs on a worker thread
instead of the event-loop thread, which numpy array reads/writes tolerate
fine since no other code touches the same staging-buffer rows until this
dispatch's `_invoke_session` call has returned (the collector never begins
forming the next batch until `await self._dispatch(batch)` completes).

**Failure handling (Requirement 5.6).** An exception raised by the
injected session, a returned result shorter than the dispatched batch, or
a non-finite win rate or move probability anywhere in the batch marks
*every* request in that batch failed -- design.md states this plainly:
"a non-finite value, a short result array, or an exception from
`run_with_iobinding` marks every request in the batch failed". Each
affected request's future has its exception set rather than its result;
`Evaluator.failure_count` accumulates the number of *requests* (not
batches) affected, for a future `report.py` to read.

**Degenerate policy substitution (Requirement 5.7).** This is a
per-request condition, distinct from 5.6: the softmax denominator (the
sum of the exponentiated, gathered legal-move logits) being 0 or
non-finite -- Property 17's own wording, "vectors whose exponentiated sum
is 0 or non-finite" -- is substituted with a uniform ``1/n`` distribution
*before* the division that would otherwise complete the softmax, and
`Evaluator.substitution_count` is incremented. A request that hits this
branch still succeeds (its future resolves normally with the uniform
policy); only 5.6's conditions fail the whole batch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import cshogi
import cshogi.dlshogi as _dlshogi
import numpy as np
import onnxruntime as ort

_LOG = logging.getLogger(__name__)

# cshogi.dlshogi exposes these as plain module attributes (measured:
# FEATURES1_NUM == 62, FEATURES2_NUM == 57); aliased here so the rest of
# this module reads design.md's own names rather than the "_dlshogi."
# prefix at every use site.
FEATURES1_NUM: int = _dlshogi.FEATURES1_NUM
FEATURES2_NUM: int = _dlshogi.FEATURES2_NUM

# How often the collector loop wakes, when idle, to re-check whether
# `stop()` has been requested. Bounded well below any plausible
# Batch_Timeout (1 to 1000 ms per Requirement 13.7) so a stop request is
# never meaningfully delayed by this poll; not itself part of the
# Requirement 5.2/5.3 dispatch decision, which is driven entirely by the
# per-batch deadline computed inside `_collector_loop` below.
_IDLE_POLL_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# The injectable inference session
# ---------------------------------------------------------------------------


class InferenceSession(Protocol):
    """The Evaluator's injectable neural-network interface (Requirement 5.1).

    ``run(features1, features2)`` takes two batched numpy arrays --
    ``features1`` of shape ``(n, FEATURES1_NUM, 9, 9)``, ``features2`` of
    shape ``(n, FEATURES2_NUM, 9, 9)``, both ``float32`` -- and returns
    ``(policy_logits, values)``: ``policy_logits`` of shape
    ``(n, 9*9*MAX_MOVE_LABEL_NUM)`` (the raw, pre-softmax policy head, 2187
    entries wide per design.md's own count) and ``values`` of shape
    ``(n,)`` or ``(n, 1)``, the win rate for the side to move at each of
    the ``n`` Board_States, already through whatever sigmoid the trained
    model applies (`dlshogi/convert_model_to_onnx.py` exports with
    ``add_sigmoid=True`` for exactly this reason -- Requirement 5.5's [0,
    1] win rate is the network's direct output, with no further transform
    needed here).

    This method is called synchronously (task 10.5's real onnxruntime
    adapter drives ``session.run_with_iobinding`` the same way
    `dlshogi/utils/usi_policy_only.py` does); the Evaluator itself is what
    moves the call off the event loop thread, via
    ``loop.run_in_executor`` (see this module's docstring). A test stub
    needs only to return two arrays of the right shape from a plain
    method or a duck-typed object with a ``run`` attribute -- no
    inheritance from this `Protocol` is required, per Python's structural
    typing.
    """

    def run(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class OnnxRuntimeInferenceSession:
    """The real `InferenceSession` adapter: onnxruntime, TensorRT-first (task 10.5).

    Implements the `InferenceSession` protocol above (structurally --
    Python's `Protocol` needs no explicit subclassing) over a real
    ``onnxruntime.InferenceSession``, following the pattern already used
    by `dlshogi/utils/usi_policy_only.py`: `io_binding.bind_cpu_input`
    for ``"input1"``/``"input2"``, `io_binding.bind_output` for
    ``"output_policy"``/``"output_value"``, and
    ``session.run_with_iobinding(io_binding)``. Nothing in `Evaluator`
    changes to accommodate this class -- it is constructed and passed to
    `Evaluator(session=..., ...)` exactly as a test stub would be, per
    task 10.1's docstring note anticipating this class.

    **Provider selection.** Tries ``TensorrtExecutionProvider`` first,
    with ``trt_fp16_enable=True`` and ``trt_engine_cache_enable=True``
    (plus an engine cache directory, so repeated runs against the same
    model and batch shape reuse the compiled engine rather than rebuilding
    it every process start), falling back to ``CUDAExecutionProvider`` --
    design.md's "Inference: ONNX through onnxruntime, TensorRT execution
    provider first" and the ``session = ort.InferenceSession(model_path,
    providers=[("TensorrtExecutionProvider", {...}), ("CUDAExecutionProvider",
    {...})])`` snippet it gives. onnxruntime's own provider fallback
    mechanism handles "TensorRT unavailable or fails to load" for us: a
    session constructed with a provider list that includes an
    unavailable provider silently falls back to the next provider in the
    list (verified empirically against this development machine, which
    has no GPU at all: constructing with
    ``providers=[("TensorrtExecutionProvider", {}), ("CUDAExecutionProvider",
    {})]`` warns and falls back all the way to ``CPUExecutionProvider``
    with no exception raised). ``CPUExecutionProvider`` is always appended
    last as a final fallback, so a session can always be constructed even
    when neither GPU provider is available -- an operator running the
    property/unit-test suite or a CPU-only development machine still gets
    a working (if slow) session rather than a startup failure; the
    resolved provider is what `get_providers()` reports after
    construction either way, and Requirement 5's own correctness criteria
    make no reference to which provider was used.

    **Provider logging (task 10.5's "log the resolved provider at
    startup").** ``__init__`` logs the requested provider list and the
    provider `onnxruntime` actually resolved (``session.get_providers()``,
    the first entry of which is always the one that will execute the
    graph), via this module's own ``_LOG`` at ``INFO`` level, so an
    Operator can see from the process log alone whether the run is
    getting the TensorRT path, the CUDA fallback, or the CPU fallback.

    **Zero-padding a short final batch.** A TensorRT engine, once built,
    is compiled for the exact input shape it was first invoked with
    (dynamic shapes force a rebuild unless the engine cache already holds
    one for that shape); design.md's own instruction is to "zero-pad a
    short final batch rather than triggering an engine rebuild, discarding
    the padded outputs". `run` therefore always invokes the underlying
    session with exactly `fixed_batch_size` rows (`Batch_Size`, the same
    ``n`` `Evaluator._dispatch` would otherwise have called with only
    partially filled): when the caller's ``features1``/``features2`` have
    fewer than `fixed_batch_size` rows, the remainder of a preallocated,
    zero-filled staging buffer supplies the padding rows, and only the
    first ``n`` rows of the two returned output arrays are sliced back
    off before returning -- so from `Evaluator._dispatch`'s point of view
    (which always calls ``session.run(f1, f2)`` with ``f1``/``f2`` already
    sliced to length ``n``, per its own ``_invoke_session``/`_dispatch`
    implementation) `run`'s return value is shaped for exactly the ``n``
    rows requested, with the padding fully hidden inside this adapter.
    When ``fixed_batch_size`` is `None` (the constructor default), no
    padding is applied and the session is invoked with whatever shape the
    caller passed in directly -- the padding behaviour is opt-in via
    `fixed_batch_size`, so this same adapter also works unpadded against a
    CUDA/CPU session with no fixed engine shape to protect, or in a test
    that wants to see the exact array it passed through.

    **`run` is synchronous.** `run` itself performs no ``await`` and is
    not a coroutine, exactly like the `InferenceSession` protocol
    requires; `Evaluator._invoke_session` is what moves the call off the
    event loop thread via ``loop.run_in_executor``, so blocking C code
    inside ``run_with_iobinding`` never stalls the rest of the process.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        device_id: int = 0,
        fixed_batch_size: Optional[int] = None,
        trt_fp16_enable: bool = True,
        trt_engine_cache_enable: bool = True,
        trt_engine_cache_path: Optional[Union[str, Path]] = None,
        provider_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
        session_options: Optional["ort.SessionOptions"] = None,
    ) -> None:
        """Build the onnxruntime session and resolve its execution provider.

        ``model_path`` is the ``.onnx`` file produced by
        `dlshogi/convert_model_to_onnx.py` (design.md: "producing the ONNX
        model the Evaluator loads"), exporting with ``input1``/``input2``
        and ``output_policy``/``output_value`` names -- exactly the names
        this adapter's `run` binds. ``device_id`` is the GPU index this
        process's Evaluator is pinned to (design.md's per-GPU-process
        model: "one process per GPU"; the ``gpu_id`` of design.md's own
        provider-options snippet). ``fixed_batch_size``, when given, is
        the shape every TensorRT engine build is compiled for -- normally
        `BookConfig.batch_size` -- and enables the zero-pad-a-short-batch
        behaviour described in this class's own docstring;
        `trt_fp16_enable`/`trt_engine_cache_enable` are passed straight
        through to the ``TensorrtExecutionProvider`` options dict.
        ``trt_engine_cache_path`` defaults to a ``trt_cache`` directory
        next to ``model_path`` when engine caching is enabled, created if
        it does not already exist, so repeated runs against the same
        model reuse the compiled engine rather than rebuilding it on
        every process start. ``provider_options``, when given, lets a
        caller override or extend the constructed provider option dicts
        by provider name (e.g. to add ``"CUDAExecutionProvider"`` options
        this constructor does not otherwise set), merged in *after* this
        constructor's own defaults so a caller's explicit choice always
        wins. ``session_options`` is passed straight through to
        ``ort.InferenceSession`` when given.
        """
        model_path = Path(model_path)
        self._fixed_batch_size = fixed_batch_size

        trt_options: dict[str, Any] = {
            "device_id": device_id,
            "trt_fp16_enable": trt_fp16_enable,
            "trt_engine_cache_enable": trt_engine_cache_enable,
        }
        if trt_engine_cache_enable:
            cache_path = (
                Path(trt_engine_cache_path)
                if trt_engine_cache_path is not None
                else model_path.parent / "trt_cache"
            )
            cache_path.mkdir(parents=True, exist_ok=True)
            trt_options["trt_engine_cache_path"] = str(cache_path)
        cuda_options: dict[str, Any] = {"device_id": device_id}

        if provider_options:
            trt_options.update(provider_options.get("TensorrtExecutionProvider", {}))
            cuda_options.update(provider_options.get("CUDAExecutionProvider", {}))

        requested_providers = [
            ("TensorrtExecutionProvider", trt_options),
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=requested_providers,
        )

        resolved_providers = self._session.get_providers()
        self.resolved_provider = resolved_providers[0] if resolved_providers else None
        _LOG.info(
            "Evaluator: onnxruntime session loaded from %s; requested providers "
            "%s, resolved provider %s (full resolved list: %s)",
            model_path,
            [p if isinstance(p, str) else p[0] for p in requested_providers],
            self.resolved_provider,
            resolved_providers,
        )

    def run(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invoke the session via `io_binding`, following `usi_policy_only.py`.

        Pads ``features1``/``features2`` up to `self._fixed_batch_size`
        rows with zeros when both a fixed batch size is configured and
        the caller supplied fewer rows than that, invokes the session with
        the padded (or, with no fixed batch size, the caller's original)
        arrays, and slices the two returned output arrays back down to
        the caller's original row count before returning -- see this
        class's own docstring, "Zero-padding a short final batch".
        """
        n = features1.shape[0]
        batch_size = self._fixed_batch_size
        if batch_size is not None and n < batch_size:
            padded1 = np.zeros((batch_size, *features1.shape[1:]), dtype=features1.dtype)
            padded2 = np.zeros((batch_size, *features2.shape[1:]), dtype=features2.dtype)
            padded1[:n] = features1
            padded2[:n] = features2
            policy, value = self._run_iobinding(padded1, padded2)
            return policy[:n], value[:n]
        return self._run_iobinding(features1, features2)

    def _run_iobinding(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The `io_binding` call itself, unpadded -- the `usi_policy_only.py` pattern."""
        io_binding = self._session.io_binding()
        io_binding.bind_cpu_input("input1", features1)
        io_binding.bind_cpu_input("input2", features2)
        io_binding.bind_output("output_policy")
        io_binding.bind_output("output_value")
        self._session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        return outputs[0], outputs[1]


class EvaluatorBatchError(Exception):
    """Set on every request's future when its batch fails (Requirement 5.6).

    Raised for none of the three triggering conditions directly by this
    module's own code paths under normal operation; it is instead
    constructed once per failed batch and handed to every affected
    request's `asyncio.Future.set_exception`, so a caller ``await``ing
    `Evaluator.enqueue` sees this exception raised at the ``await`` point,
    naming which of the three Requirement 5.6 conditions applied.
    """


# ---------------------------------------------------------------------------
# EvalRequest / EvalResult
# ---------------------------------------------------------------------------


@dataclass
class EvalRequest:
    """One pending evaluation, queued by `Evaluator.enqueue` for the collector.

    ``features1``/``features2`` are this request's own small arrays (not a
    slice of the shared staging buffer -- see this module's docstring for
    why: a shared, progressively-filled buffer would need a slot-reservation
    protocol with no natural place to enforce the "no `await` between
    reservation and the `make_input_features` write" invariant across many
    concurrently enqueuing descent tasks, whereas a private array per
    request needs no such protocol). The collector copies these into row
    ``i`` of its own staging buffer at dispatch time.

    ``legal_moves`` is ``list(board.legal_moves)`` as captured by
    `Evaluator.enqueue` at the moment the features were computed, so it is
    guaranteed consistent with those features even though the caller's
    `board` may have moved on (been pushed/popped further) by the time the
    collector actually dispatches this request. ``turn`` is
    ``board.turn`` at that same moment, needed by
    ``cshogi.dlshogi.make_move_label(move, turn)`` to decode the policy
    head (that function has no board-free equivalent that omits the
    color argument).
    """

    features1: np.ndarray
    features2: np.ndarray
    legal_moves: Sequence[int]
    turn: int
    future: "asyncio.Future[EvalResult]"


@dataclass
class EvalResult:
    """One resolved evaluation (Requirements 5.1, 5.4, 5.5).

    ``win_rate`` is the Evaluator's win rate for the side to move, in
    [0, 1] (Requirement 5.5). ``policy`` is a ``float64`` array parallel to
    the `EvalRequest.legal_moves` list that produced it -- same length,
    same order -- with each entry in [0, 1] and the array summing to 1
    within the 0.001 tolerance of Requirement 5.4 (or exactly, in the
    Requirement 5.7 uniform-substitution branch). There is deliberately no
    probability for any move that is not a legal move of the evaluated
    Board_State, because `policy` only ever has as many entries as
    `legal_moves` had.
    """

    win_rate: float
    policy: np.ndarray


def _decode_policy(
    logits_row: np.ndarray, legal_moves: Sequence[int], turn: int
) -> tuple[np.ndarray, bool]:
    """Gather ``legal_moves``' logits out of ``logits_row`` and softmax-normalise them.

    Returns ``(probabilities, substituted)``. ``substituted`` is `True`
    exactly when the softmax denominator (the sum of the exponentiated,
    max-shifted gathered logits) was 0 or non-finite, in which case
    ``probabilities`` is the uniform ``1/n`` distribution of Requirement
    5.7 rather than an actual softmax output. The max-shift
    (``gathered - gathered.max()``) is the ordinary numerically-stable
    softmax trick; it does not change which vectors are "degenerate" under
    5.7, because a non-finite or all-``-inf``-after-shift input still
    yields a non-finite or zero exponentiated sum, which is exactly the
    condition this function checks for.

    ``legal_moves`` and ``turn`` are passed straight to
    ``cshogi.dlshogi.make_move_label``, matching the calling convention
    `dlshogi/utils/usi_policy_only.py` already uses
    (``make_move_label(move, board.turn)`` with ``move`` drawn directly
    from ``board.legal_moves``, with no board reconstruction needed here).
    """
    n = len(legal_moves)
    if n == 0:
        return np.empty(0, dtype=np.float64), False

    labels = np.fromiter(
        (_dlshogi.make_move_label(move, turn) for move in legal_moves),
        dtype=np.int64,
        count=n,
    )
    gathered = logits_row[labels].astype(np.float64)
    max_logit = np.max(gathered)
    # An all -inf (or otherwise pathological) `gathered` can make
    # `gathered - max_logit` produce NaN (-inf - -inf) rather than a
    # merely-large-but-finite value; the resulting non-finite `total` is
    # exactly the condition the check below exists to catch, so the
    # invalid-value warning numpy would otherwise raise here is
    # deliberately suppressed rather than fixed by special-casing the
    # input, which would only duplicate that same check.
    with np.errstate(invalid="ignore"):
        exp_vals = np.exp(gathered - max_logit)
        total = np.sum(exp_vals)

    if not np.isfinite(total) or total == 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64), True
    return exp_vals / total, False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Batched neural-network evaluation over an injected `InferenceSession`.

    Construct with the injected ``session``, ``batch_size`` (Requirement
    13.7's Batch_Size, 1 to 4096), and ``batch_timeout_ms`` (Requirement
    13.7's Batch_Timeout, 1 to 1000 milliseconds), then call `start()`
    before any `enqueue()` call and `await stop()` when done. One
    `Evaluator` instance is one collector coroutine, matching design.md's
    "one collector coroutine per process" -- a `search` process (task 13)
    constructs exactly one.

    This class performs no validation of ``batch_size``/``batch_timeout_ms``
    beyond rejecting non-positive values: Requirement 13's full range
    validation is `config.py`'s job (`validate_config`), and a caller on
    the real startup path is expected to have already run it before
    constructing an `Evaluator` from the validated `BookConfig` fields
    (``config.batch_size``, ``config.batch_timeout``).
    """

    def __init__(
        self,
        session: InferenceSession,
        *,
        batch_size: int,
        batch_timeout_ms: float,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_timeout_ms <= 0:
            raise ValueError(f"batch_timeout_ms must be positive, got {batch_timeout_ms}")

        self._session = session
        self._batch_size = int(batch_size)
        self._batch_timeout_s = float(batch_timeout_ms) / 1000.0

        self._queue: "asyncio.Queue[EvalRequest]" = asyncio.Queue()

        # The preallocated staging buffer (design.md: "a preallocated
        # staging buffer of Batch_Size entries"). Reused across every
        # dispatch; a partial (< batch_size) final/timeout batch uses only
        # the first `n` rows of each array.
        self._features1 = np.empty(
            (self._batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32
        )
        self._features2 = np.empty(
            (self._batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32
        )

        self._collector_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Cumulative counters (Requirement 14.1's "Evaluator batches per
        # second", 14.6's Evaluator-failure count, 5.7's substitution
        # report). No real Progress_Reporter exists yet (task 16.1); these
        # are plain `int` attributes for a future report.py to read, the
        # same convention `node_store.py` follows for its own
        # `duplicate_count`/`collision_count`.
        self.batches_dispatched = 0
        self.evaluated_count = 0
        self.failure_count = 0
        self.substitution_count = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_timeout_ms(self) -> float:
        return self._batch_timeout_s * 1000.0

    @property
    def queue_size(self) -> int:
        """Requests not yet claimed by the collector's currently-forming batch."""
        return self._queue.qsize()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background collector coroutine (idempotent).

        Mirrors `node_store.py`'s `start_flusher`: calling this more than
        once while the collector is already running is a no-op, so a
        caller need not track whether it has already been started.
        """
        if self._collector_task is not None and not self._collector_task.done():
            return
        self._stop_event = asyncio.Event()
        self._collector_task = asyncio.ensure_future(
            self._collector_loop(self._stop_event)
        )

    async def stop(self) -> None:
        """Stop the background collector coroutine.

        Any request already sitting in the queue but not yet part of a
        forming batch at the moment `stop()` takes effect is left with an
        unresolved future -- this class performs no forced-failure or
        drain-and-dispatch of a leftover partial batch on stop, since that
        shutdown ordering (no further descents, no further Evaluator
        invocations, then flush and exit within 60 s) is the
        Search_Coordinator's responsibility (Requirements 10.5-10.7, task
        13), not this module's. A caller that has already stopped issuing
        `enqueue()` calls before calling `stop()` -- the ordering
        Requirement 10.6 describes -- never observes this.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._collector_task is not None:
            await self._collector_task
            self._collector_task = None

    # -- enqueue -------------------------------------------------------------

    async def enqueue(self, board: cshogi.Board) -> EvalResult:
        """Evaluate ``board``'s current Board_State, batched with concurrent callers.

        Computes ``list(board.legal_moves)`` and the input features
        synchronously (both cheap, and both must happen before this
        coroutine's first `await`, so that a caller pushing further moves
        onto the same `Board` object immediately after calling this
        cannot corrupt the features or the legal-move list this request
        carries), then queues the request and awaits its resolution.

        Raises `ValueError` if ``board`` has no legal moves: Requirement
        5.1 is stated for "a Board_State with at least one legal move",
        and a zero-legal-move Board_State is Requirement 8.5's terminal
        case, which the Search_Coordinator/Repetition_Resolver must
        classify without ever reaching the Evaluator (Requirement 8.11:
        "terminal nodes get no edges and end the descent").

        Raises `EvaluatorBatchError` (propagated from the future) if this
        request's batch fails under Requirement 5.6.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(
                "Evaluator.enqueue: board has no legal moves; a terminal "
                "Board_State must be classified before reaching the Evaluator "
                "(Requirements 8.5, 8.11)"
            )

        features1 = np.empty((FEATURES1_NUM, 9, 9), dtype=np.float32)
        features2 = np.empty((FEATURES2_NUM, 9, 9), dtype=np.float32)
        _dlshogi.make_input_features(board, features1, features2)

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[EvalResult]" = loop.create_future()
        request = EvalRequest(
            features1=features1,
            features2=features2,
            legal_moves=legal_moves,
            turn=board.turn,
            future=future,
        )
        await self._queue.put(request)
        return await future

    # -- the collector coroutine --------------------------------------------

    async def _collector_loop(self, stop_event: asyncio.Event) -> None:
        """Pop requests into the staging buffer; dispatch at Batch_Size or Batch_Timeout.

        Requirement 5.2/5.3, implemented exactly as design.md's Evaluator
        section states: the deadline is captured once, right after the
        first request of a new batch is received into the (until then)
        empty accumulating batch, and ``remaining`` is recomputed against
        that same deadline on every iteration of the inner loop via
        ``asyncio.wait_for(self._queue.get(), timeout=remaining)``.
        """
        while not stop_event.is_set():
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=_IDLE_POLL_INTERVAL_S
                )
            except asyncio.TimeoutError:
                continue

            batch = [first]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._batch_timeout_s

            while len(batch) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            try:
                await self._dispatch(batch)
            except Exception:  # noqa: BLE001 - the collector must keep running
                _LOG.exception("Evaluator collector: _dispatch failed unexpectedly")

    async def _dispatch(self, batch: list[EvalRequest]) -> None:
        """Invoke the session on one batch and resolve (or fail) every request in it."""
        if not batch:
            return

        n = len(batch)
        f1 = self._features1[:n]
        f2 = self._features2[:n]
        for i, req in enumerate(batch):
            f1[i] = req.features1
            f2[i] = req.features2

        try:
            policy_logits, values = await self._invoke_session(f1, f2)
        except Exception as exc:  # noqa: BLE001 - Requirement 5.6: an invocation error
            self._fail_batch(
                batch,
                EvaluatorBatchError(
                    f"neural network invocation failed for a batch of {n}: {exc!r}"
                ),
            )
            return

        policy_logits = np.asarray(policy_logits)
        values = np.asarray(values).reshape(-1)

        # Requirement 5.6: a short result array.
        if values.shape[0] < n or policy_logits.shape[0] < n:
            self._fail_batch(
                batch,
                EvaluatorBatchError(
                    f"neural network invocation returned results for "
                    f"{min(values.shape[0], policy_logits.shape[0])} Board_State(s) "
                    f"for a batch of {n}"
                ),
            )
            return

        # Requirement 5.6: a non-finite win rate.
        if not np.all(np.isfinite(values[:n])):
            self._fail_batch(
                batch,
                EvaluatorBatchError(
                    "neural network invocation returned a non-finite win rate"
                ),
            )
            return

        results: list[EvalResult] = []
        for i, req in enumerate(batch):
            probs, substituted = _decode_policy(policy_logits[i], req.legal_moves, req.turn)
            if substituted:
                self.substitution_count += 1
                _LOG.info(
                    "Evaluator: degenerate policy sum for a Board_State with %d "
                    "legal move(s); substituted a uniform 1/n distribution "
                    "(Requirement 5.7)",
                    len(req.legal_moves),
                )
            results.append(EvalResult(win_rate=float(values[i]), policy=probs))

        # Requirement 5.6: a non-finite move probability, checked after the
        # 5.7 substitution above (which always yields a finite result) so
        # this only fires on a genuine anomaly the substitution did not
        # already correct.
        if any(probs.size and not np.all(np.isfinite(probs)) for probs in (r.policy for r in results)):
            self._fail_batch(
                batch,
                EvaluatorBatchError(
                    "neural network invocation returned a non-finite move probability"
                ),
            )
            return

        self.batches_dispatched += 1
        self.evaluated_count += n
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)

    async def _invoke_session(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the injected session off the event-loop thread; see module docstring."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._session.run, f1, f2)

    def _fail_batch(self, batch: list[EvalRequest], exc: EvaluatorBatchError) -> None:
        """Mark every request in ``batch`` failed (Requirement 5.6)."""
        self.failure_count += len(batch)
        _LOG.error("Evaluator: %s", exc)
        for req in batch:
            if not req.future.done():
                req.future.set_exception(exc)


__all__ = [
    "FEATURES1_NUM",
    "FEATURES2_NUM",
    "InferenceSession",
    "OnnxRuntimeInferenceSession",
    "EvaluatorBatchError",
    "EvalRequest",
    "EvalResult",
    "Evaluator",
]
