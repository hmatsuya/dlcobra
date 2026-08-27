"""Progress_Reporter: run statistics emitted to the log destination.

Implements Requirement 14 (Progress Observability): one structured JSON
record per configured Report_Interval, containing elapsed run time,
Book_Node and Book_Edge counts from incrementally maintained counters,
cumulative completed descents, per-interval rates (descents/s and
Evaluator batches/s), mean and p95 Node_Store read and write latency,
cumulative Terashock/failure/duplicate counters, and the degraded-
throughput / throughput-recovered state machine.

**Public surface used by later tasks (search.py, propagate.py, etc.)
and by Properties 43 and 44:**

- `ProgressReporter`: the main reporter class, integrating with asyncio
  via `start()` / `stop()` or usable synchronously for property tests
  via `record_event(category)` and `tick(elapsed)`.
- `ThroughputStateMachine`: the separated Requirement 14.4/14.5 state
  machine, testable independently.
- `LatencyHistogram`: per-interval bucketed histogram for mean/p95
  computation.
- `merge_log_files(paths, output)`: the `--merge` mode.

**No side effects at import time.** This module performs no file read, no
network access, no logging configuration, and no asyncio loop access at
import time. The `logging` handlers are configured only inside
`ProgressReporter.start()` or the explicit `setup_logging()` call.

**Counters are plain int attributes** incremented from the single event
loop -- no atomics needed (design.md).

**Testing interface.** Properties 43 and 44 are RuleBasedStateMachine
tests that inject events and advance a virtual clock. The reporter
exposes `record_event(category)`, `record_descent()`,
`record_evaluator_batch()`, `record_latency(kind, ms)`, and `tick(dt)`
for this purpose. When no asyncio loop is involved (testing mode), the
reporter accumulates state and emits records via `tick()` alone.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TextIO, Union

import numpy as np

# ---------------------------------------------------------------------------
# Event categories for cumulative counters (Requirement 14.2, 14.6)
# ---------------------------------------------------------------------------

CATEGORIES = (
    "terashock_injection",
    "terashock_illegal_discard",
    "evaluator_failure",
    "duplicate_node_creation",
)

# ---------------------------------------------------------------------------
# Latency histogram (Requirement 14.1: mean and p95 over the interval)
# ---------------------------------------------------------------------------

# Bucket boundaries in milliseconds: 0.1ms to 10000ms in ~100 log-spaced
# buckets. One pass over a fixed bucket array stays well under the 1s
# emission deadline.
_BUCKET_EDGES: np.ndarray = np.logspace(-1, 4, num=101, dtype=np.float64)
_BUCKET_COUNT = len(_BUCKET_EDGES) - 1  # 100 buckets


class LatencyHistogram:
    """Per-interval bucketed histogram for latency percentile computation.

    Uses numpy for one-pass mean and p95 extraction from a fixed bucket
    array. Reset each interval (Requirement 14.1: "measured over the most
    recent Report_Interval").
    """

    __slots__ = ("_counts", "_total_sum", "_total_count")

    def __init__(self) -> None:
        self._counts: np.ndarray = np.zeros(_BUCKET_COUNT, dtype=np.int64)
        self._total_sum: float = 0.0
        self._total_count: int = 0

    def record(self, latency_ms: float) -> None:
        """Record a single latency observation in milliseconds."""
        idx = int(np.searchsorted(_BUCKET_EDGES, latency_ms, side="right")) - 1
        idx = max(0, min(idx, _BUCKET_COUNT - 1))
        self._counts[idx] += 1
        self._total_sum += latency_ms
        self._total_count += 1

    @property
    def count(self) -> int:
        """Total observations this interval."""
        return self._total_count

    def mean(self) -> float:
        """Mean latency in ms, or 0.0 if no observations."""
        if self._total_count == 0:
            return 0.0
        return self._total_sum / self._total_count

    def percentile(self, p: float) -> float:
        """Approximate p-th percentile from the bucketed histogram.

        Returns the midpoint of the bucket containing the p-th percentile
        observation. Returns 0.0 if no observations.
        """
        if self._total_count == 0:
            return 0.0
        threshold = self._total_count * (p / 100.0)
        cumulative = 0
        for i in range(_BUCKET_COUNT):
            cumulative += self._counts[i]
            if cumulative >= threshold:
                # Return the midpoint of this bucket
                low = _BUCKET_EDGES[i]
                high = _BUCKET_EDGES[i + 1]
                return float((low + high) / 2.0)
        # Fallback: return the midpoint of the last bucket
        return float((_BUCKET_EDGES[-2] + _BUCKET_EDGES[-1]) / 2.0)

    def p95(self) -> float:
        """95th percentile latency in ms."""
        return self.percentile(95.0)

    def reset(self) -> None:
        """Reset for the next interval."""
        self._counts[:] = 0
        self._total_sum = 0.0
        self._total_count = 0


# ---------------------------------------------------------------------------
# Throughput state machine (Requirement 14.4 / 14.5)
# ---------------------------------------------------------------------------


@dataclass
class ThroughputWarning:
    """A degraded-throughput warning record."""

    measured_rate: float
    floor: float
    duration_below: float  # seconds


@dataclass
class ThroughputRecovery:
    """A throughput-recovered record."""

    measured_rate: float


class ThroughputStateMachine:
    """The degraded-throughput and throughput-recovered state machine.

    Requirement 14.4: If descents/s stays below Throughput_Floor for
    continuous duration >= Throughput_Grace_Period, emit a warning. At most
    one warning per Throughput_Grace_Period while rate stays below floor.
    Intervals with rate=0 count toward continuous duration.

    Requirement 14.5: On recovery (rate >= floor after a warning), emit a
    throughput-recovered record and restart continuous-duration measurement.

    Separated from ProgressReporter for independent testability (Property 44).
    """

    def __init__(self, floor: float, grace_period: float, report_interval: float) -> None:
        """Initialize the state machine.

        Args:
            floor: Throughput_Floor (descents/s threshold).
            grace_period: Throughput_Grace_Period (seconds).
            report_interval: Report_Interval (seconds per tick).
        """
        self.floor = floor
        self.grace_period = grace_period
        self.report_interval = report_interval

        # Duration (seconds) the rate has been continuously below the floor.
        self._continuous_below: float = 0.0
        # Whether a warning has been emitted and not yet recovered.
        self._warned: bool = False
        # Time since the last warning was emitted (for the at-most-one-per-
        # grace-period constraint).
        self._time_since_last_warning: float = 0.0

    @property
    def warned(self) -> bool:
        """Whether a degraded-throughput warning is currently active."""
        return self._warned

    @property
    def continuous_below(self) -> float:
        """Seconds the rate has been continuously below the floor."""
        return self._continuous_below

    def feed_interval(self, rate: float) -> Optional[Union[ThroughputWarning, ThroughputRecovery]]:
        """Feed one interval's measured descent rate. Returns a warning or recovery event, or None.

        Args:
            rate: Measured descents/s for this interval.

        Returns:
            ThroughputWarning if the degraded-throughput condition triggers,
            ThroughputRecovery if recovery is detected, or None otherwise.
        """
        if rate >= self.floor:
            # Rate is at or above the floor.
            if self._warned:
                # Recovery after a warning.
                self._warned = False
                self._continuous_below = 0.0
                self._time_since_last_warning = 0.0
                return ThroughputRecovery(measured_rate=rate)
            else:
                # Normal operation, reset continuous-below counter.
                self._continuous_below = 0.0
                self._time_since_last_warning = 0.0
                return None
        else:
            # Rate is below the floor (including rate == 0).
            self._continuous_below += self.report_interval
            self._time_since_last_warning += self.report_interval

            if self._continuous_below >= self.grace_period:
                if not self._warned:
                    # First warning.
                    self._warned = True
                    self._time_since_last_warning = 0.0
                    return ThroughputWarning(
                        measured_rate=rate,
                        floor=self.floor,
                        duration_below=self._continuous_below,
                    )
                elif self._time_since_last_warning >= self.grace_period:
                    # Subsequent warning (at most one per grace period).
                    self._time_since_last_warning = 0.0
                    return ThroughputWarning(
                        measured_rate=rate,
                        floor=self.floor,
                        duration_below=self._continuous_below,
                    )

            return None


# ---------------------------------------------------------------------------
# Progress record dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProgressRecord:
    """One structured progress record (Requirement 14.1)."""

    elapsed_s: float
    node_count: int
    edge_count: int
    cumulative_descents: int
    descents_per_s: float
    evaluator_batches_per_s: float
    read_latency_mean_ms: float
    read_latency_p95_ms: float
    write_latency_mean_ms: float
    write_latency_p95_ms: float
    terashock_injections: int
    terashock_illegal_discards: int
    evaluator_failures: int
    duplicate_node_creations: int
    process_id: int = field(default_factory=os.getpid)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for JSON logging."""
        return {
            "type": "progress",
            "elapsed_s": round(self.elapsed_s, 3),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cumulative_descents": self.cumulative_descents,
            "descents_per_s": round(self.descents_per_s, 2),
            "evaluator_batches_per_s": round(self.evaluator_batches_per_s, 2),
            "read_latency_mean_ms": round(self.read_latency_mean_ms, 3),
            "read_latency_p95_ms": round(self.read_latency_p95_ms, 3),
            "write_latency_mean_ms": round(self.write_latency_mean_ms, 3),
            "write_latency_p95_ms": round(self.write_latency_p95_ms, 3),
            "terashock_injections": self.terashock_injections,
            "terashock_illegal_discards": self.terashock_illegal_discards,
            "evaluator_failures": self.evaluator_failures,
            "duplicate_node_creations": self.duplicate_node_creations,
            "process_id": self.process_id,
        }


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """The PUCT Book Builder's progress reporter (Requirement 14).

    Integrates with asyncio for production use via `start(loop)` /
    `stop()`, or can be driven synchronously for property tests via
    direct calls to `record_event`, `record_descent`, etc. plus `tick`.

    Counters are plain int attributes incremented from the single event
    loop (design.md: no atomics needed). Latency percentiles come from
    per-interval bucketed numpy histograms, reset each interval.

    The throughput state machine (Requirement 14.4/14.5) is separated
    into `ThroughputStateMachine` for independent testability.
    """

    def __init__(
        self,
        report_interval: float,
        throughput_floor: float,
        throughput_grace_period: float,
        max_book_ply: int = 0,
        verbose: bool = False,
        log_path: Optional[Union[str, Path]] = None,
        process_id: Optional[int] = None,
        time_func: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            report_interval: Seconds between progress records (1-3600).
            throughput_floor: Descents/s threshold for warnings.
            throughput_grace_period: Seconds below floor before warning.
            max_book_ply: Maximum move sequence length for verbose logging
                (0 means use 256 as limit).
            verbose: Whether to log root-to-node move sequences.
            log_path: Path for the rotating log file. None = stderr only.
            process_id: Process identifier for tagging records.
            time_func: Injectable time source for testing (default: time.time).
        """
        self.report_interval = report_interval
        self.throughput_floor = throughput_floor
        self.throughput_grace_period = throughput_grace_period
        self.max_book_ply = max_book_ply
        self.verbose = verbose
        self.log_path = Path(log_path) if log_path is not None else None
        self.process_id = process_id if process_id is not None else os.getpid()
        self._time_func = time_func if time_func is not None else time.time

        # --- Cumulative counters (Requirement 14.2, 14.6) ---
        self.terashock_injections: int = 0
        self.terashock_illegal_discards: int = 0
        self.evaluator_failures: int = 0
        self.duplicate_node_creations: int = 0

        # --- Incrementally maintained counts (NOT SELECT count(*)) ---
        self.node_count: int = 0
        self.edge_count: int = 0

        # --- Descent and batch counters ---
        self.cumulative_descents: int = 0
        self._interval_descents: int = 0
        self._interval_batches: int = 0

        # --- Latency histograms (reset each interval) ---
        self._read_latency = LatencyHistogram()
        self._write_latency = LatencyHistogram()

        # --- Throughput state machine ---
        self._throughput_sm = ThroughputStateMachine(
            floor=throughput_floor,
            grace_period=throughput_grace_period,
            report_interval=report_interval,
        )

        # --- Timing ---
        self._start_time: Optional[float] = None
        self._last_interval_time: Optional[float] = None

        # --- Logging infrastructure ---
        self._logger: Optional[logging.Logger] = None
        self._log_handler: Optional[logging.Handler] = None
        self._stderr_handler: Optional[logging.Handler] = None
        self._logging_configured: bool = False

        # --- Unwritable log destination tracking (14.7) ---
        self._unwritable_reported_this_interval: bool = False

        # --- asyncio integration ---
        self._loop: Any = None  # asyncio.AbstractEventLoop
        self._timer_handle: Any = None
        self._running: bool = False

        # --- Emitted records (for testing) ---
        self.emitted_records: list[dict[str, Any]] = []
        self.emitted_warnings: list[dict[str, Any]] = []
        self.emitted_recoveries: list[dict[str, Any]] = []
        self.emitted_verbose: list[str] = []

    # -------------------------------------------------------------------
    # Event recording interface (for production use and property tests)
    # -------------------------------------------------------------------

    def record_event(self, category: str) -> None:
        """Record a single event of the given category.

        Categories: 'terashock_injection', 'terashock_illegal_discard',
        'evaluator_failure', 'duplicate_node_creation'.
        """
        if category == "terashock_injection":
            self.terashock_injections += 1
        elif category == "terashock_illegal_discard":
            self.terashock_illegal_discards += 1
        elif category == "evaluator_failure":
            self.evaluator_failures += 1
        elif category == "duplicate_node_creation":
            self.duplicate_node_creations += 1
        else:
            raise ValueError(f"unknown event category: {category!r}")

    def record_descent(self) -> None:
        """Record one completed Selection_Descent."""
        self.cumulative_descents += 1
        self._interval_descents += 1

    def record_evaluator_batch(self) -> None:
        """Record one completed Evaluator batch invocation."""
        self._interval_batches += 1

    def record_latency(self, kind: str, latency_ms: float) -> None:
        """Record a Node_Store latency observation.

        Args:
            kind: 'read' or 'write'.
            latency_ms: Latency in milliseconds.
        """
        if kind == "read":
            self._read_latency.record(latency_ms)
        elif kind == "write":
            self._write_latency.record(latency_ms)
        else:
            raise ValueError(f"unknown latency kind: {kind!r}")

    def record_node_created(self, edge_count: int = 0) -> None:
        """Record a new Book_Node creation with its edge count.

        Incrementally maintains node_count and edge_count (not SELECT count(*)).
        """
        self.node_count += 1
        self.edge_count += edge_count

    def record_verbose_path(self, move_sequence: Sequence[str]) -> None:
        """Record a root-to-node USI move sequence in verbose mode.

        Requirement 14.3: at most Max_Book_Ply moves when > 0, else 256.
        """
        if not self.verbose:
            return
        max_moves = self.max_book_ply if self.max_book_ply > 0 else 256
        truncated = list(move_sequence[:max_moves])
        path_str = " ".join(truncated)
        self.emitted_verbose.append(path_str)
        if self._logger is not None:
            self._emit_log({"type": "verbose_path", "moves": path_str})

    # -------------------------------------------------------------------
    # Tick interface (for testing without asyncio)
    # -------------------------------------------------------------------

    def tick(self, dt: Optional[float] = None) -> Optional[ProgressRecord]:
        """Advance time by dt seconds and emit a progress record if an interval has elapsed.

        For property testing, call with dt equal to the Report_Interval to
        simulate one interval passing. Returns the emitted ProgressRecord,
        or None if no record was emitted (dt < report_interval).

        In production (asyncio mode), this is called by the timer callback.
        """
        now = self._time_func()

        if self._start_time is None:
            self._start_time = now
            self._last_interval_time = now
            return None

        if dt is not None:
            # Testing mode: advance the virtual clock.
            elapsed_since_last = dt
        else:
            # Production mode: compute elapsed from real clock.
            elapsed_since_last = now - (self._last_interval_time or now)

        if elapsed_since_last < self.report_interval:
            return None

        # --- Compute interval metrics ---
        elapsed_total = now - self._start_time
        interval_duration = elapsed_since_last if elapsed_since_last > 0 else self.report_interval

        descents_per_s = self._interval_descents / interval_duration
        batches_per_s = self._interval_batches / interval_duration

        record = ProgressRecord(
            elapsed_s=elapsed_total,
            node_count=self.node_count,
            edge_count=self.edge_count,
            cumulative_descents=self.cumulative_descents,
            descents_per_s=descents_per_s,
            evaluator_batches_per_s=batches_per_s,
            read_latency_mean_ms=self._read_latency.mean(),
            read_latency_p95_ms=self._read_latency.p95(),
            write_latency_mean_ms=self._write_latency.mean(),
            write_latency_p95_ms=self._write_latency.p95(),
            terashock_injections=self.terashock_injections,
            terashock_illegal_discards=self.terashock_illegal_discards,
            evaluator_failures=self.evaluator_failures,
            duplicate_node_creations=self.duplicate_node_creations,
            process_id=self.process_id,
        )

        # Emit the record.
        record_dict = record.to_dict()
        self.emitted_records.append(record_dict)
        self._emit_log(record_dict)

        # --- Throughput state machine ---
        event = self._throughput_sm.feed_interval(descents_per_s)
        if isinstance(event, ThroughputWarning):
            warning_dict = {
                "type": "degraded_throughput",
                "measured_rate": round(event.measured_rate, 2),
                "floor": event.floor,
                "duration_below_s": round(event.duration_below, 3),
            }
            self.emitted_warnings.append(warning_dict)
            self._emit_log(warning_dict)
        elif isinstance(event, ThroughputRecovery):
            recovery_dict = {
                "type": "throughput_recovered",
                "measured_rate": round(event.measured_rate, 2),
            }
            self.emitted_recoveries.append(recovery_dict)
            self._emit_log(recovery_dict)

        # --- Reset per-interval state ---
        self._interval_descents = 0
        self._interval_batches = 0
        self._read_latency.reset()
        self._write_latency.reset()
        self._last_interval_time = now
        self._unwritable_reported_this_interval = False

        return record

    # -------------------------------------------------------------------
    # Logging infrastructure
    # -------------------------------------------------------------------

    def setup_logging(self, log_path: Optional[Union[str, Path]] = None) -> None:
        """Configure the stdlib logging handlers.

        Creates a RotatingFileHandler (if a path is given) plus a stderr
        handler, both emitting JSON records. Called by `start()` or
        manually for non-asyncio usage.
        """
        if self._logging_configured:
            return

        path = Path(log_path) if log_path is not None else self.log_path
        logger = logging.getLogger(f"dlshogi.book.report.{self.process_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Remove any existing handlers.
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        formatter = logging.Formatter("%(message)s")

        # Stderr handler (always).
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.INFO)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        self._stderr_handler = stderr_handler

        # File handler (if path given).
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    str(path),
                    maxBytes=100 * 1024 * 1024,  # 100 MB
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                self._log_handler = file_handler
            except OSError:
                # Unwritable log destination: will be reported at most once
                # per interval (Requirement 14.7).
                pass

        self._logger = logger
        self._logging_configured = True

    def _emit_log(self, record_dict: dict[str, Any]) -> None:
        """Emit a JSON-serialized record to the log destination.

        Requirement 14.7: if the log destination is unwritable, report to
        the Operator at most once per interval, discard the record, and
        leave the run in progress.
        """
        if self._logger is None:
            return

        try:
            line = json.dumps(record_dict, ensure_ascii=False)
            self._logger.info(line)
        except OSError:
            if not self._unwritable_reported_this_interval:
                self._unwritable_reported_this_interval = True
                try:
                    # Report to stderr (the Operator) at most once per interval.
                    sys.stderr.write(
                        "WARNING: log destination is unwritable, record discarded\n"
                    )
                except OSError:
                    pass

    # -------------------------------------------------------------------
    # asyncio integration
    # -------------------------------------------------------------------

    def start(self, loop: Any = None) -> None:
        """Start periodic reporting on the given asyncio event loop.

        Schedules the first `tick` via `loop.call_later` on the interval
        boundary so the `max(1 s, 0.1 * Report_Interval)` deadline is met
        by construction (design.md).
        """
        if loop is None:
            import asyncio
            loop = asyncio.get_event_loop()

        self._loop = loop
        self._running = True
        self._start_time = self._time_func()
        self._last_interval_time = self._start_time
        self.setup_logging()
        self._schedule_next()

    def _schedule_next(self) -> None:
        """Schedule the next timer callback."""
        if not self._running or self._loop is None:
            return
        self._timer_handle = self._loop.call_later(
            self.report_interval, self._timer_callback
        )

    def _timer_callback(self) -> None:
        """Timer callback: emit a progress record and reschedule."""
        if not self._running:
            return
        self.tick()
        self._schedule_next()

    def stop(self) -> None:
        """Stop periodic reporting and cancel pending timers."""
        self._running = False
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    # -------------------------------------------------------------------
    # Factory from BookConfig
    # -------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,  # BookConfig
        verbose: bool = False,
        log_path: Optional[Union[str, Path]] = None,
        process_id: Optional[int] = None,
        time_func: Optional[Callable[[], float]] = None,
    ) -> "ProgressReporter":
        """Create a ProgressReporter from a BookConfig instance."""
        return cls(
            report_interval=config.report_interval,
            throughput_floor=config.throughput_floor,
            throughput_grace_period=config.throughput_grace_period,
            max_book_ply=config.max_book_ply,
            verbose=verbose,
            log_path=log_path,
            process_id=process_id,
            time_func=time_func,
        )


# ---------------------------------------------------------------------------
# Merge mode (design.md: "a trivial dlshogi/book/report.py --merge mode")
# ---------------------------------------------------------------------------


def merge_log_files(
    paths: Sequence[Union[str, Path]],
    output: Union[str, Path, TextIO],
) -> None:
    """Merge per-process JSON log files into a single chronological stream.

    Each line is expected to be a JSON object. Lines are merged by their
    'elapsed_s' field (or by file order if the field is missing). This is
    the `--merge` mode the design describes for aggregating per-GPU-process
    log files.

    Args:
        paths: Paths to per-process log files.
        output: Output file path or writable file-like object.
    """
    records: list[tuple[float, str]] = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    elapsed = obj.get("elapsed_s", 0.0)
                    records.append((elapsed, line))
                except (json.JSONDecodeError, TypeError):
                    # Skip malformed lines.
                    continue

    # Sort by elapsed time for chronological merge.
    records.sort(key=lambda r: r[0])

    if isinstance(output, (str, Path)):
        with open(output, "w", encoding="utf-8") as fh:
            for _, line in records:
                fh.write(line + "\n")
    else:
        for _, line in records:
            output.write(line + "\n")


# ---------------------------------------------------------------------------
# CLI entry point for --merge mode
# ---------------------------------------------------------------------------


def _main() -> None:
    """CLI entry point: `python -m dlshogi.book.report --merge <files...> -o <output>`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PUCT Book Builder progress report utilities"
    )
    subparsers = parser.add_subparsers(dest="command")

    merge_parser = subparsers.add_parser(
        "merge", help="Merge per-process log files into one chronological stream"
    )
    merge_parser.add_argument(
        "files", nargs="+", help="Per-process log file paths"
    )
    merge_parser.add_argument(
        "-o", "--output", required=True, help="Output file path"
    )

    args = parser.parse_args()

    if args.command == "merge":
        merge_log_files(args.files, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    _main()


__all__ = [
    "CATEGORIES",
    "LatencyHistogram",
    "ThroughputStateMachine",
    "ThroughputWarning",
    "ThroughputRecovery",
    "ProgressRecord",
    "ProgressReporter",
    "merge_log_files",
]
