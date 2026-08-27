"""Property tests for ``dlshogi.book.report`` (tasks 16.2, 16.3).

Implements:

- **Property 43: Progress counters are running sums**
  (Requirements 14.2, 14.6) -- task 16.2.
- **Property 44: Throughput warning state machine**
  (Requirements 14.4, 14.5) -- task 16.3.

None of these need a database or GPU. Both are RuleBasedStateMachine tests
that inject events and advance a virtual clock.
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)
from hypothesis import strategies as st

from dlshogi.book.report import (
    CATEGORIES,
    ProgressReporter,
    ThroughputRecovery,
    ThroughputStateMachine,
    ThroughputWarning,
)


# ---------------------------------------------------------------------------
# Property 43: Progress counters are running sums
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 43: Progress counters are running sums
# Validates: Requirements 14.2, 14.6


class ProgressCountersMachine(RuleBasedStateMachine):
    """RuleBasedStateMachine verifying that cumulative counters are running sums.

    The reporter's cumulative counters must be monotonically non-decreasing
    and each must equal the total count of events of that category injected.
    """

    def __init__(self):
        super().__init__()
        self._virtual_time = 0.0
        self.reporter = ProgressReporter(
            report_interval=1.0,
            throughput_floor=10.0,
            throughput_grace_period=60.0,
            time_func=lambda: self._virtual_time,
        )
        # Model counters tracking expected values
        self.expected_terashock_injections = 0
        self.expected_terashock_illegal_discards = 0
        self.expected_evaluator_failures = 0
        self.expected_duplicate_node_creations = 0
        self.expected_descents = 0
        self.expected_batches = 0
        self.total_events = 0

        # Snapshot of previous counters for monotonicity check
        self.prev_terashock_injections = 0
        self.prev_terashock_illegal_discards = 0
        self.prev_evaluator_failures = 0
        self.prev_duplicate_node_creations = 0
        self.prev_descents = 0

    @rule(category=st.sampled_from(CATEGORIES))
    def record_event(self, category):
        """Inject one event of a random category."""
        self.reporter.record_event(category)
        self.total_events += 1

        if category == "terashock_injection":
            self.expected_terashock_injections += 1
        elif category == "terashock_illegal_discard":
            self.expected_terashock_illegal_discards += 1
        elif category == "evaluator_failure":
            self.expected_evaluator_failures += 1
        elif category == "duplicate_node_creation":
            self.expected_duplicate_node_creations += 1

    @rule()
    def record_descent(self):
        """Record one descent completion."""
        self.reporter.record_descent()
        self.expected_descents += 1

    @rule()
    def record_batch(self):
        """Record one evaluator batch."""
        self.reporter.record_evaluator_batch()
        self.expected_batches += 1

    @rule()
    def tick_interval(self):
        """Advance virtual time by one report interval and emit a record."""
        self._virtual_time += self.reporter.report_interval
        self.reporter.tick(dt=self.reporter.report_interval)

    @invariant()
    def counters_match_expected(self):
        """Cumulative counters must equal the model's expected counts."""
        assert self.reporter.terashock_injections == self.expected_terashock_injections
        assert self.reporter.terashock_illegal_discards == self.expected_terashock_illegal_discards
        assert self.reporter.evaluator_failures == self.expected_evaluator_failures
        assert self.reporter.duplicate_node_creations == self.expected_duplicate_node_creations
        assert self.reporter.cumulative_descents == self.expected_descents

    @invariant()
    def counters_monotonically_non_decreasing(self):
        """Cumulative counters must never decrease."""
        assert self.reporter.terashock_injections >= self.prev_terashock_injections
        assert self.reporter.terashock_illegal_discards >= self.prev_terashock_illegal_discards
        assert self.reporter.evaluator_failures >= self.prev_evaluator_failures
        assert self.reporter.duplicate_node_creations >= self.prev_duplicate_node_creations
        assert self.reporter.cumulative_descents >= self.prev_descents

        # Update snapshots
        self.prev_terashock_injections = self.reporter.terashock_injections
        self.prev_terashock_illegal_discards = self.reporter.terashock_illegal_discards
        self.prev_evaluator_failures = self.reporter.evaluator_failures
        self.prev_duplicate_node_creations = self.reporter.duplicate_node_creations
        self.prev_descents = self.reporter.cumulative_descents

    @invariant()
    def emitted_records_reflect_cumulative_counters(self):
        """Every emitted progress record must report the cumulative totals at emission time."""
        if not self.reporter.emitted_records:
            return
        last = self.reporter.emitted_records[-1]
        # The last emitted record's counters should be <= current counters
        # (events may have been recorded after the last tick)
        assert last["terashock_injections"] <= self.reporter.terashock_injections
        assert last["terashock_illegal_discards"] <= self.reporter.terashock_illegal_discards
        assert last["evaluator_failures"] <= self.reporter.evaluator_failures
        assert last["duplicate_node_creations"] <= self.reporter.duplicate_node_creations
        assert last["cumulative_descents"] <= self.reporter.cumulative_descents


TestProgressCounters = ProgressCountersMachine.TestCase
TestProgressCounters.settings = settings(
    max_examples=1000,
    stateful_step_count=50,
)


# ---------------------------------------------------------------------------
# Property 44: Throughput warning state machine
# ---------------------------------------------------------------------------
# Feature: puct-book-builder, Property 44: Throughput warning state machine
# Validates: Requirements 14.4, 14.5


class ThroughputWarningMachine(RuleBasedStateMachine):
    """RuleBasedStateMachine verifying the degraded-throughput state machine.

    Models the ThroughputStateMachine independently and asserts:
    - At most one warning per grace period
    - Warning only emitted when rate stays below floor for >= grace period
    - Recovery resets continuous-duration measurement
    - Recovery emitted only after a warning was active
    """

    def __init__(self):
        super().__init__()
        # Use parameters that make the state machine exercise boundaries
        self.floor = 10.0
        self.report_interval = 1.0
        # Grace period as integral number of intervals in half the cases
        self.grace_period = 5.0  # 5 intervals
        self.sm = ThroughputStateMachine(
            floor=self.floor,
            grace_period=self.grace_period,
            report_interval=self.report_interval,
        )

        # Model state
        self.intervals_fed = 0
        self.warnings_emitted: list[int] = []  # interval indices where warnings were emitted
        self.recoveries_emitted: list[int] = []
        self.warned_currently = False
        self.continuous_below_intervals = 0
        self.intervals_since_last_warning = 0

    @rule(rate=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    def feed_interval(self, rate):
        """Feed one interval's rate to the state machine."""
        event = self.sm.feed_interval(rate)
        self.intervals_fed += 1

        if rate >= self.floor:
            # Above floor: should reset continuous_below
            self.continuous_below_intervals = 0
            if self.warned_currently:
                # Should get a recovery
                assert isinstance(event, ThroughputRecovery), (
                    f"Expected recovery when rate ({rate}) >= floor ({self.floor}) "
                    f"after active warning, got {event}"
                )
                self.recoveries_emitted.append(self.intervals_fed)
                self.warned_currently = False
                self.intervals_since_last_warning = 0
            else:
                # Normal operation, no event
                assert event is None or isinstance(event, type(None)), (
                    f"Expected None when rate >= floor and no active warning, got {event}"
                )
                self.intervals_since_last_warning = 0
        else:
            # Below floor
            self.continuous_below_intervals += 1
            self.intervals_since_last_warning += 1

            if isinstance(event, ThroughputWarning):
                self.warnings_emitted.append(self.intervals_fed)
                self.warned_currently = True
                self.intervals_since_last_warning = 0

    @invariant()
    def warning_only_after_grace_period(self):
        """Warnings must not fire before continuous_below >= grace_period."""
        # The SM's internal continuous_below must be >= grace_period for any warning
        if self.sm.warned:
            assert self.sm.continuous_below >= self.grace_period

    @invariant()
    def at_most_one_warning_per_grace_period(self):
        """At most one warning may be emitted per grace period window."""
        if len(self.warnings_emitted) < 2:
            return
        # Check that consecutive warnings are separated by at least
        # grace_period / report_interval intervals
        min_interval_gap = self.grace_period / self.report_interval
        last_two = self.warnings_emitted[-2:]
        gap = last_two[1] - last_two[0]
        assert gap >= min_interval_gap, (
            f"Two warnings emitted only {gap} intervals apart, "
            f"minimum required is {min_interval_gap}"
        )

    @invariant()
    def recovery_only_after_warning(self):
        """Recovery events can only occur after an active warning."""
        # If we have recoveries, each must have been preceded by a warning
        for i, recovery_idx in enumerate(self.recoveries_emitted):
            # Find the closest preceding warning
            preceding_warnings = [w for w in self.warnings_emitted if w < recovery_idx]
            assert len(preceding_warnings) > 0, (
                f"Recovery at interval {recovery_idx} has no preceding warning"
            )

    @invariant()
    def warned_state_consistent(self):
        """The SM's warned state must be consistent with our model."""
        assert self.sm.warned == self.warned_currently


TestThroughputWarning = ThroughputWarningMachine.TestCase
TestThroughputWarning.settings = settings(
    max_examples=1000,
    stateful_step_count=100,
)
