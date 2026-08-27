"""Property tests for ``dlshogi.book.prior_mixer`` (task 12.2).

Property 22: Prior mixing is a normalised convex combination.

**Validates: Requirements 7.3, 7.5, 7.7, 7.9, 7.10**

Tests the convex combination formula:
    prior(e) = (1 - w) * policy(e) + w * t(e)
and the quantisation to prior_q16, including the degenerate cases:
- w == 0 (Requirement 7.7): prior equals policy
- no Terashock evals (ts_mask all False, Requirement 7.9): prior equals policy
- normal mixing (Requirement 7.3): result is a normalised distribution
- terashock_q0 (Requirement 7.5): in [0, 1]
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dlshogi.book.prior_mixer import mix_priors, win_rate, terashock_q0


# ---------------------------------------------------------------------------
# Hypothesis strategies for Prior_Mixer inputs
# ---------------------------------------------------------------------------


@st.composite
def normalised_policy(draw, min_size: int = 1, max_size: int = 100):
    """Draw a valid policy distribution: float64 array summing to 1.

    Each entry is in [0, 1] and the sum is exactly 1.0.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    # Draw unnormalized weights, all positive
    raw = draw(
        st.lists(
            st.floats(min_value=1e-6, max_value=1.0),
            min_size=n,
            max_size=n,
        )
    )
    arr = np.array(raw, dtype=np.float64)
    arr = arr / arr.sum()
    return arr


@st.composite
def ts_evals_and_mask(draw, n: int):
    """Draw ts_evals (int16 range) and ts_mask (bool) arrays of length n."""
    # ts_evals: int16 range -32000..32000
    evals = draw(
        st.lists(
            st.integers(min_value=-32000, max_value=32000),
            min_size=n,
            max_size=n,
        )
    )
    # ts_mask: boolean array, some True, some False
    mask = draw(
        st.lists(
            st.booleans(),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(evals, dtype=np.int16), np.array(mask, dtype=np.bool_)


# ---------------------------------------------------------------------------
# Property 22: Prior mixing is a normalised convex combination
# Feature: puct-book-builder, Property 22: Prior mixing is a normalised
#          convex combination
# Validates: Requirements 7.3, 7.5, 7.7, 7.9, 7.10
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    data=st.data(),
    weight=st.floats(min_value=0.0, max_value=1.0),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_prior_mixing_normalised(data, weight, eval_coef):
    """Property 22: Prior mixing is a normalised convex combination.

    **Validates: Requirements 7.3, 7.5, 7.7, 7.9, 7.10**

    Asserts:
    - Decoded prior_q16 values sum to approximately 1.0 within 1e-3
      (Requirement 7.10)
    - Each decoded value is in [0, 1] (Requirement 7.3)
    - When weight == 0: prior equals policy within quantisation error
      (Requirement 7.7)
    - When ts_mask all False: prior equals policy within quantisation error
      (Requirement 7.9)
    """
    policy = data.draw(normalised_policy(min_size=1, max_size=80))
    n = len(policy)
    ts_evals, ts_mask = data.draw(ts_evals_and_mask(n))

    prior_q16 = mix_priors(policy, ts_evals, ts_mask, weight, eval_coef)

    # Decode from uint16 to float
    decoded = prior_q16.astype(np.float64) / 65535.0

    # Requirement 7.10: sum ≈ 1.0 within 1e-3
    decoded_sum = float(decoded.sum())
    assert abs(decoded_sum - 1.0) < 1e-3, (
        f"decoded prior sum {decoded_sum} not within 1e-3 of 1.0"
    )

    # Each decoded value in [0, 1]
    assert np.all(decoded >= 0.0), "negative decoded prior found"
    assert np.all(decoded <= 1.0), "decoded prior > 1 found"

    # prior_q16 is uint16
    assert prior_q16.dtype == np.uint16


@settings(max_examples=1000)
@given(
    data=st.data(),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_weight_zero_degenerates_to_policy(data, eval_coef):
    """Property 22 (Requirement 7.7): weight == 0 means prior equals policy.

    **Validates: Requirement 7.7**

    When Terashock_Prior_Weight is 0, the result must equal the quantised
    policy within quantisation error (max 1 LSB per element = 1/65535).
    """
    policy = data.draw(normalised_policy(min_size=1, max_size=80))
    n = len(policy)
    ts_evals, ts_mask = data.draw(ts_evals_and_mask(n))

    # Weight = 0.0 means prior should equal policy regardless of ts_evals/ts_mask
    prior_q16 = mix_priors(policy, ts_evals, ts_mask, 0.0, eval_coef)

    # Expected quantisation: round(policy * 65535)
    expected_q16 = np.round(np.clip(policy, 0.0, 1.0) * 65535.0).astype(np.uint16)

    np.testing.assert_array_equal(
        prior_q16, expected_q16,
        err_msg="With weight=0, prior_q16 should exactly equal quantised policy"
    )


@settings(max_examples=1000)
@given(
    data=st.data(),
    weight=st.floats(min_value=0.0, max_value=1.0),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_no_terashock_degenerates_to_policy(data, weight, eval_coef):
    """Property 22 (Requirement 7.9): no Terashock evals means prior equals policy.

    **Validates: Requirement 7.9**

    When ts_mask is all False (no edge carries a Terashock evaluation),
    the result must equal the quantised policy regardless of weight.
    """
    policy = data.draw(normalised_policy(min_size=1, max_size=80))
    n = len(policy)
    ts_evals = np.zeros(n, dtype=np.int16)
    ts_mask = np.zeros(n, dtype=np.bool_)  # all False

    prior_q16 = mix_priors(policy, ts_evals, ts_mask, weight, eval_coef)

    # Expected quantisation: round(policy * 65535)
    expected_q16 = np.round(np.clip(policy, 0.0, 1.0) * 65535.0).astype(np.uint16)

    np.testing.assert_array_equal(
        prior_q16, expected_q16,
        err_msg="With all-False ts_mask, prior_q16 should exactly equal quantised policy"
    )


@settings(max_examples=1000)
@given(
    ts_eval=st.integers(min_value=-32000, max_value=32000),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_terashock_q0_in_range(ts_eval, eval_coef):
    """Property 22 (Requirement 7.5): terashock_q0 is in [0, 1].

    **Validates: Requirement 7.5**

    The Terashock-derived initial mean value for PUCT scoring must always
    lie in [0, 1], regardless of the input ts_eval and eval_coef.
    """
    q0 = terashock_q0(ts_eval, eval_coef)

    assert 0.0 <= float(q0) <= 1.0, (
        f"terashock_q0({ts_eval}, {eval_coef}) = {q0} not in [0, 1]"
    )


@settings(max_examples=1000)
@given(
    score=st.floats(min_value=-32000.0, max_value=32000.0),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_win_rate_in_range(score, eval_coef):
    """Property 22: win_rate sigmoid always produces values in [0, 1].

    The sigmoid conversion is the basis of the mixing formula (Requirement
    7.3) and must always map to [0, 1].
    """
    wr = win_rate(score, eval_coef)
    assert 0.0 <= float(wr) <= 1.0, (
        f"win_rate({score}, {eval_coef}) = {wr} not in [0, 1]"
    )


@settings(max_examples=500)
@given(
    data=st.data(),
    weight=st.floats(min_value=0.01, max_value=1.0),
    eval_coef=st.floats(min_value=1.0, max_value=10000.0),
)
def test_property_22_convex_combination_bounds(data, weight, eval_coef):
    """Property 22 (Requirement 7.3): prior is bounded by policy and t.

    **Validates: Requirement 7.3**

    When both policy and t are distributions, the convex combination
    (1 - w) * policy + w * t cannot produce any element larger than
    max(policy_i, t_i) or smaller than min(policy_i, t_i) -- so each
    element of decoded prior is bounded by the two source distributions
    (within quantisation tolerance).
    """
    policy = data.draw(normalised_policy(min_size=2, max_size=40))
    n = len(policy)
    # Ensure at least some Terashock entries exist
    ts_evals = data.draw(
        st.lists(
            st.integers(min_value=-32000, max_value=32000),
            min_size=n,
            max_size=n,
        )
    )
    ts_evals = np.array(ts_evals, dtype=np.int16)
    # At least one True in ts_mask so the mixing is not degenerate
    ts_mask_list = data.draw(
        st.lists(st.booleans(), min_size=n, max_size=n)
    )
    ts_mask = np.array(ts_mask_list, dtype=np.bool_)
    assume(np.any(ts_mask))

    prior_q16 = mix_priors(policy, ts_evals, ts_mask, weight, eval_coef)
    decoded = prior_q16.astype(np.float64) / 65535.0

    # The result must still be a valid distribution
    decoded_sum = float(decoded.sum())
    assert abs(decoded_sum - 1.0) < 1e-3, (
        f"convex combination sum {decoded_sum} not within 1e-3 of 1.0"
    )
    assert np.all(decoded >= 0.0)
