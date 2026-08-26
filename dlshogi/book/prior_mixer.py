"""Prior_Mixer: combines Evaluator policy with Terashock-derived weights.

Implements Requirement 7.3 (prior mixing formula), with Requirements 7.7
and 7.9 as the ``Terashock_Prior_Weight == 0`` and "no Terashock_Entry"
degenerate cases, vectorized over numpy arrays in float64:

.. code-block:: text

    win_rate(score) = 1 / (1 + exp(-score / Eval_Coef))
    t_raw(e)        = exp(win_rate(e.ts_eval)) if Terashock eval present else 0
    t(e)            = t_raw(e) / sum(t_raw) if sum(t_raw) > 0 else 0
    prior(e)        = (1 - w) * policy(e) + w * t(e)

The result is quantised to ``prior_q16 = round(prior * 65535)`` (the
``PACKED_EDGE["prior_q16"]`` field) on write. Because ``sum(policy) == 1``
and ``sum(t) == 1`` (or ``t == 0`` for all edges when no Terashock_Move
survived legality filtering), ``sum(prior) == 1`` within accumulated float
error, satisfying Requirement 7.10.

Requirement 7.5's Terashock-derived initial mean value (``q0``) is a
scoring-time derivation from the stored ``ts_eval``, not a stored
``value_sum``, so the visit-count invariant of Requirement 4.5 stays exact.

Public API:

- ``win_rate(score, eval_coef)`` — sigmoid conversion (vectorized).
- ``mix_priors(policy, ts_evals, ts_mask, terashock_prior_weight, eval_coef)``
  — the main mixing function returning quantised ``prior_q16`` values.
- ``terashock_q0(ts_eval, eval_coef)`` — scoring-time q0 for PUCT
  (Requirement 7.5), clamped to [0, 1].
"""

from __future__ import annotations

import numpy as np


def win_rate(score: np.ndarray | float, eval_coef: float) -> np.ndarray | float:
    """Convert a centipawn-like evaluation score to a win-rate in [0, 1].

    ``win_rate(score) = 1 / (1 + exp(-score / eval_coef))``

    This is the inverse of the ``-log(1/wp - 1) * eval_coef`` conversion
    used by ``usi/UctSearch.cpp`` and ``make_book_minmax.py``'s
    ``value_to_score``.

    Parameters
    ----------
    score : array_like or scalar
        Terashock evaluation value(s). For ``PACKED_EDGE``, this is the
        ``ts_eval`` field (int16, range -32000..32000).
    eval_coef : float
        The configured Eval_Coef (range [1, 10000]).

    Returns
    -------
    numpy.ndarray or float
        Win-rate(s) in [0, 1]. Same shape as ``score``.
    """
    score = np.asarray(score, dtype=np.float64)
    # Compute sigmoid: 1 / (1 + exp(-score / eval_coef))
    # Using scipy-style stable computation via np.exp with negative argument
    return 1.0 / (1.0 + np.exp(-score / eval_coef))


def mix_priors(
    policy: np.ndarray,
    ts_evals: np.ndarray,
    ts_mask: np.ndarray,
    terashock_prior_weight: float,
    eval_coef: float,
) -> np.ndarray:
    """Compute mixed prior probabilities as quantised prior_q16 values.

    Implements the convex combination of Evaluator policy and
    Terashock-derived weights per Requirement 7.3:

    .. code-block:: text

        prior(e) = (1 - w) * policy(e) + w * t(e)

    where ``t`` is the normalised softmax over win-rates of edges carrying
    a Terashock evaluation, and ``w`` is ``terashock_prior_weight``.

    When ``w == 0`` (Requirement 7.7) or no edge carries a Terashock
    evaluation (Requirement 7.9), the result collapses to the policy term.

    Illegal Terashock_Moves must be excluded *before* calling this function
    (Requirement 7.4) — ``ts_mask`` should be False for any edge whose
    Terashock move was illegal.

    Parameters
    ----------
    policy : np.ndarray, shape (N,), dtype float64
        Evaluator probability distribution over N legal moves.
        Must sum to 1 (within float tolerance).
    ts_evals : np.ndarray, shape (N,), dtype int16 or float64
        Terashock evaluation values for each edge. Values where
        ``ts_mask`` is False are ignored.
    ts_mask : np.ndarray, shape (N,), dtype bool
        True where a valid (legal) Terashock evaluation is present
        (``PACKED_EDGE["flags"] & 0x01``). False otherwise.
    terashock_prior_weight : float
        Configured Terashock_Prior_Weight in [0, 1].
    eval_coef : float
        Configured Eval_Coef in [1, 10000].

    Returns
    -------
    np.ndarray, shape (N,), dtype uint16
        Quantised prior probabilities: ``round(prior * 65535)``.
        The values in [0, 65535] satisfy ``sum(decode) ≈ 1`` within 1e-3
        (Requirement 7.10).
    """
    n = len(policy)
    policy = np.asarray(policy, dtype=np.float64)
    ts_evals = np.asarray(ts_evals, dtype=np.float64)
    ts_mask = np.asarray(ts_mask, dtype=np.bool_)

    w = float(terashock_prior_weight)

    # Degenerate cases: w == 0 or no Terashock evals survive (Req 7.7, 7.9)
    if w == 0.0 or not np.any(ts_mask):
        # Prior equals policy, quantised
        return _quantise(policy)

    # Compute t_raw: exp(win_rate(ts_eval)) for masked edges, 0 otherwise.
    # The tau parameter is fixed at 1.0 (not configurable), so the formula
    # is simply exp(win_rate(ts_eval)).
    wr = win_rate(ts_evals, eval_coef)  # shape (N,)
    t_raw = np.where(ts_mask, np.exp(wr), 0.0)

    # Normalise to get t: t(e) = t_raw(e) / sum(t_raw)
    t_sum = t_raw.sum()
    if t_sum == 0.0:
        # Should not happen if ts_mask has any True, but guard anyway
        return _quantise(policy)

    t = t_raw / t_sum

    # Convex combination: prior = (1 - w) * policy + w * t
    prior = (1.0 - w) * policy + w * t

    return _quantise(prior)


def terashock_q0(
    ts_eval: np.ndarray | float | int,
    eval_coef: float,
) -> np.ndarray | float:
    """Compute the Terashock-derived initial mean value for PUCT scoring.

    Requirement 7.5: when a Book_Edge has a recorded Terashock evaluation,
    its mean value is initialised to ``win_rate(ts_eval, eval_coef)``
    clamped to [0, 1], expressed from the perspective of the side to move.
    This is a scoring-time derivation from the stored ``ts_eval``, not a
    stored ``value_sum``, so the visit-count invariant stays exact.

    Parameters
    ----------
    ts_eval : array_like or scalar
        Terashock evaluation value(s) (int16 range: -32000..32000).
    eval_coef : float
        Configured Eval_Coef in [1, 10000].

    Returns
    -------
    numpy.ndarray or float
        Initial mean value(s) in [0, 1], clamped.
    """
    q0 = win_rate(ts_eval, eval_coef)
    return np.clip(q0, 0.0, 1.0)


def _quantise(prior: np.ndarray) -> np.ndarray:
    """Quantise float64 prior probabilities to uint16 prior_q16 values.

    ``prior_q16 = round(prior * 65535)``

    The rounding may cause the sum to be off by ±1 LSB from 65535; this is
    within the 1e-3 tolerance of Requirement 7.10 (1/65535 ≈ 1.5e-5).
    """
    # Clip to [0, 1] for safety (float accumulation may slightly exceed)
    prior = np.clip(prior, 0.0, 1.0)
    return np.round(prior * 65535.0).astype(np.uint16)
