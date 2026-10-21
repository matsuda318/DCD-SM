"""Daily-cluster composite-score DCD utilities.

The observations within a date are not assumed independent.  Their marginal
log densities are averaged to form a weighted conditional composite
likelihood contribution.  The corresponding average composite score updates
one common density state for the next date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedStaticFit:
    weights: np.ndarray
    composite_log_likelihood: float
    penalized_objective: float
    iterations: int
    converged: bool
    max_weight_change: float


@dataclass(frozen=True)
class CompositeFilterResult:
    composite_log_likelihood: float
    contributions: np.ndarray
    states: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    cluster_sizes: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray


def fit_weighted_static_map_em(
    component_density: np.ndarray,
    observation_weights: np.ndarray,
    prior_mode: np.ndarray,
    prior_strength: float = 1.0,
    max_iter: int = 10_000,
    tolerance: float = 1e-12,
) -> WeightedStaticFit:
    """Fit a mixture by weighted MAP-EM without an N by K responsibility copy."""
    phi = np.asarray(component_density, dtype=float)
    sample_weight = np.asarray(observation_weights, dtype=float)
    if phi.ndim != 2 or sample_weight.shape != (phi.shape[0],):
        raise ValueError("component density and observation weights do not align")
    if np.any(sample_weight <= 0.0) or not np.isfinite(sample_weight).all():
        raise ValueError("observation weights must be finite and positive")
    prior = np.asarray(prior_mode, dtype=float).copy()
    prior /= prior.sum()
    weights = prior.copy()
    total_weight = float(sample_weight.sum())
    converged = False
    max_change = np.inf
    for iteration in range(1, max_iter + 1):
        fitted = np.maximum(phi @ weights, 1e-300)
        expected = weights * (phi.T @ (sample_weight / fitted))
        new_weights = (expected + prior_strength * prior) / (
            total_weight + prior_strength
        )
        new_weights /= new_weights.sum()
        max_change = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        if max_change < tolerance:
            converged = True
            break
    fitted = np.maximum(phi @ weights, 1e-300)
    composite_log_likelihood = float(
        np.dot(sample_weight, np.log(fitted))
    )
    log_prior = prior_strength * float(np.sum(prior * np.log(weights)))
    return WeightedStaticFit(
        weights=weights,
        composite_log_likelihood=composite_log_likelihood,
        penalized_objective=composite_log_likelihood + log_prior,
        iterations=iteration,
        converged=converged,
        max_weight_change=max_change,
    )


def composite_log_density_and_score(
    q: np.ndarray,
    component_density: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    inverse_cholesky: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return one cluster's average log density and average composite score."""
    phi = np.asarray(component_density, dtype=float)
    r = np.asarray(q, dtype=float) @ inverse_cholesky
    xi = r @ helmert.T
    logits = np.log(masses) + xi
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    fitted = phi @ weights
    if np.any(~np.isfinite(fitted)) or np.any(fitted <= 0.0):
        raise FloatingPointError("nonpositive composite marginal density")
    # Clamp only subnormal trial densities before taking reciprocals.  At all
    # regular states this is exactly the usual responsibility formula; the
    # floor merely keeps deliberately extreme optimizer probes finite.
    safe_fitted = np.maximum(fitted, 1e-300)
    mean_responsibility = (
        weights * (phi.T @ (1.0 / safe_fitted)) / phi.shape[0]
    )
    score_r = helmert.T @ (mean_responsibility - weights)
    score_q = inverse_cholesky @ score_r
    return float(np.mean(np.log(fitted))), score_q, weights


def filter_composite_score_arch(
    component_density: np.ndarray,
    offsets: np.ndarray,
    q0: np.ndarray,
    kappa: float,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> CompositeFilterResult:
    """Filter q[t+1] = q0 + kappa * average_composite_score[t]."""
    phi = np.asarray(component_density, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != phi.shape[0]:
        raise ValueError("offsets must partition all component-density rows")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every date cluster must contain an observation")
    clusters = offsets.size - 1
    dimension = q0.size
    states = np.empty((clusters, dimension), dtype=float)
    scores = np.empty_like(states)
    mixture_weights = np.empty((clusters, masses.size), dtype=float)
    contributions = np.empty(clusters, dtype=float)
    cluster_sizes = np.diff(offsets)
    state = np.asarray(q0 if initial_state is None else initial_state, dtype=float).copy()
    inverse_cholesky = np.linalg.inv(cholesky)
    for t, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[t] = state
        contribution, score, weights = composite_log_density_and_score(
            state,
            phi[start:stop],
            masses,
            helmert,
            inverse_cholesky,
        )
        contributions[t] = contribution
        scores[t] = score
        mixture_weights[t] = weights
        state = q0 + float(kappa) * score
    return CompositeFilterResult(
        composite_log_likelihood=float(contributions.sum()),
        contributions=contributions,
        states=states,
        scores=scores,
        weights=mixture_weights,
        cluster_sizes=cluster_sizes,
        terminal_state=state,
        terminal_score_history=np.empty((0, dimension), dtype=float),
    )


def filter_composite_score_har(
    component_density: np.ndarray,
    offsets: np.ndarray,
    q0: np.ndarray,
    betas: np.ndarray,
    windows: tuple[int, ...],
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> CompositeFilterResult:
    """Filter a finite 1/5/22-day distributed composite-score model.

    Missing presample scores are zero and every average retains its stated
    denominator. ``initial_score_history`` contains scores strictly before
    the first cluster passed to this function, ordered from oldest to newest.
    """
    phi = np.asarray(component_density, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    betas = np.asarray(betas, dtype=float)
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != phi.shape[0]:
        raise ValueError("offsets must partition all component-density rows")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every date cluster must contain an observation")
    if len(windows) != betas.size or not windows:
        raise ValueError("windows and betas must have the same positive length")
    if any(window < 1 for window in windows):
        raise ValueError("score windows must be positive")
    clusters = offsets.size - 1
    dimension = q0.size
    states = np.empty((clusters, dimension), dtype=float)
    scores = np.empty_like(states)
    mixture_weights = np.empty((clusters, masses.size), dtype=float)
    contributions = np.empty(clusters, dtype=float)
    cluster_sizes = np.diff(offsets)
    state = np.asarray(q0 if initial_state is None else initial_state, dtype=float).copy()
    inverse_cholesky = np.linalg.inv(cholesky)
    maximum_history = max(windows) - 1
    if initial_score_history is None:
        history = np.empty((0, dimension), dtype=float)
    else:
        history = np.asarray(initial_score_history, dtype=float).copy()
        if history.ndim != 2 or history.shape[1] != dimension:
            raise ValueError("initial score history has the wrong shape")
        if maximum_history == 0:
            history = np.empty((0, dimension), dtype=float)
        elif history.shape[0] > maximum_history:
            history = history[-maximum_history:]

    for t, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[t] = state
        contribution, score, weights = composite_log_density_and_score(
            state,
            phi[start:stop],
            masses,
            helmert,
            inverse_cholesky,
        )
        contributions[t] = contribution
        scores[t] = score
        mixture_weights[t] = weights
        history = np.vstack((history, score[None, :]))
        state = np.asarray(q0, dtype=float).copy()
        for coefficient, window in zip(betas, windows):
            state += coefficient * history[-window:].sum(axis=0) / window
        if maximum_history == 0:
            history = np.empty((0, dimension), dtype=float)
        elif history.shape[0] > maximum_history:
            history = history[-maximum_history:]

    return CompositeFilterResult(
        composite_log_likelihood=float(contributions.sum()),
        contributions=contributions,
        states=states,
        scores=scores,
        weights=mixture_weights,
        cluster_sizes=cluster_sizes,
        terminal_state=state,
        terminal_score_history=history,
    )
