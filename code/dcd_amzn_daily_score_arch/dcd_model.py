"""Minimal score-driven conditional-density model in Matsuda geometry.

The module is deliberately self-contained.  It implements the exact
normalized B-spline mixture, centered clr log-spline metric, Cholesky
Euclideanization, and the one-parameter recursion

    q[t + 1] = q0 + kappa * score[t].

All state vectors are stored as one-dimensional arrays.  If M = L L', the
code convention is q(row) = r(row) L, so the score in q coordinates is
solve(L, score_r(column)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import logsumexp


DEFAULT_BREAKPOINTS = np.array(
    [
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        0.85,
        0.95,
        0.98,
        0.99,
        0.995,
        1.0,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class StaticMixtureFit:
    weights: np.ndarray
    log_likelihood: float
    penalized_objective: float
    iterations: int
    converged: bool
    max_weight_change: float


@dataclass(frozen=True)
class FilterResult:
    log_likelihood: float
    contributions: np.ndarray
    states: np.ndarray
    scores: np.ndarray
    weights: np.ndarray


def make_clamped_knots(
    breakpoints: np.ndarray = DEFAULT_BREAKPOINTS,
    degree: int = 3,
) -> np.ndarray:
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or np.any(np.diff(breakpoints) <= 0.0):
        raise ValueError("breakpoints must be a strictly increasing vector")
    return np.concatenate(
        [
            np.repeat(breakpoints[0], degree + 1),
            breakpoints[1:-1],
            np.repeat(breakpoints[-1], degree + 1),
        ]
    )


def bspline_masses(knots: np.ndarray, degree: int) -> np.ndarray:
    knots = np.asarray(knots, dtype=float)
    components = knots.size - degree - 1
    masses = np.array(
        [
            (knots[j + degree + 1] - knots[j]) / (degree + 1)
            for j in range(components)
        ],
        dtype=float,
    )
    if np.any(masses <= 0.0):
        raise ValueError("B-spline masses must be positive")
    return masses


def normalized_bspline_basis(
    values: np.ndarray,
    knots: np.ndarray,
    degree: int,
    masses: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    raw = BSpline.design_matrix(
        values,
        knots,
        degree,
        extrapolate=False,
    ).toarray()
    return raw / np.asarray(masses, dtype=float)[None, :]


def helmert_submatrix(dimension: int) -> np.ndarray:
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    helmert = np.zeros((dimension, dimension - 1), dtype=float)
    for column in range(dimension - 1):
        k = column + 1
        scale = np.sqrt(k * (k + 1.0))
        helmert[:k, column] = 1.0 / scale
        helmert[k, column] = -k / scale
    return helmert


def centered_clr_metric(
    knots: np.ndarray,
    degree: int,
    masses: np.ndarray,
    helmert: np.ndarray,
) -> np.ndarray:
    """Compute H' {integral B B' - m m'/length} H."""
    nodes, weights = np.polynomial.legendre.leggauss(5)
    points_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        points_parts.append((right - left) * nodes / 2.0 + (left + right) / 2.0)
        weight_parts.append(weights * (right - left) / 2.0)
    points = np.concatenate(points_parts)
    quadrature = np.concatenate(weight_parts)
    phi = normalized_bspline_basis(points, knots, degree, masses)
    ordinary = phi * masses[None, :]
    gram = ordinary.T @ (ordinary * quadrature[:, None])
    support_length = float(knots[-degree - 1] - knots[degree])
    centered = gram - np.outer(masses, masses) / support_length
    metric = helmert.T @ centered @ helmert
    return (metric + metric.T) / 2.0


def fit_static_map_em(
    component_density: np.ndarray,
    prior_mode: np.ndarray,
    prior_strength: float = 1.0,
    max_iter: int = 10_000,
    tolerance: float = 1e-12,
) -> StaticMixtureFit:
    phi = np.asarray(component_density, dtype=float)
    prior = np.asarray(prior_mode, dtype=float).copy()
    prior /= prior.sum()
    weights = prior.copy()
    converged = False
    max_change = np.inf
    for iteration in range(1, max_iter + 1):
        fitted = np.maximum(phi @ weights, 1e-300)
        responsibilities = phi * weights[None, :] / fitted[:, None]
        expected = responsibilities.sum(axis=0)
        new_weights = (expected + prior_strength * prior) / (
            phi.shape[0] + prior_strength
        )
        new_weights /= new_weights.sum()
        max_change = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        if max_change < tolerance:
            converged = True
            break
    fitted = np.maximum(phi @ weights, 1e-300)
    log_likelihood = float(np.log(fitted).sum())
    log_prior = prior_strength * float(np.sum(prior * np.log(weights)))
    return StaticMixtureFit(
        weights=weights,
        log_likelihood=log_likelihood,
        penalized_objective=log_likelihood + log_prior,
        iterations=iteration,
        converged=converged,
        max_weight_change=max_change,
    )


def weights_to_q(
    weights: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    r = np.log(np.asarray(weights) / masses) @ helmert
    return r @ cholesky


def q_to_weights(
    q: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    inverse_cholesky: np.ndarray | None = None,
) -> np.ndarray:
    # q(row) = r(row) L, hence solve L' r' = q'.
    if inverse_cholesky is None:
        r = np.linalg.solve(cholesky.T, np.asarray(q, dtype=float))
    else:
        r = np.asarray(q, dtype=float) @ inverse_cholesky
    xi = r @ helmert.T
    logits = np.log(masses) + xi
    return np.exp(logits - logsumexp(logits))


def log_density_and_score(
    q: np.ndarray,
    phi_row: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    inverse_cholesky: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    if inverse_cholesky is None:
        inverse_cholesky = np.linalg.inv(cholesky)
    weights = q_to_weights(
        q, masses, helmert, cholesky, inverse_cholesky
    )
    fitted = float(np.dot(weights, phi_row))
    if not np.isfinite(fitted) or fitted <= 0.0:
        raise FloatingPointError("conditional mixture density is not positive")
    responsibility = weights * phi_row / fitted
    score_r = helmert.T @ (responsibility - weights)
    score_q = inverse_cholesky @ score_r
    return float(np.log(fitted)), score_q, weights


def filter_score_arch(
    component_density: np.ndarray,
    q0: np.ndarray,
    kappa: float,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
) -> FilterResult:
    phi = np.asarray(component_density, dtype=float)
    observations = phi.shape[0]
    dimension = q0.size
    states = np.empty((observations, dimension), dtype=float)
    scores = np.empty_like(states)
    mixture_weights = np.empty((observations, masses.size), dtype=float)
    contributions = np.empty(observations, dtype=float)
    state = np.asarray(q0, dtype=float).copy()
    inverse_cholesky = np.linalg.inv(cholesky)
    for t in range(observations):
        states[t] = state
        log_density, score, weights = log_density_and_score(
            state,
            phi[t],
            masses,
            helmert,
            cholesky,
            inverse_cholesky,
        )
        contributions[t] = log_density
        scores[t] = score
        mixture_weights[t] = weights
        state = q0 + float(kappa) * score
    return FilterResult(
        log_likelihood=float(contributions.sum()),
        contributions=contributions,
        states=states,
        scores=scores,
        weights=mixture_weights,
    )


def filter_score_har(
    component_density: np.ndarray,
    q0: np.ndarray,
    kappas: np.ndarray,
    windows: tuple[int, ...],
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
) -> FilterResult:
    """Filter a finite distributed-lag score-HAR density state.

    For each window ``h``, the update uses ``h**-1`` times the sum of the
    latest available scores, with missing presample scores initialized at
    zero.  Thus every average always has its stated denominator and the
    recursion is exactly static when all kappas are zero.
    """
    phi = np.asarray(component_density, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    if len(windows) != kappas.size or not windows:
        raise ValueError("windows and kappas must have the same positive length")
    if any(window < 1 for window in windows):
        raise ValueError("score windows must be positive")
    observations = phi.shape[0]
    dimension = q0.size
    states = np.empty((observations, dimension), dtype=float)
    scores = np.empty_like(states)
    mixture_weights = np.empty((observations, masses.size), dtype=float)
    contributions = np.empty(observations, dtype=float)
    state = np.asarray(q0, dtype=float).copy()
    inverse_cholesky = np.linalg.inv(cholesky)
    for t in range(observations):
        states[t] = state
        log_density, score, weights = log_density_and_score(
            state,
            phi[t],
            masses,
            helmert,
            cholesky,
            inverse_cholesky,
        )
        contributions[t] = log_density
        scores[t] = score
        mixture_weights[t] = weights
        state = q0.copy()
        for coefficient, window in zip(kappas, windows):
            start = max(0, t - window + 1)
            score_average = scores[start : t + 1].sum(axis=0) / window
            state += coefficient * score_average
    return FilterResult(
        log_likelihood=float(contributions.sum()),
        contributions=contributions,
        states=states,
        scores=scores,
        weights=mixture_weights,
    )
