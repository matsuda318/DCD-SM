"""Modular symmetric-t four-model experiment recentered at static DCD-SM q0.

Sequence:
  1. fit a static symmetric-t reference by day-equal composite MLE;
  2. hold it fixed and fit a static spline density h_0 by day-equal EM;
  3. filter location/scale or location/scale/eta with the reference score of
     f_theta(y), holding the spline block fixed and separate;
  4. add a parity DCD-SM whose zero state has mixture weights w_0.

The spline score is projected off the corresponding reference update scores
at the fixed anchor.  There is deliberately no Fisher whitening.  Thus the
only change from the earlier four-model run is the DCD-SM origin: fitted w0
instead of the uniform spline weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2

import run_sp500 as core
import run_symmetric_t_four_models as old
from estimate_score_har import hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "dcd_sp500_daily_composite_score_arch"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from composite_model import fit_weighted_static_map_em  # noqa: E402


OUTPUTS = {
    "sp500": ROOT / "output_static_q0_symmetric_t_sp500",
    "n225": ROOT / "output_static_q0_symmetric_t_n225",
}
MODEL_NAMES = {
    "fixed": "static_q0_symmetric_t_dynamic_location_scale_fixed_eta",
    "fixed_arc": "static_q0_symmetric_t_dynamic_location_scale_fixed_eta_arc_sm",
    "dynamic": "static_q0_symmetric_t_dynamic_location_scale_eta",
    "dynamic_arc": "static_q0_symmetric_t_dynamic_location_scale_eta_arc_sm",
}
STATE_NAMES = ("location", "log_scale_squared", "eta")
SCORE_NAMES = old.SCORE_NAMES
SHAPE_SCORE_STEP = 2e-5
SEED = 20260912


@dataclass(frozen=True)
class CorrectedReferenceFilter:
    states: np.ndarray
    means: np.ndarray
    log_variances: np.ndarray
    variances: np.ndarray
    eta: np.ndarray
    skew: np.ndarray
    scores: np.ndarray
    contributions: np.ndarray
    base_contributions: np.ndarray
    correction_contributions: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.contributions))


def normalized_bspline_derivative(
    u: np.ndarray, knots: np.ndarray, degree: int, masses: np.ndarray
) -> np.ndarray:
    """Derivative of the normalized B-spline component densities."""
    from scipy.interpolate import BSpline

    u = np.asarray(u, dtype=float)
    components = len(knots) - degree - 1
    lower = BSpline.design_matrix(
        u, knots, degree - 1, extrapolate=False
    ).toarray()
    derivative = np.zeros((u.size, components))
    for j in range(components):
        left_denominator = knots[j + degree] - knots[j]
        right_denominator = knots[j + degree + 1] - knots[j + 1]
        if left_denominator > 0.0:
            derivative[:, j] += degree * lower[:, j] / left_denominator
        if right_denominator > 0.0:
            derivative[:, j] -= degree * lower[:, j + 1] / right_denominator
    return derivative / np.asarray(masses)[None, :]


def static_reference_pit(values: np.ndarray, state: np.ndarray) -> np.ndarray:
    mean, log_variance = map(float, state[:2])
    z = (np.asarray(values) - mean) / np.exp(0.5 * log_variance)
    return np.clip(core.hansen_cdf(z, state[2:]), 1e-12, 1.0 - 1e-12)


def fit_static_q0(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    u = static_reference_pit(values, anchor)
    phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
    counts = np.diff(offsets)
    observation_weights = np.repeat(1.0 / counts, counts)
    fit = fit_weighted_static_map_em(
        phi,
        observation_weights,
        prior_mode=masses,
        prior_strength=0.0,
        tolerance=1e-13,
    )
    if not fit.converged:
        raise RuntimeError("static q0 EM did not converge")
    weights0 = np.asarray(fit.weights, dtype=float)
    old_q0 = helmert.T @ np.log(weights0 / masses)
    logits = np.log(masses) + helmert @ old_q0
    reconstructed = np.exp(logits - np.max(logits))
    reconstructed /= reconstructed.sum()
    return weights0, {
        "weights": weights0.tolist(),
        "old_uniform_chart_q0": old_q0.tolist(),
        "correction_composite_log_likelihood": float(
            fit.composite_log_likelihood
        ),
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
        "max_weight_change": float(fit.max_weight_change),
        "minimum_weight": float(np.min(weights0)),
        "maximum_weight": float(np.max(weights0)),
        "weight_reconstruction_error": float(
            np.max(np.abs(reconstructed - weights0))
        ),
        "new_chart": "w(q)=softmax(log(w0)+Hq), hence q=0 exactly gives w0",
    }


def corrected_logpdf_and_score(
    values: np.ndarray,
    state: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    *,
    include_eta_score: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full score of log f_theta(y)+log h_0(F_theta(y))."""
    values = np.asarray(values, dtype=float)
    state = np.asarray(state, dtype=float)
    base_log, base_score = core.reference_logpdf_and_score(values, state)
    mean, log_variance = map(float, state[:2])
    sigma = float(np.exp(0.5 * log_variance))
    z = (values - mean) / sigma
    u = np.clip(core.hansen_cdf(z, state[2:]), 1e-12, 1.0 - 1e-12)
    phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
    dphi = normalized_bspline_derivative(u, knots, core.DEGREE, masses)
    fitted = np.maximum(phi @ weights0, 1e-300)
    log_correction = np.log(fitted)
    dlog_h_du = (dphi @ weights0) / fitted

    standardized_density = np.exp(base_log + 0.5 * log_variance)
    du = np.empty((values.size, 4))
    du[:, 0] = -standardized_density / sigma
    du[:, 1] = -0.5 * z * standardized_density
    # Eta is the only dynamic shape coordinate in this symmetric-t experiment.
    # Lambda is fixed exactly at zero, so its (unused) correction derivative is
    # deliberately omitted to avoid two expensive Student-t CDF evaluations.
    if include_eta_score:
        direction = np.array([SHAPE_SCORE_STEP, 0.0])
        du[:, 2] = (
            core.hansen_cdf(z, state[2:] + direction)
            - core.hansen_cdf(z, state[2:] - direction)
        ) / (2.0 * SHAPE_SCORE_STEP)
    else:
        du[:, 2] = 0.0
    du[:, 3] = 0.0
    score = base_score + dlog_h_du[:, None] * du
    return base_log + log_correction, score, base_log, log_correction


def filter_corrected_reference(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    betas: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    *,
    score_coordinates: tuple[int, ...] = (0, 1, 2),
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> CorrectedReferenceFilter:
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if anchor.shape != (4,) or betas.shape != (3, 4):
        raise ValueError("anchor/beta dimensions do not match")
    days = offsets.size - 1
    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    contributions = np.empty(days)
    base_contributions = np.empty(days)
    correction_contributions = np.empty(days)
    state = anchor.copy() if initial_state is None else np.asarray(initial_state).copy()
    history = (
        np.empty((0, 4))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-21:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        mean, log_variance = map(float, state[:2])
        if (
            np.any(~np.isfinite(state))
            or abs(mean) > 0.5
            or log_variance < -20.0
            or log_variance > 0.0
            or np.max(np.abs(state[2:])) > 20.0
        ):
            raise FloatingPointError("unstable corrected-reference state")
        total_log, observation_score, base_log, correction_log = (
            corrected_logpdf_and_score(
                values[start:stop], state, weights0, knots, masses,
                include_eta_score=2 in score_coordinates,
            )
        )
        states[day] = state
        scores[day] = np.mean(observation_score, axis=0)
        contributions[day] = float(np.mean(total_log))
        base_contributions[day] = float(np.mean(base_log))
        correction_contributions[day] = float(np.mean(correction_log))
        history = np.vstack((history, scores[day][None, :]))[-21:]
        state = anchor.copy()
        for beta, window in zip(betas, core.WINDOWS):
            state += beta * history[-window:].sum(axis=0) / window
    eta_skew = np.array([core.state_to_parameters(s[2:])[:2] for s in states])
    return CorrectedReferenceFilter(
        states=states,
        means=states[:, 0].copy(),
        log_variances=states[:, 1].copy(),
        variances=np.exp(states[:, 1]),
        eta=eta_skew[:, 0],
        skew=eta_skew[:, 1],
        scores=scores,
        contributions=contributions,
        base_contributions=base_contributions,
        correction_contributions=correction_contributions,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_dynamic_corrected_reference(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    coordinates: tuple[int, ...],
    initial_betas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], CorrectedReferenceFilter, dict[str, object]]:
    static = filter_corrected_reference(
        values, offsets, anchor, np.zeros((3, 4)), weights0, knots, masses,
        score_coordinates=coordinates,
    )
    averages = core.fixed_denominator_score_averages(
        static.scores[:, coordinates], core.WINDOWS
    )
    target_steps = np.array([1e-4, 1e-2, 1e-2, 1e-2])[list(coordinates)]
    units = np.array(
        [
            [
                target_steps[j]
                / max(float(np.median(np.abs(average[:, j]))), 1e-12)
                for j in range(len(coordinates))
            ]
            for average in averages
        ]
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        betas = np.zeros((3, 4))
        betas[:, coordinates] = np.asarray(theta).reshape(3, len(coordinates)) * units
        return betas

    def run(theta: np.ndarray) -> CorrectedReferenceFilter:
        return filter_corrected_reference(
            values, offsets, anchor, unpack(theta), weights0, knots, masses,
            score_coordinates=coordinates,
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    size = 3 * len(coordinates)
    initial = np.zeros((3, len(coordinates)))
    if initial_betas is not None:
        initial[:, :] = np.asarray(initial_betas)[:, coordinates]
    rng = np.random.default_rng(SEED + len(coordinates))
    starts = [
        (initial / units).ravel(),
        np.zeros(size),
        np.ones(size),
        -np.ones(size),
        rng.normal(size=size),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * size,
            options={
                "maxiter": 4000,
                "ftol": 1e-14,
                "gtol": 1e-8,
                "eps": 1e-5,
                "maxls": 80,
            },
        )
        print(
            f"corrected reference {len(coordinates)}D start {index + 1}/5: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all corrected-reference starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    betas = unpack(theta)
    dynamic = run(theta)

    inference: dict[str, object] = {"available": False}
    if 2 in coordinates:
        print("corrected dynamic t: computing conditional HAC inference", flush=True)
        derivative_step = 1e-2
        hessian = numerical_hessian(objective, theta, derivative_step)
        eigenvalues = np.linalg.eigvalsh(hessian)
        if eigenvalues[0] > 1e-10:
            gradients = np.empty((offsets.size - 1, size))
            for coordinate in range(size):
                direction = np.zeros(size)
                direction[coordinate] = derivative_step
                gradients[:, coordinate] = (
                    run(theta + direction).contributions
                    - run(theta - direction).contributions
                ) / (2.0 * derivative_step)
            inverse_hessian = np.linalg.inv(hessian)
            covariance_theta = (
                inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
            )
            covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
            scaling = np.diag(units.ravel())
            covariance = scaling @ covariance_theta @ scaling
            covariance = (covariance + covariance.T) / 2.0
            parameters = betas[:, coordinates].ravel()
            standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            t_values = parameters / standard_errors
            p_values = 2.0 * ndtr(-np.abs(t_values))
            local_eta = coordinates.index(2)
            eta_indices = np.array(
                [local_eta + h * len(coordinates) for h in range(3)]
            )
            eta_beta = parameters[eta_indices]
            eta_cov = covariance[np.ix_(eta_indices, eta_indices)]
            eta_wald = float(eta_beta @ np.linalg.solve(eta_cov, eta_beta))
            inference = {
                "available": True,
                "scope": "conditional Godambe-HAC(22), static t and q0 fixed",
                "parameters": [
                    {
                        "state": STATE_NAMES[c],
                        "horizon": horizon,
                        "estimate": float(parameters[i]),
                        "hac22_se": float(standard_errors[i]),
                        "t": float(t_values[i]),
                        "p_two_sided": float(p_values[i]),
                    }
                    for i, (horizon, c) in enumerate(
                        (horizon, c)
                        for horizon in core.HORIZONS
                        for c in coordinates
                    )
                ],
                "eta_three_coefficient_wald": eta_wald,
                "eta_three_coefficient_p": float(chi2.sf(eta_wald, 3)),
                "hessian_eigenvalues_scaled": eigenvalues.tolist(),
            }
        else:
            inference = {
                "available": False,
                "reason": "nonpositive Hessian",
                "hessian_eigenvalues_scaled": eigenvalues.tolist(),
            }
    summary = {
        "coordinates": [STATE_NAMES[c] for c in coordinates],
        "betas": {
            horizon: {
                STATE_NAMES[c]: float(betas[h, c]) for c in coordinates
            }
            for h, horizon in enumerate(core.HORIZONS)
        },
        "lambda_dynamic_coefficients": "fixed exactly at zero",
        "score": "full log f_theta + log h_w0(F_theta)",
        "parameter_units": units.tolist(),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_composite_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_static": dynamic.composite_log_likelihood
        - static.composite_log_likelihood,
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(
            max(float(fit.fun) for fit in successful) - float(optimum.fun)
        ),
    }
    return betas, summary, dynamic, inference


def continue_corrected_reference(
    panel: core.RawPanel,
    n_train: int,
    anchor: np.ndarray,
    betas: np.ndarray,
    training: CorrectedReferenceFilter,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    score_coordinates: tuple[int, ...],
) -> tuple[CorrectedReferenceFilter, CorrectedReferenceFilter, float]:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = filter_corrected_reference(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        weights0,
        knots,
        masses,
        score_coordinates=score_coordinates,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full = filter_corrected_reference(
        panel.values, panel.offsets, anchor, betas, weights0, knots, masses,
        score_coordinates=score_coordinates,
    )
    error = float(
        max(
            np.max(np.abs(full.states[n_train:] - oos.states)),
            np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
        )
    )
    if error > 2e-12:
        raise RuntimeError(f"corrected-reference continuation error: {error}")
    return oos, full, error


def corrected_tangent_geometry(
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    coordinates: tuple[int, ...],
    label: str,
    quadrature_order: int = 128,
) -> core.EfficientGeometry:
    nodes0, weights_gl = np.polynomial.legendre.leggauss(quadrature_order)
    q_parts, tangent_parts, measure_parts = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes0 / 2.0
        du_weight = weights_gl * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        h0 = phi @ weights0
        responsibility = phi * weights0[None, :] / h0[:, None]
        q_parts.append((responsibility - weights0[None, :]) @ helmert)
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_parts.append(
            core.reference_logpdf_and_score(y, anchor)[1][:, coordinates]
        )
        measure_parts.append(du_weight * h0)
    q_score = np.vstack(q_parts)
    tangent = np.vstack(tangent_parts)
    weight = np.concatenate(measure_parts)
    mass_error = float(abs(weight.sum() - 1.0))
    information_q = (q_score.T * weight) @ q_score
    information_tangent = (tangent.T * weight) @ tangent
    tangent_sd = np.sqrt(np.diag(information_tangent))
    tangent_scaled = tangent / tangent_sd[None, :]
    tangent_information_scaled = (tangent_scaled.T * weight) @ tangent_scaled
    cross_scaled = (q_score.T * weight) @ tangent_scaled
    projection_scaled = cross_scaled @ np.linalg.inv(tangent_information_scaled)
    projection = projection_scaled / tangent_sd[None, :]
    efficient = q_score - tangent @ projection.T
    residual_information = (efficient.T * weight) @ efficient
    residual_information = (residual_information + residual_information.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(residual_information)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive {label} residual information: {eigenvalues}")
    reversal = np.fliplr(np.eye(weights0.size))
    reflection = helmert.T @ reversal @ helmert
    identity = np.eye(weights0.size - 1)
    plus = (identity + reflection) / 2.0
    minus = (identity - reflection) / 2.0
    post_cross = (efficient.T * weight) @ tangent
    return core.EfficientGeometry(
        projection_q_on_hansen=projection,
        residual_information=residual_information,
        inverse_sqrt=identity,
        projector_plus=plus,
        projector_minus=minus,
        diagnostics={
            "projection": label,
            "nuisance_dimension": len(coordinates),
            "anchor_measure": "h_w0(u) du",
            "q_zero_baseline": "fitted static spline weights w0",
            "nuisance_score": "modular reference-only score; h side excluded",
            "parity_interpretation": (
                "even/odd log-weight deviations around w0; w0 itself need not be symmetric"
            ),
            "residual_score_whitening": False,
            "quadrature_mass_error": mass_error,
            "tangent_information_eigenvalues_scaled": np.linalg.eigvalsh(
                tangent_information_scaled
            ).tolist(),
            "residual_information_eigenvalues": eigenvalues.tolist(),
            "residual_information_condition_number": float(
                eigenvalues[-1] / eigenvalues[0]
            ),
            "maximum_post_projection_cross_information": float(
                np.max(np.abs(post_cross))
            ),
            "information_q_condition_number": float(np.linalg.cond(information_q)),
            "plus_idempotence_error": float(np.max(np.abs(plus @ plus - plus))),
            "minus_idempotence_error": float(np.max(np.abs(minus @ minus - minus))),
            "plus_minus_orthogonality_error": float(np.max(np.abs(plus @ minus))),
        },
    )


def fit_arc_sm(
    phi: np.ndarray,
    train_offsets: np.ndarray,
    oos_offsets: np.ndarray,
    reference_train: core.FourStateFilter,
    reference_oos: core.FourStateFilter,
    reference_full: core.FourStateFilter,
    n_train: int,
    weights0: np.ndarray,
    helmert: np.ndarray,
    geometry: core.EfficientGeometry,
    score_coordinates: tuple[int, ...],
):
    fit, q_zero_train, correction_train = core.fit_residual_parity6_dcd(
        phi[: int(train_offsets[-1])],
        train_offsets,
        reference_train.scores[:, score_coordinates],
        weights0,
        helmert,
        geometry,
        nested_even_5_22=np.zeros(2),
        numerical_state_step=1e-2,
        adaptive_hessian=True,
    )
    kappa_plus = np.asarray(fit["kappa_plus"])
    kappa_minus = np.asarray(fit["kappa_minus"])
    correction_oos = core.filter_residual_parity6_dcd(
        phi[int(train_offsets[-1]) :],
        oos_offsets,
        reference_oos.scores[:, score_coordinates],
        kappa_plus,
        kappa_minus,
        weights0,
        helmert,
        geometry,
        initial_state=correction_train.terminal_state,
        initial_score_history=correction_train.terminal_score_history,
    )
    full_offsets = np.r_[train_offsets, oos_offsets[1:] + train_offsets[-1]]
    correction_full = core.filter_residual_parity6_dcd(
        phi,
        full_offsets,
        reference_full.scores[:, score_coordinates],
        kappa_plus,
        kappa_minus,
        weights0,
        helmert,
        geometry,
    )
    continuation_error = float(
        max(
            np.max(np.abs(correction_full.states[n_train:] - correction_oos.states)),
            np.max(
                np.abs(
                    correction_full.contributions[n_train:]
                    - correction_oos.contributions
                )
            ),
        )
    )
    q_zero_state_error = float(np.max(np.abs(q_zero_train.states)))
    q_zero_weight_error = float(
        np.max(np.abs(q_zero_train.weights - weights0[None, :]))
    )
    static_log = np.empty(train_offsets.size - 1)
    for day, (start, stop) in enumerate(zip(train_offsets[:-1], train_offsets[1:])):
        static_log[day] = float(np.mean(np.log(phi[start:stop] @ weights0)))
    static_log_error = float(
        np.max(np.abs(q_zero_train.contributions - static_log))
    )
    if max(
        continuation_error, q_zero_state_error, q_zero_weight_error, static_log_error
    ) > 2e-12:
        raise RuntimeError("recentered DCD-SM continuation/nesting check failed")
    fit["gain_vs_static_q0"] = float(
        correction_train.composite_log_likelihood
        - q_zero_train.composite_log_likelihood
    )
    return fit, correction_train, correction_oos, {
        "continuation_error": continuation_error,
        "q_zero_state_error": q_zero_state_error,
        "q_zero_weight_error": q_zero_weight_error,
        "q_zero_static_log_error": static_log_error,
    }


def score_oos_pair(
    values: np.ndarray,
    offsets: np.ndarray,
    dates: pd.DatetimeIndex,
    training_values: np.ndarray,
    reference: core.FourStateFilter,
    correction,
    knots: np.ndarray,
    masses: np.ndarray,
    weights0: np.ndarray,
    reference_name: str,
    arc_name: str,
) -> pd.DataFrame:
    u_grid, component_cdf = core.component_cdfs(knots, masses)
    probability_nodes = (np.arange(core.QUANTILE_NODES) + 0.5) / core.QUANTILE_NODES
    bin_edges = {
        count: np.concatenate(
            ([-np.inf], np.quantile(training_values, np.arange(1, count) / count), [np.inf])
        )
        for count in core.BIN_COUNTS
    }
    thresholds = {
        alpha: float(np.quantile(training_values, alpha)) for alpha in core.TAIL_LEVELS
    }
    static_cdf = np.maximum.accumulate(component_cdf @ weights0)
    static_cdf[0], static_cdf[-1] = 0.0, 1.0
    rows: list[dict[str, object]] = []
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if day % 50 == 0:
            print(f"scoring OOS day {day + 1}/{len(dates)}", flush=True)
        observations = values[start:stop]
        state = reference.states[day]
        mean, log_variance = state[:2]
        sigma = float(np.exp(0.5 * log_variance))
        shape_state = state[2:]
        dynamic_cdf = np.maximum.accumulate(
            component_cdf @ correction.weights[day]
        )
        dynamic_cdf[0], dynamic_cdf[-1] = 0.0, 1.0
        mean_log_h0 = float(
            np.mean(
                np.log(
                    core.normalized_bspline_basis(
                        np.clip(
                            core.hansen_cdf(
                                (observations - mean) / sigma, shape_state
                            ),
                            1e-12,
                            1.0 - 1e-12,
                        ),
                        knots,
                        core.DEGREE,
                        masses,
                    )
                    @ weights0
                )
            )
        )
        model_data = (
            (
                reference_name,
                static_cdf,
                float(reference.contributions[day] + mean_log_h0),
            ),
            (
                arc_name,
                dynamic_cdf,
                float(reference.contributions[day] + correction.contributions[day]),
            ),
        )
        for model, correction_cdf, continuous_log in model_data:
            corrected_u = np.interp(probability_nodes, correction_cdf, u_grid)
            nodes = mean + sigma * core.hansen_ppf(corrected_u, shape_state)
            row: dict[str, object] = {
                "date": str(dates[day].date()),
                "model": model,
                "n_constituents": int(stop - start),
                "continuous_log_score": continuous_log,
                "crps": core.score_ordered_predictive_sample(nodes, observations),
            }
            for count in core.BIN_COUNTS:
                edges = bin_edges[count]
                base_u = np.concatenate(
                    (
                        [0.0],
                        core.hansen_cdf(
                            (edges[1:-1] - mean) / sigma, shape_state
                        ),
                        [1.0],
                    )
                )
                model_cdf = np.interp(base_u, u_grid, correction_cdf)
                row[f"categorical_log_score_{count}"] = core.categorical_log_score(
                    np.maximum(np.diff(model_cdf), 0.0), observations, edges
                )
            for alpha in core.TAIL_LEVELS:
                quantile = float(np.interp(alpha, probability_nodes, nodes))
                base_probability = float(
                    core.hansen_cdf(
                        np.array([(thresholds[alpha] - mean) / sigma]), shape_state
                    )[0]
                )
                probability = float(
                    np.interp(base_probability, u_grid, correction_cdf)
                )
                row[f"q_loss_{alpha:g}"] = core.q_loss(observations, quantile, alpha)
                row[f"brier_{alpha:g}"] = core.brier_loss(
                    observations, thresholds[alpha], probability
                )
            rows.append(row)
    return pd.DataFrame(rows)


def validate_score(
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = rng.normal(anchor[0], np.exp(0.5 * anchor[1]), 401)
    _, analytic, _, _ = corrected_logpdf_and_score(
        values, anchor, weights0, knots, masses
    )
    steps = np.array([1e-7, 1e-6, SHAPE_SCORE_STEP])
    numerical = np.empty((values.size, 3))
    for coordinate, step in enumerate(steps):
        direction = np.zeros(4)
        direction[coordinate] = step
        numerical[:, coordinate] = (
            corrected_logpdf_and_score(
                values, anchor + direction, weights0, knots, masses
            )[0]
            - corrected_logpdf_and_score(
                values, anchor - direction, weights0, knots, masses
            )[0]
        ) / (2.0 * step)
    analytic = analytic[:, :3]
    absolute = float(np.max(np.abs(analytic - numerical)))
    relative = float(
        np.max(np.abs(analytic - numerical) / np.maximum(1.0, np.abs(numerical)))
    )
    if absolute > 3e-3 or relative > 3e-5:
        raise RuntimeError(f"corrected score validation failed: {absolute}, {relative}")
    return {"maximum_absolute_error": absolute, "maximum_relative_error": relative}


def run_market(market: str) -> None:
    output = OUTPUTS[market]
    output.mkdir(parents=True, exist_ok=True)
    panel = old.load_market(market)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= core.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    training_values = panel.values[:training_rows]

    previous_path = old.OUTPUTS[market] / "estimation.json"
    if not previous_path.exists():
        raise FileNotFoundError(
            f"missing prior symmetric-t four-model result: {previous_path}"
        )
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    static_t = previous["static_symmetric_t"]
    anchor = np.asarray(static_t["state"], dtype=float)

    def stored_betas(section: str, coordinates: tuple[int, ...]) -> np.ndarray:
        output_betas = np.zeros((3, 4))
        stored = previous[section]["betas"]
        for h, horizon in enumerate(core.HORIZONS):
            for coordinate in coordinates:
                output_betas[h, coordinate] = stored[horizon][STATE_NAMES[coordinate]]
        return output_betas

    fixed_betas = stored_betas("fixed_eta_dynamic_location_scale", (0, 1))
    dynamic_betas = stored_betas("dynamic_location_scale_eta", (0, 1, 2))
    fixed_summary = previous["fixed_eta_dynamic_location_scale"]
    dynamic_summary = previous["dynamic_location_scale_eta"]
    dynamic_inference = previous["dynamic_location_scale_eta_inference"]
    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    print(f"{market}: fitting static conditional q0", flush=True)
    weights0, static_q0 = fit_static_q0(
        training_values, train_offsets, anchor, knots, masses, helmert
    )
    print(f"{market}: reproducing modular reference paths", flush=True)
    fixed_train = core.filter_reference(
        training_values, train_offsets, anchor, fixed_betas
    )
    fixed_oos, fixed_full, fixed_continuation = old.continue_reference(
        panel, n_train, anchor, fixed_betas, fixed_train
    )
    dynamic_train = core.filter_reference(
        training_values, train_offsets, anchor, dynamic_betas
    )
    dynamic_oos, dynamic_full, dynamic_continuation = old.continue_reference(
        panel, n_train, anchor, dynamic_betas, dynamic_train
    )
    fixed_likelihood_error = float(
        abs(
            fixed_train.composite_log_likelihood
            - fixed_summary["dynamic_composite_log_likelihood"]
        )
    )
    dynamic_likelihood_error = float(
        abs(
            dynamic_train.composite_log_likelihood
            - dynamic_summary["dynamic_composite_log_likelihood"]
        )
    )
    if max(fixed_likelihood_error, dynamic_likelihood_error) > 1e-9:
        raise RuntimeError("stored modular reference paths do not reproduce")
    if max(
        np.max(np.abs(fixed_full.states[:, 3])),
        np.max(np.abs(dynamic_full.states[:, 3])),
    ) > 1e-15:
        raise RuntimeError("lambda moved away from zero")

    fixed_u = core.dynamic_reference_pit(panel.values, panel.offsets, fixed_full.states)
    dynamic_u = core.dynamic_reference_pit(
        panel.values, panel.offsets, dynamic_full.states
    )
    fixed_phi = core.normalized_bspline_basis(fixed_u, knots, core.DEGREE, masses)
    dynamic_phi = core.normalized_bspline_basis(dynamic_u, knots, core.DEGREE, masses)
    fixed_geometry = corrected_tangent_geometry(
        anchor,
        weights0,
        knots,
        masses,
        helmert,
        (0, 1),
        "modular reference location/log-scale-squared update scores",
    )
    dynamic_geometry = corrected_tangent_geometry(
        anchor,
        weights0,
        knots,
        masses,
        helmert,
        (0, 1, 2),
        "modular reference location/log-scale-squared/eta update scores",
    )

    print(f"{market}: fitting fixed-eta DCD-SM around q0", flush=True)
    fixed_arc, _, fixed_correction_oos, fixed_arc_checks = fit_arc_sm(
        fixed_phi,
        train_offsets,
        oos_offsets,
        fixed_train,
        fixed_oos,
        fixed_full,
        n_train,
        weights0,
        helmert,
        fixed_geometry,
        (0, 1),
    )
    print(f"{market}: fitting dynamic-eta DCD-SM around q0", flush=True)
    dynamic_arc, _, dynamic_correction_oos, dynamic_arc_checks = fit_arc_sm(
        dynamic_phi,
        train_offsets,
        oos_offsets,
        dynamic_train,
        dynamic_oos,
        dynamic_full,
        n_train,
        weights0,
        helmert,
        dynamic_geometry,
        (0, 1, 2),
    )

    fixed_scores = score_oos_pair(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        fixed_oos,
        fixed_correction_oos,
        knots,
        masses,
        weights0,
        MODEL_NAMES["fixed"],
        MODEL_NAMES["fixed_arc"],
    )
    dynamic_scores = score_oos_pair(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        dynamic_oos,
        dynamic_correction_oos,
        knots,
        masses,
        weights0,
        MODEL_NAMES["dynamic"],
        MODEL_NAMES["dynamic_arc"],
    )
    scores = pd.concat((fixed_scores, dynamic_scores), ignore_index=True)
    summary = scores.groupby("model", as_index=False)[list(SCORE_NAMES)].mean()
    structural_pairs = (
        (MODEL_NAMES["fixed_arc"], MODEL_NAMES["fixed"]),
        (MODEL_NAMES["dynamic_arc"], MODEL_NAMES["dynamic"]),
        (MODEL_NAMES["fixed"], MODEL_NAMES["dynamic"]),
        (MODEL_NAMES["fixed_arc"], MODEL_NAMES["dynamic_arc"]),
    )
    comparisons = [
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in structural_pairs
        for score in SCORE_NAMES
    ]

    def arc_rows(specification: str, fit: dict[str, object]):
        return [
            {
                "specification": specification,
                "block": block,
                "horizon": horizon,
                "estimate": estimate,
                "hac22_se_conditional_on_reference": se,
                "p_two_sided": p,
            }
            for block, horizon, estimate, se, p in zip(
                ["even"] * 3 + ["odd"] * 3,
                [*core.WINDOWS, *core.WINDOWS],
                [*fit["kappa_plus"], *fit["kappa_minus"]],
                fit["se_godambe_hac22_conditional_on_hansen"],
                fit["p_two_sided_conditional_on_hansen"],
            )
        ]

    arc_parameters = pd.DataFrame(
        arc_rows("fixed_eta", fixed_arc) + arc_rows("dynamic_eta", dynamic_arc)
    )
    dynamic_parameters = pd.DataFrame(dynamic_inference.get("parameters", []))
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "four_model_oos_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "four_model_oos_comparisons.csv", index=False
    )
    arc_parameters.to_csv(output / "arc_sm_parameters.csv", index=False)
    dynamic_parameters.to_csv(output / "dynamic_t_parameters.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(masses.size),
            "uniform_chart_weight": masses,
            "fitted_static_q0_weight": weights0,
        }
    ).to_csv(output / "static_q0_weights.csv", index=False)

    result = {
        "analysis": f"{market} symmetric-t four-model comparison around fitted static q0",
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
            "total_observations": int(panel.values.size),
        },
        "specification": {
            "estimation_sequence": [
                "static symmetric-t MLE",
                "conditional static q0 MLE with reference fixed",
                "modular reference-only score dynamics with q0 fixed and excluded",
                "conditional parity DCD-SM dynamics around q=0 (weights w0)",
            ],
            "reference_score": "derivative of log f_theta(y); h side excluded",
            "arc_projection": "off the modular reference update scores",
            "residual_score_whitening": False,
            "parity": "even/odd log-weight deviations around fitted w0",
            "windows": list(core.WINDOWS),
        },
        "static_symmetric_t": static_t,
        "static_q0": static_q0,
        "fixed_eta_dynamic_location_scale": fixed_summary,
        "fixed_eta_arc_sm": fixed_arc,
        "dynamic_location_scale_eta": dynamic_summary,
        "dynamic_location_scale_eta_inference": dynamic_inference,
        "dynamic_eta_arc_sm": dynamic_arc,
        "geometries": {
            "fixed_eta_arc_sm": fixed_geometry.diagnostics,
            "dynamic_eta_arc_sm": dynamic_geometry.diagnostics,
        },
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons,
        "checks": {
            "stored_fixed_reference_likelihood_reproduction_error": fixed_likelihood_error,
            "stored_dynamic_reference_likelihood_reproduction_error": dynamic_likelihood_error,
            "lambda_fixed_maximum_absolute_state": float(
                max(
                    np.max(np.abs(fixed_full.states[:, 3])),
                    np.max(np.abs(dynamic_full.states[:, 3])),
                )
            ),
            "fixed_reference_continuation": fixed_continuation,
            "dynamic_reference_continuation": dynamic_continuation,
            "fixed_eta_arc_sm": fixed_arc_checks,
            "dynamic_eta_arc_sm": dynamic_arc_checks,
        },
    }
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output, result, summary, pd.DataFrame(comparisons), arc_parameters)
    print(summary.to_string(index=False), flush=True)


def write_report(
    output: Path,
    result: dict[str, object],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    arc_parameters: pd.DataFrame,
) -> None:
    labels = {
        MODEL_NAMES["fixed"]: "static q0; dynamic m,sigma2; fixed eta",
        MODEL_NAMES["fixed_arc"]: "static q0; dynamic m,sigma2; fixed eta + DCD-SM",
        MODEL_NAMES["dynamic_arc"]: "static q0; dynamic m,sigma2,eta + DCD-SM",
        MODEL_NAMES["dynamic"]: "static q0; dynamic m,sigma2,eta",
    }
    lines = [
        f"# {result['analysis']}",
        "",
        "The static symmetric-t reference is fitted first and held fixed while q0 is "
        "estimated. Reference dynamics remain modular and use only the reference "
        "score, with the h side excluded. DCD-SM is recentered so q=0 exactly equals "
        "q0, projected off those reference update scores, and updated by unwhitened "
        "parity HAR(1,5,22).",
        "",
        "## Static q0",
        "",
        f"Static correction composite log likelihood: "
        f"`{result['static_q0']['correction_composite_log_likelihood']:.9f}`; "
        f"weight range: `[{result['static_q0']['minimum_weight']:.6g}, "
        f"{result['static_q0']['maximum_weight']:.6g}]`.",
        "",
        "## Strict-OOS four-model comparison",
        "",
        "| model | continuous log score (higher) | CRPS (lower) |",
        "|---|---:|---:|",
    ]
    for row in summary.sort_values("continuous_log_score", ascending=False).to_dict(
        orient="records"
    ):
        lines.append(
            f"| {labels[row['model']]} | {row['continuous_log_score']:.9f} | "
            f"{row['crps']:.9f} |"
        )
    lines.extend(["", "## DCD-SM score-coefficient tests", ""])
    for specification in ("fixed_eta", "dynamic_eta"):
        fit = result[
            "fixed_eta_arc_sm" if specification == "fixed_eta" else "dynamic_eta_arc_sm"
        ]
        lines.extend(
            [
                f"### {specification}",
                "",
                "| block | horizon | estimate | HAC(22) p |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in arc_parameters.loc[
            arc_parameters["specification"] == specification
        ].to_dict(orient="records"):
            lines.append(
                f"| {row['block']} | {row['horizon']} | {row['estimate']:.9g} | "
                f"{row['p_two_sided']:.6g} |"
            )
        lines.extend(
            [
                "",
                f"Joint six-coefficient p: `{fit['joint_all_p']:.6g}`; "
                f"joint odd-block p: `{fit['joint_odd_p']:.6g}`; "
                f"training gain over static q0: `{fit['gain_vs_static_q0']:.9f}`.",
                "",
            ]
        )
    inference = result["dynamic_location_scale_eta_inference"]
    eta_text = (
        f"Eta three-coefficient Wald p: `{inference['eta_three_coefficient_p']:.6g}`."
        if "eta_three_coefficient_p" in inference
        else f"Eta inference unavailable: `{inference.get('reason', 'unknown')}`."
    )
    lines.extend(
        [
            "## Dynamic eta reference test",
            "",
            eta_text,
            "",
            "## Structural OOS comparisons",
            "",
            "Positive improvement favors the proposed model.",
            "",
            "| proposed | benchmark | score | improvement | HAC(22) p |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in comparisons.loc[
        comparisons["score"].isin(("continuous_log_score", "crps"))
    ].to_dict(orient="records"):
        lines.append(
            f"| {labels[row['proposed']]} | {labels[row['benchmark']]} | "
            f"{row['score']} | {row['improvement_mean']:.9g} | "
            f"{row['hac22_two_sided_p']:.6g} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    args = parser.parse_args()
    markets = ("sp500", "n225") if args.market == "both" else (args.market,)
    for market in markets:
        run_market(market)


if __name__ == "__main__":
    main()
