"""Four-model fitted-static-q0 Hansen skew-t experiment.

M1: static Hansen skew-t plus fitted static spline q0.
M2: M1 plus dynamic location and log scale-squared; shape and q0 fixed.
M3: frozen M2 path plus joint dynamic eta/lambda, with both reference shape
    scores Fisher-projected off location/log-scale at the static fitted density.
M4: frozen M2 path plus dynamic spline q, split into its conditional eta
    direction and eta-orthogonal others after projecting off location/log-scale.

Reference updates use reference-only scores: derivatives through the fixed
static spline via its PIT are deliberately excluded.  There is no whitening
and no parity split.  M3 and M4 are sibling branches from M2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2

import run_direct_dynamic_skewt_shape_test as direct
import run_identity_sm_skewt_fx_five_models as fx
import run_identity_sm_skewt_sector_five_models as sectors
import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_single_stocks as stocks
import run_static_q0_symmetric_t_four_models as q0
from estimate_score_har import fixed_denominator_score_averages, hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_q0_skewt_four_models"
DATASETS = five.DATASETS
MODEL_NAMES = {
    "m1": "m1_static_skewt_static_sm",
    "m2": "m2_dynamic_location_scale_fixed_eta_lambda_static_sm",
    "m3": "m3_dynamic_location_scale_eta_lambda_static_sm",
    "m4": "m4_dynamic_location_scale_fixed_eta_lambda_qeta_others_arc_sm",
}
STATE_NAMES = ("location", "log_scale_squared", "eta", "lambda")
SEED = 20260911


@dataclass(frozen=True)
class ModularReferenceFilter:
    states: np.ndarray
    scores: np.ndarray
    base_contributions: np.ndarray
    static_sm_contributions: np.ndarray
    total_contributions: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.total_contributions))

    def as_reference(self) -> core.FourStateFilter:
        parameters = np.array([core.state_to_parameters(state[2:])[:2] for state in self.states])
        return core.FourStateFilter(
            states=self.states,
            means=self.states[:, 0],
            log_variances=self.states[:, 1],
            variances=np.exp(self.states[:, 1]),
            eta=parameters[:, 0],
            skew=parameters[:, 1],
            scores=self.scores,
            contributions=self.base_contributions,
            terminal_state=self.terminal_state,
            terminal_score_history=self.terminal_score_history,
        )


@dataclass(frozen=True)
class ShapeGeometry:
    projection_shape_on_ms: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ShapeLayerFilter:
    states: np.ndarray
    scores: np.ndarray
    efficient_shape_scores: np.ndarray
    base_contributions: np.ndarray
    static_sm_contributions: np.ndarray
    total_contributions: np.ndarray
    terminal_shape_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.total_contributions))

    def as_reference(self) -> core.FourStateFilter:
        parameters = np.array([core.state_to_parameters(state[2:])[:2] for state in self.states])
        return core.FourStateFilter(
            states=self.states,
            means=self.states[:, 0],
            log_variances=self.states[:, 1],
            variances=np.exp(self.states[:, 1]),
            eta=parameters[:, 0],
            skew=parameters[:, 1],
            scores=self.scores,
            contributions=self.base_contributions,
            terminal_state=np.r_[self.states[-1, :2], self.terminal_shape_state],
            terminal_score_history=np.empty((0, 4)),
        )


def static_log_h(
    observations: np.ndarray,
    state: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    z = (observations - state[0]) / np.exp(0.5 * state[1])
    u = np.clip(core.hansen_cdf(z, state[2:]), 1e-12, 1.0 - 1e-12)
    phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
    return np.log(np.maximum(phi @ weights0, 1e-300))


def filter_modular_reference(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    betas: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ModularReferenceFilter:
    days = offsets.size - 1
    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    base_contributions = np.empty(days)
    static_contributions = np.empty(days)
    total_contributions = np.empty(days)
    state = anchor.copy() if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    history = (
        np.empty((0, 4))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-21:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if (
            np.any(~np.isfinite(state)) or abs(state[0]) > 0.5
            or state[1] < -20.0 or state[1] > 0.0 or np.max(np.abs(state[2:])) > 20.0
        ):
            raise FloatingPointError("unstable modular reference state")
        observations = values[start:stop]
        base_log, observation_score = core.reference_logpdf_and_score(observations, state)
        log_h0 = static_log_h(observations, state, weights0, knots, masses)
        states[day] = state
        scores[day] = np.mean(observation_score, axis=0)
        base_contributions[day] = float(np.mean(base_log))
        static_contributions[day] = float(np.mean(log_h0))
        total_contributions[day] = base_contributions[day] + static_contributions[day]
        history = np.vstack((history, scores[day][None, :]))[-21:]
        state = anchor.copy()
        for beta, window in zip(betas, core.WINDOWS):
            state += beta * history[-window:].sum(axis=0) / window
    return ModularReferenceFilter(
        states=states,
        scores=scores,
        base_contributions=base_contributions,
        static_sm_contributions=static_contributions,
        total_contributions=total_contributions,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_location_scale(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    initial_betas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], ModularReferenceFilter]:
    static = filter_modular_reference(
        values, offsets, anchor, np.zeros((3, 4)), weights0, knots, masses
    )
    averages = fixed_denominator_score_averages(static.scores[:, :2], core.WINDOWS)
    targets = np.array([1e-4, 1e-2])
    units = np.array([
        [targets[j] / max(float(np.median(np.abs(average[:, j]))), 1e-12) for j in range(2)]
        for average in averages
    ])

    def unpack(theta: np.ndarray) -> np.ndarray:
        betas = np.zeros((3, 4))
        betas[:, :2] = np.asarray(theta).reshape(3, 2) * units
        return betas

    def run(theta: np.ndarray) -> ModularReferenceFilter:
        return filter_modular_reference(values, offsets, anchor, unpack(theta), weights0, knots, masses)

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    initial = np.zeros((3, 2)) if initial_betas is None else np.asarray(initial_betas)[:, :2]
    rng = np.random.default_rng(SEED + 1)
    starts = [(initial / units).ravel(), np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6)]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective, np.clip(start, -500.0, 500.0), method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 6,
            options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-9, "eps": 1e-5, "maxls": 80},
        )
        print(f"M2 start {index + 1}/5: success={fit.success} objective={float(fit.fun):.12g}", flush=True)
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all Model-2 starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    betas = unpack(np.asarray(optimum.x))
    dynamic = run(np.asarray(optimum.x))
    return betas, {
        "betas": {
            horizon: {"location": float(betas[i, 0]), "log_scale_squared": float(betas[i, 1])}
            for i, horizon in enumerate(core.HORIZONS)
        },
        "parameter_units": units.tolist(),
        "static_log_likelihood": static.composite_log_likelihood,
        "dynamic_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_model1": dynamic.composite_log_likelihood - static.composite_log_likelihood,
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(max(fit.fun for fit in successful) - optimum.fun),
    }, dynamic


def weighted_shape_geometry(
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    quadrature_order: int = 128,
) -> ShapeGeometry:
    nodes, legendre_weights = np.polynomial.legendre.leggauss(quadrature_order)
    score_parts, measure_parts = [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes / 2.0
        du = legendre_weights * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        h0 = phi @ weights0
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        score_parts.append(core.reference_logpdf_and_score(y, anchor)[1])
        measure_parts.append(du * h0)
    scores = np.vstack(score_parts)
    measure = np.concatenate(measure_parts)
    ms, shape = scores[:, :2], scores[:, 2:]
    information_ms = (ms.T * measure) @ ms
    projection = ((shape.T * measure) @ ms) @ np.linalg.inv(information_ms)
    efficient = shape - ms @ projection.T
    information = (efficient.T * measure) @ efficient
    information = (information + information.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(information)
    post_cross = (efficient.T * measure) @ ms
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive M3 shape information: {eigenvalues}")
    return ShapeGeometry(
        projection_shape_on_ms=projection,
        diagnostics={
            "measure": "f_skewt(y;theta0) h0(F_theta0(y))dy = h0(u)du",
            "reference_score": "derivative of log f_skewt only; fixed h0 PIT derivative excluded",
            "projection": "(eta,lambda) off (location,log scale-squared)",
            "score_whitening": False,
            "eta_lambda_mutual_orthogonalization": False,
            "quadrature_mass_error": float(abs(np.sum(measure) - 1.0)),
            "maximum_post_projection_cross_information": float(np.max(np.abs(post_cross))),
            "efficient_shape_information_eigenvalues": eigenvalues.tolist(),
        },
    )


def filter_shape_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen: ModularReferenceFilter,
    anchor: np.ndarray,
    betas: np.ndarray,
    geometry: ShapeGeometry,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    *,
    initial_shape_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ShapeLayerFilter:
    days = offsets.size - 1
    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    efficient_scores = np.empty((days, 2))
    base_contributions = np.empty(days)
    static_contributions = np.empty(days)
    total_contributions = np.empty(days)
    shape_state = anchor[2:].copy() if initial_shape_state is None else np.asarray(initial_shape_state).copy()
    history = (
        np.empty((0, 2))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-21:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = frozen.states[day].copy()
        state[2:] = shape_state
        if np.any(~np.isfinite(state)) or np.max(np.abs(shape_state)) > 20.0:
            raise FloatingPointError("unstable M3 shape state")
        observations = values[start:stop]
        base_log, observation_score = core.reference_logpdf_and_score(observations, state)
        log_h0 = static_log_h(observations, state, weights0, knots, masses)
        daily_score = np.mean(observation_score, axis=0)
        efficient = daily_score[2:] - geometry.projection_shape_on_ms @ daily_score[:2]
        states[day], scores[day], efficient_scores[day] = state, daily_score, efficient
        base_contributions[day] = float(np.mean(base_log))
        static_contributions[day] = float(np.mean(log_h0))
        total_contributions[day] = base_contributions[day] + static_contributions[day]
        history = np.vstack((history, efficient[None, :]))[-21:]
        shape_state = anchor[2:].copy()
        for beta, window in zip(betas, core.WINDOWS):
            shape_state += beta * history[-window:].sum(axis=0) / window
    return ShapeLayerFilter(
        states, scores, efficient_scores, base_contributions, static_contributions,
        total_contributions, shape_state, history,
    )


def wald(beta: np.ndarray, covariance: np.ndarray, indices: list[int]) -> dict[str, float | int]:
    selected = np.asarray(indices)
    value = beta[selected]
    variance = covariance[np.ix_(selected, selected)]
    statistic = float(value @ np.linalg.solve(variance, value))
    return {"wald": statistic, "df": int(selected.size), "p": float(chi2.sf(statistic, selected.size))}


def fit_shape_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen: ModularReferenceFilter,
    anchor: np.ndarray,
    geometry: ShapeGeometry,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], ShapeLayerFilter, ShapeLayerFilter]:
    nested = filter_shape_layer(
        values, offsets, frozen, anchor, np.zeros((3, 2)), geometry, weights0, knots, masses
    )
    averages = fixed_denominator_score_averages(nested.efficient_shape_scores, core.WINDOWS)
    units = np.array([
        [1e-2 / max(float(np.median(np.abs(average[:, j]))), 1e-12) for j in range(2)]
        for average in averages
    ])

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta).reshape(3, 2) * units

    def run(theta: np.ndarray) -> ShapeLayerFilter:
        return filter_shape_layer(values, offsets, frozen, anchor, unpack(theta), geometry, weights0, knots, masses)

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    rng = np.random.default_rng(SEED + 2)
    starts = [np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6), 3.0 * rng.normal(size=6)]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective, np.clip(start, -500.0, 500.0), method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 6,
            options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-9, "eps": 1e-5, "maxls": 80},
        )
        print(f"M3 start {index + 1}/5: success={fit.success} objective={float(fit.fun):.12g}", flush=True)
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all Model-3 starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x)
    betas = unpack(theta)
    dynamic = run(theta)

    hessian = eigenvalues = derivative_step = None
    attempts = []
    for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        candidate = numerical_hessian(objective, theta, step)
        candidate = (candidate + candidate.T) / 2.0
        candidate_eigenvalues = np.linalg.eigvalsh(candidate)
        attempts.append({"step": step, "eigenvalues": candidate_eigenvalues.tolist()})
        if candidate_eigenvalues[0] > 1e-10:
            hessian, eigenvalues, derivative_step = candidate, candidate_eigenvalues, step
            break
    if hessian is None:
        raise RuntimeError(f"nonpositive M3 Hessian: {attempts}")
    gradients = np.empty((offsets.size - 1, 6))
    for coordinate in range(6):
        direction = np.zeros(6)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).total_contributions - run(theta - direction).total_contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units.ravel())
    covariance = scaling @ covariance_theta @ scaling
    covariance = (covariance + covariance.T) / 2.0
    beta = betas.ravel()
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    parameters = []
    for index, (horizon, state) in enumerate(
        (horizon, state) for horizon in core.HORIZONS for state in ("eta", "lambda")
    ):
        parameters.append({
            "state": state, "horizon": horizon, "window": int(core.WINDOWS[index // 2]),
            "estimate": float(beta[index]), "hac22_se_conditional_on_model2": float(standard_errors[index]),
            "t": float(t_values[index]), "p_two_sided": float(p_values[index]),
        })
    tests = {
        "eta_three_coefficients": wald(beta, covariance, [0, 2, 4]),
        "lambda_three_coefficients": wald(beta, covariance, [1, 3, 5]),
        "eta_lambda_six_coefficients": wald(beta, covariance, list(range(6))),
    }
    return betas, {
        "betas": {
            horizon: {"eta": float(betas[i, 0]), "lambda": float(betas[i, 1])}
            for i, horizon in enumerate(core.HORIZONS)
        },
        "parameter_units": units.tolist(),
        "parameters": parameters,
        "wald_tests": tests,
        "model2_log_likelihood": nested.composite_log_likelihood,
        "dynamic_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_model2": dynamic.composite_log_likelihood - nested.composite_log_likelihood,
        "successful_starts": len(successful), "total_starts": len(starts),
        "largest_successful_objective_gap": float(max(fit.fun for fit in successful) - optimum.fun),
        "hessian_derivative_step": derivative_step,
        "hessian_eigenvalues_scaled": eigenvalues.tolist(),
        "maximum_absolute_mean_day_gradient": float(np.max(np.abs(gradients.mean(axis=0)))),
        "inference": "conditional Godambe-HAC(22), static anchor/q0 and Model-2 path fixed",
    }, nested, dynamic


def continue_modular(
    panel: core.RawPanel, n_train: int, anchor: np.ndarray, betas: np.ndarray,
    weights0: np.ndarray, knots: np.ndarray, masses: np.ndarray,
    training: ModularReferenceFilter,
) -> ModularReferenceFilter:
    training_rows = int(panel.offsets[n_train])
    return filter_modular_reference(
        panel.values[training_rows:], panel.offsets[n_train:] - training_rows,
        anchor, betas, weights0, knots, masses,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )


def continue_shape(
    panel: core.RawPanel, n_train: int, frozen_oos: ModularReferenceFilter,
    anchor: np.ndarray, betas: np.ndarray, geometry: ShapeGeometry,
    weights0: np.ndarray, knots: np.ndarray, masses: np.ndarray,
    training: ShapeLayerFilter,
) -> ShapeLayerFilter:
    training_rows = int(panel.offsets[n_train])
    return filter_shape_layer(
        panel.values[training_rows:], panel.offsets[n_train:] - training_rows,
        frozen_oos, anchor, betas, geometry, weights0, knots, masses,
        initial_shape_state=training.terminal_shape_state,
        initial_score_history=training.terminal_score_history,
    )


def continue_arc(
    phi_oos: np.ndarray,
    offsets_oos: np.ndarray,
    reference_scores_oos: np.ndarray,
    coefficients: np.ndarray,
    weights0: np.ndarray,
    q_matrix: np.ndarray,
    transform,
    training: split.BlockFilter,
) -> split.BlockFilter:
    return split.filter_block_arc_sm(
        phi_oos, offsets_oos, reference_scores_oos, coefficients,
        weights0, q_matrix, transform,
        initial_state=training.terminal_state,
        initial_component_history=training.terminal_component_history,
    )


def score_all_models(
    panel: core.RawPanel,
    dates: pd.DatetimeIndex,
    n_train: int,
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    model2_oos: ModularReferenceFilter,
    model3_oos: ShapeLayerFilter,
    model4_oos: split.BlockFilter,
) -> pd.DataFrame:
    training_rows = int(panel.offsets[n_train])
    values = panel.values[training_rows:]
    offsets = panel.offsets[n_train:] - training_rows
    oos_dates = dates[n_train:]
    static_reference = core.filter_reference(values, offsets, anchor, np.zeros((3, 4)))
    static_u = core.dynamic_reference_pit(values, offsets, static_reference.states)
    static_phi = core.normalized_bspline_basis(static_u, knots, core.DEGREE, masses)
    static_correction = five.static_correction(static_phi, offsets, weights0)
    m1 = q0.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], static_reference,
        static_correction, knots, masses, weights0, MODEL_NAMES["m1"], "_discard_m1",
    )
    m2_m4 = q0.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], model2_oos.as_reference(),
        model4_oos, knots, masses, weights0, MODEL_NAMES["m2"], MODEL_NAMES["m4"],
    )
    m3_u = core.dynamic_reference_pit(values, offsets, model3_oos.states)
    m3_phi = core.normalized_bspline_basis(m3_u, knots, core.DEGREE, masses)
    m3_correction = five.static_correction(m3_phi, offsets, weights0)
    m3 = q0.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], model3_oos.as_reference(),
        m3_correction, knots, masses, weights0, MODEL_NAMES["m3"], "_discard_m3",
    )
    return pd.concat((
        m1.loc[m1["model"] == MODEL_NAMES["m1"]],
        m2_m4,
        m3.loc[m3["model"] == MODEL_NAMES["m3"]],
    ), ignore_index=True)


def run_dataset(dataset: str) -> dict[str, object]:
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    if dataset in fx.FX:
        panel, dataset_info = fx.load_fx(dataset)
    elif dataset in sectors.SECTORS:
        panel, dataset_info = sectors.load_sector(dataset)
    else:
        panel, dataset_info = five.load_dataset(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    training_end = core.TRAINING_END if dataset in ("sp500", "n225") else stocks.TRAINING_END
    n_train = int(np.sum(dates <= training_end))
    training_rows = int(panel.offsets[n_train])
    train_values = panel.values[:training_rows]
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)

    if dataset in fx.FX:
        direct_root = fx.DIRECT_OUTPUT_ROOT
    elif dataset in sectors.SECTORS:
        direct_root = sectors.DIRECT_OUTPUT_ROOT
    else:
        direct_root = direct.OUTPUT_ROOT
    direct_saved = direct_root / dataset / "estimation.json"
    if direct_saved.exists():
        direct_result = json.loads(direct_saved.read_text(encoding="utf-8"))
        anchor = np.asarray(direct_result["static_reference"]["state"], dtype=float)
        static_summary = direct_result["static_reference"]
        initial_betas = np.asarray(direct_result["dynamic_location_scale_betas_matrix"], dtype=float)
        static_source = str(direct_saved)
    else:
        anchor, static_summary = core.fit_static_reference(train_values, train_offsets)
        initial_betas = None
        static_source = "fitted in this run"
    print(f"{dataset}: fitting static q0 around skew-t", flush=True)
    weights0, q0_summary = q0.fit_static_q0(
        train_values, train_offsets, anchor, knots, masses, q_matrix
    )
    print(f"{dataset}: fitting Model 2", flush=True)
    model2_betas, model2_summary, model2_train = fit_location_scale(
        train_values, train_offsets, anchor, weights0, knots, masses, initial_betas
    )
    model2_oos = continue_modular(
        panel, n_train, anchor, model2_betas, weights0, knots, masses, model2_train
    )

    shape_geometry = weighted_shape_geometry(anchor, weights0, knots, masses)
    print(f"{dataset}: fitting Model 3 eta/lambda", flush=True)
    model3_betas, model3_summary, model3_nested, model3_train = fit_shape_layer(
        train_values, train_offsets, model2_train, anchor, shape_geometry, weights0, knots, masses
    )
    model3_oos = continue_shape(
        panel, n_train, model2_oos, anchor, model3_betas, shape_geometry,
        weights0, knots, masses, model3_train,
    )

    spline_geometry = five.sequential_geometry(anchor, weights0, knots, masses, q_matrix)
    full_u = core.dynamic_reference_pit(panel.values, panel.offsets, np.vstack((model2_train.states, model2_oos.states)))
    full_phi = core.normalized_bspline_basis(full_u, knots, core.DEGREE, masses)
    transform = five.eta_direction_transform(spline_geometry)
    print(f"{dataset}: fitting Model 4 q_eta/others", flush=True)
    model4_summary, model4_nested, model4_train = split.fit_block_arc_sm(
        full_phi[:training_rows], train_offsets, model2_train.scores,
        weights0, q_matrix, transform, ("q_eta", "others"),
    )
    model4_coefficients = np.asarray(model4_summary["coefficients"])
    model4_oos = continue_arc(
        full_phi[training_rows:], oos_offsets, model2_oos.scores, model4_coefficients,
        weights0, q_matrix, transform, model4_train,
    )

    scores = score_all_models(
        panel, dates, n_train, anchor, weights0, knots, masses,
        model2_oos, model3_oos, model4_oos,
    )
    summary = scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean()
    comparisons = pd.DataFrame([
        core.score_comparison(scores, MODEL_NAMES[model], MODEL_NAMES["m2"], score)
        for model in ("m1", "m3", "m4")
        for score in q0.SCORE_NAMES
    ])

    model2_nesting = float(max(
        np.max(np.abs(model3_nested.states - model2_train.states)),
        np.max(np.abs(model3_nested.total_contributions - model2_train.total_contributions)),
    ))
    model4_nesting = float(max(
        np.max(np.abs(model4_nested.states)),
        np.max(np.abs(model4_nested.weights - weights0[None, :])),
    ))
    if max(model2_nesting, model4_nesting) > 2e-12:
        raise RuntimeError(f"nesting check failed: {model2_nesting}, {model4_nesting}")

    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "four_model_oos_summary.csv", index=False)
    comparisons.to_csv(output / "oos_comparisons_vs_model2.csv", index=False)
    pd.DataFrame(model3_summary["parameters"]).to_csv(output / "model3_eta_lambda_parameters.csv", index=False)
    pd.DataFrame(model4_summary["parameters"]).to_csv(output / "model4_qeta_others_parameters.csv", index=False)
    result = {
        "analysis": "fitted-static-q0 skew-t four-model comparison",
        "dataset": dataset_info,
        "sample": {
            "start": str(dates[0].date()), "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()), "end": str(dates[-1].date()),
            "training_days": n_train, "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows, "total_observations": int(panel.values.size),
        },
        "specification": {
            "models": {
                "M1": "static skew-t plus fitted static q0",
                "M2": "M1 plus dynamic m/log sigma2; eta/lambda/q0 fixed",
                "M3": "M2 plus joint dynamic eta/lambda, shape score orthogonal to m/log sigma2",
                "M4": "M2 plus dynamic q_eta/others after m/log sigma2 projection; lambda remains in others",
            },
            "reference_score": "derivative of log f_skewt only; fixed h0 PIT derivative excluded",
            "projection_measure": "h0(u)du", "score_whitening": False,
            "parity_split": False, "windows": list(core.WINDOWS),
            "estimation": "M2 fixed for sibling branches M3 and M4",
        },
        "static_reference_source": static_source,
        "static_skewt": static_summary,
        "static_q0": q0_summary,
        "model2": model2_summary,
        "model3": model3_summary,
        "model4": model4_summary,
        "shape_geometry": shape_geometry.diagnostics,
        "spline_geometry": spline_geometry.diagnostics,
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons_vs_model2": comparisons.to_dict(orient="records"),
        "checks": {"model3_zero_nesting_error": model2_nesting, "model4_zero_nesting_error": model4_nesting},
    }
    (output / "estimation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"{dataset}: M3 eta p={model3_summary['wald_tests']['eta_three_coefficients']['p']:.6g}, "
        f"lambda p={model3_summary['wald_tests']['lambda_three_coefficients']['p']:.6g}; "
        f"M4 q_eta p={model4_summary['joint_blocks']['q_eta']['p']:.6g}, "
        f"others p={model4_summary['joint_blocks']['others']['p']:.6g}", flush=True,
    )
    return result


def write_combined(results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"]["label"],
            "m3_eta_p": result["model3"]["wald_tests"]["eta_three_coefficients"]["p"],
            "m3_lambda_p": result["model3"]["wald_tests"]["lambda_three_coefficients"]["p"],
            "m3_joint_p": result["model3"]["wald_tests"]["eta_lambda_six_coefficients"]["p"],
            "m4_q_eta_p": result["model4"]["joint_blocks"]["q_eta"]["p"],
            "m4_others_p": result["model4"]["joint_blocks"]["others"]["p"],
            "m4_joint_p": result["model4"]["joint_all"]["p"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "combined_tests.csv", index=False)
    lines = [
        "# Fitted-static-q0 skew-t four-model results", "",
        "All models share the same static Hansen skew-t and fitted static spline q0.",
        "M3 and M4 are sibling branches from the frozen dynamic-location/scale Model 2.",
        "Reference scores exclude the fixed-spline PIT derivative; no whitening or parity is used.",
        "", "| dataset | M3 eta p | M3 lambda p | M3 joint p | M4 q_eta p | M4 others p | M4 joint p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['m3_eta_p']:.6g} | {row['m3_lambda_p']:.6g} | "
            f"{row['m3_joint_p']:.6g} | {row['m4_q_eta_p']:.6g} | "
            f"{row['m4_others_p']:.6g} | {row['m4_joint_p']:.6g} |"
        )
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(
            *DATASETS, *sectors.SECTORS, *fx.FX,
            "markets", "stocks", "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    if args.dataset == "markets":
        datasets = ("sp500", "n225")
    elif args.dataset == "stocks":
        datasets = tuple(stocks.ASSETS)
    elif args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = (args.dataset,)
    results = [run_dataset(dataset) for dataset in datasets]
    if tuple(datasets) == tuple(DATASETS):
        write_combined(results)


if __name__ == "__main__":
    main()
