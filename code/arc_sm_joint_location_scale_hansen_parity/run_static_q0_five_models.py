"""Sequential five-model symmetric-t DCD-SM experiment around fitted static q0.

The fitted static spline density is part of every model.  Reference dynamics
remain modular: update scores are derivatives of log f_theta only, deliberately
excluding the derivative of log h_0(F_theta).  All Fisher projections are
evaluated once at the fitted static anchor under the fitted density
f_theta0(y) h_0(F_theta0(y)).  There is no Fisher whitening and no parity split.

M1: static m, sigma, eta and static q0.
M2: dynamic m,sigma; fixed eta and static q0 (loaded/reproduced).
M3: M2 path plus dynamic eta, conditionally estimated with an eta score
    orthogonal to the m,sigma reference scores.
M4: M2 path plus DCD-SM, split into eta and eta-orthogonal directions.
M5: M3 path plus DCD-SM, using one q-score vector orthogonal to m,sigma,eta.
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

import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_single_stocks as stocks
import run_static_q0_symmetric_t_four_models as q0
import run_symmetric_t_four_models as symmetric
from estimate_score_har import fixed_denominator_score_averages, hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_q0_five_models"
DATASETS = ("sp500", "n225", *stocks.ASSETS)
MODEL_NAMES = {
    "m1": "m1_static_symmetric_t_static_sm",
    "m2": "m2_dynamic_location_scale_fixed_eta_static_sm",
    "m3": "m3_dynamic_location_scale_eta_static_sm",
    "m4": "m4_dynamic_location_scale_fixed_eta_eta_noneta_arc_sm",
    "m5": "m5_dynamic_location_scale_eta_residual_arc_sm",
}
LABELS = {
    MODEL_NAMES["m1"]: "M1: static t + static SM",
    MODEL_NAMES["m2"]: "M2: dynamic m,sigma; fixed eta + static SM",
    MODEL_NAMES["m3"]: "M3: M2 + dynamic eta",
    MODEL_NAMES["m4"]: "M4: M2 + eta/others DCD-SM",
    MODEL_NAMES["m5"]: "M5: M3 + residual DCD-SM",
}
SEED = 20260910


@dataclass(frozen=True)
class SequentialGeometry:
    projection_ms: np.ndarray
    eta_on_ms: np.ndarray
    eta_loading: np.ndarray
    projection_mseta: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class EtaLayerFilter:
    states: np.ndarray
    scores: np.ndarray
    eta_update_scores: np.ndarray
    base_contributions: np.ndarray
    static_sm_contributions: np.ndarray
    total_contributions: np.ndarray
    terminal_eta_state: float
    terminal_eta_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.total_contributions))

    def as_reference(self) -> core.FourStateFilter:
        eta = np.array([core.state_to_parameters(state[2:])[0] for state in self.states])
        skew = np.zeros(self.states.shape[0])
        return core.FourStateFilter(
            states=self.states,
            means=self.states[:, 0],
            log_variances=self.states[:, 1],
            variances=np.exp(self.states[:, 1]),
            eta=eta,
            skew=skew,
            scores=self.scores,
            contributions=self.base_contributions,
            terminal_state=self.states[-1].copy(),
            terminal_score_history=np.empty((0, 4)),
        )


@dataclass(frozen=True)
class StaticCorrection:
    weights: np.ndarray
    contributions: np.ndarray


def saved_output(dataset: str) -> Path:
    if dataset in q0.OUTPUTS:
        return q0.OUTPUTS[dataset]
    return stocks.OUTPUT_ROOT / dataset


def load_dataset(dataset: str) -> tuple[core.RawPanel, dict[str, object]]:
    if dataset in ("sp500", "n225"):
        return symmetric.load_market(dataset), {"dataset": dataset, "label": dataset.upper()}
    return stocks.load_asset(dataset)


def stored_betas(saved: dict[str, object], section: str, coordinates: tuple[int, ...]) -> np.ndarray:
    betas = np.zeros((3, 4))
    source = saved[section]["betas"]
    for horizon_index, horizon in enumerate(core.HORIZONS):
        for coordinate in coordinates:
            name = q0.STATE_NAMES[coordinate]
            if name in source[horizon]:
                betas[horizon_index, coordinate] = source[horizon][name]
    return betas


def sequential_geometry(
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    q_matrix: np.ndarray,
    quadrature_order: int = 128,
) -> SequentialGeometry:
    """Fixed-anchor decomposition under h0(u)du using reference-only tangents."""
    nodes, legendre_weights = np.polynomial.legendre.leggauss(quadrature_order)
    q_parts, tangent_parts, measure_parts = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes / 2.0
        du_weight = legendre_weights * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        h0 = phi @ weights0
        responsibility = phi * weights0[None, :] / h0[:, None]
        q_parts.append((responsibility - weights0[None, :]) @ q_matrix)
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_parts.append(core.reference_logpdf_and_score(y, anchor)[1][:, :3])
        measure_parts.append(du_weight * h0)

    q_score = np.vstack(q_parts)
    tangent = np.vstack(tangent_parts)
    measure = np.concatenate(measure_parts)
    ms = tangent[:, :2]
    eta = tangent[:, 2]
    information_ms = (ms.T * measure) @ ms
    projection_ms = ((q_score.T * measure) @ ms) @ np.linalg.inv(information_ms)
    q_ms = q_score - ms @ projection_ms.T
    eta_on_ms = np.linalg.solve(information_ms, (ms.T * measure) @ eta)
    eta_residual = eta - ms @ eta_on_ms
    information_eta_given_ms = float(np.sum(measure * eta_residual**2))
    eta_loading = ((q_ms.T * measure) @ eta_residual) / information_eta_given_ms
    eta_component = eta_residual[:, None] * eta_loading[None, :]
    noneta = q_ms - eta_component
    projection_mseta = np.column_stack(
        (projection_ms - np.outer(eta_loading, eta_on_ms), eta_loading)
    )
    direct_noneta = q_score - tangent @ projection_mseta.T
    noneta_information = (noneta.T * measure) @ noneta
    noneta_eigenvalues = np.linalg.eigvalsh(
        (noneta_information + noneta_information.T) / 2.0
    )
    if information_eta_given_ms <= 1e-12 or noneta_eigenvalues[0] <= 1e-10:
        raise RuntimeError("nonpositive sequential score geometry")
    return SequentialGeometry(
        projection_ms=projection_ms,
        eta_on_ms=eta_on_ms,
        eta_loading=eta_loading,
        projection_mseta=projection_mseta,
        diagnostics={
            "anchor_measure": "f_eta0(y) h0(F_eta0(y)) dy = h0(u) du",
            "reference_score": "derivative of log f_theta(y) only; derivative of log h0(F_theta(y)) excluded",
            "projection": "fixed-anchor Fisher projection: first (m,log sigma2), then eta",
            "score_whitening": False,
            "parity_split": False,
            "quadrature_mass_error": float(abs(np.sum(measure) - 1.0)),
            "eta_information_given_location_scale": information_eta_given_ms,
            "eta_component_rank": int(
                np.linalg.matrix_rank((eta_component.T * measure) @ eta_component)
            ),
            "eta_loading_norm": float(np.linalg.norm(eta_loading)),
            "maximum_post_location_scale_cross_information": float(
                np.max(np.abs((q_ms.T * measure) @ ms))
            ),
            "maximum_post_eta_cross_information": float(
                np.max(np.abs((noneta.T * measure) @ eta_residual))
            ),
            "sequential_vs_joint_projection_max_abs_error": float(
                np.max(np.abs(noneta - direct_noneta))
            ),
            "noneta_information_eigenvalues": noneta_eigenvalues.tolist(),
            "noneta_information_condition_number": float(
                noneta_eigenvalues[-1] / noneta_eigenvalues[0]
            ),
        },
    )


def filter_eta_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen_reference: core.FourStateFilter,
    anchor: np.ndarray,
    coefficients: np.ndarray,
    eta_on_ms: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    *,
    initial_eta_state: float | None = None,
    initial_eta_score_history: np.ndarray | None = None,
) -> EtaLayerFilter:
    """Update eta on a frozen M2 m,sigma path with a reference-only score."""
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    coefficients = np.asarray(coefficients, dtype=float)
    days = offsets.size - 1
    if coefficients.shape != (3,) or frozen_reference.states.shape[0] != days:
        raise ValueError("eta coefficients or frozen reference path do not align")
    eta_state = float(anchor[2] if initial_eta_state is None else initial_eta_state)
    history = (
        np.empty(0)
        if initial_eta_score_history is None
        else np.asarray(initial_eta_score_history, dtype=float).copy()[-21:]
    )
    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    eta_update_scores = np.empty(days)
    base_contributions = np.empty(days)
    static_sm_contributions = np.empty(days)
    total_contributions = np.empty(days)
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = frozen_reference.states[day].copy()
        state[2] = eta_state
        state[3] = 0.0
        if not np.isfinite(eta_state) or abs(eta_state) > 20.0:
            raise FloatingPointError("unstable eta layer")
        observations = values[start:stop]
        base_log, observation_scores = core.reference_logpdf_and_score(observations, state)
        daily_score = np.mean(observation_scores, axis=0)
        z = (observations - state[0]) / np.exp(0.5 * state[1])
        u = np.clip(core.hansen_cdf(z, state[2:]), 1e-12, 1.0 - 1e-12)
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        log_h0 = np.log(np.maximum(phi @ weights0, 1e-300))
        efficient_eta = float(daily_score[2] - eta_on_ms @ daily_score[:2])
        states[day] = state
        scores[day] = daily_score
        eta_update_scores[day] = efficient_eta
        base_contributions[day] = float(np.mean(base_log))
        static_sm_contributions[day] = float(np.mean(log_h0))
        total_contributions[day] = base_contributions[day] + static_sm_contributions[day]
        history = np.r_[history, efficient_eta][-21:]
        eta_state = float(anchor[2])
        for coefficient, window in zip(coefficients, core.WINDOWS):
            eta_state += coefficient * float(np.sum(history[-window:]) / window)
    return EtaLayerFilter(
        states=states,
        scores=scores,
        eta_update_scores=eta_update_scores,
        base_contributions=base_contributions,
        static_sm_contributions=static_sm_contributions,
        total_contributions=total_contributions,
        terminal_eta_state=eta_state,
        terminal_eta_score_history=history,
    )


def fit_eta_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen_reference: core.FourStateFilter,
    anchor: np.ndarray,
    eta_on_ms: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    initial_coefficients: np.ndarray | None = None,
) -> tuple[dict[str, object], EtaLayerFilter, EtaLayerFilter]:
    static = filter_eta_layer(
        values, offsets, frozen_reference, anchor, np.zeros(3), eta_on_ms,
        weights0, knots, masses,
    )
    averages = fixed_denominator_score_averages(
        static.eta_update_scores[:, None], core.WINDOWS
    )
    units = np.array([
        1e-2 / max(float(np.median(np.abs(average[:, 0]))), 1e-12)
        for average in averages
    ])

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float) * units

    def run(theta: np.ndarray) -> EtaLayerFilter:
        return filter_eta_layer(
            values, offsets, frozen_reference, anchor, unpack(theta), eta_on_ms,
            weights0, knots, masses,
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    rng = np.random.default_rng(SEED)
    starts = [np.zeros(3), np.ones(3), -np.ones(3), rng.normal(size=3)]
    if initial_coefficients is not None:
        starts.insert(0, np.asarray(initial_coefficients, dtype=float) / units)
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 3,
            options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9, "eps": 1e-5, "maxls": 60},
        )
        print(
            f"eta layer start {index + 1}/{len(starts)}: success={fit.success} "
            f"objective={float(fit.fun):.12g}", flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all eta-layer starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    coefficients = unpack(theta)
    dynamic = run(theta)

    hessian = None
    eigenvalues = None
    derivative_step = None
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
        raise RuntimeError(f"nonpositive eta-layer Hessian: {attempts}")
    gradients = np.empty((offsets.size - 1, 3))
    for coordinate in range(3):
        direction = np.zeros(3)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).total_contributions
            - run(theta - direction).total_contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units)
    covariance = scaling @ covariance_theta @ scaling
    covariance = (covariance + covariance.T) / 2.0
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = coefficients / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    wald = float(coefficients @ np.linalg.solve(covariance, coefficients))
    parameters = [
        {
            "horizon": int(window),
            "estimate": float(coefficients[index]),
            "hac22_se_conditional_on_model2": float(standard_errors[index]),
            "t": float(t_values[index]),
            "p_two_sided": float(p_values[index]),
        }
        for index, window in enumerate(core.WINDOWS)
    ]
    summary = {
        "coefficients": coefficients.tolist(),
        "parameter_units": units.tolist(),
        "parameters": parameters,
        "joint_three_coefficient_wald": wald,
        "joint_three_coefficient_p": float(chi2.sf(wald, 3)),
        "model2_composite_log_likelihood_with_static_sm": static.composite_log_likelihood,
        "model3_composite_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_model2": dynamic.composite_log_likelihood - static.composite_log_likelihood,
        "optimizer_success": bool(optimum.success),
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(max(fit.fun for fit in successful) - optimum.fun),
        "hessian_derivative_step": derivative_step,
        "hessian_eigenvalues_scaled": eigenvalues.tolist(),
        "maximum_absolute_mean_day_gradient": float(np.max(np.abs(gradients.mean(axis=0)))),
        "inference_scope": "conditional Godambe-HAC(22), holding static anchor, static SM, and Model-2 m/sigma path fixed",
    }
    return summary, static, dynamic


def eta_direction_transform(geometry: SequentialGeometry):
    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        q_ms = raw_q_score - geometry.projection_ms @ reference_score[:2]
        eta_residual = float(reference_score[2] - geometry.eta_on_ms @ reference_score[:2])
        eta_component = geometry.eta_loading * eta_residual
        return np.vstack((eta_component, q_ms - eta_component))
    return transform


def residual_vector_transform(geometry: SequentialGeometry):
    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        return (raw_q_score - geometry.projection_mseta @ reference_score[:3])[None, :]
    return transform


def fit_and_continue_arc(
    phi: np.ndarray,
    panel: core.RawPanel,
    n_train: int,
    reference_train: core.FourStateFilter,
    reference_oos: core.FourStateFilter,
    reference_full: core.FourStateFilter,
    weights0: np.ndarray,
    q_matrix: np.ndarray,
    transform,
    block_names: tuple[str, ...],
) -> tuple[dict[str, object], split.BlockFilter, dict[str, float]]:
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    fit, static, training = split.fit_block_arc_sm(
        phi[:training_rows], train_offsets, reference_train.scores,
        weights0, q_matrix, transform, block_names,
    )
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    oos = split.filter_block_arc_sm(
        phi[training_rows:], oos_offsets, reference_oos.scores, coefficients,
        weights0, q_matrix, transform,
        initial_state=training.terminal_state,
        initial_component_history=training.terminal_component_history,
    )
    full = split.filter_block_arc_sm(
        phi, panel.offsets, reference_full.scores, coefficients,
        weights0, q_matrix, transform,
    )
    continuation_error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    static_log = np.empty(train_offsets.size - 1)
    for day, (start, stop) in enumerate(zip(train_offsets[:-1], train_offsets[1:])):
        static_log[day] = float(np.mean(np.log(np.maximum(phi[start:stop] @ weights0, 1e-300))))
    nesting_error = float(max(
        np.max(np.abs(static.states)),
        np.max(np.abs(static.weights - weights0[None, :])),
        np.max(np.abs(static.contributions - static_log)),
    ))
    if max(continuation_error, nesting_error) > 2e-12:
        raise RuntimeError("static-q0 DCD-SM continuation or nesting failed")
    return fit, oos, {
        "continuation_error": continuation_error,
        "q_zero_nesting_error": nesting_error,
    }


def continue_fixed_reference(
    panel: core.RawPanel, n_train: int, anchor: np.ndarray, betas: np.ndarray
) -> tuple[core.FourStateFilter, core.FourStateFilter, core.FourStateFilter, float]:
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    training = core.filter_reference(panel.values[:training_rows], train_offsets, anchor, betas)
    oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, betas,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full = core.filter_reference(panel.values, panel.offsets, anchor, betas)
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"Model-2 reference continuation failed: {error}")
    return training, oos, full, error


def continue_eta_layer(
    panel: core.RawPanel,
    n_train: int,
    fixed_oos: core.FourStateFilter,
    fixed_full: core.FourStateFilter,
    anchor: np.ndarray,
    coefficients: np.ndarray,
    eta_on_ms: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    training: EtaLayerFilter,
) -> tuple[EtaLayerFilter, EtaLayerFilter, float]:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = filter_eta_layer(
        panel.values[training_rows:], oos_offsets, fixed_oos, anchor, coefficients,
        eta_on_ms, weights0, knots, masses,
        initial_eta_state=training.terminal_eta_state,
        initial_eta_score_history=training.terminal_eta_score_history,
    )
    full = filter_eta_layer(
        panel.values, panel.offsets, fixed_full, anchor, coefficients,
        eta_on_ms, weights0, knots, masses,
    )
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.total_contributions[n_train:] - oos.total_contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"Model-3 continuation failed: {error}")
    return oos, full, error


def static_correction(phi: np.ndarray, offsets: np.ndarray, weights0: np.ndarray) -> StaticCorrection:
    contributions = np.array([
        float(np.mean(np.log(np.maximum(phi[start:stop] @ weights0, 1e-300))))
        for start, stop in zip(offsets[:-1], offsets[1:])
    ])
    return StaticCorrection(
        weights=np.repeat(weights0[None, :], offsets.size - 1, axis=0),
        contributions=contributions,
    )


def score_pair(
    panel: core.RawPanel,
    n_train: int,
    dates: pd.DatetimeIndex,
    reference: core.FourStateFilter,
    correction,
    knots: np.ndarray,
    masses: np.ndarray,
    weights0: np.ndarray,
    reference_name: str,
    arc_name: str,
) -> pd.DataFrame:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    return q0.score_oos_pair(
        panel.values[training_rows:], oos_offsets, dates[n_train:],
        panel.values[:training_rows], reference, correction, knots, masses,
        weights0, reference_name, arc_name,
    )


def run_dataset(dataset: str) -> None:
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    source_output = saved_output(dataset)
    saved = json.loads((source_output / "estimation.json").read_text(encoding="utf-8"))
    panel, dataset_info = load_dataset(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    training_end = core.TRAINING_END if dataset in ("sp500", "n225") else stocks.TRAINING_END
    n_train = int(np.sum(dates <= training_end))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    anchor = np.asarray(saved["static_symmetric_t"]["state"], dtype=float)
    weights0 = np.asarray(saved["static_q0"]["weights"], dtype=float)
    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)
    geometry = sequential_geometry(anchor, weights0, knots, masses, q_matrix)
    print(f"{dataset}: geometry checks {geometry.diagnostics}", flush=True)

    fixed_betas = stored_betas(saved, "fixed_eta_dynamic_location_scale", (0, 1))
    fixed_train, fixed_oos, fixed_full, fixed_continuation = continue_fixed_reference(
        panel, n_train, anchor, fixed_betas
    )
    stored_likelihood_error = float(abs(
        fixed_train.composite_log_likelihood
        - saved["fixed_eta_dynamic_location_scale"]["dynamic_composite_log_likelihood"]
    ))
    # Yahoo adjusted-price histories can be revised after a saved single-stock
    # run.  Preserve the exact discrepancy as an audit field, but only stop for
    # a material mismatch; cross-market repository panels should still match at
    # floating-point precision.
    if stored_likelihood_error > 1e-2:
        raise RuntimeError(
            f"stored Model-2 path materially differs on current input: {stored_likelihood_error}"
        )
    if stored_likelihood_error > 1e-9:
        print(
            f"{dataset}: warning: saved/current Model-2 likelihood difference "
            f"{stored_likelihood_error:.9g}", flush=True,
        )

    initial_eta = np.array([
        saved.get("dynamic_location_scale_eta", {}).get("betas", {})
        .get(horizon, {}).get("eta", 0.0)
        for horizon in core.HORIZONS
    ])
    print(f"{dataset}: fitting Model 3 (dynamic eta on frozen Model 2)", flush=True)
    model3_fit, model2_nested, model3_train = fit_eta_layer(
        panel.values[:training_rows], train_offsets, fixed_train, anchor,
        geometry.eta_on_ms, weights0, knots, masses, initial_eta,
    )
    model3_oos, model3_full, model3_continuation = continue_eta_layer(
        panel, n_train, fixed_oos, fixed_full, anchor,
        np.asarray(model3_fit["coefficients"]), geometry.eta_on_ms,
        weights0, knots, masses, model3_train,
    )
    model3_train_reference = model3_train.as_reference()
    model3_oos_reference = model3_oos.as_reference()
    model3_full_reference = model3_full.as_reference()

    fixed_u = core.dynamic_reference_pit(panel.values, panel.offsets, fixed_full.states)
    model3_u = core.dynamic_reference_pit(panel.values, panel.offsets, model3_full.states)
    fixed_phi = core.normalized_bspline_basis(fixed_u, knots, core.DEGREE, masses)
    model3_phi = core.normalized_bspline_basis(model3_u, knots, core.DEGREE, masses)

    print(f"{dataset}: reproducing Model 4 at fitted q0 (eta/others, no parity)", flush=True)
    model4_fit, model4_oos, model4_checks = fit_and_continue_arc(
        fixed_phi, panel, n_train, fixed_train, fixed_oos, fixed_full,
        weights0, q_matrix, eta_direction_transform(geometry),
        ("eta_direction", "noneta_direction"),
    )
    print(f"{dataset}: fitting Model 5 (fully orthogonal residual vector)", flush=True)
    model5_fit, model5_oos, model5_checks = fit_and_continue_arc(
        model3_phi, panel, n_train, model3_train_reference, model3_oos_reference,
        model3_full_reference, weights0, q_matrix,
        residual_vector_transform(geometry), ("residual_vector",),
    )

    constant_betas = np.zeros((3, 4))
    m1_oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, constant_betas
    )
    m1_u = core.dynamic_reference_pit(
        panel.values[training_rows:], oos_offsets, m1_oos.states
    )
    m1_phi = core.normalized_bspline_basis(m1_u, knots, core.DEGREE, masses)
    m1_correction = static_correction(m1_phi, oos_offsets, weights0)
    fixed_oos_phi = fixed_phi[training_rows:]
    model2_correction = static_correction(fixed_oos_phi, oos_offsets, weights0)

    m1_pair = score_pair(
        panel, n_train, dates, m1_oos, m1_correction, knots, masses, weights0,
        MODEL_NAMES["m1"], "_discard_m1_duplicate",
    )
    m2_m4_pair = score_pair(
        panel, n_train, dates, fixed_oos, model4_oos, knots, masses, weights0,
        MODEL_NAMES["m2"], MODEL_NAMES["m4"],
    )
    m3_m5_pair = score_pair(
        panel, n_train, dates, model3_oos_reference, model5_oos, knots, masses,
        weights0, MODEL_NAMES["m3"], MODEL_NAMES["m5"],
    )
    scores = pd.concat((
        m1_pair.loc[m1_pair["model"] == MODEL_NAMES["m1"]],
        m2_m4_pair,
        m3_m5_pair,
    ), ignore_index=True)
    if scores.groupby("date")["model"].nunique().min() != 5:
        raise RuntimeError("five-model OOS panel is incomplete")
    summary = scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean()
    pairs = (
        (MODEL_NAMES["m2"], MODEL_NAMES["m1"]),
        (MODEL_NAMES["m3"], MODEL_NAMES["m2"]),
        (MODEL_NAMES["m4"], MODEL_NAMES["m2"]),
        (MODEL_NAMES["m5"], MODEL_NAMES["m3"]),
        (MODEL_NAMES["m5"], MODEL_NAMES["m4"]),
    )
    comparisons = pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in q0.SCORE_NAMES
    ])
    model3_parameters = pd.DataFrame(model3_fit["parameters"])
    model5_parameters = pd.DataFrame(model5_fit["parameters"])
    model4_parameters = pd.DataFrame(model4_fit["parameters"])
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "five_model_oos_summary.csv", index=False)
    comparisons.to_csv(output / "five_model_oos_comparisons.csv", index=False)
    model3_parameters.to_csv(output / "model3_eta_parameters.csv", index=False)
    model4_parameters.to_csv(output / "model4_arc_sm_parameters.csv", index=False)
    model5_parameters.to_csv(output / "model5_arc_sm_parameters.csv", index=False)

    model2_static_log = np.array([
        float(np.mean(np.log(np.maximum(fixed_phi[start:stop] @ weights0, 1e-300))))
        for start, stop in zip(train_offsets[:-1], train_offsets[1:])
    ])
    model2_nesting_error = float(max(
        np.max(np.abs(model2_nested.states - fixed_train.states)),
        np.max(np.abs(model2_nested.static_sm_contributions - model2_static_log)),
        np.max(np.abs(
            model2_nested.total_contributions
            - fixed_train.contributions
            - model2_static_log
        )),
    ))
    result = {
        "analysis": f"{dataset} fitted-static-q0 sequential five-model comparison",
        "dataset": dataset_info,
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
            "static_sm_origin": "w(q)=softmax(log(w0)+Qq), so q=0 exactly reproduces fitted w0",
            "reference_score": "derivative of log f_theta(y) only; derivative of log h0(F_theta(y)) deliberately excluded",
            "projection_anchor": "static symmetric-t plus fitted static SM",
            "projection_measure": "h0(u) du",
            "score_whitening": False,
            "parity_split": False,
            "windows": list(core.WINDOWS),
            "estimation": "sequential/conditional: M2 fixed for M3 and M4; M3 fixed for M5",
        },
        "geometry": geometry.diagnostics,
        "model1": {
            "static_symmetric_t": saved["static_symmetric_t"],
            "static_q0": saved["static_q0"],
        },
        "model2": saved["fixed_eta_dynamic_location_scale"],
        "model3": model3_fit,
        "model4": model4_fit,
        "model5": model5_fit,
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": {
            "stored_model2_likelihood_reproduction_error": stored_likelihood_error,
            "model2_static_sm_nesting_error": model2_nesting_error,
            "model2_continuation_error": fixed_continuation,
            "model3_continuation_error": model3_continuation,
            "model4": model4_checks,
            "model5": model5_checks,
            "lambda_fixed_maximum_absolute_state": float(np.max(np.abs(model3_full.states[:, 3]))),
        },
    }
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output, result, summary, comparisons, model3_parameters, model4_parameters, model5_parameters)
    print(summary.to_string(index=False), flush=True)


def write_report(
    output: Path,
    result: dict[str, object],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    model3_parameters: pd.DataFrame,
    model4_parameters: pd.DataFrame,
    model5_parameters: pd.DataFrame,
) -> None:
    lines = [
        f"# {result['analysis']}",
        "",
        "Every model includes the same fitted static spline density. Model 3 uses",
        "only the symmetric-t reference score for eta: the derivative of the static",
        "spline term through the PIT is deliberately omitted. Fisher projections use",
        "the fitted static anchor, there is no whitening, and Models 4/5 use no parity split.",
        "",
        "## Strict-OOS score levels",
        "",
        "| model | log score (higher) | CRPS (lower) |",
        "|---|---:|---:|",
    ]
    for row in summary.sort_values("continuous_log_score", ascending=False).to_dict("records"):
        lines.append(
            f"| {LABELS[row['model']]} | {row['continuous_log_score']:.9f} | {row['crps']:.9f} |"
        )
    lines += [
        "",
        "## Model 3: eta update",
        "",
        "| horizon | estimate | HAC(22) SE | t | p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in model3_parameters.to_dict("records"):
        lines.append(
            f"| {int(row['horizon'])} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_model2']:.9g} | {row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines += [
        "",
        f"Joint three-coefficient Wald p: `{result['model3']['joint_three_coefficient_p']:.6g}`; "
        f"training gain over Model 2: `{result['model3']['gain_vs_model2']:.9f}`.",
        "",
        "## Model 4 verification: eta direction and others",
        "",
        "| block | horizon | estimate | HAC(22) p |",
        "|---|---:|---:|---:|",
    ]
    for row in model4_parameters.to_dict("records"):
        lines.append(
            f"| {row['block']} | {int(row['horizon'])} | {row['estimate']:.9g} | {row['p_two_sided']:.6g} |"
        )
    lines += [
        "",
        f"Joint all-six p: `{result['model4']['joint_all']['p']:.6g}`; "
        f"eta-direction p: `{result['model4']['joint_blocks']['eta_direction']['p']:.6g}`; "
        f"others p: `{result['model4']['joint_blocks']['noneta_direction']['p']:.6g}`.",
        "",
        "## Model 5: residual DCD-SM update",
        "",
        "| horizon | estimate | HAC(22) SE | t | p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in model5_parameters.to_dict("records"):
        lines.append(
            f"| {int(row['horizon'])} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_reference']:.9g} | {row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines += [
        "",
        f"Joint three-coefficient Wald p: `{result['model5']['joint_all']['p']:.6g}`; "
        f"training gain over Model 3: `{result['model5']['gain_vs_static']:.9f}`.",
        "",
        "## Strict-OOS comparisons",
        "",
        "Positive improvement favors the proposed model for every row.",
        "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ]
    headline = comparisons.loc[comparisons["score"].isin(("continuous_log_score", "crps"))]
    for row in headline.to_dict("records"):
        lines.append(
            f"| {LABELS[row['proposed']]} | {LABELS[row['benchmark']]} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    lines += [
        "",
        "All nine loss comparisons are stored in `five_model_oos_comparisons.csv`;",
        "daily strict-OOS observations are stored in `daily_oos_scores.csv`.",
        "",
        "## Numerical audits",
        "",
        f"- Model 2 stored-likelihood reproduction error: `{result['checks']['stored_model2_likelihood_reproduction_error']:.3g}`.",
        f"- Model 2 static-SM nesting error: `{result['checks']['model2_static_sm_nesting_error']:.3g}`.",
        f"- Model 3 continuation error: `{result['checks']['model3_continuation_error']:.3g}`.",
        f"- Model 4 q=0 nesting error: `{result['checks']['model4']['q_zero_nesting_error']:.3g}`.",
        f"- Model 5 q=0 nesting error: `{result['checks']['model5']['q_zero_nesting_error']:.3g}`.",
        f"- Maximum post-eta cross-information: `{result['geometry']['maximum_post_eta_cross_information']:.3g}`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "markets", "stocks", "all"), default="all")
    args = parser.parse_args()
    if args.dataset == "markets":
        datasets = ("sp500", "n225")
    elif args.dataset == "stocks":
        datasets = tuple(stocks.ASSETS)
    elif args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
