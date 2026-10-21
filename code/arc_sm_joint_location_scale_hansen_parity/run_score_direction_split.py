"""Split symmetric-t DCD-SM scores into eta and non-eta directions.

Two additional strict-OOS models are fitted for S&P 500 and N225:

1. A dynamic-location/scale, fixed-eta reference plus DCD-SM.  After removing
   the location and log-scale-squared tangents, the spline score is decomposed
   into its conditional eta projection and the remaining eta-orthogonal score.
   Each component has its own HAR(1,5,22) scalar response.
2. A dynamic-location/scale/eta reference plus DCD-SM.  After removing all
   three tangents, the residual spline score itself has one HAR(1,5,22)
   scalar response.

The construction is sequential Fisher projection at the fixed training
anchor.  Scores are deliberately not Fisher-whitened and are not parity split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2

import run_sp500 as core
import run_static_q0_single_stocks as stocks
import run_symmetric_t_four_models as previous
from estimate_score_har import fixed_denominator_score_averages, hac_meat, numerical_hessian
from run_componentwise_lag5_dcd import cluster_density_and_component_score


ROOT = Path(__file__).resolve().parent
OUTPUTS = {
    "sp500": ROOT / "output_score_direction_split_sp500",
    "n225": ROOT / "output_score_direction_split_n225",
    **{
        asset: ROOT / "output_score_direction_split_single_stocks" / asset
        for asset in stocks.ASSETS
    },
}
SAVED_REFERENCE = {
    "sp500": ROOT / "output_symmetric_t_sp500" / "estimation.json",
    "n225": ROOT / "output_symmetric_t_n225" / "estimation.json",
    **{
        asset: stocks.OUTPUT_ROOT / asset / "estimation.json"
        for asset in stocks.ASSETS
    },
}
SAVED_SCORES = {
    "sp500": ROOT / "output_symmetric_t_sp500" / "daily_oos_scores.csv",
    "n225": ROOT / "output_symmetric_t_n225" / "daily_oos_scores.csv",
    **{
        asset: stocks.OUTPUT_ROOT / asset / "uniform_q0_daily_oos_scores.csv"
        for asset in stocks.ASSETS
    },
}
MODEL_NAMES = {
    "fixed_split": "symmetric_t_fixed_eta_eta_noneta_split_arc_sm",
    "dynamic_vector": "symmetric_t_dynamic_eta_residual_vector_arc_sm",
}


def reference_model_names(dataset: str) -> dict[str, str]:
    if dataset in ("sp500", "n225"):
        return previous.MODEL_NAMES
    return {key: f"uniform_q0_{key}" for key in previous.MODEL_NAMES}


@dataclass(frozen=True)
class SplitGeometry:
    projection_ms: np.ndarray
    eta_on_ms: np.ndarray
    eta_loading: np.ndarray
    projection_mseta: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class BlockFilter:
    contributions: np.ndarray
    states: np.ndarray
    component_scores: np.ndarray
    weights: np.ndarray
    terminal_state: np.ndarray
    terminal_component_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(self.contributions.sum())


def score_direction_geometry(
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    quadrature_order: int = 128,
) -> SplitGeometry:
    """Build the sequential (m,sigma), then eta, score decomposition."""
    nodes0, weights0 = np.polynomial.legendre.leggauss(quadrature_order)
    q_parts, tangent_parts, weight_parts = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes0 / 2.0
        weight = weights0 * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        q_parts.append((phi * masses[None, :] - masses[None, :]) @ helmert)
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_parts.append(core.reference_logpdf_and_score(y, anchor)[1][:, :3])
        weight_parts.append(weight)

    q_score = np.vstack(q_parts)
    tangent = np.vstack(tangent_parts)
    weight = np.concatenate(weight_parts)
    ms = tangent[:, :2]
    eta = tangent[:, 2]

    information_ms = (ms.T * weight) @ ms
    cross_q_ms = (q_score.T * weight) @ ms
    projection_ms = cross_q_ms @ np.linalg.inv(information_ms)
    q_ms = q_score - ms @ projection_ms.T

    eta_on_ms = np.linalg.solve(information_ms, (ms.T * weight) @ eta)
    eta_residual = eta - ms @ eta_on_ms
    information_eta_given_ms = float(np.sum(weight * eta_residual**2))
    eta_loading = ((q_ms.T * weight) @ eta_residual) / information_eta_given_ms
    eta_component = eta_residual[:, None] * eta_loading[None, :]
    noneta = q_ms - eta_component

    projection_mseta = np.column_stack(
        (
            projection_ms - np.outer(eta_loading, eta_on_ms),
            eta_loading,
        )
    )
    direct_noneta = q_score - tangent @ projection_mseta.T
    decomposition_error = float(np.max(np.abs(noneta - direct_noneta)))
    post_ms_cross = (q_ms.T * weight) @ ms
    post_eta_cross = (noneta.T * weight) @ eta_residual
    eta_rank = int(np.linalg.matrix_rank((eta_component.T * weight) @ eta_component))
    noneta_information = (noneta.T * weight) @ noneta
    eigenvalues = np.linalg.eigvalsh((noneta_information + noneta_information.T) / 2.0)
    if information_eta_given_ms <= 1e-12 or eigenvalues[0] <= 1e-10:
        raise RuntimeError("nonpositive score-direction residual information")

    return SplitGeometry(
        projection_ms=projection_ms,
        eta_on_ms=eta_on_ms,
        eta_loading=eta_loading,
        projection_mseta=projection_mseta,
        diagnostics={
            "projection": "sequential fixed-anchor Fisher projection: (m,sigma2), then eta",
            "score_whitening": False,
            "eta_component_rank": eta_rank,
            "eta_information_given_location_scale": information_eta_given_ms,
            "eta_loading_norm": float(np.linalg.norm(eta_loading)),
            "noneta_information_eigenvalues": eigenvalues.tolist(),
            "noneta_information_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "maximum_post_location_scale_cross_information": float(np.max(np.abs(post_ms_cross))),
            "maximum_post_eta_cross_information": float(np.max(np.abs(post_eta_cross))),
            "sequential_vs_joint_projection_max_abs_error": decomposition_error,
        },
    )


def fixed_split_transform(
    geometry: SplitGeometry,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        ms_score = raw_q_score - geometry.projection_ms @ reference_score[:2]
        eta_residual = float(reference_score[2] - geometry.eta_on_ms @ reference_score[:2])
        eta_component = geometry.eta_loading * eta_residual
        return np.vstack((eta_component, ms_score - eta_component))

    return transform


def dynamic_vector_transform(
    geometry: SplitGeometry,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        residual = raw_q_score - geometry.projection_mseta @ reference_score[:3]
        return residual[None, :]

    return transform


def filter_block_arc_sm(
    phi: np.ndarray,
    offsets: np.ndarray,
    reference_scores: np.ndarray,
    coefficients: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray | None = None,
    initial_component_history: np.ndarray | None = None,
) -> BlockFilter:
    phi = np.asarray(phi, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    reference_scores = np.asarray(reference_scores, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    clusters = offsets.size - 1
    blocks = coefficients.shape[0]
    dimension = masses.size - 1
    if coefficients.shape != (blocks, 3) or reference_scores.shape[0] != clusters:
        raise ValueError("block coefficients or daily reference scores do not align")
    state = np.zeros(dimension) if initial_state is None else np.asarray(initial_state).copy()
    if initial_component_history is None:
        history = np.empty((0, blocks, dimension))
    else:
        history = np.asarray(initial_component_history, dtype=float).copy()[-21:]
        if history.ndim != 3 or history.shape[1:] != (blocks, dimension):
            raise ValueError("initial component-score history has wrong shape")

    states = np.empty((clusters, dimension))
    component_scores = np.empty((clusters, blocks, dimension))
    weights_path = np.empty((clusters, masses.size))
    contributions = np.empty(clusters)
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[day] = state
        contribution, component_score, weights = cluster_density_and_component_score(
            state, phi[start:stop], masses, helmert
        )
        raw_q_score = helmert.T @ component_score
        components = transform(raw_q_score, reference_scores[day])
        if components.shape != (blocks, dimension):
            raise ValueError("score transform returned the wrong shape")
        contributions[day] = contribution
        component_scores[day] = components
        weights_path[day] = weights
        history = np.concatenate((history, components[None, :, :]), axis=0)[-21:]
        state = np.zeros(dimension)
        for horizon_index, window in enumerate(core.WINDOWS):
            average = history[-window:].sum(axis=0) / window
            state += np.sum(coefficients[:, horizon_index, None] * average, axis=0)
        if np.any(~np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("unstable score-direction DCD-SM state")

    return BlockFilter(
        contributions=contributions,
        states=states,
        component_scores=component_scores,
        weights=weights_path,
        terminal_state=state,
        terminal_component_history=history,
    )


def fit_block_arc_sm(
    phi: np.ndarray,
    offsets: np.ndarray,
    reference_scores: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
    block_names: tuple[str, ...],
    numerical_state_step: float = 1e-2,
) -> tuple[dict[str, object], BlockFilter, BlockFilter]:
    blocks = len(block_names)
    static = filter_block_arc_sm(
        phi, offsets, reference_scores, np.zeros((blocks, 3)), masses, helmert, transform
    )
    units = np.empty((blocks, 3))
    for block in range(blocks):
        averages = fixed_denominator_score_averages(
            static.component_scores[:, block, :], core.WINDOWS
        )
        for horizon, values in enumerate(averages):
            scale = float(np.median(np.linalg.norm(values, axis=1)))
            units[block, horizon] = numerical_state_step / max(scale, 1e-12)

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(blocks, 3) * units

    def run(theta: np.ndarray) -> BlockFilter:
        return filter_block_arc_sm(
            phi, offsets, reference_scores, unpack(theta), masses, helmert, transform
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    parameter_count = blocks * 3
    rng = np.random.default_rng(core.SEED + 131 + blocks)
    starts = [
        np.zeros(parameter_count),
        np.ones(parameter_count),
        -np.ones(parameter_count),
        rng.normal(size=parameter_count),
        3.0 * rng.normal(size=parameter_count),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * parameter_count,
            options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9, "eps": 1e-5, "maxls": 60},
        )
        print(
            f"score-direction {blocks}-block start {index + 1}/{len(starts)}: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all score-direction DCD-SM starts failed")
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
        candidate_eigenvalues = np.linalg.eigvalsh(candidate)
        attempts.append({"step": step, "eigenvalues": candidate_eigenvalues.tolist()})
        if candidate_eigenvalues[0] > 1e-10:
            hessian, eigenvalues, derivative_step = candidate, candidate_eigenvalues, step
            break
    if hessian is None:
        raise RuntimeError(f"nonpositive score-direction Hessian: {attempts}")

    gradients = np.empty((offsets.size - 1, parameter_count))
    for coordinate in range(parameter_count):
        direction = np.zeros(parameter_count)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units.ravel())
    covariance = scaling @ covariance_theta @ scaling
    covariance = (covariance + covariance.T) / 2.0
    parameters = coefficients.ravel()
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = parameters / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))

    def wald(indices: np.ndarray) -> dict[str, float | int | list[float]]:
        selected = parameters[indices]
        selected_covariance = covariance[np.ix_(indices, indices)]
        selected_covariance = (selected_covariance + selected_covariance.T) / 2.0
        covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(selected_covariance)
        largest = float(max(np.max(np.abs(covariance_eigenvalues)), 1.0))
        tolerance = float(indices.size * np.finfo(float).eps * largest)
        materially_negative = covariance_eigenvalues < -1e-8 * largest
        if np.any(materially_negative):
            return {
                "wald": float("nan"),
                "df": 0,
                "p": float("nan"),
                "covariance_rank": int(np.sum(covariance_eigenvalues > tolerance)),
                "covariance_eigenvalues": covariance_eigenvalues.tolist(),
            }
        retained = covariance_eigenvalues > tolerance
        rank = int(np.sum(retained))
        if rank == 0:
            return {
                "wald": float("nan"),
                "df": 0,
                "p": float("nan"),
                "covariance_rank": 0,
                "covariance_eigenvalues": covariance_eigenvalues.tolist(),
            }
        rotated = covariance_eigenvectors[:, retained].T @ selected
        statistic = float(np.sum(rotated**2 / covariance_eigenvalues[retained]))
        return {
            "wald": statistic,
            "df": rank,
            "p": float(chi2.sf(statistic, rank)),
            "covariance_rank": rank,
            "covariance_eigenvalues": covariance_eigenvalues.tolist(),
        }

    block_tests = {
        name: wald(np.arange(block * 3, block * 3 + 3))
        for block, name in enumerate(block_names)
    }
    rows = []
    for block, name in enumerate(block_names):
        for horizon, window in enumerate(core.WINDOWS):
            index = block * 3 + horizon
            rows.append(
                {
                    "block": name,
                    "horizon": int(window),
                    "estimate": float(parameters[index]),
                    "hac22_se_conditional_on_reference": float(standard_errors[index]),
                    "t": float(t_values[index]),
                    "p_two_sided": float(p_values[index]),
                }
            )
    summary = {
        "block_names": list(block_names),
        "coefficients": coefficients.tolist(),
        "parameter_units": units.tolist(),
        "parameters": rows,
        "joint_all": wald(np.arange(parameter_count)),
        "joint_blocks": block_tests,
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_composite_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_static": dynamic.composite_log_likelihood - static.composite_log_likelihood,
        "optimizer_success": bool(optimum.success),
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(max(fit.fun for fit in successful) - optimum.fun),
        "hessian_derivative_step": derivative_step,
        "hessian_eigenvalues_scaled": eigenvalues.tolist(),
        "sandwich_covariance_eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
        "maximum_absolute_mean_day_gradient": float(np.max(np.abs(gradients.mean(axis=0)))),
        "inference_scope": "conditional Godambe-HAC(22), holding the fitted reference path and static anchor fixed",
    }
    return summary, static, dynamic


def reference_betas(saved: dict[str, object], key: str) -> np.ndarray:
    names = ("location", "log_scale_squared", "eta")
    betas = np.zeros((3, 4))
    source = saved[key]["betas"]
    for h, horizon in enumerate(core.HORIZONS):
        for coordinate, name in enumerate(names):
            if name in source[horizon]:
                betas[h, coordinate] = source[horizon][name]
    return betas


def continue_reference(panel, n_train, anchor, betas):
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    train = core.filter_reference(panel.values[:training_rows], train_offsets, anchor, betas)
    oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, betas,
        initial_state=train.terminal_state,
        initial_score_history=train.terminal_score_history,
    )
    full = core.filter_reference(panel.values, panel.offsets, anchor, betas)
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"reference continuation failed: {error}")
    return train, oos, full, error


def fit_and_continue_correction(
    phi, panel, n_train, reference_train, reference_oos, reference_full,
    masses, helmert, transform, block_names,
):
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    fit, static, train = fit_block_arc_sm(
        phi[:training_rows], train_offsets, reference_train.scores,
        masses, helmert, transform, block_names,
    )
    coefficients = np.asarray(fit["coefficients"])
    oos = filter_block_arc_sm(
        phi[training_rows:], oos_offsets, reference_oos.scores, coefficients,
        masses, helmert, transform,
        initial_state=train.terminal_state,
        initial_component_history=train.terminal_component_history,
    )
    full = filter_block_arc_sm(
        phi, panel.offsets, reference_full.scores, coefficients,
        masses, helmert, transform,
    )
    continuation = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    nesting = float(np.max(np.abs(static.contributions)))
    if max(continuation, nesting) > 2e-12:
        raise RuntimeError("correction continuation or q=0 nesting failed")
    return fit, oos, {"continuation_error": continuation, "q_zero_nesting_error": nesting}


def run_dataset(dataset: str) -> None:
    output = OUTPUTS[dataset]
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(SAVED_REFERENCE[dataset].read_text(encoding="utf-8"))
    panel = (
        previous.load_market(dataset)
        if dataset in ("sp500", "n225")
        else stocks.load_asset(dataset)[0]
    )
    benchmark_names = reference_model_names(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= core.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    anchor = np.asarray(saved["static_symmetric_t"]["state"], dtype=float)

    fixed_betas = reference_betas(saved, "fixed_eta_dynamic_location_scale")
    dynamic_betas = reference_betas(saved, "dynamic_location_scale_eta")
    fixed_train, fixed_oos, fixed_full, fixed_reference_error = continue_reference(
        panel, n_train, anchor, fixed_betas
    )
    dynamic_train, dynamic_oos, dynamic_full, dynamic_reference_error = continue_reference(
        panel, n_train, anchor, dynamic_betas
    )

    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    geometry = score_direction_geometry(anchor, knots, masses, helmert)
    print(f"{dataset}: geometry {geometry.diagnostics}", flush=True)
    fixed_u = core.dynamic_reference_pit(panel.values, panel.offsets, fixed_full.states)
    dynamic_u = core.dynamic_reference_pit(panel.values, panel.offsets, dynamic_full.states)
    fixed_phi = core.normalized_bspline_basis(fixed_u, knots, core.DEGREE, masses)
    dynamic_phi = core.normalized_bspline_basis(dynamic_u, knots, core.DEGREE, masses)

    print(f"{dataset}: fitting eta/noneta split DCD-SM", flush=True)
    fixed_fit, fixed_correction_oos, fixed_checks = fit_and_continue_correction(
        fixed_phi, panel, n_train, fixed_train, fixed_oos, fixed_full,
        masses, helmert, fixed_split_transform(geometry), ("eta_direction", "noneta_direction"),
    )
    print(f"{dataset}: fitting fully orthogonal residual-vector DCD-SM", flush=True)
    dynamic_fit, dynamic_correction_oos, dynamic_checks = fit_and_continue_correction(
        dynamic_phi, panel, n_train, dynamic_train, dynamic_oos, dynamic_full,
        masses, helmert, dynamic_vector_transform(geometry), ("residual_vector",),
    )

    fixed_scores, _, _ = core.score_oos(
        panel.values[training_rows:], oos_offsets, dates[n_train:], panel.values[:training_rows],
        fixed_oos, fixed_correction_oos, knots, masses,
        reference_model_name=benchmark_names["fixed"],
        proposed_model_name=MODEL_NAMES["fixed_split"],
    )
    dynamic_scores, _, _ = core.score_oos(
        panel.values[training_rows:], oos_offsets, dates[n_train:], panel.values[:training_rows],
        dynamic_oos, dynamic_correction_oos, knots, masses,
        reference_model_name=benchmark_names["dynamic"],
        proposed_model_name=MODEL_NAMES["dynamic_vector"],
    )
    old_scores = pd.read_csv(SAVED_SCORES[dataset])
    new_only = pd.concat((
        fixed_scores.loc[fixed_scores["model"] == MODEL_NAMES["fixed_split"]],
        dynamic_scores.loc[dynamic_scores["model"] == MODEL_NAMES["dynamic_vector"]],
    ))
    scores = pd.concat((old_scores, new_only), ignore_index=True)
    summary = scores.groupby("model", as_index=False)[list(previous.SCORE_NAMES)].mean()
    comparisons_to_make = (
        (MODEL_NAMES["fixed_split"], benchmark_names["fixed"]),
        (MODEL_NAMES["fixed_split"], benchmark_names["fixed_arc"]),
        (MODEL_NAMES["dynamic_vector"], benchmark_names["dynamic"]),
        (MODEL_NAMES["dynamic_vector"], benchmark_names["dynamic_arc"]),
        (MODEL_NAMES["fixed_split"], MODEL_NAMES["dynamic_vector"]),
    )
    comparisons = [
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in comparisons_to_make
        for score in previous.SCORE_NAMES
    ]
    parameters = pd.DataFrame(
        [{"specification": "fixed_eta_split", **row} for row in fixed_fit["parameters"]]
        + [{"specification": "dynamic_eta_residual_vector", **row} for row in dynamic_fit["parameters"]]
    )
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "oos_score_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "oos_comparisons.csv", index=False)
    parameters.to_csv(output / "score_direction_parameters.csv", index=False)
    result = {
        "analysis": f"{dataset} symmetric-t DCD-SM score-direction decomposition",
        "specification": {
            "fixed_eta": "after (m,sigma2) projection, separate eta-direction and noneta-direction HAR(1,5,22) responses",
            "dynamic_eta": "after (m,sigma2,eta) projection, one residual-score-vector HAR(1,5,22) response",
            "parity_split": False,
            "score_whitening": False,
            "static_q0": "uniform (q0=0)",
        },
        "sample": saved["sample"],
        "geometry": geometry.diagnostics,
        "fixed_eta_split_arc_sm": fixed_fit,
        "dynamic_eta_residual_vector_arc_sm": dynamic_fit,
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons,
        "checks": {
            "fixed_reference_continuation": fixed_reference_error,
            "dynamic_reference_continuation": dynamic_reference_error,
            "fixed_eta_split": fixed_checks,
            "dynamic_eta_residual_vector": dynamic_checks,
        },
    }
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(
        output, dataset, result, summary, pd.DataFrame(comparisons), parameters,
        benchmark_names,
    )


def write_report(
    output, dataset, result, summary, comparisons, parameters, benchmark_names
) -> None:
    labels = {
        benchmark_names["fixed"]: "dynamic m,sigma2; fixed eta",
        benchmark_names["fixed_arc"]: "fixed eta + current parity DCD-SM",
        MODEL_NAMES["fixed_split"]: "fixed eta + eta/noneta split DCD-SM",
        benchmark_names["dynamic"]: "dynamic m,sigma2,eta",
        benchmark_names["dynamic_arc"]: "dynamic eta + current parity DCD-SM",
        MODEL_NAMES["dynamic_vector"]: "dynamic eta + residual-vector DCD-SM",
    }
    lines = [
        f"# {dataset.upper()} symmetric-t DCD-SM score-direction decomposition",
        "",
        "No Fisher whitening is used. The new models do not use a parity split.",
        "The fixed-eta model separates the location/scale-orthogonal score into",
        "its conditional eta projection and its eta-orthogonal remainder. The",
        "dynamic-eta model regresses directly on the fully orthogonal residual vector.",
        "",
        "## Strict-OOS scores",
        "",
        "| model | log score | CRPS |",
        "|---|---:|---:|",
    ]
    for row in summary.sort_values("continuous_log_score", ascending=False).to_dict("records"):
        lines.append(f"| {labels[row['model']]} | {row['continuous_log_score']:.9f} | {row['crps']:.9f} |")
    lines += ["", "## New score coefficients", "", "| specification | block | horizon | estimate | HAC(22) p |", "|---|---|---:|---:|---:|"]
    for row in parameters.to_dict("records"):
        lines.append(
            f"| {row['specification']} | {row['block']} | {row['horizon']} | "
            f"{row['estimate']:.9g} | {row['p_two_sided']:.6g} |"
        )
    fixed = result["fixed_eta_split_arc_sm"]
    dynamic = result["dynamic_eta_residual_vector_arc_sm"]
    lines += [
        "",
        "## Joint coefficient tests",
        "",
        f"- Fixed eta, all six coefficients: `{fixed['joint_all']['p']:.6g}`.",
        f"- Fixed eta, eta-direction block: `{fixed['joint_blocks']['eta_direction']['p']:.6g}`.",
        f"- Fixed eta, noneta-direction block: `{fixed['joint_blocks']['noneta_direction']['p']:.6g}`.",
        f"- Dynamic eta, residual-vector block: `{dynamic['joint_all']['p']:.6g}`.",
        "",
        "## Strict-OOS comparisons",
        "",
        "Positive improvement favors the proposed model.",
        "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ]
    for row in comparisons.loc[comparisons["score"].isin(("continuous_log_score", "crps"))].to_dict("records"):
        lines.append(
            f"| {labels[row['proposed']]} | {labels[row['benchmark']]} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    checks = result["geometry"]
    lines += [
        "",
        "## Geometry checks",
        "",
        f"- Eta component rank: `{checks['eta_component_rank']}`.",
        f"- Maximum noneta/eta cross-information: `{checks['maximum_post_eta_cross_information']:.3g}`.",
        f"- Sequential-versus-joint projection error: `{checks['sequential_vs_joint_projection_max_abs_error']:.3g}`.",
        "- Fisher whitening: `false`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    choices = ("sp500", "n225", *stocks.ASSETS, "both", "stocks", "all")
    parser.add_argument("--market", choices=choices, default="both")
    args = parser.parse_args()
    if args.market == "both":
        datasets = ("sp500", "n225")
    elif args.market == "stocks":
        datasets = tuple(stocks.ASSETS)
    elif args.market == "all":
        datasets = ("sp500", "n225", *stocks.ASSETS)
    else:
        datasets = (args.market,)
    for dataset in datasets:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
