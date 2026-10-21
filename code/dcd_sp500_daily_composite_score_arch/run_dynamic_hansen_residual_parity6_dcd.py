"""Fit a six-coefficient parity residual DCD over dynamic diagonal Hansen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_dynamic_hansen_residual_parity6_dcd"
TWO_COEFFICIENT_OUTPUT = ROOT / "output_dynamic_hansen_residual_dcd"
BREAKPOINTS = np.linspace(0.0, 1.0, 11)
WINDOWS = (1, 5, 22)
BIN_COUNTS = (50, 100, 200)
TAIL_LEVELS = (0.01, 0.05)
QUANTILE_NODES = 8192
SEED = 20260901

sys.path.insert(0, str(ROOT))
from estimate import DEGREE, TRAINING_END, load_and_standardize  # noqa: E402
sys.path.insert(0, str(ROOT))
from estimate_score_har import (  # noqa: E402
    fixed_denominator_score_averages,
    hac_meat,
    numerical_hessian,
)
from run_componentwise_lag5_dcd import (  # noqa: E402
    cluster_density_and_component_score,
    component_cdfs,
)
from run_dynamic_hansen_residual_dcd import (  # noqa: E402
    EfficientGeometry,
    ResidualFilter,
    brier_loss,
    dynamic_hansen_pit,
    efficient_score_geometry,
    filter_residual_dcd,
    q_loss,
    score_comparison,
)
from run_empirical_reference import hac_mean_test  # noqa: E402
from run_hansen_skewt_gas_benchmark import (  # noqa: E402
    categorical_log_score,
    filter_hansen_score_har,
    hansen_cdf,
    hansen_ppf,
    score_ordered_predictive_sample,
)

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))
from dcd_model import (  # noqa: E402
    bspline_masses,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


def filter_residual_parity6_dcd(
    component_density: np.ndarray,
    offsets: np.ndarray,
    hansen_daily_scores: np.ndarray,
    kappa_plus: np.ndarray,
    kappa_minus: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    geometry: EfficientGeometry,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ResidualFilter:
    """Filter daily/weekly/monthly even and odd orthogonal residual scores."""
    phi = np.asarray(component_density, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    hansen_daily_scores = np.asarray(hansen_daily_scores, dtype=float)
    kappa_plus = np.asarray(kappa_plus, dtype=float)
    kappa_minus = np.asarray(kappa_minus, dtype=float)
    clusters = offsets.size - 1
    dimension = masses.size - 1
    nuisance_dimension = geometry.projection_q_on_hansen.shape[1]
    if hansen_daily_scores.shape != (clusters, nuisance_dimension):
        raise ValueError(
            "daily nuisance scores do not align with DCD clusters and geometry"
        )
    if kappa_plus.shape != (3,) or kappa_minus.shape != (3,):
        raise ValueError("three even and three odd coefficients are required")
    states = np.empty((clusters, dimension))
    scaled_scores = np.empty_like(states)
    weights_path = np.empty((clusters, masses.size))
    contributions = np.empty(clusters)
    state = (
        np.zeros(dimension)
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    if initial_score_history is None:
        history = np.empty((0, dimension))
    else:
        history = np.asarray(initial_score_history, dtype=float).copy()
        if history.ndim != 2 or history.shape[1] != dimension:
            raise ValueError("initial residual-score history has wrong shape")
        history = history[-21:]

    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[day] = state
        contribution, component_score, weights = (
            cluster_density_and_component_score(
                state, phi[start:stop], masses, helmert
            )
        )
        contributions[day] = contribution
        weights_path[day] = weights
        raw_q_score = helmert.T @ component_score
        efficient_score = (
            raw_q_score
            - geometry.projection_q_on_hansen @ hansen_daily_scores[day]
        )
        scaled_score = geometry.inverse_sqrt @ efficient_score
        scaled_scores[day] = scaled_score
        history = np.vstack((history, scaled_score[None, :]))
        state = np.zeros(dimension)
        for index, window in enumerate(WINDOWS):
            average = history[-window:].sum(axis=0) / window
            state += (
                kappa_plus[index] * (geometry.projector_plus @ average)
                + kappa_minus[index] * (geometry.projector_minus @ average)
            )
        history = history[-21:]
        if np.any(~np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("unstable six-coefficient residual DCD state")
    return ResidualFilter(
        contributions=contributions,
        states=states,
        scaled_scores=scaled_scores,
        weights=weights_path,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_residual_parity6_dcd(
    phi: np.ndarray,
    offsets: np.ndarray,
    hansen_daily_scores: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    geometry: EfficientGeometry,
    nested_even_5_22: np.ndarray,
    numerical_state_step: float = 1e-3,
    scaled_starts_override: list[np.ndarray] | None = None,
    refine_derivative_free: bool = False,
    adaptive_hessian: bool = False,
) -> tuple[dict[str, object], ResidualFilter, ResidualFilter]:
    """Estimate six coefficients with conditional Godambe-HAC inference."""
    if not np.isfinite(numerical_state_step) or numerical_state_step <= 0.0:
        raise ValueError("numerical_state_step must be positive")
    static = filter_residual_parity6_dcd(
        phi,
        offsets,
        hansen_daily_scores,
        np.zeros(3),
        np.zeros(3),
        masses,
        helmert,
        geometry,
    )
    averages = fixed_denominator_score_averages(static.scaled_scores, WINDOWS)
    units_plus = np.array(
        [
            numerical_state_step
            / max(
                float(
                    np.median(
                        np.linalg.norm(values @ geometry.projector_plus, axis=1)
                    )
                ),
                1e-12,
            )
            for values in averages
        ]
    )
    units_minus = np.array(
        [
            numerical_state_step
            / max(
                float(
                    np.median(
                        np.linalg.norm(values @ geometry.projector_minus, axis=1)
                    )
                ),
                1e-12,
            )
            for values in averages
        ]
    )
    units = np.concatenate((units_plus, units_minus))

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray(theta, dtype=float) * units
        return parameters[:3], parameters[3:]

    def run(theta: np.ndarray) -> ResidualFilter:
        kappa_plus, kappa_minus = unpack(theta)
        return filter_residual_parity6_dcd(
            phi,
            offsets,
            hansen_daily_scores,
            kappa_plus,
            kappa_minus,
            masses,
            helmert,
            geometry,
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    nested_plus = np.array([0.0, nested_even_5_22[0], nested_even_5_22[1]])
    nested_minus = np.zeros(3)
    nested_parameters = np.concatenate((nested_plus, nested_minus))
    nested_theta = nested_parameters / units
    nested = run(nested_theta)
    rng = np.random.default_rng(SEED + 17)
    starts = (
        [np.asarray(start, dtype=float) for start in scaled_starts_override]
        if scaled_starts_override is not None
        else [
            nested_theta,
            np.zeros(6),
            nested_theta + 0.5 * rng.normal(size=6),
            nested_theta + 2.0 * rng.normal(size=6),
            rng.normal(size=6),
            5.0 * rng.normal(size=6),
        ]
    )
    if not starts or any(start.shape != (6,) for start in starts):
        raise ValueError("scaled start overrides must contain six-vectors")
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 6,
            options={
                "maxiter": 3000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        print(
            f"residual parity6 start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.10g} theta={np.asarray(fit.x)}",
            flush=True,
        )
        fits.append(fit)
    successful_fits = [
        result
        for result in fits
        if result.success and np.isfinite(float(result.fun))
    ]
    if not successful_fits:
        best_attempt = min(fits, key=lambda result: float(result.fun))
        raise RuntimeError(
            f"residual parity6 optimization failed: {best_attempt.message}"
        )
    # A failed L-BFGS-B line search can end at the same optimum as successful
    # starts and differ only below the displayed objective precision.  Never let
    # that numerical tie displace a genuinely converged solution.
    optimum = min(successful_fits, key=lambda result: float(result.fun))
    refinement = None
    if refine_derivative_free:
        refinement = minimize(
            objective,
            np.asarray(optimum.x, dtype=float),
            method="COBYQA",
            bounds=[(-500.0, 500.0)] * 6,
            options={
                "maxfev": 2000,
                "initial_tr_radius": 0.25,
                "final_tr_radius": 1e-5,
            },
        )
        print(
            f"residual parity6 COBYQA refinement: success={refinement.success} "
            f"objective={float(refinement.fun):.10g} theta={np.asarray(refinement.x)}",
            flush=True,
        )
        if np.isfinite(refinement.fun) and float(refinement.fun) < float(optimum.fun):
            optimum = refinement
    theta = np.asarray(optimum.x, dtype=float)
    if np.max(np.abs(theta)) > 499.9:
        raise RuntimeError("residual parity6 optimum hit scaled search boundary")
    kappa_plus, kappa_minus = unpack(theta)
    dynamic = run(theta)
    if dynamic.composite_log_likelihood + 1e-9 < nested.composite_log_likelihood:
        raise RuntimeError("six-coefficient fit is below nested two-coefficient fit")

    derivative_step = None
    hessian = None
    hessian_eigenvalues = None
    hessian_attempts = []
    candidate_steps = (
        (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)
        if adaptive_hessian
        else (1e-2,)
    )
    for candidate_step in candidate_steps:
        candidate_hessian = numerical_hessian(objective, theta, candidate_step)
        candidate_eigenvalues = np.linalg.eigvalsh(candidate_hessian)
        hessian_attempts.append(
            {
                "step": candidate_step,
                "eigenvalues": candidate_eigenvalues.tolist(),
            }
        )
        if candidate_eigenvalues[0] > 1e-10:
            derivative_step = candidate_step
            hessian = candidate_hessian
            hessian_eigenvalues = candidate_eigenvalues
            break
    if hessian is None or derivative_step is None or hessian_eigenvalues is None:
        raise RuntimeError(f"nonpositive parity6 Hessian: {hessian_attempts}")
    gradients = np.empty((offsets.size - 1, 6))
    for coordinate in range(6):
        direction = np.zeros(6)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions
            - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units)
    covariance = scaling @ covariance_theta @ scaling
    covariance = (covariance + covariance.T) / 2.0
    parameters = np.concatenate((kappa_plus, kappa_minus))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = parameters / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))

    def wald(indices: list[int]) -> tuple[float, float]:
        selected = parameters[indices]
        selected_covariance = covariance[np.ix_(indices, indices)]
        statistic = float(
            selected @ np.linalg.solve(selected_covariance, selected)
        )
        return statistic, float(chi2.sf(statistic, df=len(indices)))

    joint_all = wald(list(range(6)))
    joint_odd = wald([3, 4, 5])
    added_vs_two = wald([0, 3, 4, 5])
    contrast = np.column_stack((np.eye(3), -np.eye(3)))
    difference = contrast @ parameters
    difference_covariance = contrast @ covariance @ contrast.T
    equality_statistic = float(
        difference @ np.linalg.solve(difference_covariance, difference)
    )
    summary: dict[str, object] = {
        "kappa_plus": kappa_plus.tolist(),
        "kappa_minus": kappa_minus.tolist(),
        "parameter_units": units.tolist(),
        "numerical_state_step": numerical_state_step,
        "se_godambe_hac22_conditional_on_hansen": standard_errors.tolist(),
        "t_godambe_hac22_conditional_on_hansen": t_values.tolist(),
        "p_two_sided_conditional_on_hansen": p_values.tolist(),
        "nested_two_coefficient_composite_log_likelihood": nested.composite_log_likelihood,
        "parity6_composite_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_two_coefficient": (
            dynamic.composite_log_likelihood - nested.composite_log_likelihood
        ),
        "joint_all_wald": joint_all[0],
        "joint_all_p": joint_all[1],
        "joint_odd_wald": joint_odd[0],
        "joint_odd_p": joint_odd[1],
        "added_daily_plus_and_all_odd_wald": added_vs_two[0],
        "added_daily_plus_and_all_odd_p": added_vs_two[1],
        "equal_plus_minus_wald": equality_statistic,
        "equal_plus_minus_p": float(chi2.sf(equality_statistic, df=3)),
        "hessian_eigenvalues_theta": hessian_eigenvalues.tolist(),
        "hessian_derivative_step": derivative_step,
        "hessian_attempts": hessian_attempts,
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "best_of_starts": len(starts),
        "derivative_free_refinement": (
            {
                "success": bool(refinement.success),
                "objective": float(refinement.fun),
                "message": str(refinement.message),
            }
            if refinement is not None
            else None
        ),
    }
    return summary, nested, dynamic


def main() -> None:
    prerequisite = TWO_COEFFICIENT_OUTPUT / "estimation.json"
    if not prerequisite.exists():
        raise FileNotFoundError(
            f"{prerequisite} is missing; run run_dynamic_hansen_residual_dcd.py first"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    two_result = json.loads(prerequisite.read_text(encoding="utf-8"))

    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(offsets[n_train])
    standardized = returns["standardized_return"].to_numpy(dtype=float)
    train_offsets = offsets[: n_train + 1]
    oos_offsets = offsets[n_train:] - training_rows
    anchor = np.asarray(
        two_result["hansen_static_training"]["state_anchor"], dtype=float
    )
    diagonal = two_result["hansen_diagonal_training"]["betas"]
    diagonal_betas = np.array(
        [
            [diagonal[horizon]["eta"], diagonal[horizon]["lambda"]]
            for horizon in ("daily", "weekly", "monthly")
        ]
    )
    diagonal_train = filter_hansen_score_har(
        standardized[:training_rows],
        train_offsets,
        anchor,
        diagonal_betas,
    )
    diagonal_oos = filter_hansen_score_har(
        standardized[training_rows:],
        oos_offsets,
        anchor,
        diagonal_betas,
        initial_state=diagonal_train.terminal_state,
        initial_score_history=diagonal_train.terminal_score_history,
    )
    stored_training_likelihood = two_result["hansen_diagonal_training"][
        "diagonal_composite_log_likelihood"
    ]
    backbone_reproduction_error = float(
        abs(diagonal_train.composite_log_likelihood - stored_training_likelihood)
    )
    if backbone_reproduction_error > 1e-9:
        raise RuntimeError(
            f"diagonal Hansen backbone does not reproduce: {backbone_reproduction_error}"
        )

    all_states = np.vstack((diagonal_train.states, diagonal_oos.states))
    all_scores = np.vstack((diagonal_train.scores, diagonal_oos.scores))
    u = dynamic_hansen_pit(standardized, offsets, all_states)
    knots = make_clamped_knots(BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    helmert = helmert_submatrix(masses.size)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)
    geometry = efficient_score_geometry(anchor, knots, masses, helmert)
    nested_even = np.array(
        [
            two_result["training_residual_dcd"]["kappa_5_even"],
            two_result["training_residual_dcd"]["kappa_22_even"],
        ]
    )

    print("Fitting six-coefficient orthogonal residual parity DCD", flush=True)
    fit_summary, nested_train, parity6_train = fit_residual_parity6_dcd(
        phi[:training_rows],
        train_offsets,
        diagonal_train.scores,
        masses,
        helmert,
        geometry,
        nested_even,
    )
    kappa_plus = np.asarray(fit_summary["kappa_plus"])
    kappa_minus = np.asarray(fit_summary["kappa_minus"])
    parity6_oos = filter_residual_parity6_dcd(
        phi[training_rows:],
        oos_offsets,
        diagonal_oos.scores,
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
        initial_state=parity6_train.terminal_state,
        initial_score_history=parity6_train.terminal_score_history,
    )
    two_oos = filter_residual_dcd(
        phi[training_rows:],
        oos_offsets,
        diagonal_oos.scores,
        nested_even,
        masses,
        helmert,
        geometry,
        initial_state=nested_train.terminal_state,
        initial_score_history=nested_train.terminal_score_history,
    )
    nested_oos_equivalence_error = float(
        np.max(
            np.abs(
                two_oos.contributions
                - filter_residual_parity6_dcd(
                    phi[training_rows:],
                    oos_offsets,
                    diagonal_oos.scores,
                    np.array([0.0, *nested_even]),
                    np.zeros(3),
                    masses,
                    helmert,
                    geometry,
                    initial_state=nested_train.terminal_state,
                    initial_score_history=nested_train.terminal_score_history,
                ).contributions
            )
        )
    )
    if nested_oos_equivalence_error > 1e-12:
        raise RuntimeError(
            f"two-coefficient nesting error: {nested_oos_equivalence_error}"
        )

    full_parity6 = filter_residual_parity6_dcd(
        phi,
        offsets,
        all_scores,
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
    )
    continuation_error = float(
        np.max(np.abs(full_parity6.states[n_train:] - parity6_oos.states))
    )
    if continuation_error > 1e-12:
        raise RuntimeError(f"training/OOS continuation error: {continuation_error}")

    print("Scoring six-coefficient strict-OOS distributions", flush=True)
    u_grid, cdfs = component_cdfs(knots, masses)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    ordered_training = np.sort(standardized[:training_rows])
    bin_edges = {
        count: np.concatenate(
            (
                [-np.inf],
                np.quantile(ordered_training, np.arange(1, count) / count),
                [np.inf],
            )
        )
        for count in BIN_COUNTS
    }
    thresholds = {
        alpha: float(np.quantile(ordered_training, alpha))
        for alpha in TAIL_LEVELS
    }
    rows: list[dict[str, object]] = []
    residual_paths = {
        "dynamic_hansen_residual_even2": two_oos,
        "dynamic_hansen_residual_parity6": parity6_oos,
    }
    for local_day, target in enumerate(range(n_train, len(dates))):
        if local_day % 50 == 0:
            print(f"scoring day {local_day + 1}/{len(dates)-n_train}", flush=True)
        start, stop = offsets[target : target + 2]
        observations = standardized[start:stop]
        hansen_state = diagonal_oos.states[local_day]
        model_data: list[tuple[str, np.ndarray, np.ndarray | None, float]] = [
            (
                "diagonal_hansen",
                hansen_ppf(probability_nodes, hansen_state),
                None,
                0.0,
            )
        ]
        for model, filtered in residual_paths.items():
            mixture_cdf = np.maximum.accumulate(
                cdfs @ filtered.weights[local_day]
            )
            mixture_cdf[0], mixture_cdf[-1] = 0.0, 1.0
            residual_nodes = np.interp(probability_nodes, mixture_cdf, u_grid)
            model_data.append(
                (
                    model,
                    hansen_ppf(residual_nodes, hansen_state),
                    mixture_cdf,
                    float(filtered.contributions[local_day]),
                )
            )
        for model, ordered_sample, mixture_cdf, correction_log in model_data:
            row: dict[str, object] = {
                "date": str(dates[target].date()),
                "model": model,
                "n_constituents": int(stop - start),
                "continuous_log_score": float(
                    diagonal_oos.contributions[local_day] + correction_log
                ),
                "crps": score_ordered_predictive_sample(
                    ordered_sample, observations
                ),
            }
            for count in BIN_COUNTS:
                hansen_bin_u = np.concatenate(
                    (
                        [0.0],
                        hansen_cdf(bin_edges[count][1:-1], hansen_state),
                        [1.0],
                    )
                )
                model_bin_cdf = (
                    hansen_bin_u
                    if mixture_cdf is None
                    else np.interp(hansen_bin_u, u_grid, mixture_cdf)
                )
                row[f"categorical_log_score_{count}"] = categorical_log_score(
                    np.maximum(np.diff(model_bin_cdf), 0.0),
                    observations,
                    bin_edges[count],
                )
            for alpha in TAIL_LEVELS:
                quantile = float(
                    np.interp(alpha, probability_nodes, ordered_sample)
                )
                threshold_u = float(
                    hansen_cdf(np.array([thresholds[alpha]]), hansen_state)[0]
                )
                probability = (
                    threshold_u
                    if mixture_cdf is None
                    else float(np.interp(threshold_u, u_grid, mixture_cdf))
                )
                row[f"q_loss_{alpha:g}"] = q_loss(
                    observations, quantile, alpha
                )
                row[f"brier_{alpha:g}"] = brier_loss(
                    observations, thresholds[alpha], probability
                )
            rows.append(row)

    scores = pd.DataFrame(rows)
    score_names = [
        "continuous_log_score",
        "crps",
        *(f"categorical_log_score_{count}" for count in BIN_COUNTS),
        *(f"q_loss_{alpha:g}" for alpha in TAIL_LEVELS),
        *(f"brier_{alpha:g}" for alpha in TAIL_LEVELS),
    ]
    pairs = (
        ("dynamic_hansen_residual_parity6", "diagonal_hansen"),
        ("dynamic_hansen_residual_parity6", "dynamic_hansen_residual_even2"),
    )
    comparisons = [
        score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in score_names
    ]
    summary = scores.groupby("model", as_index=False)[score_names].mean()

    parameter_names = [
        "kappa_plus_1",
        "kappa_plus_5",
        "kappa_plus_22",
        "kappa_minus_1",
        "kappa_minus_5",
        "kappa_minus_22",
    ]
    pd.DataFrame(
        {
            "parameter": parameter_names,
            "estimate": [*kappa_plus, *kappa_minus],
            "hac22_se_conditional_on_hansen": fit_summary[
                "se_godambe_hac22_conditional_on_hansen"
            ],
            "t_conditional_on_hansen": fit_summary[
                "t_godambe_hac22_conditional_on_hansen"
            ],
            "p_conditional_on_hansen": fit_summary[
                "p_two_sided_conditional_on_hansen"
            ],
        }
    ).to_csv(OUTPUT / "parameters.csv", index=False)
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        OUTPUT / "oos_score_comparisons.csv", index=False
    )
    output = {
        "analysis": (
            "six-coefficient even/odd HAR(1,5,22) orthogonal residual DCD "
            "over diagonal dynamic Hansen"
        ),
        "sample": two_result["sample"],
        "density": two_result["density"],
        "residual_geometry": {
            **two_result["residual_score"],
            "parity": "separate even and odd PIT-reflection responses",
            "windows": list(WINDOWS),
        },
        "training": fit_summary,
        "oos_score_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons,
        "checks": {
            "diagonal_hansen_training_likelihood_reproduction_error": backbone_reproduction_error,
            "nested_two_coefficient_oos_max_abs_error": nested_oos_equivalence_error,
            "training_oos_continuation_max_abs_error": continuation_error,
            "inference_scope": (
                "conditional Godambe-HAC(22), holding the first-stage Hansen "
                "anchor and diagonal dynamic coefficients fixed"
            ),
        },
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
