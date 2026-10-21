"""Four symmetric-t location/scale/shape comparisons for SP500 and N225.

The four models are:
  1. dynamic location/log-scale-squared with fixed static-MLE eta;
  2. model 1 plus location-scale-orthogonal parity DCD-SM;
  3. dynamic location/log-scale-squared/eta plus three-tangent-orthogonal DCD-SM;
  4. dynamic location/log-scale-squared/eta.

Hansen lambda is fixed exactly at zero throughout.  No reference or residual
score is Fisher-whitened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2

import n225_config
import run_sp500 as core
import run_sp500_fixed_shape as fixed
from estimate_score_har import hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
OUTPUTS = {
    "sp500": ROOT / "output_symmetric_t_sp500",
    "n225": ROOT / "output_symmetric_t_n225",
}
MODEL_NAMES = {
    "fixed": "symmetric_t_dynamic_location_scale_fixed_eta",
    "fixed_arc": "symmetric_t_dynamic_location_scale_fixed_eta_arc_sm",
    "dynamic": "symmetric_t_dynamic_location_scale_eta",
    "dynamic_arc": "symmetric_t_dynamic_location_scale_eta_arc_sm",
}
STATE_NAMES = ("location", "log_scale_squared", "eta")
SCORE_NAMES = (
    "continuous_log_score",
    "crps",
    *(f"categorical_log_score_{count}" for count in core.BIN_COUNTS),
    *(f"q_loss_{alpha:g}" for alpha in core.TAIL_LEVELS),
    *(f"brier_{alpha:g}" for alpha in core.TAIL_LEVELS),
)


def load_market(market: str) -> core.RawPanel:
    if market == "sp500":
        return core.load_raw_panel()
    source = n225_config.SOURCE
    if not source.exists():
        raise FileNotFoundError(
            f"readable N225 panel not found: {source}; run rebuild_n225_panel.py"
        )
    returns = pd.read_parquet(source, columns=["date", "log_return"])
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[np.isfinite(returns["log_return"])].copy()
    returns = returns.sort_values("date", kind="stable").reset_index(drop=True)
    daily = returns.groupby("date", sort=True).size().rename("n_constituents").reset_index()
    counts = daily["n_constituents"].to_numpy(dtype=int)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    values = returns["log_return"].to_numpy(dtype=float, copy=True)
    if offsets[-1] != values.size:
        raise RuntimeError("N225 offsets do not cover the input panel")
    return core.RawPanel(daily=daily, values=values, offsets=offsets)


def fit_static_symmetric_t(
    values: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    counts = np.diff(offsets)
    days = counts.size
    observation_weights = np.repeat(1.0 / counts, counts)
    mean0 = float(np.sum(observation_weights * values) / days)
    variance0 = float(
        np.sum(observation_weights * ((values - mean0) ** 2)) / days
    )

    def expand(parameter: np.ndarray) -> np.ndarray:
        return np.r_[parameter, 0.0]

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        state = expand(np.asarray(parameter, dtype=float))
        contribution = 0.0
        gradient = np.zeros(3)
        try:
            for start in range(0, values.size, 50_000):
                stop = min(start + 50_000, values.size)
                log_density, score = core.reference_logpdf_and_score(
                    values[start:stop], state
                )
                weight = observation_weights[start:stop]
                contribution += float(np.sum(weight * log_density))
                gradient += np.sum(weight[:, None] * score[:, :3], axis=0)
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100, np.zeros(3)
        return -contribution, -gradient

    starts = [
        np.r_[mean0, np.log(variance0), core.parameters_to_state(eta, 0.0)[0]]
        for eta in (3.5, 5.0, 8.0, 15.0)
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-0.10, 0.10), (-20.0, 0.0), (-12.0, 12.0)],
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
        )
        print(
            f"static symmetric-t start {index + 1}/{len(starts)}: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all static symmetric-t starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    anchor = expand(np.asarray(optimum.x, dtype=float))
    eta, skew, _, _ = core.state_to_parameters(anchor[2:])
    if abs(skew) > 1e-15:
        raise RuntimeError(f"symmetric-t lambda is not zero: {skew}")
    return anchor, {
        "mean": float(anchor[0]),
        "log_scale_squared": float(anchor[1]),
        "scale": float(np.exp(0.5 * anchor[1])),
        "eta": float(eta),
        "lambda": float(skew),
        "state": anchor.tolist(),
        "composite_log_likelihood": float(-optimum.fun),
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(
            max(float(fit.fun) for fit in successful) - float(optimum.fun)
        ),
    }


def fit_dynamic_symmetric_t(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    initial_location_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], core.FourStateFilter, dict[str, object]]:
    coordinates = (0, 1, 2)
    static = core.filter_reference(values, offsets, anchor, np.zeros((3, 4)))
    averages = core.fixed_denominator_score_averages(
        static.scores[:, coordinates], core.WINDOWS
    )
    target_steps = np.array([1e-4, 1e-2, 1e-2])
    units = np.array(
        [
            [
                target_steps[coordinate]
                / max(float(np.median(np.abs(average[:, coordinate]))), 1e-12)
                for coordinate in range(3)
            ]
            for average in averages
        ]
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        betas = np.zeros((3, 4))
        betas[:, coordinates] = np.asarray(theta).reshape(3, 3) * units
        return betas

    def run(theta: np.ndarray) -> core.FourStateFilter:
        return core.filter_reference(values, offsets, anchor, unpack(theta))

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    initial = np.zeros((3, 3))
    initial[:, :2] = initial_location_scale[:, :2]
    rng = np.random.default_rng(core.SEED + 11)
    starts = [
        (initial / units).ravel(),
        np.zeros(9),
        np.ones(9),
        -np.ones(9),
        rng.normal(size=9),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500, 500),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 9,
            options={
                "maxiter": 4000,
                "ftol": 1e-14,
                "gtol": 1e-8,
                "eps": 1e-5,
                "maxls": 80,
            },
        )
        print(
            f"dynamic symmetric-t start {index + 1}/{len(starts)}: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all dynamic symmetric-t starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    betas = unpack(theta)
    dynamic = run(theta)

    print("dynamic symmetric-t: computing conditional HAC inference", flush=True)
    derivative_step = 1e-2
    hessian = numerical_hessian(objective, theta, derivative_step)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive dynamic-t Hessian: {eigenvalues}")
    gradients = np.empty((offsets.size - 1, theta.size))
    for coordinate in range(theta.size):
        direction = np.zeros(theta.size)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions
            - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units.ravel())
    covariance_beta = scaling @ covariance_theta @ scaling
    covariance_beta = (covariance_beta + covariance_beta.T) / 2.0
    beta = betas[:, :3].ravel()
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_beta), 0.0))
    t_values = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    eta_indices = np.array([2, 5, 8])
    eta_beta = beta[eta_indices]
    eta_covariance = covariance_beta[np.ix_(eta_indices, eta_indices)]
    eta_wald = float(eta_beta @ np.linalg.solve(eta_covariance, eta_beta))
    inference_rows = [
        {
            "state": state,
            "horizon": horizon,
            "estimate": float(beta[index]),
            "hac22_se_conditional_on_static_anchor": float(standard_errors[index]),
            "t": float(t_values[index]),
            "p_two_sided": float(p_values[index]),
        }
        for index, (horizon, state) in enumerate(
            (horizon, state)
            for horizon in core.HORIZONS
            for state in STATE_NAMES
        )
    ]
    inference = {
        "scope": (
            "conditional Godambe-HAC(22), holding the training static symmetric-t "
            "anchor fixed and jointly treating all nine dynamic coefficients as estimated"
        ),
        "parameters": inference_rows,
        "eta_three_coefficient_wald": eta_wald,
        "eta_three_coefficient_p": float(chi2.sf(eta_wald, df=3)),
        "hessian_eigenvalues_scaled": eigenvalues.tolist(),
        "hessian_condition_number_scaled": float(eigenvalues[-1] / eigenvalues[0]),
        "maximum_absolute_mean_day_gradient": float(
            np.max(np.abs(gradients.mean(axis=0)))
        ),
    }
    summary = {
        "betas": {
            horizon: {
                name: float(betas[index, coordinate])
                for coordinate, name in enumerate(STATE_NAMES)
            }
            for index, horizon in enumerate(core.HORIZONS)
        },
        "lambda_dynamic_coefficients": "fixed exactly at zero",
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


def selected_tangent_geometry(
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    coordinates: tuple[int, ...],
    label: str,
    quadrature_order: int = 128,
):
    nodes0, weights0 = np.polynomial.legendre.leggauss(quadrature_order)
    q_pieces, tangent_pieces, weight_pieces = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes0 / 2.0
        weight = weights0 * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        q_pieces.append((phi * masses[None, :] - masses[None, :]) @ helmert)
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_pieces.append(
            core.reference_logpdf_and_score(y, anchor)[1][:, coordinates]
        )
        weight_pieces.append(weight)
    q_score = np.vstack(q_pieces)
    tangent = np.vstack(tangent_pieces)
    weight = np.concatenate(weight_pieces)
    information_q = (q_score.T * weight) @ q_score
    information_tangent = (tangent.T * weight) @ tangent
    tangent_sd = np.sqrt(np.diag(information_tangent))
    tangent_scaled = tangent / tangent_sd[None, :]
    information_tangent_scaled = (tangent_scaled.T * weight) @ tangent_scaled
    cross_scaled = (q_score.T * weight) @ tangent_scaled
    projection_scaled = cross_scaled @ np.linalg.inv(information_tangent_scaled)
    projection = projection_scaled / tangent_sd[None, :]
    efficient = q_score - tangent @ projection.T
    residual_information = (efficient.T * weight) @ efficient
    residual_information = (residual_information + residual_information.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(residual_information)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive {label} residual information: {eigenvalues}")
    reversal = np.fliplr(np.eye(masses.size))
    reflection = helmert.T @ reversal @ helmert
    identity = np.eye(masses.size - 1)
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
            "residual_score_whitening": False,
            "tangent_information_eigenvalues_scaled": np.linalg.eigvalsh(
                information_tangent_scaled
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
    masses: np.ndarray,
    helmert: np.ndarray,
    geometry,
    score_coordinates: tuple[int, ...],
):
    fit, q_zero_train, correction_train = core.fit_residual_parity6_dcd(
        phi[: int(train_offsets[-1])],
        train_offsets,
        reference_train.scores[:, score_coordinates],
        masses,
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
        masses,
        helmert,
        geometry,
        initial_state=correction_train.terminal_state,
        initial_score_history=correction_train.terminal_score_history,
    )
    correction_full = core.filter_residual_parity6_dcd(
        phi,
        np.r_[train_offsets, oos_offsets[1:] + train_offsets[-1]],
        reference_full.scores[:, score_coordinates],
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
    )
    continuation_error = float(
        np.max(np.abs(correction_full.states[n_train:] - correction_oos.states))
    )
    q_zero_error = float(np.max(np.abs(q_zero_train.contributions)))
    if max(continuation_error, q_zero_error) > 2e-12:
        raise RuntimeError("DCD-SM continuation or nesting check failed")
    return fit, correction_train, correction_oos, {
        "continuation_error": continuation_error,
        "q_zero_nesting_error": q_zero_error,
    }


def continue_reference(
    panel: core.RawPanel,
    n_train: int,
    anchor: np.ndarray,
    betas: np.ndarray,
    training: core.FourStateFilter,
) -> tuple[core.FourStateFilter, core.FourStateFilter, float]:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = core.filter_reference(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full_path = core.filter_reference(panel.values, panel.offsets, anchor, betas)
    error = float(np.max(np.abs(full_path.states[n_train:] - oos.states)))
    if error > 2e-12:
        raise RuntimeError(f"reference continuation error: {error}")
    return oos, full_path, error


def run_market(market: str) -> None:
    output = OUTPUTS[market]
    output.mkdir(parents=True, exist_ok=True)
    panel = load_market(market)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= core.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    training_values = panel.values[:training_rows]

    print(f"{market}: fitting static symmetric-t anchor", flush=True)
    anchor, static_summary = fit_static_symmetric_t(training_values, train_offsets)
    print(f"{market}: static anchor {static_summary}", flush=True)

    print(f"{market}: fitting fixed-eta dynamic location-scale", flush=True)
    fixed_betas, fixed_summary, fixed_train = fixed.fit_dynamic_location_scale(
        training_values, train_offsets, anchor
    )
    fixed_oos, fixed_full, fixed_continuation = continue_reference(
        panel, n_train, anchor, fixed_betas, fixed_train
    )

    print(f"{market}: fitting dynamic location-scale-eta symmetric-t", flush=True)
    dynamic_betas, dynamic_summary, dynamic_train, dynamic_inference = (
        fit_dynamic_symmetric_t(
            training_values, train_offsets, anchor, fixed_betas
        )
    )
    dynamic_oos, dynamic_full, dynamic_continuation = continue_reference(
        panel, n_train, anchor, dynamic_betas, dynamic_train
    )
    if max(
        np.max(np.abs(fixed_full.states[:, 3])),
        np.max(np.abs(dynamic_full.states[:, 3])),
    ) > 1e-15:
        raise RuntimeError("lambda moved away from zero")

    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    fixed_u = core.dynamic_reference_pit(panel.values, panel.offsets, fixed_full.states)
    dynamic_u = core.dynamic_reference_pit(
        panel.values, panel.offsets, dynamic_full.states
    )
    fixed_phi = core.normalized_bspline_basis(
        fixed_u, knots, core.DEGREE, masses
    )
    dynamic_phi = core.normalized_bspline_basis(
        dynamic_u, knots, core.DEGREE, masses
    )
    fixed_geometry = selected_tangent_geometry(
        anchor,
        knots,
        masses,
        helmert,
        (0, 1),
        "fixed-anchor location and log-scale-squared tangents",
    )
    dynamic_geometry = selected_tangent_geometry(
        anchor,
        knots,
        masses,
        helmert,
        (0, 1, 2),
        "fixed-anchor location, log-scale-squared, and eta tangents",
    )

    print(f"{market}: fitting fixed-eta DCD-SM", flush=True)
    fixed_arc, _, fixed_correction_oos, fixed_arc_checks = fit_arc_sm(
        fixed_phi,
        train_offsets,
        oos_offsets,
        fixed_train,
        fixed_oos,
        fixed_full,
        n_train,
        masses,
        helmert,
        fixed_geometry,
        (0, 1),
    )
    print(f"{market}: fitting dynamic-eta DCD-SM", flush=True)
    dynamic_arc, _, dynamic_correction_oos, dynamic_arc_checks = fit_arc_sm(
        dynamic_phi,
        train_offsets,
        oos_offsets,
        dynamic_train,
        dynamic_oos,
        dynamic_full,
        n_train,
        masses,
        helmert,
        dynamic_geometry,
        (0, 1, 2),
    )

    fixed_scores, _, _ = core.score_oos(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        fixed_oos,
        fixed_correction_oos,
        knots,
        masses,
        reference_model_name=MODEL_NAMES["fixed"],
        proposed_model_name=MODEL_NAMES["fixed_arc"],
    )
    dynamic_scores, _, _ = core.score_oos(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        dynamic_oos,
        dynamic_correction_oos,
        knots,
        masses,
        reference_model_name=MODEL_NAMES["dynamic"],
        proposed_model_name=MODEL_NAMES["dynamic_arc"],
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

    def arc_rows(specification: str, fit: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "specification": specification,
                "block": block,
                "horizon": horizon,
                "estimate": estimate,
                "hac22_se_conditional_on_reference": standard_error,
                "p_two_sided": p_value,
            }
            for block, horizon, estimate, standard_error, p_value in zip(
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
    dynamic_parameters = pd.DataFrame(dynamic_inference["parameters"])
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "four_model_oos_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "four_model_oos_comparisons.csv", index=False
    )
    arc_parameters.to_csv(output / "arc_sm_parameters.csv", index=False)
    dynamic_parameters.to_csv(output / "dynamic_t_parameters.csv", index=False)

    result = {
        "analysis": f"{market} symmetric-t four-model raw-return comparison",
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
        "static_symmetric_t": static_summary,
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


def write_report(
    output: Path,
    result: dict[str, object],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    arc_parameters: pd.DataFrame,
) -> None:
    labels = {
        MODEL_NAMES["fixed"]: "dynamic m,sigma2; fixed eta",
        MODEL_NAMES["fixed_arc"]: "dynamic m,sigma2; fixed eta + DCD-SM",
        MODEL_NAMES["dynamic_arc"]: "dynamic m,sigma2,eta + DCD-SM",
        MODEL_NAMES["dynamic"]: "dynamic m,sigma2,eta",
    }
    lines = [
        f"# {result['analysis']}",
        "",
        "Hansen lambda is fixed exactly at zero. Scores are tangent-orthogonalized "
        "without Fisher whitening and updated by parity HAR(1,5,22).",
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
                f"joint odd-block p: `{fit['joint_odd_p']:.6g}`.",
                "",
            ]
        )
    inference = result["dynamic_location_scale_eta_inference"]
    lines.extend(
        [
            "## Dynamic eta reference test",
            "",
            f"Eta three-coefficient Wald p: `{inference['eta_three_coefficient_p']:.6g}`.",
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
