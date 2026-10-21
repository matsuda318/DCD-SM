"""Four-state Hansen score-HAR plus four-tangent orthogonal parity DCD-SM.

The model is fitted to raw daily S&P 500 constituent log returns.  The
reference state contains location, log scale-squared, Hansen tail thickness,
and Hansen skewness.  Conditional on that fitted reference path, the usual
decile-spline score is projected off all four reference tangents and updated
with separate even/odd HAR(1,5,22) coefficients.
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


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LEGACY = REPO / "dcd_sp500_daily_composite_score_arch"
GARCH = REPO / "dcd_sp500_cross_sectional_garch_correction"
AMZN = REPO / "dcd_amzn_daily_score_arch"
OUTPUT = ROOT / "output_sp500"
SOURCE = REPO / "data" / "weekly_density_em" / "daily_returns" / "usa.parquet"
CACHE = REPO / "data" / "arc_sm_joint_location_scale_hansen_parity_input.npz"

TRAINING_END = pd.Timestamp("2023-12-29")
WINDOWS = (1, 5, 22)
HORIZONS = ("daily", "weekly", "monthly")
BREAKPOINTS = np.linspace(0.0, 1.0, 11)
DEGREE = 3
BIN_COUNTS = (50, 100, 200)
TAIL_LEVELS = (0.01, 0.05)
QUANTILE_NODES = 8192
SEED = 20260910

sys.path.insert(0, str(LEGACY))
from estimate_score_har import fixed_denominator_score_averages  # noqa: E402
from run_componentwise_lag5_dcd import component_cdfs  # noqa: E402
from run_dynamic_hansen_residual_dcd import (  # noqa: E402
    EfficientGeometry,
    brier_loss,
    q_loss,
    score_comparison,
)
from run_dynamic_hansen_residual_parity6_dcd import (  # noqa: E402
    filter_residual_parity6_dcd,
    fit_residual_parity6_dcd,
)
from run_hansen_skewt_gas_benchmark import (  # noqa: E402
    categorical_log_score,
    hansen_cdf,
    hansen_constants,
    hansen_logpdf_and_score,
    hansen_ppf,
    parameters_to_state,
    score_ordered_predictive_sample,
    state_to_parameters,
)

sys.path.insert(0, str(AMZN))
from dcd_model import (  # noqa: E402
    bspline_masses,
    centered_clr_metric,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


@dataclass(frozen=True)
class FourStateFilter:
    states: np.ndarray
    means: np.ndarray
    log_variances: np.ndarray
    variances: np.ndarray
    eta: np.ndarray
    skew: np.ndarray
    scores: np.ndarray
    contributions: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.contributions))


@dataclass(frozen=True)
class RawPanel:
    daily: pd.DataFrame
    values: np.ndarray
    offsets: np.ndarray


def load_raw_panel() -> RawPanel:
    """Load only the two columns needed by this memory-conscious experiment."""
    if CACHE.exists():
        cached = np.load(CACHE)
        values = np.asarray(cached["values"], dtype=float)
        counts = np.asarray(cached["counts"], dtype=int)
        dates = pd.to_datetime(cached["dates"])
        offsets = np.concatenate(([0], np.cumsum(counts)))
        if offsets[-1] != values.size:
            raise ValueError("cached daily offsets do not cover the raw-return panel")
        return RawPanel(
            daily=pd.DataFrame({"date": dates, "n_constituents": counts}),
            values=values,
            offsets=offsets,
        )
    returns = pd.read_parquet(SOURCE, columns=["date", "log_return"])
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[np.isfinite(returns["log_return"])].copy()
    returns = returns.sort_values("date", kind="stable").reset_index(drop=True)
    daily = (
        returns.groupby("date", sort=True)
        .size()
        .rename("n_constituents")
        .reset_index()
    )
    counts = daily["n_constituents"].to_numpy(dtype=int)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    values = returns["log_return"].to_numpy(dtype=float, copy=True)
    if offsets[-1] != values.size:
        raise ValueError("daily offsets do not cover the raw-return panel")
    del returns
    return RawPanel(daily=daily, values=values, offsets=offsets)


def reference_logpdf_and_score(
    values: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw-return log density and score for the four unconstrained states."""
    values = np.asarray(values, dtype=float)
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError("the reference state must have four coordinates")
    mean, log_variance = map(float, state[:2])
    if not np.isfinite(log_variance) or log_variance < -40.0 or log_variance > 10.0:
        raise FloatingPointError("invalid log scale-squared state")
    sigma = float(np.exp(0.5 * log_variance))
    z = (values - mean) / sigma
    shape_state = state[2:]
    log_standardized, shape_score = hansen_logpdf_and_score(z, shape_state)
    eta, skew, _, _ = state_to_parameters(shape_state)
    a, b, _ = hansen_constants(eta, skew)
    side = np.where(z < -a / b, -1.0, 1.0)
    denominator = 1.0 + side * skew
    transformed = (b * z + a) / denominator
    d_transformed_dz = b / denominator
    d_log_density_dz = -(
        (eta + 1.0)
        * transformed
        * d_transformed_dz
        / (eta - 2.0 + transformed**2)
    )
    location_score = -d_log_density_dz / sigma
    log_variance_score = -0.5 - 0.5 * z * d_log_density_dz
    score = np.column_stack(
        (location_score, log_variance_score, shape_score)
    )
    return log_standardized - 0.5 * log_variance, score


def validate_reference_score() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = rng.normal(0.001, 0.025, size=401)
    state = np.r_[0.002, np.log(0.0005), parameters_to_state(5.3, -0.12)]
    log_density, analytic = reference_logpdf_and_score(values, state)
    steps = np.array([1e-7, 1e-6, 1e-6, 1e-6])
    numerical = np.empty_like(analytic)
    for coordinate, step in enumerate(steps):
        direction = np.zeros(4)
        direction[coordinate] = step
        numerical[:, coordinate] = (
            reference_logpdf_and_score(values, state + direction)[0]
            - reference_logpdf_and_score(values, state - direction)[0]
        ) / (2.0 * step)
    error = float(np.max(np.abs(analytic - numerical)))
    relative = float(
        np.max(
            np.abs(analytic - numerical)
            / np.maximum(1.0, np.abs(numerical))
        )
    )
    if error > 2e-4 or relative > 2e-6 or np.any(~np.isfinite(log_density)):
        raise RuntimeError(f"four-state score validation failed: {error}, {relative}")
    return {"maximum_absolute_error": error, "maximum_relative_error": relative}


def filter_reference(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    betas: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> FourStateFilter:
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if anchor.shape != (4,) or betas.shape != (3, 4):
        raise ValueError("anchor/beta dimensions do not match the four-state model")
    days = offsets.size - 1
    states = np.empty((days, 4))
    means = np.empty(days)
    log_variances = np.empty(days)
    variances = np.empty(days)
    eta_path = np.empty(days)
    skew_path = np.empty(days)
    scores = np.empty((days, 4))
    contributions = np.empty(days)
    state = anchor.copy() if initial_state is None else np.asarray(initial_state).copy()
    history = (
        np.empty((0, 4))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-21:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if stop <= start:
            raise ValueError("empty daily cross section")
        mean, log_variance = map(float, state[:2])
        if (
            np.any(~np.isfinite(state))
            or abs(mean) > 0.5
            or log_variance < -20.0
            or log_variance > 0.0
            or np.max(np.abs(state[2:])) > 20.0
        ):
            raise FloatingPointError("unstable four-state reference")
        log_density, observation_score = reference_logpdf_and_score(
            values[start:stop], state
        )
        eta, skew, _, _ = state_to_parameters(state[2:])
        states[day] = state
        means[day] = mean
        log_variances[day] = log_variance
        variances[day] = np.exp(log_variance)
        eta_path[day] = eta
        skew_path[day] = skew
        scores[day] = np.mean(observation_score, axis=0)
        contributions[day] = float(np.mean(log_density))
        history = np.vstack((history, scores[day][None, :]))[-21:]
        state = anchor.copy()
        for beta, window in zip(betas, WINDOWS):
            state += beta * history[-window:].sum(axis=0) / window
    return FourStateFilter(
        states=states,
        means=means,
        log_variances=log_variances,
        variances=variances,
        eta=eta_path,
        skew=skew_path,
        scores=scores,
        contributions=contributions,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_static_reference(
    values: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    print("static reference: preparing day-equal weights", flush=True)
    counts = np.diff(offsets)
    days = counts.size
    observation_weights = np.repeat(1.0 / counts, counts)
    # Avoid a large BLAS vector dot: the available PyArrow-compatible Conda
    # environment has an unstable BLAS path for million-element vectors.
    mean0 = float(np.sum(observation_weights * values) / days)
    variance0 = float(
        np.sum(observation_weights * ((values - mean0) ** 2)) / days
    )
    print(
        f"static reference initialization: mean={mean0:.9g} variance={variance0:.9g}",
        flush=True,
    )

    objective_calls = 0

    def objective(state: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal objective_calls
        objective_calls += 1
        if objective_calls <= 3:
            print(f"static objective call {objective_calls}: begin", flush=True)
        try:
            contribution = 0.0
            gradient = np.zeros(4)
            # Chunking matches the day-equal objective exactly and avoids the
            # large temporary arrays created by evaluating all 0.8m training
            # observations at once on Windows.
            chunk_size = 50_000
            for start in range(0, values.size, chunk_size):
                stop = min(start + chunk_size, values.size)
                log_density, score = reference_logpdf_and_score(
                    values[start:stop], state
                )
                chunk_weights = observation_weights[start:stop]
                contribution += float(np.sum(chunk_weights * log_density))
                gradient += np.sum(chunk_weights[:, None] * score, axis=0)
            value = -contribution
            gradient = -gradient
        except (FloatingPointError, ValueError, OverflowError):
            return 1e100, np.zeros(4)
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            return 1e100, np.zeros(4)
        if objective_calls <= 3:
            print(
                f"static objective call {objective_calls}: value={value:.12g}",
                flush=True,
            )
        return value, np.asarray(gradient, dtype=float)

    starts = [
        np.r_[mean0, np.log(variance0), parameters_to_state(eta, skew)]
        for eta in (3.5, 5.0, 8.0, 15.0)
        for skew in (-0.12, 0.0, 0.12)
    ]
    bounds = [(-0.10, 0.10), (-20.0, 0.0), (-12.0, 12.0), (-12.0, 12.0)]
    fits = []
    for index, start in enumerate(starts):
        print(f"static reference start {index + 1}/{len(starts)}", flush=True)
        fit = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
        )
        print(
            f"static reference start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all static four-state reference starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    anchor = np.asarray(optimum.x, dtype=float)
    eta, skew, _, _ = state_to_parameters(anchor[2:])
    return anchor, {
        "mean": float(anchor[0]),
        "log_scale_squared": float(anchor[1]),
        "scale": float(np.exp(0.5 * anchor[1])),
        "eta": eta,
        "lambda": skew,
        "state": anchor.tolist(),
        "composite_log_likelihood": float(-optimum.fun),
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(
            max(float(fit.fun) for fit in successful) - float(optimum.fun)
        ),
    }


def fit_dynamic_reference(
    values: np.ndarray, offsets: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, dict[str, object], FourStateFilter]:
    static = filter_reference(values, offsets, anchor, np.zeros((3, 4)))
    averages = fixed_denominator_score_averages(static.scores, WINDOWS)
    target_steps = np.array([1e-4, 1e-2, 1e-2, 1e-2])
    units = np.array(
        [
            [
                target_steps[coordinate]
                / max(float(np.median(np.abs(average[:, coordinate]))), 1e-12)
                for coordinate in range(4)
            ]
            for average in averages
        ]
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(3, 4) * units

    def run(theta: np.ndarray) -> FourStateFilter:
        return filter_reference(values, offsets, anchor, unpack(theta))

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, ValueError, OverflowError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    rng = np.random.default_rng(SEED)
    starts = [
        np.zeros(12),
        np.ones(12),
        -np.ones(12),
        rng.normal(size=12),
        3.0 * rng.normal(size=12),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 12,
            options={
                "maxiter": 4000,
                "ftol": 1e-14,
                "gtol": 1e-8,
                "eps": 1e-5,
                "maxls": 80,
            },
        )
        print(
            f"four-state reference start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        best = min(fits, key=lambda fit: float(fit.fun))
        raise RuntimeError(f"dynamic reference optimization failed: {best.message}")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    if np.max(np.abs(theta)) > 499.9:
        raise RuntimeError("dynamic reference optimum hit the scaled bound")
    betas = unpack(theta)
    dynamic = run(theta)
    return betas, {
        "betas": {
            horizon: {
                name: float(betas[index, coordinate])
                for coordinate, name in enumerate(
                    ("location", "log_scale_squared", "eta", "lambda")
                )
            }
            for index, horizon in enumerate(HORIZONS)
        },
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
        "maximum_scaled_parameter_fraction_of_bound": float(
            np.max(np.abs(theta)) / 500.0
        ),
    }, dynamic


def dynamic_reference_pit(
    values: np.ndarray, offsets: np.ndarray, states: np.ndarray
) -> np.ndarray:
    output = np.empty_like(np.asarray(values, dtype=float))
    if states.shape != (offsets.size - 1, 4):
        raise ValueError("reference path does not align with daily offsets")
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        mean, log_variance = states[day, :2]
        z = (values[start:stop] - mean) / np.exp(0.5 * log_variance)
        output[start:stop] = hansen_cdf(z, states[day, 2:])
    return np.clip(output, 1e-12, 1.0 - 1e-12)


def four_tangent_geometry(
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    quadrature_order: int = 128,
) -> EfficientGeometry:
    nodes0, weights0 = np.polynomial.legendre.leggauss(quadrature_order)
    q_pieces: list[np.ndarray] = []
    tangent_pieces: list[np.ndarray] = []
    weight_pieces: list[np.ndarray] = []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        if right <= left:
            continue
        u = (left + right) / 2.0 + (right - left) * nodes0 / 2.0
        weight = weights0 * (right - left) / 2.0
        phi = normalized_bspline_basis(u, knots, DEGREE, masses)
        responsibility = phi * masses[None, :]
        q_pieces.append((responsibility - masses[None, :]) @ helmert)
        z = hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_pieces.append(reference_logpdf_and_score(y, anchor)[1])
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
    eigenvalues, eigenvectors = np.linalg.eigh(residual_information)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive four-tangent residual information: {eigenvalues}")
    # Deliberately do not Fisher-whiten the residual score.  Direction-specific
    # whitening is not a coordinate-invariant operation under the scalar
    # even/odd response restriction and can amplify near-null directions.
    inverse_sqrt = np.eye(masses.size - 1)
    reversal = np.fliplr(np.eye(masses.size))
    reflection = helmert.T @ reversal @ helmert
    identity = np.eye(masses.size - 1)
    plus = (identity + reflection) / 2.0
    minus = (identity - reflection) / 2.0
    post_cross = (efficient.T * weight) @ tangent
    metric = centered_clr_metric(knots, DEGREE, masses, helmert)
    return EfficientGeometry(
        projection_q_on_hansen=projection,
        residual_information=residual_information,
        inverse_sqrt=inverse_sqrt,
        projector_plus=plus,
        projector_minus=minus,
        diagnostics={
            "projection": "fixed-anchor location, log-scale-squared, eta, lambda tangent",
            "residual_score_whitening": False,
            "nuisance_dimension": 4,
            "quadrature_order_per_knot_interval": quadrature_order,
            "tangent_information_eigenvalues_scaled": np.linalg.eigvalsh(
                information_tangent_scaled
            ).tolist(),
            "tangent_information_condition_number_unscaled": float(
                np.linalg.cond(information_tangent)
            ),
            "information_q_vs_locked_metric_max_abs_error": float(
                np.max(np.abs(information_q - metric))
            ),
            "residual_information_eigenvalues": eigenvalues.tolist(),
            "residual_information_condition_number": float(
                eigenvalues[-1] / eigenvalues[0]
            ),
            "maximum_post_projection_cross_information": float(
                np.max(np.abs(post_cross))
            ),
            "plus_idempotence_error": float(np.max(np.abs(plus @ plus - plus))),
            "minus_idempotence_error": float(np.max(np.abs(minus @ minus - minus))),
            "plus_minus_orthogonality_error": float(np.max(np.abs(plus @ minus))),
        },
    )


def score_oos(
    values: np.ndarray,
    offsets: np.ndarray,
    dates: pd.DatetimeIndex,
    training_values: np.ndarray,
    reference: FourStateFilter,
    correction,
    knots: np.ndarray,
    masses: np.ndarray,
    reference_model_name: str = "four_state_dynamic_hansen",
    proposed_model_name: str = "four_state_dynamic_hansen_orthogonal_parity_arc_sm",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    u_grid, component_cdf = component_cdfs(knots, masses)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    bin_edges = {
        count: np.concatenate(
            ([-np.inf], np.quantile(training_values, np.arange(1, count) / count), [np.inf])
        )
        for count in BIN_COUNTS
    }
    thresholds = {
        alpha: float(np.quantile(training_values, alpha)) for alpha in TAIL_LEVELS
    }
    rows: list[dict[str, object]] = []
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if day % 50 == 0:
            print(f"scoring OOS day {day + 1}/{len(dates)}", flush=True)
        observations = values[start:stop]
        state = reference.states[day]
        mean, log_variance = state[:2]
        sigma = float(np.exp(0.5 * log_variance))
        shape_state = state[2:]
        base_nodes = mean + sigma * hansen_ppf(probability_nodes, shape_state)
        mixture_cdf = np.maximum.accumulate(component_cdf @ correction.weights[day])
        mixture_cdf[0], mixture_cdf[-1] = 0.0, 1.0
        corrected_u = np.interp(probability_nodes, mixture_cdf, u_grid)
        corrected_nodes = mean + sigma * hansen_ppf(corrected_u, shape_state)
        models = (
            (
                reference_model_name,
                base_nodes,
                None,
                float(reference.contributions[day]),
            ),
            (
                proposed_model_name,
                corrected_nodes,
                mixture_cdf,
                float(reference.contributions[day] + correction.contributions[day]),
            ),
        )
        for model, ordered_sample, correction_cdf, continuous_log in models:
            row: dict[str, object] = {
                "date": str(dates[day].date()),
                "model": model,
                "n_constituents": int(stop - start),
                "continuous_log_score": continuous_log,
                "crps": score_ordered_predictive_sample(ordered_sample, observations),
            }
            for count in BIN_COUNTS:
                edges = bin_edges[count]
                base_u = np.concatenate(
                    (
                        [0.0],
                        hansen_cdf((edges[1:-1] - mean) / sigma, shape_state),
                        [1.0],
                    )
                )
                model_cdf = (
                    base_u
                    if correction_cdf is None
                    else np.interp(base_u, u_grid, correction_cdf)
                )
                row[f"categorical_log_score_{count}"] = categorical_log_score(
                    np.maximum(np.diff(model_cdf), 0.0), observations, edges
                )
            for alpha in TAIL_LEVELS:
                quantile = float(np.interp(alpha, probability_nodes, ordered_sample))
                base_probability = float(
                    hansen_cdf(np.array([(thresholds[alpha] - mean) / sigma]), shape_state)[0]
                )
                probability = (
                    base_probability
                    if correction_cdf is None
                    else float(np.interp(base_probability, u_grid, correction_cdf))
                )
                row[f"q_loss_{alpha:g}"] = q_loss(observations, quantile, alpha)
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
    summary = scores.groupby("model", as_index=False)[score_names].mean()
    comparisons = [
        score_comparison(
            scores,
            proposed_model_name,
            reference_model_name,
            score,
        )
        for score in score_names
    ]
    return scores, summary, comparisons


def write_report(result: dict[str, object]) -> None:
    fit = result["residual_parity_arc_sm"]
    comparisons = pd.DataFrame(result["oos_comparisons"])
    headline = comparisons.loc[
        comparisons["score"].isin(("continuous_log_score", "crps"))
    ]
    lines = [
        "# Four-state dynamic Hansen reference + orthogonal parity DCD-SM",
        "",
        "## Specification",
        "",
        "- Raw S&P 500 constituent log returns; one day-equal composite likelihood contribution per date.",
        "- Reference states: location, log scale-squared, Hansen eta, and Hansen lambda.",
        "- Each state uses its own score-HAR(1,5,22) coefficients.",
        "- The decile-spline score is projected off all four fixed-anchor score tangents.",
        "- Neither the four reference scores nor the residual DCD-SM score is Fisher-whitened.",
        "- DCD-SM uses separate even and odd HAR(1,5,22) coefficients.",
        "",
        "## Static anchor",
        "",
        "| location | scale | eta | lambda |",
        "|---:|---:|---:|---:|",
        "| {mean:.9g} | {scale:.9g} | {eta:.9g} | {lambda:.9g} |".format(
            **result["static_reference"]
        ),
        "",
        "## Dynamic reference coefficients",
        "",
        "| state | horizon | estimate |",
        "|---|---:|---:|",
    ]
    for horizon in HORIZONS:
        for state_name, estimate in result["dynamic_reference"]["betas"][horizon].items():
            lines.append(f"| {state_name} | {horizon} | {estimate:.9g} |")
    lines.extend(
        [
        "",
        "The four-state reference improves training composite log likelihood by "
        f"`{result['dynamic_reference']['gain_vs_static']:.9g}` over its static anchor.",
        "",
        "## DCD-SM coefficients",
        "",
        "| block | horizon | estimate | HAC(22) p |",
        "|---|---:|---:|---:|",
        ]
    )
    estimates = [*fit["kappa_plus"], *fit["kappa_minus"]]
    p_values = fit["p_two_sided_conditional_on_hansen"]
    for block, horizon, estimate, p_value in zip(
        ["even"] * 3 + ["odd"] * 3,
        [*WINDOWS, *WINDOWS],
        estimates,
        p_values,
    ):
        lines.append(f"| {block} | {horizon} | {estimate:.9g} | {p_value:.6g} |")
    lines.extend(
        [
            "",
            f"Joint six-coefficient Wald p-value: `{fit['joint_all_p']:.6g}`.",
            f"Joint odd-block Wald p-value: `{fit['joint_odd_p']:.6g}`.",
            "",
            "## Strict-OOS comparison",
            "",
            "Positive improvement favors DCD-SM over the four-state reference.",
            "",
            "| score | improvement | HAC(22) p |",
            "|---|---:|---:|",
        ]
    )
    for row in headline.to_dict(orient="records"):
        lines.append(
            f"| {row['score']} | {row['improvement_mean']:.9g} | "
            f"{row['hac22_two_sided_p']:.6g} |"
        )
    condition = result["residual_specification"]["geometry"][
        "residual_information_condition_number"
    ]
    log_row = headline.loc[headline["score"] == "continuous_log_score"].iloc[0]
    crps_row = headline.loc[headline["score"] == "crps"].iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation and numerical caution",
            "",
            f"The six DCD-SM coefficients are not jointly significant "
            f"(`p={fit['joint_all_p']:.6g}`), and none is individually significant "
            "at 5%.  Thus the unwhitened specification does not find training-sample "
            "evidence of residual dynamicity after the four reference states.",
            "",
            f"Strict-OOS continuous log score improves by "
            f"`{log_row['improvement_mean']:.9g}` (`p={log_row['hac22_two_sided_p']:.6g}`) "
            f"and CRPS improves by `{crps_row['improvement_mean']:.9g}` "
            f"(`p={crps_row['hac22_two_sided_p']:.6g}`).  These are weak positive "
            "forecasting signs, not conventional 5% evidence.",
            "",
            f"The unwhitened residual Fisher condition number is `{condition:.6g}`.  "
            "It is reported as a geometry diagnostic only: because no inverse-Fisher "
            "whitening is applied, its small eigenvalues are not amplified in the "
            "DCD-SM update.",
        ]
    )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="reuse the saved static anchor and dynamic reference coefficients",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = validate_reference_score()
    print(f"reference score validation: {validation}", flush=True)
    panel = load_raw_panel()
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows

    if args.reuse_reference:
        saved_path = OUTPUT / "estimation.json"
        if not saved_path.exists():
            raise FileNotFoundError(f"cannot reuse missing {saved_path}")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        static_summary = saved["static_reference"]
        dynamic_summary = saved["dynamic_reference"]
        anchor = np.asarray(static_summary["state"], dtype=float)
        state_names = ("location", "log_scale_squared", "eta", "lambda")
        betas = np.array(
            [
                [dynamic_summary["betas"][horizon][name] for name in state_names]
                for horizon in HORIZONS
            ],
            dtype=float,
        )
        print("reusing saved four-state reference", flush=True)
        reference_train = filter_reference(
            panel.values[:training_rows], train_offsets, anchor, betas
        )
    else:
        print("fitting static four-parameter Hansen reference", flush=True)
        anchor, static_summary = fit_static_reference(
            panel.values[:training_rows], train_offsets
        )
        print(f"static reference: {static_summary}", flush=True)
        print("fitting diagonal four-state score-HAR reference", flush=True)
        betas, dynamic_summary, reference_train = fit_dynamic_reference(
            panel.values[:training_rows], train_offsets, anchor
        )
    reference_oos = filter_reference(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        initial_state=reference_train.terminal_state,
        initial_score_history=reference_train.terminal_score_history,
    )
    reference_full = filter_reference(panel.values, panel.offsets, anchor, betas)
    reference_continuation_error = float(
        max(
            np.max(np.abs(reference_full.states[n_train:] - reference_oos.states)),
            np.max(
                np.abs(
                    reference_full.contributions[n_train:]
                    - reference_oos.contributions
                )
            ),
        )
    )
    if reference_continuation_error > 2e-12:
        raise RuntimeError("four-state reference continuation check failed")

    u = dynamic_reference_pit(panel.values, panel.offsets, reference_full.states)
    knots = make_clamped_knots(BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    helmert = helmert_submatrix(masses.size)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)
    geometry = four_tangent_geometry(anchor, knots, masses, helmert)
    print(f"four-tangent geometry: {geometry.diagnostics}", flush=True)

    print("fitting six-coefficient orthogonal parity DCD-SM", flush=True)
    fit, q_zero_train, correction_train = fit_residual_parity6_dcd(
        phi[:training_rows],
        train_offsets,
        reference_train.scores,
        masses,
        helmert,
        geometry,
        nested_even_5_22=np.zeros(2),
        numerical_state_step=1e-2,
        adaptive_hessian=True,
    )
    kappa_plus = np.asarray(fit["kappa_plus"], dtype=float)
    kappa_minus = np.asarray(fit["kappa_minus"], dtype=float)
    correction_oos = filter_residual_parity6_dcd(
        phi[training_rows:],
        oos_offsets,
        reference_oos.scores,
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
        initial_state=correction_train.terminal_state,
        initial_score_history=correction_train.terminal_score_history,
    )
    correction_full = filter_residual_parity6_dcd(
        phi,
        panel.offsets,
        reference_full.scores,
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
    )
    correction_continuation_error = float(
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
    q_zero_error = float(
        max(
            np.max(np.abs(q_zero_train.states)),
            np.max(np.abs(q_zero_train.contributions)),
        )
    )
    if max(correction_continuation_error, q_zero_error) > 2e-12:
        raise RuntimeError("DCD-SM continuation or nesting check failed")

    scores, score_summary, comparisons = score_oos(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        panel.values[:training_rows],
        reference_oos,
        correction_oos,
        knots,
        masses,
    )
    parameters = pd.DataFrame(
        {
            "parameter": [
                *(f"even_{window}" for window in WINDOWS),
                *(f"odd_{window}" for window in WINDOWS),
            ],
            "estimate": [*fit["kappa_plus"], *fit["kappa_minus"]],
            "hac22_se_conditional_on_reference": fit[
                "se_godambe_hac22_conditional_on_hansen"
            ],
            "t_conditional_on_reference": fit[
                "t_godambe_hac22_conditional_on_hansen"
            ],
            "p_conditional_on_reference": fit[
                "p_two_sided_conditional_on_hansen"
            ],
        }
    )
    parameters.to_csv(OUTPUT / "arc_sm_parameters.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "sample": np.where(np.arange(len(dates)) < n_train, "training", "oos"),
            "location": reference_full.means,
            "log_scale_squared": reference_full.log_variances,
            "scale": np.sqrt(reference_full.variances),
            "eta": reference_full.eta,
            "lambda": reference_full.skew,
            **{
                f"reference_score_{name}": reference_full.scores[:, index]
                for index, name in enumerate(
                    ("location", "log_scale_squared", "eta", "lambda")
                )
            },
        }
    ).to_csv(OUTPUT / "reference_daily_path.csv", index=False)
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    score_summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "oos_comparisons.csv", index=False)
    result = {
        "analysis": "raw-return four-state dynamic Hansen reference plus four-tangent orthogonal parity DCD-SM",
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
        "static_reference": static_summary,
        "dynamic_reference": dynamic_summary,
        "residual_specification": {
            "breakpoints": BREAKPOINTS.tolist(),
            "degree": DEGREE,
            "windows": list(WINDOWS),
            "parity_coefficients": 6,
            "projection": "location/log-scale-squared/eta/lambda at fixed training anchor",
            "score_scaling": "none after four-tangent projection",
            "geometry": geometry.diagnostics,
        },
        "residual_parity_arc_sm": fit,
        "oos_score_summary": score_summary.to_dict(orient="records"),
        "oos_comparisons": comparisons,
        "checks": {
            "reference_score_gradient": validation,
            "q_zero_exact_nesting_max_abs_error": q_zero_error,
            "reference_training_oos_continuation_max_abs_error": reference_continuation_error,
            "arc_sm_training_oos_continuation_max_abs_error": correction_continuation_error,
            "oos_minimum_mixture_weight": float(correction_oos.weights.min()),
            "oos_maximum_absolute_q": float(np.max(np.abs(correction_oos.states))),
        },
        "inference_scope": "DCD-SM Godambe-HAC(22) conditional on the estimated four-state reference path",
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
