"""Estimate a one-day composite-score DCD for the S&P 500 cross section.

There is one univariate standardized-return density state per trading day:

    q[t + 1] = q0 + kappa * s[t].

The score is the derivative of the average marginal log density across all
point-in-time S&P 500 constituents observed on date t.  This is a conditional
composite likelihood; no within-day cross-sectional independence claim is
made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))

from dcd_model import (  # noqa: E402
    DEFAULT_BREAKPOINTS,
    bspline_masses,
    centered_clr_metric,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
    weights_to_q,
)
from composite_model import (  # noqa: E402
    filter_composite_score_arch,
    fit_weighted_static_map_em,
)


SOURCE = REPO / "data" / "weekly_density_em" / "daily_returns" / "usa.parquet"
OUTPUT = ROOT / "output"
TRAINING_END = pd.Timestamp("2023-12-29")
DEGREE = 3
PRIOR_STRENGTH = 1.0


def newey_west_long_run_variance(values: np.ndarray, bandwidth: int) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    output = float(np.dot(centered, centered))
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        covariance_sum = float(np.dot(centered[lag:], centered[:-lag]))
        output += 2.0 * weight * covariance_sum
    return max(output, 0.0)


def load_and_standardize() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    returns = pd.read_parquet(
        SOURCE,
        columns=["date", "firm_id", "ticker", "log_return"],
    )
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[np.isfinite(returns["log_return"])].copy()
    returns = returns.sort_values(["date", "firm_id"]).reset_index(drop=True)
    if returns.duplicated(["date", "firm_id"]).any():
        raise ValueError("duplicate point-in-time constituent-date observations")

    grouped = returns.groupby("date", sort=True)["log_return"]
    location = grouped.transform("mean").to_numpy(dtype=float)
    centered = returns["log_return"].to_numpy(dtype=float) - location
    returns["centered_squared"] = centered**2
    variance = returns.groupby("date", sort=True)["centered_squared"].transform(
        "mean"
    ).to_numpy(dtype=float)
    scale = np.sqrt(variance)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("nonpositive S&P 500 daily cross-sectional scale")
    standardized = centered / scale
    returns["cross_sectional_location"] = location
    returns["cross_sectional_scale"] = scale
    returns["standardized_return"] = standardized
    returns = returns.drop(columns=["centered_squared"])

    daily = returns.groupby("date", sort=True).agg(
        n_constituents=("firm_id", "size"),
        n_unique_constituents=("firm_id", "nunique"),
        cross_sectional_location=("cross_sectional_location", "first"),
        cross_sectional_scale=("cross_sectional_scale", "first"),
        standardized_mean=("standardized_return", "mean"),
        standardized_second_moment=(
            "standardized_return",
            lambda values: float(np.mean(np.asarray(values) ** 2)),
        ),
    ).reset_index()
    if not np.array_equal(
        daily["n_constituents"].to_numpy(),
        daily["n_unique_constituents"].to_numpy(),
    ):
        raise ValueError("constituent counts are not unique within date")
    counts = daily["n_constituents"].to_numpy(dtype=int)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if offsets[-1] != len(returns):
        raise ValueError("daily offsets do not cover the return panel")
    return returns, daily, offsets


def optimize_kappa(
    phi: np.ndarray,
    offsets: np.ndarray,
    q0: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
) -> tuple[dict, object, object, pd.DataFrame]:
    static = filter_composite_score_arch(
        phi, offsets, q0, 0.0, masses, helmert, cholesky
    )
    median_score_norm = float(np.median(np.linalg.norm(static.scores, axis=1)))
    kappa_unit = 1e-3 / max(median_score_norm, 1e-12)

    def run(theta: float, initial_state: np.ndarray | None = None):
        return filter_composite_score_arch(
            phi,
            offsets,
            q0,
            float(theta) * kappa_unit,
            masses,
            helmert,
            cholesky,
            initial_state=initial_state,
        )

    def objective(theta: float) -> float:
        try:
            value = -run(float(theta)).composite_log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    magnitudes = np.logspace(-3.0, 2.0, 16)
    grid = np.unique(np.concatenate((-magnitudes, [0.0], magnitudes)))
    grid.sort()
    grid_objective = np.array([objective(value) for value in grid])
    best = int(np.argmin(grid_objective))
    if best in (0, grid.size - 1):
        raise RuntimeError("composite kappa optimum is on the search boundary")
    optimum = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    if not optimum.success:
        raise RuntimeError(f"composite DCD optimization failed: {optimum.message}")
    theta_hat = float(optimum.x)
    kappa_hat = theta_hat * kappa_unit
    dynamic = run(theta_hat)

    nonnegative = np.flatnonzero(grid >= 0.0)
    constrained_best = int(
        nonnegative[int(np.argmin(grid_objective[nonnegative]))]
    )
    if grid[constrained_best] == 0.0:
        theta_nonnegative = 0.0
        constrained_gain = 0.0
    else:
        constrained = minimize_scalar(
            objective,
            bounds=(
                float(grid[constrained_best - 1]),
                float(grid[constrained_best + 1]),
            ),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        if not constrained.success:
            raise RuntimeError("nonnegative composite DCD optimization failed")
        theta_nonnegative = float(constrained.x)
        constrained_gain = (
            run(theta_nonnegative).composite_log_likelihood
            - static.composite_log_likelihood
        )

    derivative_step = max(1e-4, abs(theta_hat) * 1e-4)
    plus = run(theta_hat + derivative_step)
    minus = run(theta_hat - derivative_step)
    curvature = -(
        plus.composite_log_likelihood
        - 2.0 * dynamic.composite_log_likelihood
        + minus.composite_log_likelihood
    ) / derivative_step**2
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise RuntimeError(f"nonpositive composite curvature: {curvature}")
    day_gradient = (
        plus.contributions - minus.contributions
    ) / (2.0 * derivative_step)
    variability_hac22 = newey_west_long_run_variance(day_gradient, 22)
    se_theta_hac22 = float(np.sqrt(variability_hac22) / curvature)
    se_kappa_hac22 = kappa_unit * se_theta_hac22
    t_hac22 = float(kappa_hat / se_kappa_hac22)
    p_hac22 = float(2.0 * ndtr(-abs(t_hac22)))

    derivative_sensitivity = []
    for check_step in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        check_plus = run(theta_hat + check_step)
        check_minus = run(theta_hat - check_step)
        check_curvature = -(
            check_plus.composite_log_likelihood
            - 2.0 * dynamic.composite_log_likelihood
            + check_minus.composite_log_likelihood
        ) / check_step**2
        check_gradient = (
            check_plus.contributions - check_minus.contributions
        ) / (2.0 * check_step)
        check_variability = newey_west_long_run_variance(check_gradient, 22)
        check_se_kappa = float(
            kappa_unit * np.sqrt(check_variability) / check_curvature
        )
        derivative_sensitivity.append(
            {
                "step_theta": check_step,
                "sensitivity_curvature_theta": float(check_curvature),
                "variability_hac22_theta": float(check_variability),
                "se_kappa_godambe_hac22_conditional": check_se_kappa,
                "t_kappa_godambe_hac22_conditional": float(
                    kappa_hat / check_se_kappa
                ),
            }
        )

    summary = {
        "kappa": kappa_hat,
        "theta_scaled": theta_hat,
        "kappa_unit": kappa_unit,
        "kappa_nonnegative": theta_nonnegative * kappa_unit,
        "nonnegative_composite_gain": constrained_gain,
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_composite_log_likelihood": dynamic.composite_log_likelihood,
        "dynamic_composite_gain": (
            dynamic.composite_log_likelihood - static.composite_log_likelihood
        ),
        "sensitivity_curvature_theta": curvature,
        "variability_hac22_theta": variability_hac22,
        "se_kappa_godambe_hac22_conditional": se_kappa_hac22,
        "t_kappa_godambe_hac22_conditional": t_hac22,
        "p_two_sided_normal_godambe_hac22_conditional": p_hac22,
        "median_static_composite_score_norm": median_score_norm,
        "derivative_step_theta": derivative_step,
        "derivative_sensitivity": derivative_sensitivity,
    }
    likelihood_grid = pd.DataFrame(
        {
            "theta_scaled": grid,
            "kappa": grid * kappa_unit,
            "composite_log_likelihood": -grid_objective,
        }
    )
    return summary, static, dynamic, likelihood_grid


def oos_comparison(
    phi: np.ndarray,
    offsets: np.ndarray,
    q0: np.ndarray,
    kappa: float,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[dict, object, object]:
    static = filter_composite_score_arch(
        phi, offsets, q0, 0.0, masses, helmert, cholesky
    )
    dynamic = filter_composite_score_arch(
        phi,
        offsets,
        q0,
        kappa,
        masses,
        helmert,
        cholesky,
        initial_state=initial_state,
    )
    difference = dynamic.contributions - static.contributions
    hac22 = newey_west_long_run_variance(difference, 22)
    standard_error_mean = float(np.sqrt(hac22) / difference.size)
    mean_gain = float(difference.mean())
    t_value = mean_gain / standard_error_mean if standard_error_mean > 0 else np.nan
    p_value = float(2.0 * ndtr(-abs(t_value))) if np.isfinite(t_value) else np.nan
    return (
        {
            "days": int(difference.size),
            "static_mean_composite_log_score": float(static.contributions.mean()),
            "dynamic_mean_composite_log_score": float(dynamic.contributions.mean()),
            "dynamic_minus_static_mean_log_score": mean_gain,
            "hac22_standard_error_of_mean": standard_error_mean,
            "hac22_t": float(t_value),
            "hac22_two_sided_p": p_value,
            "days_dynamic_better": int(np.sum(difference > 0.0)),
            "days_static_better": int(np.sum(difference < 0.0)),
        },
        static,
        dynamic,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    if n_train <= 22 or n_train >= len(dates):
        raise ValueError("invalid training/OOS split")
    training_rows = int(offsets[n_train])

    u = np.clip(
        ndtr(returns["standardized_return"].to_numpy(dtype=float)),
        1e-12,
        1.0 - 1e-12,
    )
    knots = make_clamped_knots(DEFAULT_BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)
    helmert = helmert_submatrix(masses.size)
    metric = centered_clr_metric(knots, DEGREE, masses, helmert)
    cholesky = np.linalg.cholesky(metric)

    training_counts = np.diff(offsets[: n_train + 1])
    observation_weights = np.repeat(1.0 / training_counts, training_counts)
    static_fit = fit_weighted_static_map_em(
        phi[:training_rows],
        observation_weights,
        prior_mode=masses,
        prior_strength=PRIOR_STRENGTH,
    )
    if not static_fit.converged:
        raise RuntimeError("weighted static MAP-EM did not converge")
    q0 = weights_to_q(static_fit.weights, masses, helmert, cholesky)

    estimation, static_train, dynamic_train, likelihood_grid = optimize_kappa(
        phi[:training_rows],
        offsets[: n_train + 1],
        q0,
        masses,
        helmert,
        cholesky,
    )
    static_reproduction_error = abs(
        static_train.composite_log_likelihood
        - static_fit.composite_log_likelihood
    )
    oos_offsets = offsets[n_train:] - offsets[n_train]
    oos, static_oos, dynamic_oos = oos_comparison(
        phi[training_rows:],
        oos_offsets,
        q0,
        estimation["kappa"],
        masses,
        helmert,
        cholesky,
        initial_state=dynamic_train.terminal_state,
    )

    daily["sample"] = np.where(
        np.arange(len(daily)) < n_train, "training", "oos"
    )
    daily["static_composite_log_score"] = np.concatenate(
        [static_train.contributions, static_oos.contributions]
    )
    daily["dynamic_composite_log_score"] = np.concatenate(
        [dynamic_train.contributions, dynamic_oos.contributions]
    )
    daily["dynamic_minus_static_log_score"] = (
        daily["dynamic_composite_log_score"]
        - daily["static_composite_log_score"]
    )
    daily["dynamic_q_distance_from_q0"] = np.linalg.norm(
        np.vstack([dynamic_train.states, dynamic_oos.states]) - q0[None, :],
        axis=1,
    )
    daily["dynamic_composite_score_norm"] = np.linalg.norm(
        np.vstack([dynamic_train.scores, dynamic_oos.scores]), axis=1
    )
    daily.to_csv(OUTPUT / "daily_composite_filter.csv", index=False)
    likelihood_grid.to_csv(OUTPUT / "kappa_likelihood_grid.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(masses.size),
            "basis_mass": masses,
            "static_weight": static_fit.weights,
        }
    ).to_csv(OUTPUT / "static_weights.csv", index=False)
    np.save(OUTPUT / "q0.npy", q0)
    np.save(OUTPUT / "metric.npy", metric)

    summary = {
        "model": "S&P 500 daily cross-sectional composite-score DCD",
        "recursion": "q[t+1] = q0 + kappa * s[t]",
        "density_state": "one common univariate density state per trading day",
        "source": str(SOURCE.relative_to(REPO)),
        "sample_start": str(dates[0].date()),
        "sample_end": str(dates[-1].date()),
        "training_end": str(TRAINING_END.date()),
        "training_days": n_train,
        "oos_days": int(len(dates) - n_train),
        "total_returns": int(len(returns)),
        "minimum_daily_constituents": int(daily["n_constituents"].min()),
        "median_daily_constituents": float(daily["n_constituents"].median()),
        "maximum_daily_constituents": int(daily["n_constituents"].max()),
        "location_scale": (
            "realized daily cross-sectional population mean and standard deviation"
        ),
        "reference": "standard normal CDF fixed a priori",
        "composite_likelihood": (
            "equal-weight sum over dates of within-date average marginal log densities; "
            "no constituent independence assumption"
        ),
        "basis": "20 normalized cubic B-spline components",
        "metric": "centered clr log-spline Matsuda metric",
        "static_center": {
            "estimator": "equal-day-weighted gamma-one MAP-EM",
            "iterations": int(static_fit.iterations),
            "converged": bool(static_fit.converged),
            "max_weight_change": static_fit.max_weight_change,
            "static_filter_reproduction_error": static_reproduction_error,
        },
        "training_estimation": {
            **estimation,
            "inference_scope": (
                "conditional Godambe sandwich with day-cluster HAC(22), holding "
                "the weighted MAP-EM q0 fixed"
            ),
        },
        "oos_comparison": oos,
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
