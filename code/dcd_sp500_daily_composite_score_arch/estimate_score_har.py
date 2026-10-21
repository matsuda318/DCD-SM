"""Estimate the 1/5/22-day S&P 500 composite-score density model."""

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
sys.path.insert(0, str(ROOT))

from estimate import (  # noqa: E402
    DEGREE,
    OUTPUT as BASELINE_OUTPUT,
    TRAINING_END,
    load_and_standardize,
    newey_west_long_run_variance,
)

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))

from dcd_model import (  # noqa: E402
    DEFAULT_BREAKPOINTS,
    bspline_masses,
    centered_clr_metric,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)
from composite_model import (  # noqa: E402
    filter_composite_score_arch,
    filter_composite_score_har,
)
OUTPUT = ROOT / "output_score_har"
WINDOWS = (1, 5, 22)


def hac_meat(gradients: np.ndarray, bandwidth: int) -> np.ndarray:
    centered = gradients - gradients.mean(axis=0, keepdims=True)
    meat = centered.T @ centered
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        covariance = centered[lag:].T @ centered[:-lag]
        meat += weight * (covariance + covariance.T)
    return (meat + meat.T) / 2.0


def numerical_hessian(function, point: np.ndarray, step: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    dimension = point.size
    output = np.empty((dimension, dimension), dtype=float)
    center = float(function(point))
    for i in range(dimension):
        direction = np.zeros(dimension)
        direction[i] = step
        output[i, i] = (
            function(point + direction)
            - 2.0 * center
            + function(point - direction)
        ) / step**2
        for j in range(i):
            other = np.zeros(dimension)
            other[j] = step
            value = (
                function(point + direction + other)
                - function(point + direction - other)
                - function(point - direction + other)
                + function(point - direction - other)
            ) / (4.0 * step**2)
            output[i, j] = value
            output[j, i] = value
    return (output + output.T) / 2.0


def fixed_denominator_score_averages(
    scores: np.ndarray, windows: tuple[int, ...]
) -> list[np.ndarray]:
    output = []
    cumulative = np.vstack((np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)))
    for window in windows:
        averages = np.empty_like(scores)
        for t in range(scores.shape[0]):
            start = max(0, t - window + 1)
            averages[t] = (cumulative[t + 1] - cumulative[start]) / window
        output.append(averages)
    return output


def compare_oos(
    name: str,
    proposed: np.ndarray,
    benchmark: np.ndarray,
) -> dict:
    difference = np.asarray(proposed) - np.asarray(benchmark)
    hac22 = newey_west_long_run_variance(difference, 22)
    standard_error = float(np.sqrt(hac22) / difference.size)
    mean = float(difference.mean())
    t_value = mean / standard_error if standard_error > 0.0 else np.nan
    return {
        "comparison": name,
        "days": int(difference.size),
        "proposed_minus_benchmark_mean_log_score": mean,
        "hac22_standard_error": standard_error,
        "hac22_t": float(t_value),
        "hac22_two_sided_p": (
            float(2.0 * ndtr(-abs(t_value))) if np.isfinite(t_value) else np.nan
        ),
        "days_proposed_better": int(np.sum(difference > 0.0)),
        "days_benchmark_better": int(np.sum(difference < 0.0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(
        (BASELINE_OUTPUT / "estimation.json").read_text(encoding="utf-8")
    )
    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
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
    q0 = np.load(BASELINE_OUTPUT / "q0.npy")
    phi_train = phi[:training_rows]
    offsets_train = offsets[: n_train + 1]

    static_train = filter_composite_score_har(
        phi_train,
        offsets_train,
        q0,
        np.zeros(3),
        WINDOWS,
        masses,
        helmert,
        cholesky,
    )
    score_averages = fixed_denominator_score_averages(
        static_train.scores, WINDOWS
    )
    beta_units = np.array(
        [
            1e-3 / max(float(np.median(np.linalg.norm(values, axis=1))), 1e-12)
            for values in score_averages
        ]
    )

    def run(theta: np.ndarray):
        return filter_composite_score_har(
            phi_train,
            offsets_train,
            q0,
            np.asarray(theta, dtype=float) * beta_units,
            WINDOWS,
            masses,
            helmert,
            cholesky,
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    daily_kappa = float(baseline["training_estimation"]["kappa"])
    daily_start = np.array([daily_kappa / beta_units[0], 0.0, 0.0])
    rng = np.random.default_rng(20260824)
    starts = [
        np.zeros(3),
        daily_start,
        np.ones(3),
        -np.ones(3),
        rng.normal(size=3),
        3.0 * rng.normal(size=3),
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-100.0, 100.0)] * 3,
            options={
                "maxiter": 2_000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        for start in starts
    ]
    optimum = min(fits, key=lambda result: float(result.fun))
    if not optimum.success:
        raise RuntimeError(f"score-HAR optimization failed: {optimum.message}")
    theta_hat = np.asarray(optimum.x, dtype=float)
    betas_hat = theta_hat * beta_units
    dynamic_train = run(theta_hat)

    derivative_step = 1e-3
    hessian = numerical_hessian(objective, theta_hat, derivative_step)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    if hessian_eigenvalues[0] <= 0.0:
        raise RuntimeError(f"nonpositive score-HAR Hessian: {hessian_eigenvalues}")
    gradients = np.empty((n_train, 3), dtype=float)
    for coordinate in range(3):
        direction = np.zeros(3)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta_hat + direction).contributions
            - run(theta_hat - direction).contributions
        ) / (2.0 * derivative_step)
    meat = hac_meat(gradients, 22)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ meat @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scale_matrix = np.diag(beta_units)
    covariance_beta = scale_matrix @ covariance_theta @ scale_matrix
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_beta), 0.0))
    t_values = betas_hat / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    joint_wald = float(
        betas_hat @ np.linalg.inv(covariance_beta) @ betas_hat
    )
    joint_p = float(chi2.sf(joint_wald, df=3))

    sensitivity = []
    for check_step in (3e-2, 1e-2, 3e-3, 1e-3):
        check_hessian = numerical_hessian(objective, theta_hat, check_step)
        check_gradients = np.empty((n_train, 3), dtype=float)
        for coordinate in range(3):
            direction = np.zeros(3)
            direction[coordinate] = check_step
            check_gradients[:, coordinate] = (
                run(theta_hat + direction).contributions
                - run(theta_hat - direction).contributions
            ) / (2.0 * check_step)
        check_inverse = np.linalg.inv(check_hessian)
        check_covariance_theta = (
            check_inverse @ hac_meat(check_gradients, 22) @ check_inverse
        )
        check_covariance_beta = (
            scale_matrix @ check_covariance_theta @ scale_matrix
        )
        check_covariance_beta = (
            check_covariance_beta + check_covariance_beta.T
        ) / 2.0
        check_se = np.sqrt(np.maximum(np.diag(check_covariance_beta), 0.0))
        check_wald = float(
            betas_hat @ np.linalg.inv(check_covariance_beta) @ betas_hat
        )
        sensitivity.append(
            {
                "step_theta": check_step,
                "hessian_min_eigenvalue": float(
                    np.linalg.eigvalsh(check_hessian)[0]
                ),
                "se_beta": check_se.tolist(),
                "joint_wald": check_wald,
                "joint_p": float(chi2.sf(check_wald, df=3)),
            }
        )

    daily_train = filter_composite_score_arch(
        phi_train,
        offsets_train,
        q0,
        daily_kappa,
        masses,
        helmert,
        cholesky,
    )
    oos_phi = phi[training_rows:]
    oos_offsets = offsets[n_train:] - offsets[n_train]
    full_oos = filter_composite_score_har(
        oos_phi,
        oos_offsets,
        q0,
        betas_hat,
        WINDOWS,
        masses,
        helmert,
        cholesky,
        initial_state=dynamic_train.terminal_state,
        initial_score_history=dynamic_train.terminal_score_history,
    )
    static_oos = filter_composite_score_arch(
        oos_phi, oos_offsets, q0, 0.0, masses, helmert, cholesky
    )
    daily_oos = filter_composite_score_arch(
        oos_phi,
        oos_offsets,
        q0,
        daily_kappa,
        masses,
        helmert,
        cholesky,
        initial_state=daily_train.terminal_state,
    )
    oos_comparisons = [
        compare_oos(
            "score-HAR versus static",
            full_oos.contributions,
            static_oos.contributions,
        ),
        compare_oos(
            "score-HAR versus one-day score",
            full_oos.contributions,
            daily_oos.contributions,
        ),
        compare_oos(
            "one-day score versus static",
            daily_oos.contributions,
            static_oos.contributions,
        ),
    ]

    pd.DataFrame(
        {
            "date": dates[n_train:],
            "static_composite_log_score": static_oos.contributions,
            "daily_composite_log_score": daily_oos.contributions,
            "score_har_composite_log_score": full_oos.contributions,
        }
    ).to_csv(OUTPUT / "oos_daily_scores.csv", index=False)
    pd.DataFrame(oos_comparisons).to_csv(
        OUTPUT / "oos_comparisons.csv", index=False
    )
    summary = {
        "model": (
            "q[t] = q0 + beta1*s[t-1] + beta5*avg(s[t-1:t-5]) "
            "+ beta22*avg(s[t-1:t-22])"
        ),
        "reference": "standard normal CDF fixed a priori",
        "training_days": n_train,
        "oos_days": int(len(dates) - n_train),
        "windows": list(WINDOWS),
        "presample": "missing scores are zero; denominators fixed at 1, 5, and 22",
        "beta_units": beta_units.tolist(),
        "training": {
            "static_composite_log_likelihood": static_train.composite_log_likelihood,
            "daily_composite_log_likelihood": daily_train.composite_log_likelihood,
            "score_har_composite_log_likelihood": dynamic_train.composite_log_likelihood,
            "score_har_gain_vs_static": (
                dynamic_train.composite_log_likelihood
                - static_train.composite_log_likelihood
            ),
            "score_har_gain_vs_daily": (
                dynamic_train.composite_log_likelihood
                - daily_train.composite_log_likelihood
            ),
            "betas": betas_hat.tolist(),
            "se_beta_godambe_hac22_conditional": standard_errors.tolist(),
            "t_beta_godambe_hac22_conditional": t_values.tolist(),
            "p_beta_two_sided_normal_conditional": p_values.tolist(),
            "joint_wald_godambe_hac22_conditional": joint_wald,
            "joint_wald_df": 3,
            "joint_wald_p": joint_p,
            "hessian_min_eigenvalue": float(hessian_eigenvalues[0]),
            "hessian_condition_number": float(
                hessian_eigenvalues[-1] / hessian_eigenvalues[0]
            ),
            "optimizer_success": bool(optimum.success),
            "optimizer_message": str(optimum.message),
            "best_of_starts": len(starts),
            "derivative_sensitivity": sensitivity,
            "inference_scope": (
                "conditional Godambe sandwich with day-cluster HAC(22), holding q0 fixed"
            ),
        },
        "oos_comparisons": oos_comparisons,
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
