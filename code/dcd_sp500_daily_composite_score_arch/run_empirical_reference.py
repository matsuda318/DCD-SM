"""Re-estimate the S&P 500 composite-score DCD with an empirical reference.

The reference is the pooled training-only empirical distribution of raw
point-in-time constituent returns.  No same-day location or scale removal is
performed.  One common density state is filtered per trading day using the
within-day average composite score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from composite_model import (  # noqa: E402
    filter_composite_score_arch,
    filter_composite_score_har,
    fit_weighted_static_map_em,
)
from estimate import (  # noqa: E402
    DEGREE,
    PRIOR_STRENGTH,
    SOURCE,
    TRAINING_END,
    newey_west_long_run_variance,
    oos_comparison,
    optimize_kappa,
)
sys.path.insert(0, str(ROOT))
from estimate_score_har import (  # noqa: E402
    WINDOWS,
    fixed_denominator_score_averages,
    hac_meat,
    numerical_hessian,
)

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


OUTPUT = ROOT / "output_empirical_reference"
HAR_THETA_BOUND = 100.0
TAIL_LEVELS = (0.01, 0.05)
MODELS = ("static", "one_day_score", "score_har")
SCORES = ("crps", "q_loss_0.01", "q_loss_0.05", "brier_0.01", "brier_0.05")
QUANTILE_NODES = 8192
CDF_GRID_SIZE = 8193
RAW_LOG_SCORE_BINS = 100
BIN_LOG_SCORE = "raw_bin_log_score_100"


def load_raw_panel() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    returns = pd.read_parquet(
        SOURCE, columns=["date", "firm_id", "ticker", "log_return"]
    )
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[np.isfinite(returns["log_return"])].copy()
    returns = returns.sort_values(["date", "firm_id"]).reset_index(drop=True)
    if returns.duplicated(["date", "firm_id"]).any():
        raise ValueError("duplicate point-in-time constituent-date observations")
    daily = (
        returns.groupby("date", sort=True)
        .agg(
            n_constituents=("firm_id", "size"),
            n_unique_constituents=("firm_id", "nunique"),
            raw_mean=("log_return", "mean"),
            raw_population_scale=("log_return", lambda x: float(np.std(x, ddof=0))),
        )
        .reset_index()
    )
    if not np.array_equal(
        daily["n_constituents"].to_numpy(),
        daily["n_unique_constituents"].to_numpy(),
    ):
        raise ValueError("constituent counts are not unique within date")
    offsets = np.concatenate(
        ([0], np.cumsum(daily["n_constituents"].to_numpy(dtype=int)))
    )
    if offsets[-1] != len(returns):
        raise ValueError("daily offsets do not cover the return panel")
    return returns, daily, offsets


def fixed_empirical_pit(training_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply the deterministic smoothed mid-ECDF used in prior experiments."""
    ordered = np.sort(np.asarray(training_values, dtype=float))
    values = np.asarray(values, dtype=float)
    if ordered.ndim != 1 or values.ndim != 1 or ordered.size == 0:
        raise ValueError("empirical PIT inputs must be nonempty vectors")
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right + 1.0) / (2.0 * (ordered.size + 1.0))


def empirical_quantile(ordered_training: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Linearly interpolate the training empirical quantile function."""
    ordered = np.asarray(ordered_training, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    positions = probabilities * (ordered.size - 1.0)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, ordered.size - 1)
    fraction = positions - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def states_to_weights(
    states: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    inverse_cholesky = np.linalg.inv(cholesky)
    r = np.asarray(states, dtype=float) @ inverse_cholesky
    logits = np.log(masses)[None, :] + r @ helmert.T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def component_cdfs(
    knots: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, CDF_GRID_SIZE)
    basis = normalized_bspline_basis(grid, knots, DEGREE, masses)
    cdfs = cumulative_trapezoid(basis, grid, axis=0, initial=0.0)
    cdfs /= cdfs[-1][None, :]
    cdfs[0] = 0.0
    cdfs[-1] = 1.0
    return grid, cdfs


def fit_score_har(
    phi_train: np.ndarray,
    offsets_train: np.ndarray,
    q0: np.ndarray,
    daily_kappa: float,
    masses: np.ndarray,
    helmert: np.ndarray,
    cholesky: np.ndarray,
    theta_bound: float = 100.0,
) -> tuple[dict, object, object]:
    n_train = offsets_train.size - 1
    static = filter_composite_score_har(
        phi_train,
        offsets_train,
        q0,
        np.zeros(3),
        WINDOWS,
        masses,
        helmert,
        cholesky,
    )
    averages = fixed_denominator_score_averages(static.scores, WINDOWS)
    beta_units = np.array(
        [
            1e-3 / max(float(np.median(np.linalg.norm(x, axis=1))), 1e-12)
            for x in averages
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
            bounds=[(-theta_bound, theta_bound)] * 3,
            options={
                "maxiter": 2000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        for start in starts
    ]
    for index, fit in enumerate(fits):
        print(
            f"HAR start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.10g} theta={np.asarray(fit.x)}",
            flush=True,
        )
    optimum = min(fits, key=lambda result: float(result.fun))
    if not optimum.success:
        raise RuntimeError(f"score-HAR optimization failed: {optimum.message}")
    powell = minimize(
        objective,
        np.asarray(optimum.x, dtype=float),
        method="Powell",
        bounds=[(-theta_bound, theta_bound)] * 3,
        options={"maxiter": 1000, "xtol": 1e-7, "ftol": 1e-13},
    )
    print(
        f"HAR Powell refinement: success={powell.success} "
        f"objective={float(powell.fun):.10g} theta={np.asarray(powell.x)}",
        flush=True,
    )
    candidates = [optimum, powell]
    optimum = min(candidates, key=lambda result: float(result.fun))
    theta = np.asarray(optimum.x, dtype=float)
    betas = theta * beta_units
    dynamic = run(theta)

    derivative_step = 1e-2
    hessian = numerical_hessian(objective, theta, derivative_step)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(f"nonpositive score-HAR Hessian: {eigenvalues}")
    gradients = np.empty((n_train, 3), dtype=float)
    for coordinate in range(3):
        direction = np.zeros(3)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions
            - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scale_matrix = np.diag(beta_units)
    covariance_beta = scale_matrix @ covariance_theta @ scale_matrix
    covariance_beta = (covariance_beta + covariance_beta.T) / 2.0
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_beta), 0.0))
    t_values = betas / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    joint_wald = float(betas @ np.linalg.inv(covariance_beta) @ betas)
    summary = {
        "betas": betas.tolist(),
        "beta_units": beta_units.tolist(),
        "se_beta_godambe_hac22_conditional": standard_errors.tolist(),
        "t_beta_godambe_hac22_conditional": t_values.tolist(),
        "p_beta_two_sided_normal_conditional": p_values.tolist(),
        "joint_wald_godambe_hac22_conditional": joint_wald,
        "joint_wald_p": float(chi2.sf(joint_wald, df=3)),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "score_har_composite_log_likelihood": dynamic.composite_log_likelihood,
        "score_har_gain_vs_static": (
            dynamic.composite_log_likelihood - static.composite_log_likelihood
        ),
        "hessian_min_eigenvalue": float(eigenvalues[0]),
        "hessian_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "best_of_starts": len(starts),
        "inference_scope": (
            "conditional Godambe sandwich with day-cluster HAC(22), holding q0 fixed"
        ),
    }
    return summary, static, dynamic


def hac_mean_test(values: np.ndarray, lags: int = 22) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = values.size
    long_run = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    standard_error = float(np.sqrt(max(long_run, 0.0) / n))
    statistic = values.mean() / standard_error if standard_error > 0.0 else np.nan
    p_value = float(2.0 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return float(values.mean()), standard_error, p_value


def empirical_absolute_difference(ordered: np.ndarray) -> float:
    n = ordered.size
    coefficients = 2.0 * np.arange(1, n + 1) - n - 1.0
    return float(2.0 * np.dot(coefficients, ordered) / (n * n))


def empirical_mean_absolute_error(
    ordered: np.ndarray, observations: np.ndarray
) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(ordered)))
    positions = np.searchsorted(ordered, observations, side="right")
    left = observations * positions - prefix[positions]
    right = prefix[-1] - prefix[positions] - observations * (
        ordered.size - positions
    )
    return (left + right) / ordered.size


def score_predictive_distribution(
    weights: np.ndarray,
    grid: np.ndarray,
    cdfs: np.ndarray,
    ordered_reference: np.ndarray,
    observations: np.ndarray,
    thresholds: dict[float, float],
    threshold_u: dict[float, float],
    log_bin_edges: np.ndarray,
    log_bin_u: np.ndarray,
    log_score_effective_sample_size: int,
) -> dict[str, float]:
    mixture_cdf = np.maximum.accumulate(cdfs @ weights)
    mixture_cdf[0] = 0.0
    mixture_cdf[-1] = 1.0
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    u_nodes = np.interp(probability_nodes, mixture_cdf, grid)
    ordered_samples = empirical_quantile(ordered_reference, u_nodes)
    pairwise = empirical_absolute_difference(ordered_samples)
    crps = (
        empirical_mean_absolute_error(ordered_samples, observations)
        - 0.5 * pairwise
    )
    result = {"crps": float(crps.mean())}
    bin_cdf = np.interp(log_bin_u, grid, mixture_cdf)
    bin_probabilities = np.maximum(np.diff(bin_cdf), 0.0)
    bin_probabilities /= bin_probabilities.sum()
    bin_probabilities = (
        bin_probabilities * log_score_effective_sample_size + 0.5
    ) / (
        log_score_effective_sample_size + 0.5 * bin_probabilities.size
    )
    observation_bins = np.searchsorted(
        log_bin_edges[1:-1], observations, side="right"
    )
    result[BIN_LOG_SCORE] = float(
        np.mean(np.log(bin_probabilities[observation_bins]))
    )
    for alpha in TAIL_LEVELS:
        u_quantile = float(np.interp(alpha, mixture_cdf, grid))
        forecast_quantile = float(
            empirical_quantile(ordered_reference, np.array([u_quantile]))[0]
        )
        residual = observations - forecast_quantile
        q_loss = residual * (alpha - (residual < 0.0).astype(float))
        tail_probability = float(np.interp(threshold_u[alpha], grid, mixture_cdf))
        event = (observations <= thresholds[alpha]).astype(float)
        result[f"q_loss_{alpha:g}"] = float(q_loss.mean())
        result[f"brier_{alpha:g}"] = float(
            np.mean((event - tail_probability) ** 2)
        )
        result[f"var_{alpha:g}"] = forecast_quantile
        result[f"tail_probability_{alpha:g}"] = tail_probability
        result[f"realized_tail_frequency_{alpha:g}"] = float(event.mean())
    return result


def compare_log_scores(
    name: str, proposed: np.ndarray, benchmark: np.ndarray
) -> dict:
    difference = np.asarray(proposed) - np.asarray(benchmark)
    hac = newey_west_long_run_variance(difference, 22)
    standard_error = float(np.sqrt(hac) / difference.size)
    mean = float(difference.mean())
    statistic = mean / standard_error if standard_error > 0.0 else np.nan
    return {
        "comparison": name,
        "mean_gain": mean,
        "hac22_standard_error": standard_error,
        "hac22_t": float(statistic),
        "hac22_two_sided_p": (
            float(2.0 * ndtr(-abs(statistic))) if np.isfinite(statistic) else np.nan
        ),
        "days_proposed_better": int(np.sum(difference > 0.0)),
        "days_benchmark_better": int(np.sum(difference < 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-har",
        action="store_true",
        help="estimate and score only the static and one-day models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    returns, daily, offsets = load_raw_panel()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    if n_train <= max(WINDOWS) or n_train >= len(dates):
        raise ValueError("invalid training/OOS split")
    training_rows = int(offsets[n_train])
    raw = returns["log_return"].to_numpy(dtype=float)
    ordered_reference = np.sort(raw[:training_rows])
    u = fixed_empirical_pit(ordered_reference, raw)

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

    print("Estimating one-day score model", flush=True)
    baseline, static_train, daily_train, likelihood_grid = optimize_kappa(
        phi[:training_rows],
        offsets[: n_train + 1],
        q0,
        masses,
        helmert,
        cholesky,
    )
    oos_phi = phi[training_rows:]
    oos_offsets = offsets[n_train:] - offsets[n_train]
    daily_oos_summary, static_oos, daily_oos = oos_comparison(
        oos_phi,
        oos_offsets,
        q0,
        baseline["kappa"],
        masses,
        helmert,
        cholesky,
        initial_state=daily_train.terminal_state,
    )
    train_dynamic_weights = states_to_weights(
        daily_train.states, masses, helmert, cholesky
    )
    oos_dynamic_weights = states_to_weights(
        daily_oos.states, masses, helmert, cholesky
    )
    one_day_state_diagnostics = {
        "training_minimum_component_weight": float(train_dynamic_weights.min()),
        "training_fraction_component_weights_below_1e-12": float(
            np.mean(train_dynamic_weights < 1e-12)
        ),
        "training_maximum_q_distance_from_q0": float(
            np.max(np.linalg.norm(daily_train.states - q0[None, :], axis=1))
        ),
        "oos_minimum_component_weight": float(oos_dynamic_weights.min()),
        "oos_fraction_component_weights_below_1e-12": float(
            np.mean(oos_dynamic_weights < 1e-12)
        ),
        "oos_maximum_q_distance_from_q0": float(
            np.max(np.linalg.norm(daily_oos.states - q0[None, :], axis=1))
        ),
    }

    har_summary = None
    har_oos = None
    if not args.skip_har:
        print("Estimating score-HAR model", flush=True)
        har_summary, _, har_train = fit_score_har(
            phi[:training_rows],
            offsets[: n_train + 1],
            q0,
            baseline["kappa"],
            masses,
            helmert,
            cholesky,
            theta_bound=HAR_THETA_BOUND,
        )
        har_oos = filter_composite_score_har(
            oos_phi,
            oos_offsets,
            q0,
            np.asarray(har_summary["betas"]),
            WINDOWS,
            masses,
            helmert,
            cholesky,
            initial_state=har_train.terminal_state,
            initial_score_history=har_train.terminal_score_history,
        )
    log_comparisons = [
        compare_log_scores(
            "one-day score versus static",
            daily_oos.contributions,
            static_oos.contributions,
        )
    ]
    if har_oos is not None:
        log_comparisons[0:0] = [
            compare_log_scores(
                "score-HAR versus static",
                har_oos.contributions,
                static_oos.contributions,
            ),
            compare_log_scores(
                "score-HAR versus one-day score",
                har_oos.contributions,
                daily_oos.contributions,
            ),
        ]

    print("Evaluating raw-return proper scores", flush=True)
    static_weights = states_to_weights(q0[None, :], masses, helmert, cholesky)[0]
    daily_weights = states_to_weights(daily_oos.states, masses, helmert, cholesky)
    har_weights = (
        states_to_weights(har_oos.states, masses, helmert, cholesky)
        if har_oos is not None
        else None
    )
    grid, cdfs = component_cdfs(knots, masses)
    thresholds = {
        alpha: float(np.quantile(ordered_reference, alpha)) for alpha in TAIL_LEVELS
    }
    threshold_u = {
        alpha: float(
            fixed_empirical_pit(
                ordered_reference, np.array([thresholds[alpha]])
            )[0]
        )
        for alpha in TAIL_LEVELS
    }
    log_bin_edges = np.concatenate(
        (
            [-np.inf],
            np.quantile(
                ordered_reference,
                np.arange(1, RAW_LOG_SCORE_BINS) / RAW_LOG_SCORE_BINS,
            ),
            [np.inf],
        )
    )
    log_bin_u = np.concatenate(
        (
            [0.0],
            fixed_empirical_pit(ordered_reference, log_bin_edges[1:-1]),
            [1.0],
        )
    )
    log_score_effective_sample_size = 1842 * 32
    rows: list[dict] = []
    for local_day, target in enumerate(range(n_train, len(dates))):
        if local_day % 50 == 0:
            print(f"Scoring OOS day {local_day + 1}/{len(dates) - n_train}", flush=True)
        start, stop = offsets[target : target + 2]
        observations = raw[start:stop]
        model_weights = {
            "static": static_weights,
            "one_day_score": daily_weights[local_day],
        }
        if har_weights is not None:
            model_weights["score_har"] = har_weights[local_day]
        for model in model_weights:
            score = score_predictive_distribution(
                model_weights[model],
                grid,
                cdfs,
                ordered_reference,
                observations,
                thresholds,
                threshold_u,
                log_bin_edges,
                log_bin_u,
                log_score_effective_sample_size,
            )
            rows.append(
                {
                    "date": str(dates[target].date()),
                    "model": model,
                    "n_constituents": int(stop - start),
                    **score,
                }
            )
    scores = pd.DataFrame(rows)
    score_summary = scores.groupby("model", as_index=False)[
        [*SCORES, BIN_LOG_SCORE]
    ].mean()
    comparisons = []
    pairs = [("one_day_score", "static")]
    if har_oos is not None:
        pairs.extend(
            [("score_har", "static"), ("score_har", "one_day_score")]
        )
    for proposed, benchmark in pairs:
        left = scores.loc[scores["model"] == proposed].set_index("date")
        right = scores.loc[scores["model"] == benchmark].set_index("date")
        for score_name in SCORES:
            reduction = right[score_name] - left[score_name]
            mean, standard_error, p_value = hac_mean_test(reduction.to_numpy(), 22)
            benchmark_loss = float(right[score_name].mean())
            comparisons.append(
                {
                    "model": proposed,
                    "benchmark": benchmark,
                    "score": score_name,
                    "model_mean_loss": float(left[score_name].mean()),
                    "benchmark_mean_loss": benchmark_loss,
                    "mean_loss_reduction": mean,
                    "relative_loss_reduction_percent": 100.0 * mean / benchmark_loss,
                    "hac22_standard_error": standard_error,
                    "hac22_two_sided_p": p_value,
                    "positive_day_fraction": float(np.mean(reduction > 0.0)),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)

    likelihood_grid.to_csv(OUTPUT / "kappa_likelihood_grid.csv", index=False)
    scores.to_csv(OUTPUT / "daily_return_scores.csv", index=False)
    score_summary.to_csv(OUTPUT / "score_summary.csv", index=False)
    comparison_frame.to_csv(OUTPUT / "score_comparisons.csv", index=False)
    log_score_columns = {
        "date": dates[n_train:],
        "static_composite_log_score": static_oos.contributions,
        "daily_composite_log_score": daily_oos.contributions,
    }
    if har_oos is not None:
        log_score_columns["score_har_composite_log_score"] = har_oos.contributions
    pd.DataFrame(log_score_columns).to_csv(
        OUTPUT / "oos_composite_log_scores.csv", index=False
    )
    pd.DataFrame(
        {
            "component": np.arange(masses.size),
            "basis_mass": masses,
            "static_weight": static_fit.weights,
        }
    ).to_csv(OUTPUT / "static_weights.csv", index=False)
    np.save(OUTPUT / "q0.npy", q0)
    np.save(OUTPUT / "metric.npy", metric)

    output = {
        "model": "S&P 500 raw-return composite-score DCD",
        "reference": (
            "pooled training-only deterministic smoothed empirical CDF; "
            "no same-day location or scale standardization"
        ),
        "empirical_pit": "(left rank + right rank + 1) / (2 * (N_train + 1))",
        "source": str(SOURCE.relative_to(REPO)),
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(TRAINING_END.date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_returns": training_rows,
            "total_returns": int(len(raw)),
        },
        "composite_likelihood": (
            "equal-weight sum over dates of within-date average log h(u;q); "
            "the common empirical-reference term is omitted"
        ),
        "state_innovation": "none; observation-driven conditional density",
        "basis": (
            f"{masses.size} normalized cubic B-spline components; "
            f"state dimension {masses.size - 1}"
        ),
        "static_center": {
            "iterations": int(static_fit.iterations),
            "converged": bool(static_fit.converged),
            "max_weight_change": static_fit.max_weight_change,
        },
        "one_day_score": {
            "training": baseline,
            "oos": daily_oos_summary,
            "state_diagnostics": one_day_state_diagnostics,
        },
        "score_har": {
            "status": "estimated" if har_summary is not None else "skipped",
            "training": har_summary,
        },
        "oos_composite_log_score_comparisons": log_comparisons,
        "proper_scores": {
            "quantile_nodes": QUANTILE_NODES,
            "cdf_grid_size": CDF_GRID_SIZE,
            "tail_thresholds": {str(k): v for k, v in thresholds.items()},
            "raw_bin_log_score": {
                "name": BIN_LOG_SCORE,
                "bins": RAW_LOG_SCORE_BINS,
                "training_quantile_bins": True,
                "jeffreys_pseudocount": 0.5,
                "effective_sample_size": log_score_effective_sample_size,
            },
            "summary": score_summary.to_dict(orient="records"),
            "comparisons": comparison_frame.to_dict(orient="records"),
        },
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
