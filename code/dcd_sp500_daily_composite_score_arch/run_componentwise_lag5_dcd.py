"""Fit a componentwise lag-5 DCD to normalized S&P 500 cross sections."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, ndtr
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_componentwise_lag5_dcd"
LOCKED_DCD = ROOT / "output_decile_knot_comparison"
FISHER_DCD = ROOT / "output_fixed_fisher_sqrt_scalar_dcd"
HANSEN = ROOT / "output_hansen_skewt_gas_benchmark"
WINDOW = 5
BREAKPOINTS = np.linspace(0.0, 1.0, 11)
LOG_SCORE_BIN_COUNTS = (50, 100, 200)
QUANTILE_NODES = 8192
SEED = 20260901

sys.path.insert(0, str(ROOT))
from estimate import DEGREE, TRAINING_END, load_and_standardize  # noqa: E402
sys.path.insert(0, str(ROOT))
from estimate_score_har import hac_meat, numerical_hessian  # noqa: E402
from run_empirical_reference import (  # noqa: E402
    compare_log_scores,
    empirical_quantile,
    fixed_empirical_pit,
    hac_mean_test,
)
from run_hansen_skewt_gas_benchmark import (  # noqa: E402
    categorical_log_score,
    score_ordered_predictive_sample,
)
from run_unwhitened_scalar_dcd import component_cdfs  # noqa: E402

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))
from dcd_model import (  # noqa: E402
    bspline_masses,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


@dataclass(frozen=True)
class ComponentFilter:
    contributions: np.ndarray
    states: np.ndarray
    component_scores: np.ndarray
    weights: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(self.contributions.sum())


def cluster_density_and_component_score(
    q: np.ndarray,
    component_density: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits = np.log(masses) + helmert @ np.asarray(q, dtype=float)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    fitted = component_density @ weights
    if np.any(~np.isfinite(fitted)) or np.any(fitted <= 0.0):
        raise FloatingPointError("nonpositive fitted density")
    fitted = np.maximum(fitted, 1e-300)
    mean_responsibility = (
        weights * (component_density.T @ (1.0 / fitted))
        / component_density.shape[0]
    )
    component_score = mean_responsibility - weights
    return float(np.mean(np.log(fitted))), component_score, weights


def filter_componentwise_lag5(
    component_density: np.ndarray,
    offsets: np.ndarray,
    kappas: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ComponentFilter:
    phi = np.asarray(component_density, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    kappas = np.asarray(kappas, dtype=float)
    clusters = offsets.size - 1
    dimension = masses.size - 1
    if kappas.shape != masses.shape:
        raise ValueError("one kappa is required for every mixture component")
    states = np.empty((clusters, dimension))
    scores = np.empty((clusters, masses.size))
    weights_path = np.empty_like(scores)
    contributions = np.empty(clusters)
    state = (
        np.zeros(dimension)
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    if initial_score_history is None:
        history = np.empty((0, masses.size))
    else:
        history = np.asarray(initial_score_history, dtype=float).copy()
        if history.ndim != 2 or history.shape[1] != masses.size:
            raise ValueError("initial component-score history has wrong shape")
        history = history[-(WINDOW - 1) :]

    for t, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[t] = state
        contribution, score, weights = cluster_density_and_component_score(
            state, phi[start:stop], masses, helmert
        )
        contributions[t] = contribution
        scores[t] = score
        weights_path[t] = weights
        history = np.vstack((history, score[None, :]))
        average_score = history[-WINDOW:].sum(axis=0) / WINDOW
        state = helmert.T @ (kappas * average_score)
        history = history[-(WINDOW - 1) :]
    return ComponentFilter(
        contributions=contributions,
        states=states,
        component_scores=scores,
        weights=weights_path,
        terminal_state=state,
        terminal_score_history=history,
    )


def fixed_denominator_average(scores: np.ndarray) -> np.ndarray:
    cumulative = np.vstack((np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)))
    output = np.empty_like(scores)
    for t in range(scores.shape[0]):
        start = max(0, t - WINDOW + 1)
        output[t] = (cumulative[t + 1] - cumulative[start]) / WINDOW
    return output


def fit_models(
    phi: np.ndarray,
    offsets: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
) -> tuple[dict[str, object], ComponentFilter, ComponentFilter, ComponentFilter]:
    components = masses.size
    static = filter_componentwise_lag5(
        phi, offsets, np.zeros(components), masses, helmert
    )
    averages = fixed_denominator_average(static.component_scores)
    median_absolute = np.median(np.abs(averages), axis=0)
    units = 1e-3 / np.maximum(median_absolute, 1e-8)

    def run_kappa(kappas: np.ndarray) -> ComponentFilter:
        return filter_componentwise_lag5(phi, offsets, kappas, masses, helmert)

    def scalar_objective(kappa: float) -> float:
        try:
            value = -run_kappa(np.full(components, float(kappa))).composite_log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    scalar = minimize_scalar(
        scalar_objective,
        bounds=(-100.0, 100.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not scalar.success or abs(float(scalar.x)) > 99.99:
        raise RuntimeError(f"scalar lag-5 fit failed or hit boundary: {scalar}")
    scalar_kappa = float(scalar.x)
    scalar_fit = run_kappa(np.full(components, scalar_kappa))

    def run_theta(theta: np.ndarray) -> ComponentFilter:
        return run_kappa(np.asarray(theta, dtype=float) * units)

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run_theta(theta).composite_log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    scalar_start = np.clip(scalar_kappa / units, -450.0, 450.0)
    rng = np.random.default_rng(SEED)
    starts = [
        np.zeros(components),
        scalar_start,
        scalar_start + rng.normal(scale=0.5, size=components),
        rng.normal(size=components),
        5.0 * rng.normal(size=components),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * components,
            options={
                "maxiter": 3000,
                "ftol": 1e-14,
                "gtol": 1e-8,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        print(
            f"component start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.10g}",
            flush=True,
        )
        fits.append(fit)
    optimum = min(fits, key=lambda result: float(result.fun))
    if not optimum.success:
        raise RuntimeError(f"componentwise optimization failed: {optimum.message}")
    theta = np.asarray(optimum.x, dtype=float)
    if np.max(np.abs(theta)) > 499.9:
        raise RuntimeError("componentwise optimum hit scaled search boundary")
    kappas = theta * units
    dynamic = run_theta(theta)

    print("Computing 13-parameter HAC inference", flush=True)
    step = 1e-2
    hessian = numerical_hessian(objective, theta, step)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    if hessian_eigenvalues[0] <= 0.0:
        raise RuntimeError(f"nonpositive component Hessian: {hessian_eigenvalues}")
    gradients = np.empty((offsets.size - 1, components))
    for coordinate in range(components):
        direction = np.zeros(components)
        direction[coordinate] = step
        gradients[:, coordinate] = (
            run_theta(theta + direction).contributions
            - run_theta(theta - direction).contributions
        ) / (2.0 * step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scale = np.diag(units)
    covariance_kappa = scale @ covariance_theta @ scale
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_kappa), 0.0))
    t_values = kappas / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    joint_wald = float(kappas @ np.linalg.solve(covariance_kappa, kappas))

    contrast = np.column_stack((np.eye(components - 1), -np.ones(components - 1)))
    difference = contrast @ kappas
    difference_covariance = contrast @ covariance_kappa @ contrast.T
    equality_wald = float(
        difference @ np.linalg.solve(difference_covariance, difference)
    )
    summary: dict[str, object] = {
        "scalar_lag5_kappa": scalar_kappa,
        "scalar_lag5_composite_log_likelihood": scalar_fit.composite_log_likelihood,
        "component_kappas": kappas.tolist(),
        "component_kappa_units": units.tolist(),
        "component_se_godambe_hac22": standard_errors.tolist(),
        "component_t_godambe_hac22": t_values.tolist(),
        "component_p_two_sided": p_values.tolist(),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "component_composite_log_likelihood": dynamic.composite_log_likelihood,
        "component_gain_vs_static": (
            dynamic.composite_log_likelihood - static.composite_log_likelihood
        ),
        "component_gain_vs_scalar_lag5": (
            dynamic.composite_log_likelihood - scalar_fit.composite_log_likelihood
        ),
        "joint_zero_wald": joint_wald,
        "joint_zero_p": float(chi2.sf(joint_wald, df=components)),
        "equal_kappas_wald": equality_wald,
        "equal_kappas_p": float(chi2.sf(equality_wald, df=components - 1)),
        "hessian_min_eigenvalue": float(hessian_eigenvalues[0]),
        "hessian_condition_number": float(hessian_eigenvalues[-1] / hessian_eigenvalues[0]),
        "optimizer_success": bool(optimum.success),
    }
    return summary, static, scalar_fit, dynamic


def model_weights(states: np.ndarray, masses: np.ndarray, helmert: np.ndarray) -> np.ndarray:
    logits = np.log(masses)[None, :] + np.asarray(states) @ helmert.T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def score_comparison(
    scores: pd.DataFrame, proposed: str, benchmark: str, score: str
) -> dict[str, object]:
    left = scores.loc[scores["model"] == proposed].set_index("date")
    right = scores.loc[scores["model"] == benchmark].set_index("date")
    if not left.index.equals(right.index):
        raise ValueError(f"dates do not align: {proposed}, {benchmark}")
    improvement = (
        right[score] - left[score]
        if score == "crps"
        else left[score] - right[score]
    ).to_numpy(dtype=float)
    mean, standard_error, p_value = hac_mean_test(improvement, 22)
    return {
        "proposed": proposed,
        "benchmark": benchmark,
        "score": score,
        "improvement_mean": mean,
        "hac22_standard_error": standard_error,
        "hac22_two_sided_p": p_value,
        "positive_day_fraction": float(np.mean(improvement > 0.0)),
    }


def main() -> None:
    required = (
        LOCKED_DCD / "oos_composite_log_scores.csv",
        FISHER_DCD / "oos_composite_log_scores.csv",
        FISHER_DCD / "daily_oos_scores_with_benchmarks.csv",
        HANSEN / "daily_oos_scores.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing benchmark outputs: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(offsets[n_train])
    standardized = returns["standardized_return"].to_numpy(dtype=float)
    ordered_reference = np.sort(standardized[:training_rows])
    u = fixed_empirical_pit(ordered_reference, standardized)
    knots = make_clamped_knots(BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    helmert = helmert_submatrix(masses.size)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)

    print("Fitting scalar and componentwise lag-5 DCD", flush=True)
    fit_summary, static_train, scalar_train, component_train = fit_models(
        phi[:training_rows], offsets[: n_train + 1], masses, helmert
    )
    oos_phi = phi[training_rows:]
    oos_offsets = offsets[n_train:] - offsets[n_train]
    static_oos = filter_componentwise_lag5(
        oos_phi, oos_offsets, np.zeros(masses.size), masses, helmert
    )
    scalar_oos = filter_componentwise_lag5(
        oos_phi,
        oos_offsets,
        np.full(masses.size, fit_summary["scalar_lag5_kappa"]),
        masses,
        helmert,
        scalar_train.terminal_state,
        scalar_train.terminal_score_history,
    )
    component_oos = filter_componentwise_lag5(
        oos_phi,
        oos_offsets,
        np.asarray(fit_summary["component_kappas"]),
        masses,
        helmert,
        component_train.terminal_state,
        component_train.terminal_score_history,
    )

    locked_log = pd.read_csv(LOCKED_DCD / "oos_composite_log_scores.csv")
    fisher_log = pd.read_csv(FISHER_DCD / "oos_composite_log_scores.csv")
    log_comparisons = [
        compare_log_scores("component lag5 versus uniform static", component_oos.contributions, static_oos.contributions),
        compare_log_scores("component lag5 versus scalar lag5", component_oos.contributions, scalar_oos.contributions),
        compare_log_scores("component lag5 versus locked DCD-HAR", component_oos.contributions, locked_log["score_har_composite_log_score"].to_numpy()),
        compare_log_scores("component lag5 versus Fisher-sqrt DCD-HAR", component_oos.contributions, fisher_log["fisher_sqrt_scalar_dcd_har"].to_numpy()),
    ]

    print("Scoring OOS componentwise distributions", flush=True)
    component_weights = model_weights(component_oos.states, masses, helmert)
    scalar_weights = model_weights(scalar_oos.states, masses, helmert)
    u_grid, cdfs = component_cdfs(knots, masses)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    bin_edges = {
        count: np.concatenate(([-np.inf], np.quantile(ordered_reference, np.arange(1, count) / count), [np.inf]))
        for count in LOG_SCORE_BIN_COUNTS
    }
    bin_u = {
        count: np.concatenate(([0.0], fixed_empirical_pit(ordered_reference, bin_edges[count][1:-1]), [1.0]))
        for count in LOG_SCORE_BIN_COUNTS
    }
    rows: list[dict[str, object]] = []
    for local_day, target in enumerate(range(n_train, len(dates))):
        if local_day % 50 == 0:
            print(f"Scoring day {local_day + 1}/{len(dates)-n_train}", flush=True)
        start, stop = offsets[target : target + 2]
        observations = standardized[start:stop]
        for label, weights in (
            ("scalar_lag5_dcd", scalar_weights[local_day]),
            ("componentwise_lag5_dcd", component_weights[local_day]),
        ):
            mixture_cdf = np.maximum.accumulate(cdfs @ weights)
            mixture_cdf[0], mixture_cdf[-1] = 0.0, 1.0
            u_nodes = np.interp(probability_nodes, mixture_cdf, u_grid)
            ordered_sample = empirical_quantile(ordered_reference, u_nodes)
            row: dict[str, object] = {
                "date": str(dates[target].date()),
                "model": label,
                "n_constituents": int(stop - start),
                "crps": score_ordered_predictive_sample(ordered_sample, observations),
            }
            for count in LOG_SCORE_BIN_COUNTS:
                bin_cdf = np.interp(bin_u[count], u_grid, mixture_cdf)
                row[f"categorical_log_score_{count}"] = categorical_log_score(
                    np.maximum(np.diff(bin_cdf), 0.0), observations, bin_edges[count]
                )
            rows.append(row)

    benchmarks = pd.read_csv(FISHER_DCD / "daily_oos_scores_with_benchmarks.csv")
    scores = pd.concat((benchmarks, pd.DataFrame(rows)), ignore_index=True)
    score_names = ["crps", *(f"categorical_log_score_{count}" for count in LOG_SCORE_BIN_COUNTS)]
    pairs = (
        ("componentwise_lag5_dcd", "dcd_static"),
        ("componentwise_lag5_dcd", "scalar_lag5_dcd"),
        ("componentwise_lag5_dcd", "dcd_score_har"),
        ("componentwise_lag5_dcd", "fisher_sqrt_scalar_dcd_har"),
        ("componentwise_lag5_dcd", "hansen_diagonal_score_har"),
        ("componentwise_lag5_dcd", "hansen_score_har"),
    )
    comparisons = [
        score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in score_names
    ]
    summary = scores.groupby("model", as_index=False)[score_names].mean()

    pd.DataFrame(
        {
            "component": np.arange(masses.size),
            "basis_mass": masses,
            "kappa": fit_summary["component_kappas"],
            "hac22_se": fit_summary["component_se_godambe_hac22"],
            "t": fit_summary["component_t_godambe_hac22"],
            "p": fit_summary["component_p_two_sided"],
        }
    ).to_csv(OUTPUT / "component_kappas.csv", index=False)
    pd.DataFrame(
        {
            "date": dates[n_train:],
            "uniform_static": static_oos.contributions,
            "scalar_lag5_dcd": scalar_oos.contributions,
            "componentwise_lag5_dcd": component_oos.contributions,
        }
    ).to_csv(OUTPUT / "oos_composite_log_scores.csv", index=False)
    scores.to_csv(OUTPUT / "daily_oos_scores_with_benchmarks.csv", index=False)
    summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "oos_score_comparisons.csv", index=False)

    output = {
        "analysis": "componentwise lag-5 normalized S&P 500 DCD",
        "sample": {
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
        },
        "specification": {
            "q0": "exact zero vector",
            "whitening": False,
            "recursion": "q[t+1] = H' diag(kappa_1,...,kappa_K) avg5(mean_tau[t]-w[t])",
            "fixed_denominator": WINDOW,
            "component_count": int(masses.size),
        },
        "training": fit_summary,
        "oos_composite_log_score_comparisons": log_comparisons,
        "cross_model_scoring": {
            "summary": summary.to_dict(orient="records"),
            "comparisons": comparisons,
        },
        "state_diagnostics": {
            "training_min_weight": float(component_train.weights.min()),
            "oos_min_weight": float(component_oos.weights.min()),
            "training_max_abs_q": float(np.max(np.abs(component_train.states))),
            "oos_max_abs_q": float(np.max(np.abs(component_oos.states))),
        },
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
