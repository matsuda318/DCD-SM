"""Fit simple scalar DCD-HAR geometry variants to normalized S&P 500 returns.

The fixed training empirical CDF is the reference, q0 is exactly zero, and
the natural spline coefficient enters the mixture logits directly.  The
default uses no metric.  ``--score-scaling fisher-sqrt`` applies fixed
symmetric I0^(-1/2) scaling to the raw natural-parameter score:

    w_t = softmax(log(m) + H q_t)
    s_t = H' (mean(tau_t) - w_t)
    q_{t+1} = beta_1 S s_t + beta_5 S avg_5(s)_t + beta_22 S avg_22(s)_t,
    S = I0^(-1/2),

Each beta is a scalar shared by every density direction.  The script compares
the resulting model with its exact uniform-PIT static anchor, the locked
metric-whitened DCD-HAR, and the locked diagonal Hansen skew-t GAS-HAR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CURRENT_DCD_OUTPUT = ROOT / "output_decile_knot_comparison"
HANSEN_OUTPUT = ROOT / "output_hansen_skewt_gas_benchmark"

BREAKPOINTS = np.linspace(0.0, 1.0, 11)
WINDOWS = (1, 5, 22)
LOG_SCORE_BIN_COUNTS = (50, 100, 200)
QUANTILE_NODES = 8192

sys.path.insert(0, str(ROOT))
from composite_model import filter_composite_score_har  # noqa: E402
from estimate import DEGREE, TRAINING_END, load_and_standardize  # noqa: E402
from run_empirical_reference import (  # noqa: E402
    compare_log_scores,
    empirical_quantile,
    fit_score_har,
    fixed_empirical_pit,
    hac_mean_test,
)
from run_hansen_skewt_gas_benchmark import (  # noqa: E402
    categorical_log_score,
    score_ordered_predictive_sample,
)

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))
from dcd_model import (  # noqa: E402
    bspline_masses,
    centered_clr_metric,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


def states_to_weights(
    states: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    inverse_geometry: np.ndarray,
) -> np.ndarray:
    """Map internal states through the selected geometry to mixture weights."""
    natural_states = np.asarray(states) @ inverse_geometry
    logits = np.log(masses)[None, :] + natural_states @ helmert.T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def component_cdfs(
    knots: np.ndarray, masses: np.ndarray, grid_size: int = 8193
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, grid_size)
    basis = normalized_bspline_basis(grid, knots, DEGREE, masses)
    cdfs = cumulative_trapezoid(basis, grid, axis=0, initial=0.0)
    cdfs /= cdfs[-1][None, :]
    cdfs[0] = 0.0
    cdfs[-1] = 1.0
    return grid, cdfs


def proper_score_comparison(
    scores: pd.DataFrame, proposed: str, benchmark: str, score: str
) -> dict[str, object]:
    left = scores.loc[scores["model"] == proposed].set_index("date")
    right = scores.loc[scores["model"] == benchmark].set_index("date")
    if not left.index.equals(right.index):
        raise ValueError(f"score dates do not align: {proposed}, {benchmark}")
    if score == "crps":
        improvement = (right[score] - left[score]).to_numpy(dtype=float)
    else:
        improvement = (left[score] - right[score]).to_numpy(dtype=float)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-scaling",
        choices=("none", "fisher-sqrt"),
        default="none",
        help="fixed anchor scaling applied to the natural-parameter score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fisher_sqrt = args.score_scaling == "fisher-sqrt"
    model_name = (
        "fisher_sqrt_scalar_dcd_har"
        if fisher_sqrt
        else "unwhitened_scalar_dcd_har"
    )
    output_directory = ROOT / (
        "output_fixed_fisher_sqrt_scalar_dcd"
        if fisher_sqrt
        else "output_unwhitened_scalar_dcd"
    )
    required = (
        CURRENT_DCD_OUTPUT / "estimation.json",
        CURRENT_DCD_OUTPUT / "oos_composite_log_scores.csv",
        HANSEN_OUTPUT / "daily_oos_scores.csv",
        HANSEN_OUTPUT / "estimation.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked benchmark artifacts: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)

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
    dimension = masses.size - 1
    metric = centered_clr_metric(knots, DEGREE, masses, helmert)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues[0] <= 0.0:
        raise ValueError(f"anchor Fisher information is not positive: {eigenvalues}")
    # The existing filter uses the same inverse geometry A in the logits and
    # score.  Taking A=I0^(-1/4) makes the natural state a=A'p obey
    # a[t+1]=sum_h beta_h I0^(-1/2) raw_score_h exactly.
    if fisher_sqrt:
        geometry = (
            eigenvectors
            @ np.diag(eigenvalues ** 0.25)
            @ eigenvectors.T
        )
    else:
        geometry = np.eye(dimension)
    inverse_geometry = np.linalg.inv(geometry)
    q0 = np.zeros(dimension)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)
    phi_train = phi[:training_rows]
    offsets_train = offsets[: n_train + 1]

    print(f"Fitting {model_name} DCD-HAR(1,5,22)", flush=True)
    har_summary, static_train, dynamic_train = fit_score_har(
        phi_train,
        offsets_train,
        q0,
        0.0,
        masses,
        helmert,
        geometry,
        theta_bound=500.0,
    )

    oos_phi = phi[training_rows:]
    oos_offsets = offsets[n_train:] - offsets[n_train]
    static_oos = filter_composite_score_har(
        oos_phi,
        oos_offsets,
        q0,
        np.zeros(3),
        WINDOWS,
        masses,
        helmert,
        geometry,
    )
    dynamic_oos = filter_composite_score_har(
        oos_phi,
        oos_offsets,
        q0,
        np.asarray(har_summary["betas"], dtype=float),
        WINDOWS,
        masses,
        helmert,
        geometry,
        initial_state=dynamic_train.terminal_state,
        initial_score_history=dynamic_train.terminal_score_history,
    )

    locked_dcd_log = pd.read_csv(CURRENT_DCD_OUTPUT / "oos_composite_log_scores.csv")
    if not np.array_equal(
        pd.to_datetime(locked_dcd_log["date"]).to_numpy(),
        dates[n_train:].to_numpy(),
    ):
        raise ValueError("locked DCD composite-log dates do not align")
    composite_log_scores = pd.DataFrame(
        {
            "date": dates[n_train:],
            "uniform_static": static_oos.contributions,
            model_name: dynamic_oos.contributions,
            "locked_whitened_dcd_static": locked_dcd_log[
                "static_composite_log_score"
            ].to_numpy(dtype=float),
            "locked_whitened_dcd_har": locked_dcd_log[
                "score_har_composite_log_score"
            ].to_numpy(dtype=float),
        }
    )
    log_comparisons = [
        compare_log_scores(
            f"{model_name} versus uniform static",
            dynamic_oos.contributions,
            static_oos.contributions,
        ),
        compare_log_scores(
            f"{model_name} versus locked whitened DCD-HAR",
            dynamic_oos.contributions,
            locked_dcd_log["score_har_composite_log_score"].to_numpy(dtype=float),
        ),
        compare_log_scores(
            f"{model_name} versus locked whitened DCD static",
            dynamic_oos.contributions,
            locked_dcd_log["static_composite_log_score"].to_numpy(dtype=float),
        ),
    ]

    print("Scoring normalized OOS distributions", flush=True)
    weights = states_to_weights(
        dynamic_oos.states, masses, helmert, inverse_geometry
    )
    u_grid, cdfs = component_cdfs(knots, masses)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    bin_edges = {
        count: np.concatenate(
            (
                [-np.inf],
                np.quantile(ordered_reference, np.arange(1, count) / count),
                [np.inf],
            )
        )
        for count in LOG_SCORE_BIN_COUNTS
    }
    bin_u = {
        count: np.concatenate(
            (
                [0.0],
                fixed_empirical_pit(ordered_reference, bin_edges[count][1:-1]),
                [1.0],
            )
        )
        for count in LOG_SCORE_BIN_COUNTS
    }
    new_rows: list[dict[str, object]] = []
    for local_day, target in enumerate(range(n_train, len(dates))):
        if local_day % 50 == 0:
            print(f"Scoring day {local_day + 1}/{len(dates) - n_train}", flush=True)
        mixture_cdf = np.maximum.accumulate(cdfs @ weights[local_day])
        mixture_cdf[0] = 0.0
        mixture_cdf[-1] = 1.0
        u_nodes = np.interp(probability_nodes, mixture_cdf, u_grid)
        ordered_sample = empirical_quantile(ordered_reference, u_nodes)
        start, stop = offsets[target : target + 2]
        observations = standardized[start:stop]
        row: dict[str, object] = {
            "date": str(dates[target].date()),
            "model": model_name,
            "n_constituents": int(stop - start),
            "crps": score_ordered_predictive_sample(ordered_sample, observations),
        }
        for count in LOG_SCORE_BIN_COUNTS:
            bin_cdf = np.interp(bin_u[count], u_grid, mixture_cdf)
            probabilities = np.maximum(np.diff(bin_cdf), 0.0)
            row[f"categorical_log_score_{count}"] = categorical_log_score(
                probabilities, observations, bin_edges[count]
            )
        new_rows.append(row)

    benchmark_scores = pd.read_csv(HANSEN_OUTPUT / "daily_oos_scores.csv")
    scores = pd.concat((benchmark_scores, pd.DataFrame(new_rows)), ignore_index=True)
    score_names = [
        "crps",
        *(f"categorical_log_score_{count}" for count in LOG_SCORE_BIN_COUNTS),
    ]
    pairs = (
        (model_name, "dcd_static"),
        (model_name, "dcd_score_har"),
        (model_name, "hansen_diagonal_score_har"),
        (model_name, "hansen_score_har"),
        (model_name, "hansen_static"),
    )
    proper_comparisons = [
        proper_score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in score_names
    ]
    summary = scores.groupby("model", as_index=False)[score_names].mean()

    state_frame = pd.DataFrame({"date": dates, "sample": "training"})
    state_frame.loc[n_train:, "sample"] = "oos"
    all_internal_states = np.vstack((dynamic_train.states, dynamic_oos.states))
    all_internal_scores = np.vstack((dynamic_train.scores, dynamic_oos.scores))
    all_states = all_internal_states @ inverse_geometry
    all_scores = all_internal_scores @ inverse_geometry
    for coordinate in range(dimension):
        state_frame[f"q_{coordinate}"] = all_states[:, coordinate]
        state_frame[f"score_{coordinate}"] = all_scores[:, coordinate]

    composite_log_scores.to_csv(
        output_directory / "oos_composite_log_scores.csv", index=False
    )
    scores.to_csv(
        output_directory / "daily_oos_scores_with_benchmarks.csv", index=False
    )
    summary.to_csv(output_directory / "oos_score_summary.csv", index=False)
    pd.DataFrame(proper_comparisons).to_csv(
        output_directory / "oos_score_comparisons.csv", index=False
    )
    state_frame.to_csv(output_directory / "daily_state_and_score.csv", index=False)

    output = {
        "analysis": f"{model_name} normalized S&P 500 DCD-HAR",
        "sample": {
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
        },
        "specification": {
            "reference": "pooled training-only empirical CDF of daily standardized returns",
            "q0": "exact zero vector",
            "static_weights": masses.tolist(),
            "score_scaling": (
                "fixed symmetric I0^(-1/2), where I0 is Fisher information at q=0"
                if fisher_sqrt
                else "none"
            ),
            "logits": "log(m) + H q_natural[t]",
            "raw_score": "H' (mean responsibility[t] - w[t])",
            "recursion": (
                "q_natural[t+1] = beta1*S*s[t] + beta5*S*avg5(s)[t] "
                "+ beta22*S*avg22(s)[t], S=I0^(-1/2)"
                if fisher_sqrt
                else "q_natural[t+1] = beta1*s[t] + beta5*avg5(s)[t] + beta22*avg22(s)[t]"
            ),
            "coefficients": "three horizon-specific scalars shared by all density directions",
            "anchor_fisher_eigenvalues": eigenvalues.tolist(),
            "implementation": (
                "symmetric I0^(-1/4) internal reparameterization; exactly equivalent "
                "to I0^(-1/2) raw-score scaling in the natural density state"
                if fisher_sqrt
                else "natural density state used directly"
            ),
        },
        "training": {
            "uniform_static_composite_log_likelihood": static_train.composite_log_likelihood,
            "score_har": har_summary,
        },
        "oos_composite_log_score_comparisons": log_comparisons,
        "cross_model_scoring": {
            "summary": summary.to_dict(orient="records"),
            "comparisons": proper_comparisons,
        },
        "state_diagnostics": {
            "training_max_abs_natural_q": float(
                np.max(np.abs(dynamic_train.states @ inverse_geometry))
            ),
            "oos_max_abs_natural_q": float(
                np.max(np.abs(dynamic_oos.states @ inverse_geometry))
            ),
            "training_min_weight": float(
                np.min(
                    states_to_weights(
                        dynamic_train.states,
                        masses,
                        helmert,
                        inverse_geometry,
                    )
                )
            ),
            "oos_min_weight": float(np.min(weights)),
        },
    }
    (output_directory / "estimation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
