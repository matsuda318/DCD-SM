"""Fit an even residual DCD on top of diagonal dynamic Hansen skew-t.

The predictive density is

    g_t(z) = f_H(z; theta_t) h(F_H(z; theta_t); q_t),

where theta_t is the fitted diagonal Hansen GAS-HAR state.  The residual DCD
score is projected off the two-dimensional Hansen score tangent at the static
training anchor.  Only even five- and 22-day residual-score responses are
estimated.  Setting both coefficients to zero gives diagonal Hansen exactly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_dynamic_hansen_residual_dcd"
BREAKPOINTS = np.linspace(0.0, 1.0, 11)
WINDOWS = (5, 22)
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
    model_weights,
)
from run_empirical_reference import (  # noqa: E402
    empirical_quantile,
    hac_mean_test,
)
from run_hansen_skewt_gas_benchmark import (  # noqa: E402
    categorical_log_score,
    fit_diagonal_dynamic_hansen,
    fit_dynamic_hansen,
    fit_static_hansen,
    filter_hansen_score_har,
    hansen_cdf,
    hansen_logpdf_and_score,
    hansen_ppf,
    score_ordered_predictive_sample,
    validate_hansen_implementation,
)

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))
from dcd_model import (  # noqa: E402
    bspline_masses,
    centered_clr_metric,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


@dataclass(frozen=True)
class EfficientGeometry:
    projection_q_on_hansen: np.ndarray
    residual_information: np.ndarray
    inverse_sqrt: np.ndarray
    projector_plus: np.ndarray
    projector_minus: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ResidualFilter:
    contributions: np.ndarray
    states: np.ndarray
    scaled_scores: np.ndarray
    weights: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(self.contributions.sum())


def efficient_score_geometry(
    anchor_hansen: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    quadrature_order: int = 32,
) -> EfficientGeometry:
    """Project the uniform-anchor spline score off the Hansen score tangent."""
    nodes0, weights0 = np.polynomial.legendre.leggauss(quadrature_order)
    q_score_pieces: list[np.ndarray] = []
    hansen_score_pieces: list[np.ndarray] = []
    quadrature_pieces: list[np.ndarray] = []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        if right <= left:
            continue
        u = (left + right) / 2.0 + (right - left) * nodes0 / 2.0
        quadrature_weight = weights0 * (right - left) / 2.0
        phi = normalized_bspline_basis(u, knots, DEGREE, masses)
        responsibility = phi * masses[None, :]
        q_score = (responsibility - masses[None, :]) @ helmert
        z = hansen_ppf(u, anchor_hansen)
        _, hansen_score = hansen_logpdf_and_score(z, anchor_hansen)
        q_score_pieces.append(q_score)
        hansen_score_pieces.append(hansen_score)
        quadrature_pieces.append(quadrature_weight)

    q_score = np.vstack(q_score_pieces)
    hansen_score = np.vstack(hansen_score_pieces)
    quadrature_weight = np.concatenate(quadrature_pieces)
    information_q = (q_score.T * quadrature_weight) @ q_score
    information_hansen = (hansen_score.T * quadrature_weight) @ hansen_score
    cross_information = (q_score.T * quadrature_weight) @ hansen_score
    projection = cross_information @ np.linalg.inv(information_hansen)
    efficient_score = q_score - hansen_score @ projection.T
    residual_information = (
        efficient_score.T * quadrature_weight
    ) @ efficient_score
    residual_information = (residual_information + residual_information.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(residual_information)
    if eigenvalues[0] <= 1e-10:
        raise ValueError(
            f"nonpositive residual Fisher information: {eigenvalues}"
        )
    inverse_sqrt = (
        eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    )

    reversal = np.fliplr(np.eye(masses.size))
    reflection_q = helmert.T @ reversal @ helmert
    identity = np.eye(masses.size - 1)
    plus = (identity + reflection_q) / 2.0
    minus = (identity - reflection_q) / 2.0
    metric = centered_clr_metric(knots, DEGREE, masses, helmert)
    remaining_cross = (
        efficient_score.T * quadrature_weight
    ) @ hansen_score
    diagnostics: dict[str, object] = {
        "quadrature_order_per_knot_interval": quadrature_order,
        "information_q_vs_locked_fisher_max_abs_error": float(
            np.max(np.abs(information_q - metric))
        ),
        "hansen_information_eigenvalues": np.linalg.eigvalsh(
            information_hansen
        ).tolist(),
        "residual_information_eigenvalues": eigenvalues.tolist(),
        "maximum_post_projection_cross_information": float(
            np.max(np.abs(remaining_cross))
        ),
        "plus_idempotence_error": float(np.max(np.abs(plus @ plus - plus))),
        "minus_idempotence_error": float(np.max(np.abs(minus @ minus - minus))),
        "plus_minus_orthogonality_error": float(np.max(np.abs(plus @ minus))),
    }
    return EfficientGeometry(
        projection_q_on_hansen=projection,
        residual_information=residual_information,
        inverse_sqrt=inverse_sqrt,
        projector_plus=plus,
        projector_minus=minus,
        diagnostics=diagnostics,
    )


def dynamic_hansen_pit(
    values: np.ndarray,
    offsets: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Apply each date's predetermined Hansen CDF to its cross section."""
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    if states.shape != (offsets.size - 1, 2):
        raise ValueError("Hansen state path does not align with daily offsets")
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        output[start:stop] = hansen_cdf(values[start:stop], states[day])
    return np.clip(output, 1e-12, 1.0 - 1e-12)


def filter_residual_dcd(
    component_density: np.ndarray,
    offsets: np.ndarray,
    hansen_daily_scores: np.ndarray,
    kappas: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    geometry: EfficientGeometry,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ResidualFilter:
    """Filter even 5/22-day residual DCD scores over a fixed Hansen path."""
    phi = np.asarray(component_density, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    hansen_daily_scores = np.asarray(hansen_daily_scores, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    clusters = offsets.size - 1
    dimension = masses.size - 1
    if hansen_daily_scores.shape != (clusters, 2):
        raise ValueError("daily Hansen scores do not align with DCD clusters")
    if kappas.shape != (2,):
        raise ValueError("weekly and monthly residual coefficients are required")
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
        for kappa, window in zip(kappas, WINDOWS):
            average = history[-window:].sum(axis=0) / window
            state += kappa * (geometry.projector_plus @ average)
        history = history[-21:]
        if np.any(~np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("unstable dynamic-Hansen residual DCD state")
    return ResidualFilter(
        contributions=contributions,
        states=states,
        scaled_scores=scaled_scores,
        weights=weights_path,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_residual_dcd(
    phi: np.ndarray,
    offsets: np.ndarray,
    hansen_daily_scores: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    geometry: EfficientGeometry,
) -> tuple[dict[str, object], ResidualFilter, ResidualFilter]:
    """Estimate the two even residual response coefficients and HAC inference."""
    static = filter_residual_dcd(
        phi,
        offsets,
        hansen_daily_scores,
        np.zeros(2),
        masses,
        helmert,
        geometry,
    )
    if np.max(np.abs(static.contributions)) > 1e-12:
        raise RuntimeError("q=0 residual correction does not nest Hansen exactly")
    averages = fixed_denominator_score_averages(static.scaled_scores, WINDOWS)
    units = np.array(
        [
            1e-3
            / max(
                float(
                    np.median(
                        np.linalg.norm(
                            values @ geometry.projector_plus, axis=1
                        )
                    )
                ),
                1e-12,
            )
            for values in averages
        ]
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float) * units

    def run(theta: np.ndarray) -> ResidualFilter:
        return filter_residual_dcd(
            phi,
            offsets,
            hansen_daily_scores,
            unpack(theta),
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

    rng = np.random.default_rng(SEED)
    starts = [
        np.zeros(2),
        np.ones(2),
        -np.ones(2),
        rng.normal(size=2),
        5.0 * rng.normal(size=2),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 2,
            options={
                "maxiter": 3000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        print(
            f"residual DCD start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.10g} theta={np.asarray(fit.x)}",
            flush=True,
        )
        fits.append(fit)
    optimum = min(fits, key=lambda result: float(result.fun))
    if not optimum.success:
        raise RuntimeError(f"residual DCD optimization failed: {optimum.message}")
    theta = np.asarray(optimum.x, dtype=float)
    if np.max(np.abs(theta)) > 499.9:
        raise RuntimeError("residual DCD optimum hit scaled search boundary")
    kappas = unpack(theta)
    dynamic = run(theta)

    derivative_step = 1e-2
    hessian = numerical_hessian(objective, theta, derivative_step)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    if hessian_eigenvalues[0] <= 1e-10:
        raise RuntimeError(
            f"nonpositive residual DCD Hessian: {hessian_eigenvalues}"
        )
    gradients = np.empty((offsets.size - 1, 2))
    for coordinate in range(2):
        direction = np.zeros(2)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions
            - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units)
    covariance_kappa = scaling @ covariance_theta @ scaling
    covariance_kappa = (covariance_kappa + covariance_kappa.T) / 2.0
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_kappa), 0.0))
    t_values = kappas / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    joint_wald = float(kappas @ np.linalg.solve(covariance_kappa, kappas))
    summary: dict[str, object] = {
        "kappa_5_even": float(kappas[0]),
        "kappa_22_even": float(kappas[1]),
        "parameter_units": units.tolist(),
        "se_godambe_hac22_conditional_on_hansen": standard_errors.tolist(),
        "t_godambe_hac22_conditional_on_hansen": t_values.tolist(),
        "p_two_sided_conditional_on_hansen": p_values.tolist(),
        "joint_zero_wald_conditional_on_hansen": joint_wald,
        "joint_zero_p_conditional_on_hansen": float(chi2.sf(joint_wald, df=2)),
        "static_correction_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_correction_composite_log_likelihood": dynamic.composite_log_likelihood,
        "hessian_eigenvalues_theta": hessian_eigenvalues.tolist(),
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "best_of_starts": len(starts),
    }
    return summary, static, dynamic


def score_comparison(
    scores: pd.DataFrame,
    proposed: str,
    benchmark: str,
    score: str,
) -> dict[str, object]:
    proposed_frame = scores.loc[scores["model"] == proposed].set_index("date")
    benchmark_frame = scores.loc[scores["model"] == benchmark].set_index("date")
    if not proposed_frame.index.equals(benchmark_frame.index):
        raise ValueError("model score dates do not align")
    lower_is_better = score == "crps" or score.startswith(("q_loss", "brier"))
    improvement = (
        benchmark_frame[score] - proposed_frame[score]
        if lower_is_better
        else proposed_frame[score] - benchmark_frame[score]
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


def q_loss(observations: np.ndarray, quantile: float, alpha: float) -> float:
    residual = observations - quantile
    return float(np.mean(residual * (alpha - (residual < 0.0))))


def brier_loss(
    observations: np.ndarray, threshold: float, probability: float
) -> float:
    event = (observations <= threshold).astype(float)
    return float(np.mean((event - probability) ** 2))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = validate_hansen_implementation()
    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(offsets[n_train])
    standardized = returns["standardized_return"].to_numpy(dtype=float)
    train_values = standardized[:training_rows]
    train_offsets = offsets[: n_train + 1]

    print("Fitting static and shared-coefficient Hansen models", flush=True)
    anchor_hansen, static_hansen_summary = fit_static_hansen(
        train_values, train_offsets
    )
    shared_betas, shared_hansen_summary, _, _ = fit_dynamic_hansen(
        train_values, train_offsets, anchor_hansen
    )
    print("Fitting diagonal Hansen GAS-HAR backbone", flush=True)
    diagonal_betas, diagonal_hansen_summary, _, diagonal_train = (
        fit_diagonal_dynamic_hansen(
            train_values, train_offsets, anchor_hansen, shared_betas
        )
    )
    oos_values = standardized[training_rows:]
    oos_offsets = offsets[n_train:] - training_rows
    diagonal_oos = filter_hansen_score_har(
        oos_values,
        oos_offsets,
        anchor_hansen,
        diagonal_betas,
        initial_state=diagonal_train.terminal_state,
        initial_score_history=diagonal_train.terminal_score_history,
    )
    all_hansen_states = np.vstack((diagonal_train.states, diagonal_oos.states))
    all_hansen_scores = np.vstack((diagonal_train.scores, diagonal_oos.scores))

    print("Constructing dynamic Hansen PITs and orthogonal score geometry", flush=True)
    u = dynamic_hansen_pit(standardized, offsets, all_hansen_states)
    knots = make_clamped_knots(BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    helmert = helmert_submatrix(masses.size)
    phi = normalized_bspline_basis(u, knots, DEGREE, masses)
    geometry = efficient_score_geometry(
        anchor_hansen, knots, masses, helmert
    )
    print(f"Efficient-score diagnostics: {geometry.diagnostics}", flush=True)

    print("Fitting two-coefficient even residual DCD", flush=True)
    fit_summary, _, residual_train = fit_residual_dcd(
        phi[:training_rows],
        train_offsets,
        diagonal_train.scores,
        masses,
        helmert,
        geometry,
    )
    residual_oos = filter_residual_dcd(
        phi[training_rows:],
        oos_offsets,
        diagonal_oos.scores,
        np.array(
            [fit_summary["kappa_5_even"], fit_summary["kappa_22_even"]]
        ),
        masses,
        helmert,
        geometry,
        initial_state=residual_train.terminal_state,
        initial_score_history=residual_train.terminal_score_history,
    )
    continuation_states = filter_residual_dcd(
        phi,
        offsets,
        all_hansen_scores,
        np.array(
            [fit_summary["kappa_5_even"], fit_summary["kappa_22_even"]]
        ),
        masses,
        helmert,
        geometry,
    ).states[n_train:]
    continuation_error = float(
        np.max(np.abs(continuation_states - residual_oos.states))
    )
    if continuation_error > 1e-12:
        raise RuntimeError(f"training/OOS continuation error: {continuation_error}")

    print("Scoring strict-OOS predictive distributions", flush=True)
    u_grid, cdfs = component_cdfs(knots, masses)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    ordered_training = np.sort(train_values)
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
    for local_day, target in enumerate(range(n_train, len(dates))):
        if local_day % 50 == 0:
            print(
                f"scoring day {local_day + 1}/{len(dates) - n_train}",
                flush=True,
            )
        start, stop = offsets[target : target + 2]
        observations = standardized[start:stop]
        hansen_state = diagonal_oos.states[local_day]
        weights = residual_oos.weights[local_day]
        mixture_cdf = np.maximum.accumulate(cdfs @ weights)
        mixture_cdf[0], mixture_cdf[-1] = 0.0, 1.0

        residual_u_nodes = np.interp(
            probability_nodes, mixture_cdf, u_grid
        )
        predictive_samples = {
            "diagonal_hansen": hansen_ppf(probability_nodes, hansen_state),
            "dynamic_hansen_residual_dcd": hansen_ppf(
                residual_u_nodes, hansen_state
            ),
        }
        for model, ordered_sample in predictive_samples.items():
            is_residual = model == "dynamic_hansen_residual_dcd"
            row: dict[str, object] = {
                "date": str(dates[target].date()),
                "model": model,
                "n_constituents": int(stop - start),
                "continuous_log_score": float(
                    diagonal_oos.contributions[local_day]
                    + (residual_oos.contributions[local_day] if is_residual else 0.0)
                ),
                "crps": score_ordered_predictive_sample(
                    ordered_sample, observations
                ),
            }
            for count in BIN_COUNTS:
                hansen_bin_u = np.concatenate(
                    (
                        [0.0],
                        hansen_cdf(
                            bin_edges[count][1:-1], hansen_state
                        ),
                        [1.0],
                    )
                )
                model_bin_cdf = (
                    np.interp(hansen_bin_u, u_grid, mixture_cdf)
                    if is_residual
                    else hansen_bin_u
                )
                row[f"categorical_log_score_{count}"] = categorical_log_score(
                    np.maximum(np.diff(model_bin_cdf), 0.0),
                    observations,
                    bin_edges[count],
                )
            for alpha in TAIL_LEVELS:
                model_quantile = float(
                    np.interp(alpha, probability_nodes, ordered_sample)
                )
                hansen_threshold_u = float(
                    hansen_cdf(np.array([thresholds[alpha]]), hansen_state)[0]
                )
                model_probability = (
                    float(np.interp(hansen_threshold_u, u_grid, mixture_cdf))
                    if is_residual
                    else hansen_threshold_u
                )
                row[f"q_loss_{alpha:g}"] = q_loss(
                    observations, model_quantile, alpha
                )
                row[f"brier_{alpha:g}"] = brier_loss(
                    observations, thresholds[alpha], model_probability
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
    comparisons = [
        score_comparison(
            scores,
            "dynamic_hansen_residual_dcd",
            "diagonal_hansen",
            score,
        )
        for score in score_names
    ]
    score_summary = scores.groupby("model", as_index=False)[score_names].mean()

    parameter_frame = pd.DataFrame(
        {
            "parameter": ["kappa_5_even", "kappa_22_even"],
            "estimate": [
                fit_summary["kappa_5_even"],
                fit_summary["kappa_22_even"],
            ],
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
    )
    parameter_frame.to_csv(OUTPUT / "parameters.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "sample": np.where(
                np.arange(len(dates)) < n_train, "training", "oos"
            ),
            "hansen_eta": np.concatenate(
                (diagonal_train.eta, diagonal_oos.eta)
            ),
            "hansen_lambda": np.concatenate(
                (diagonal_train.skew, diagonal_oos.skew)
            ),
        }
    ).to_csv(OUTPUT / "hansen_backbone_states.csv", index=False)
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    score_summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        OUTPUT / "oos_score_comparisons.csv", index=False
    )
    np.save(OUTPUT / "projection_q_on_hansen.npy", geometry.projection_q_on_hansen)
    np.save(OUTPUT / "residual_fisher_information.npy", geometry.residual_information)

    output = {
        "analysis": (
            "two-coefficient even residual DCD over a diagonal dynamic Hansen "
            "skew-t GAS-HAR reference"
        ),
        "sample": {
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
        },
        "density": (
            "g_t(z)=f_H(z;theta_t) h(F_H(z;theta_t);q_t); q_t=0 exactly "
            "nests diagonal Hansen"
        ),
        "hansen_implementation_validation": validation,
        "hansen_static_training": static_hansen_summary,
        "hansen_shared_training": shared_hansen_summary,
        "hansen_diagonal_training": diagonal_hansen_summary,
        "residual_score": {
            "projection": (
                "fixed-anchor Fisher projection off the two-dimensional Hansen "
                "unconstrained-state score tangent"
            ),
            "scaling": "inverse square root of residual Fisher information",
            "parity": "even PIT-reflection subspace only",
            "windows": list(WINDOWS),
            "geometry_diagnostics": geometry.diagnostics,
        },
        "training_residual_dcd": fit_summary,
        "oos_score_summary": score_summary.to_dict(orient="records"),
        "oos_comparisons_vs_diagonal_hansen": comparisons,
        "checks": {
            "training_oos_continuation_max_abs_error": continuation_error,
            "inference_scope": (
                "conditional Godambe-HAC(22), holding the first-stage fitted "
                "Hansen anchor and diagonal dynamic coefficients fixed"
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
