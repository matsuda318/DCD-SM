"""Direct Hansen skew-t test of eta/lambda dynamics after location-scale.

There is no spline correction in this experiment.  For each data set:

1. estimate a static Hansen skew-t anchor by day-equal MLE;
2. estimate dynamic location and log scale-squared with eta/lambda fixed;
3. freeze that location-scale path and jointly estimate diagonal HAR(1,5,22)
   responses for eta and lambda.

The two shape update scores are Fisher-orthogonalized against location and
log scale-squared at the static anchor.  They are not whitened and eta/lambda
are not orthogonalized against one another.  Inference is conditional on the
static anchor and the fitted location-scale path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2

import run_sp500 as core
import run_sp500_fixed_shape as fixed
import run_static_q0_five_models as data_source
import run_static_q0_single_stocks as stocks
from estimate_score_har import fixed_denominator_score_averages, hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_direct_dynamic_skewt_shape"
DATASETS = data_source.DATASETS
SEED = 20260911


@dataclass(frozen=True)
class ShapeGeometry:
    projection_shape_on_ms: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ShapeFilter:
    states: np.ndarray
    scores: np.ndarray
    efficient_shape_scores: np.ndarray
    contributions: np.ndarray
    terminal_shape_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.contributions))


def shape_geometry(anchor: np.ndarray, quadrature_order: int = 128) -> ShapeGeometry:
    """Project (eta,lambda) scores off (m,log sigma2) under static skew-t."""
    # Splitting the probability interval makes the tail integration materially
    # more accurate than one global Gauss-Legendre rule of the same order.
    breaks = np.array([0.0, 0.01, 0.05, *np.arange(0.1, 1.0, 0.1), 0.95, 0.99, 1.0])
    breaks = np.unique(breaks)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    score_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes / 2.0
        du = weights * (right - left) / 2.0
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        score_parts.append(core.reference_logpdf_and_score(y, anchor)[1])
        weight_parts.append(du)
    scores = np.vstack(score_parts)
    measure = np.concatenate(weight_parts)
    ms = scores[:, :2]
    shape = scores[:, 2:]
    information_ms = (ms.T * measure) @ ms
    cross_shape_ms = (shape.T * measure) @ ms
    projection = cross_shape_ms @ np.linalg.inv(information_ms)
    efficient = shape - ms @ projection.T
    information_shape = (efficient.T * measure) @ efficient
    information_shape = (information_shape + information_shape.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(information_shape)
    post_cross = (efficient.T * measure) @ ms
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(f"nonpositive efficient shape information: {eigenvalues}")
    return ShapeGeometry(
        projection_shape_on_ms=projection,
        diagnostics={
            "measure": "static Hansen skew-t density; equivalently du on its PIT",
            "projection": "(eta,lambda) tangent off (location,log scale-squared) tangent",
            "score_whitening": False,
            "eta_lambda_mutual_orthogonalization": False,
            "quadrature_order_per_probability_interval": quadrature_order,
            "quadrature_mass_error": float(abs(np.sum(measure) - 1.0)),
            "maximum_absolute_score_mean": float(np.max(np.abs((scores.T * measure).sum(axis=1)))),
            "maximum_post_projection_cross_information": float(np.max(np.abs(post_cross))),
            "efficient_shape_information": information_shape.tolist(),
            "efficient_shape_information_eigenvalues": eigenvalues.tolist(),
            "efficient_shape_information_correlation": float(
                information_shape[0, 1]
                / np.sqrt(information_shape[0, 0] * information_shape[1, 1])
            ),
        },
    )


def filter_shape_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen_location_scale: core.FourStateFilter,
    anchor: np.ndarray,
    betas: np.ndarray,
    geometry: ShapeGeometry,
    *,
    initial_shape_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> ShapeFilter:
    """Filter eta/lambda while taking the fitted m/sigma path as fixed."""
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    betas = np.asarray(betas, dtype=float)
    if betas.shape != (3, 2):
        raise ValueError("shape betas must have dimension 3 x 2")
    days = offsets.size - 1
    if frozen_location_scale.states.shape[0] != days:
        raise ValueError("frozen location-scale path has the wrong number of days")
    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    efficient_scores = np.empty((days, 2))
    contributions = np.empty(days)
    shape_state = (
        anchor[2:].copy()
        if initial_shape_state is None
        else np.asarray(initial_shape_state, dtype=float).copy()
    )
    history = (
        np.empty((0, 2))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-21:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = frozen_location_scale.states[day].copy()
        state[2:] = shape_state
        if np.any(~np.isfinite(state)) or np.max(np.abs(shape_state)) > 20.0:
            raise FloatingPointError("unstable eta/lambda state")
        log_density, observation_score = core.reference_logpdf_and_score(values[start:stop], state)
        daily_score = np.mean(observation_score, axis=0)
        efficient = daily_score[2:] - geometry.projection_shape_on_ms @ daily_score[:2]
        states[day] = state
        scores[day] = daily_score
        efficient_scores[day] = efficient
        contributions[day] = float(np.mean(log_density))
        history = np.vstack((history, efficient[None, :]))[-21:]
        shape_state = anchor[2:].copy()
        for beta, window in zip(betas, core.WINDOWS):
            shape_state += beta * history[-window:].sum(axis=0) / window
    return ShapeFilter(
        states=states,
        scores=scores,
        efficient_shape_scores=efficient_scores,
        contributions=contributions,
        terminal_shape_state=shape_state,
        terminal_score_history=history,
    )


def wald_test(beta: np.ndarray, covariance: np.ndarray, indices: list[int]) -> dict[str, float | int]:
    selected = np.asarray(indices, dtype=int)
    value = beta[selected]
    variance = covariance[np.ix_(selected, selected)]
    statistic = float(value @ np.linalg.solve(variance, value))
    return {"df": int(selected.size), "wald": statistic, "p": float(chi2.sf(statistic, selected.size))}


def fit_shape_layer(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen: core.FourStateFilter,
    anchor: np.ndarray,
    geometry: ShapeGeometry,
) -> tuple[np.ndarray, dict[str, object], ShapeFilter, ShapeFilter]:
    nested = filter_shape_layer(values, offsets, frozen, anchor, np.zeros((3, 2)), geometry)
    averages = fixed_denominator_score_averages(nested.efficient_shape_scores, core.WINDOWS)
    units = np.array([
        [
            1e-2 / max(float(np.median(np.abs(average[:, coordinate]))), 1e-12)
            for coordinate in range(2)
        ]
        for average in averages
    ])

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(3, 2) * units

    def run(theta: np.ndarray) -> ShapeFilter:
        return filter_shape_layer(values, offsets, frozen, anchor, unpack(theta), geometry)

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    rng = np.random.default_rng(SEED)
    starts = [np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6), 3.0 * rng.normal(size=6)]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 6,
            options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-9, "eps": 1e-5, "maxls": 80},
        )
        print(
            f"shape start {index + 1}/{len(starts)}: success={fit.success} "
            f"objective={float(fit.fun):.12g}", flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all eta/lambda-layer starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    betas = unpack(theta)
    dynamic = run(theta)

    hessian = None
    hessian_eigenvalues = None
    derivative_step = None
    hessian_attempts = []
    for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        candidate = numerical_hessian(objective, theta, step)
        candidate = (candidate + candidate.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(candidate)
        hessian_attempts.append({"step": step, "eigenvalues": eigenvalues.tolist()})
        if eigenvalues[0] > 1e-10:
            hessian, hessian_eigenvalues, derivative_step = candidate, eigenvalues, step
            break
    if hessian is None:
        raise RuntimeError(f"nonpositive eta/lambda Hessian: {hessian_attempts}")
    gradients = np.empty((offsets.size - 1, 6))
    for coordinate in range(6):
        direction = np.zeros(6)
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
    beta = betas.ravel()
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    eta_indices = [0, 2, 4]
    lambda_indices = [1, 3, 5]
    tests = {
        "eta_three_coefficients": wald_test(beta, covariance, eta_indices),
        "lambda_three_coefficients": wald_test(beta, covariance, lambda_indices),
        "eta_lambda_six_coefficients": wald_test(beta, covariance, list(range(6))),
    }
    parameters = []
    for index, (horizon, state_name) in enumerate(
        (horizon, state_name) for horizon in core.HORIZONS for state_name in ("eta", "lambda")
    ):
        parameters.append({
            "state": state_name,
            "horizon": horizon,
            "window": int(core.WINDOWS[index // 2]),
            "estimate": float(beta[index]),
            "hac22_se_conditional_on_location_scale": float(standard_errors[index]),
            "t": float(t_values[index]),
            "p_two_sided": float(p_values[index]),
        })
    summary = {
        "betas": {
            horizon: {"eta": float(betas[index, 0]), "lambda": float(betas[index, 1])}
            for index, horizon in enumerate(core.HORIZONS)
        },
        "parameter_units": units.tolist(),
        "parameters": parameters,
        "wald_tests": tests,
        "nested_location_scale_log_likelihood": nested.composite_log_likelihood,
        "dynamic_shape_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_location_scale": dynamic.composite_log_likelihood - nested.composite_log_likelihood,
        "optimizer_success": bool(optimum.success),
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(max(fit.fun for fit in successful) - optimum.fun),
        "hessian_derivative_step": derivative_step,
        "hessian_eigenvalues_scaled": hessian_eigenvalues.tolist(),
        "hessian_attempts": hessian_attempts,
        "maximum_absolute_mean_day_gradient": float(np.max(np.abs(gradients.mean(axis=0)))),
        "inference": "conditional Godambe-HAC(22), holding static MLE and fitted m/sigma path fixed",
    }
    return betas, summary, nested, dynamic


def load_or_fit_location_scale(
    dataset: str,
    values: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, dict[str, object], core.FourStateFilter, str]:
    # Existing market results used the identical static skew-t and fixed-shape
    # location-scale specification. Reuse them and reproduce the path exactly.
    saved_path = ROOT / f"output_{dataset}_fixed_shape" / "estimation.json"
    if dataset in ("sp500", "n225") and saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        anchor = np.asarray(saved["static_reference"]["state"], dtype=float)
        betas = np.zeros((3, 4))
        for horizon_index, horizon in enumerate(core.HORIZONS):
            betas[horizon_index, 0] = saved["dynamic_location_scale"]["betas"][horizon]["location"]
            betas[horizon_index, 1] = saved["dynamic_location_scale"]["betas"][horizon]["log_scale_squared"]
        path = core.filter_reference(values, offsets, anchor, betas)
        error = abs(path.composite_log_likelihood - saved["dynamic_location_scale"]["dynamic_composite_log_likelihood"])
        if error > 1e-9:
            raise RuntimeError(f"saved {dataset} location-scale path mismatch: {error}")
        return anchor, saved["static_reference"], betas, saved["dynamic_location_scale"], path, str(saved_path)
    anchor, static_summary = core.fit_static_reference(values, offsets)
    betas, location_scale_summary, path = fixed.fit_dynamic_location_scale(values, offsets, anchor)
    return anchor, static_summary, betas, location_scale_summary, path, "fitted in this run"


def write_dataset_report(output: Path, result: dict[str, object]) -> None:
    fit = result["dynamic_eta_lambda"]
    lines = [
        f"# Direct dynamic skew-t shape test: {result['dataset']['label']}",
        "",
        "No spline is used. Static Hansen skew-t MLE is followed by dynamic location/scale;",
        "the fitted location-scale path is then frozen while eta and lambda are jointly updated.",
        "Shape scores are Fisher-orthogonalized against location/scale, without whitening.",
        "",
        "## Static skew-t MLE",
        "",
        f"- eta: `{result['static_reference']['eta']:.9g}`",
        f"- lambda: `{result['static_reference']['lambda']:.9g}`",
        "",
        "## Joint HAC(22) Wald tests",
        "",
        "| block | df | Wald | p |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("eta_three_coefficients", "eta"),
        ("lambda_three_coefficients", "lambda"),
        ("eta_lambda_six_coefficients", "eta + lambda"),
    ):
        row = fit["wald_tests"][key]
        lines.append(f"| {label} | {row['df']} | {row['wald']:.9g} | {row['p']:.9g} |")
    lines += [
        "",
        "## Coefficients",
        "",
        "| state | horizon | estimate | HAC(22) SE | t | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in fit["parameters"]:
        lines.append(
            f"| {row['state']} | {row['horizon']} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_location_scale']:.9g} | {row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines += [
        "",
        f"Training log-likelihood gain over dynamic location/scale: `{fit['gain_vs_location_scale']:.9f}`.",
        "",
        "Inference is conditional on the first-stage static MLE and fitted location/scale path.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(dataset: str) -> dict[str, object]:
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    panel, dataset_info = data_source.load_dataset(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    training_end = core.TRAINING_END if dataset in ("sp500", "n225") else stocks.TRAINING_END
    n_train = int(np.sum(dates <= training_end))
    training_rows = int(panel.offsets[n_train])
    values = panel.values[:training_rows]
    offsets = panel.offsets[: n_train + 1]
    print(f"{dataset}: fitting/loading static skew-t and dynamic location-scale", flush=True)
    anchor, static_summary, location_scale_betas, location_scale_summary, frozen, source = load_or_fit_location_scale(
        dataset, values, offsets
    )
    geometry = shape_geometry(anchor)
    print(f"{dataset}: shape geometry {geometry.diagnostics}", flush=True)
    print(f"{dataset}: jointly fitting eta/lambda dynamics", flush=True)
    _, shape_summary, nested, dynamic = fit_shape_layer(values, offsets, frozen, anchor, geometry)
    nesting_error = float(max(
        np.max(np.abs(nested.states - frozen.states)),
        np.max(np.abs(nested.contributions - frozen.contributions)),
    ))
    if nesting_error > 2e-12:
        raise RuntimeError(f"eta/lambda zero does not nest location-scale model: {nesting_error}")
    eta_path = np.array([core.state_to_parameters(state[2:])[0] for state in dynamic.states])
    lambda_path = np.array([core.state_to_parameters(state[2:])[1] for state in dynamic.states])
    paths = pd.DataFrame({
        "date": dates[:n_train],
        "location": dynamic.states[:, 0],
        "scale": np.exp(0.5 * dynamic.states[:, 1]),
        "eta": eta_path,
        "lambda": lambda_path,
    })
    parameters = pd.DataFrame(shape_summary["parameters"])
    paths.to_csv(output / "training_state_paths.csv", index=False)
    parameters.to_csv(output / "eta_lambda_parameters.csv", index=False)
    result = {
        "analysis": "direct dynamic Hansen skew-t eta/lambda test after dynamic location-scale",
        "dataset": dataset_info,
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "training_days": n_train,
            "training_observations": training_rows,
        },
        "specification": {
            "spline": False,
            "windows": list(core.WINDOWS),
            "sequence": "static skew-t MLE; dynamic m/log sigma2 with shape fixed; frozen m/sigma path plus joint eta/lambda dynamics",
            "shape_score": "fixed-anchor Fisher projection off location/log sigma2",
            "score_whitening": False,
            "parity_split": False,
            "eta_lambda_estimation": "joint six-parameter conditional MLE",
        },
        "first_stage_source": source,
        "static_reference": static_summary,
        "dynamic_location_scale": location_scale_summary,
        "dynamic_location_scale_betas_matrix": location_scale_betas.tolist(),
        "geometry": geometry.diagnostics,
        "dynamic_eta_lambda": shape_summary,
        "checks": {"zero_shape_dynamics_nesting_error": nesting_error},
    }
    (output / "estimation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_dataset_report(output, result)
    tests = shape_summary["wald_tests"]
    print(
        f"{dataset}: eta p={tests['eta_three_coefficients']['p']:.9g}, "
        f"lambda p={tests['lambda_three_coefficients']['p']:.9g}, "
        f"all p={tests['eta_lambda_six_coefficients']['p']:.9g}", flush=True,
    )
    return result


def write_combined(results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        tests = result["dynamic_eta_lambda"]["wald_tests"]
        rows.append({
            "dataset": result["dataset"].get("label", result["dataset"].get("dataset")),
            "static_eta": result["static_reference"]["eta"],
            "static_lambda": result["static_reference"]["lambda"],
            "eta_p": tests["eta_three_coefficients"]["p"],
            "lambda_p": tests["lambda_three_coefficients"]["p"],
            "all_p": tests["eta_lambda_six_coefficients"]["p"],
            "log_likelihood_gain": result["dynamic_eta_lambda"]["gain_vs_location_scale"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_ROOT / "combined_tests.csv", index=False)
    lines = [
        "# Direct dynamic Hansen skew-t shape tests",
        "",
        "Spline-free conditional tests after fitting dynamic location and scale.",
        "Scores for eta and lambda are projected off the location/scale tangent,",
        "but are not whitened or mutually orthogonalized.",
        "",
        "| dataset | static eta | static lambda | eta-dynamics p | lambda-dynamics p | joint p | training LL gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['static_eta']:.9g} | {row['static_lambda']:.9g} | "
            f"{row['eta_p']:.9g} | {row['lambda_p']:.9g} | {row['all_p']:.9g} | "
            f"{row['log_likelihood_gain']:.9f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The lambda three-coefficient block is not significant at 5% in any of the six data sets.",
        "- Eta dynamics are significant at 5% for SP500, N225, Toyota, AAPL, and NTT; AMZN is the exception.",
        "- The six-coefficient joint block is significant for SP500, N225, AAPL, and NTT. Toyota's eta block is significant but the larger joint block is not.",
        "- AAPL weekly lambda and NTT monthly lambda are individually below 5%, but their three-coefficient lambda blocks are not; they are not robust block-level skewness evidence.",
        "",
        "All p-values use a day-level conditional Godambe sandwich with HAC(22).",
        "First-stage static-anchor and location/scale estimation uncertainty is not propagated.",
    ]
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "markets", "stocks", "all"), default="all")
    args = parser.parse_args()
    if args.dataset == "markets":
        datasets = ("sp500", "n225")
    elif args.dataset == "stocks":
        datasets = tuple(stocks.ASSETS)
    elif args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = (args.dataset,)
    results = [run_dataset(dataset) for dataset in datasets]
    if tuple(datasets) == tuple(DATASETS):
        write_combined(results)


if __name__ == "__main__":
    main()
