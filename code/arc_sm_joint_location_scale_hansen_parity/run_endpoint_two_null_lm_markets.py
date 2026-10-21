"""Conditional endpoint-persistence LM tests under two static-shape nulls.

For each of M4|M2 and M5|M3, compare (i) q0=0, kappa=0 and
(ii) q0 unrestricted, kappa=0.  Both tests have six dynamic coefficients:
common/even and asymmetric/odd, each at HAR(1,5,22).  The M2/M3 reference
paths and their location/scale scores are treated as fixed first-stage output.

The current-day likelihood derivative is the *raw* q score.  Lagged driving
scores are projected off location/scale, exactly as in the standard M4/M5
filter.  Under the free-q0 null, the static q0 is fitted on that model's own
training PITs and the LM variance is adjusted for its estimation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import run_identity_sm_standard_endpoint_five_models as identity
import run_sp500 as core
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_standard_even_odd_model5 as endpoint


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_endpoint_two_null_lm_sp500_n225"
DATASETS = ("sp500", "n225")
BLOCKS = ("common_even", "asymmetric_odd")


def weights_from_q(q: np.ndarray, masses: np.ndarray, helmert: np.ndarray) -> np.ndarray:
    logits = np.log(masses) + helmert @ q
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def fit_static_q0(
    phi: np.ndarray, offsets: np.ndarray, masses: np.ndarray,
    helmert: np.ndarray, initial_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    counts = np.diff(offsets)
    observation_weights = np.repeat(1.0 / counts, counts)
    days = counts.size
    start = helmert.T @ np.log(initial_weights / masses)

    def objective(q: np.ndarray) -> tuple[float, np.ndarray]:
        weights = weights_from_q(q, masses, helmert)
        fitted = np.maximum(phi @ weights, 1e-300)
        value = -float(np.dot(observation_weights, np.log(fitted)) / days)
        response = weights * (phi.T @ (observation_weights / fitted))
        gradient = -(helmert.T @ (response - days * weights)) / days
        return value, gradient

    result = minimize(
        objective, start, jac=True, method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-14, "gtol": 2e-9,
                 "maxls": 60, "maxcor": 30},
    )
    q0 = np.asarray(result.x, dtype=float)
    weights = weights_from_q(q0, masses, helmert)
    objective_value, gradient = objective(q0)
    diagnostics = {
        "optimizer_success": bool(result.success),
        "iterations": int(result.nit),
        "static_mean_log_h": -objective_value,
        "mean_score_max_abs": float(np.max(np.abs(gradient))),
        "minimum_weight": float(weights.min()),
        "maximum_abs_q0": float(np.max(np.abs(q0))),
    }
    if diagnostics["mean_score_max_abs"] > 2e-6:
        raise RuntimeError(f"restricted q0 score not near zero: {diagnostics}")
    return q0, weights, diagnostics


def static_daily_score_and_hessian(
    phi: np.ndarray, offsets: np.ndarray, weights: np.ndarray,
    helmert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw daily q score and observed derivative w.r.t. the static q chart."""
    days = len(offsets) - 1
    dimension = helmert.shape[1]
    score = np.empty((days, dimension))
    derivative = np.empty((days, dimension, dimension))
    weighted_helmert = weights @ helmert
    weight_covariance = (
        (helmert.T * weights) @ helmert
        - np.outer(weighted_helmert, weighted_helmert)
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        day_phi = phi[start:stop]
        fitted = day_phi @ weights
        if np.any(fitted <= 0) or not np.all(np.isfinite(fitted)):
            raise FloatingPointError("nonpositive static spline density")
        responsibility = (day_phi * weights[None, :]) / fitted[:, None]
        mean_response = responsibility.mean(axis=0)
        projected_response = responsibility @ helmert
        score[day] = (mean_response - weights) @ helmert
        derivative[day] = (
            (helmert.T * mean_response) @ helmert
            - projected_response.T @ projected_response / (stop - start)
            - weight_covariance
        )
    return score, derivative


def fixed_denominator_past_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Same early-day convention as filter_block_arc_sm: divide by window."""
    values = np.asarray(series)
    cumulative = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)))
    day = np.arange(len(values))
    return (cumulative[day] - cumulative[np.maximum(day - window, 0)]) / window


def dynamic_score_and_sensitivity(
    raw_score: np.ndarray, derivative: np.ndarray,
    reference_scores: np.ndarray, projection: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Six kappa scores and their derivatives w.r.t. q0, holding P fixed.

    Current-day likelihood derivative is d' raw_score; lagged driver is
    d' (raw_score - P reference_score).  The derivative of P(q0) does not
    enter the population H_k0 under the correctly specified conditional null:
    it multiplies a predictable regressor by a current-day MDS score.
    """
    current = raw_score @ directions.T
    driver = (raw_score - reference_scores[:, :2] @ projection.T) @ directions.T
    current_derivative = np.einsum("bd,tdj->tbj", directions, derivative)
    rows, sensitivities = [], []
    for block in range(2):
        for window in core.WINDOWS:
            past = fixed_denominator_past_mean(driver[:, block], window)
            past_derivative = fixed_denominator_past_mean(
                current_derivative[:, block, :], window
            )
            rows.append(current[:, block] * past)
            sensitivities.append(
                current_derivative[:, block, :] * past[:, None]
                + current[:, block, None] * past_derivative
            )
    return np.column_stack(rows), np.stack(sensitivities, axis=1)


def lm_rows(
    dataset: str, model: str, null: str, m: np.ndarray, raw_score: np.ndarray,
    score_derivative: np.ndarray, m_derivative: np.ndarray,
    q0_diagnostics: dict[str, float | int | bool],
) -> tuple[list[dict[str, object]], np.ndarray]:
    days = len(m)
    if null == "q0_fitted":
        h00 = -score_derivative.mean(axis=0)
        hk0 = -m_derivative.mean(axis=0)
        eig = np.linalg.eigvalsh((h00 + h00.T) / 2)
        if eig[0] < 1e-10:
            raise RuntimeError(f"{dataset} {model}: weak static-q0 information {eig[0]}")
        adjustment = np.linalg.solve(h00.T, hk0.T).T
        zeta = m - raw_score @ adjustment.T
        q0_mean_score = float(np.max(np.abs(raw_score.mean(axis=0))))
    else:
        zeta = m
        eig = np.array([np.nan])
        q0_mean_score = np.nan
    covariance = zeta.T @ zeta / days
    numerator = np.sqrt(days) * m.mean(axis=0)
    result = []
    for block, indices in (
        ("common_even", np.arange(3)),
        ("asymmetric_odd", np.arange(3, 6)),
        ("joint_six", np.arange(6)),
    ):
        matrix = covariance[np.ix_(indices, indices)]
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        if eigenvalues[0] <= 1e-12:
            raise RuntimeError(
                f"{dataset} {model} {null} {block}: singular LM variance {eigenvalues}"
            )
        statistic = float(numerator[indices] @ np.linalg.solve(matrix, numerator[indices]))
        result.append({
            "dataset": dataset, "model": model, "null": null,
            "block": block, "df": int(len(indices)), "training_days": days,
            "LM": statistic, "p_chi2": float(chi2.sf(statistic, len(indices))),
            "max_abs_restricted_q0_score": q0_mean_score,
            "min_h00_eigenvalue": float(eig[0]),
            "min_variance_eigenvalue": float(eigenvalues[0]),
            **q0_diagnostics,
        })
    return result, zeta


def run_dataset(dataset: str) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    print(f"{dataset}: loading locked reference paths", flush=True)
    panel, _info, saved, _label = identity.load_dataset(dataset)
    standard = json.loads(
        (identity.OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8")
    )
    train_days = int(standard["sample"]["training_days"])
    train_rows = int(panel.offsets[train_days])
    train_values = panel.values[:train_rows]
    offsets = panel.offsets[:train_days + 1]
    anchor = np.asarray(saved["model1"]["state"], dtype=float)
    knots = core.make_clamped_knots(base.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(len(masses))
    directions = np.vstack(endpoint.endpoint_directions(helmert))
    m2_train, _, _, m3_train, _, _, checks = base.reference_paths(
        panel, train_days, anchor, saved
    )
    if max(checks.values()) > 1e-2:
        raise RuntimeError(f"reference-path reproduction failed: {checks}")
    initial_weights = np.asarray(json.loads(
        (endpoint.standard.OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8")
    )["static_q0"]["weights"], dtype=float)

    results: list[dict[str, object]] = []
    null_fits: list[dict[str, object]] = []
    daily_scores: list[dict[str, object]] = []
    dates = pd.DatetimeIndex(panel.daily["date"])[:train_days]
    for model, reference in (("M4_after_M2", m2_train), ("M5_after_M3", m3_train)):
        print(f"{dataset} {model}: PIT and null scores", flush=True)
        u = core.dynamic_reference_pit(train_values, offsets, reference.states)
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        for null in ("q0_fixed_zero", "q0_fitted"):
            if null == "q0_fixed_zero":
                q0 = np.zeros(len(masses) - 1)
                weights = masses.copy()
                fit_diagnostics: dict[str, float | int | bool] = {}
            else:
                q0, weights, fit_diagnostics = fit_static_q0(
                    phi, offsets, masses, helmert, initial_weights
                )
            raw_score, score_derivative = static_daily_score_and_hessian(
                phi, offsets, weights, helmert
            )
            geometry = scoring.corrected_tangent_geometry(
                anchor, weights, knots, masses, helmert, (0, 1),
                f"{dataset} {model} {null} q score off m/log-variance",
            )
            m, m_derivative = dynamic_score_and_sensitivity(
                raw_score, score_derivative, reference.scores,
                geometry.projection_q_on_hansen, directions,
            )
            block_results, adjusted_score = lm_rows(
                dataset, model, null, m, raw_score, score_derivative,
                m_derivative, fit_diagnostics,
            )
            results.extend(block_results)
            null_fits.append({
                "dataset": dataset, "model": model, "null": null,
                "q0": q0.tolist(), "weights": weights.tolist(),
                "projection_q_on_location_scale":
                    geometry.projection_q_on_hansen.tolist(),
                "diagnostics": fit_diagnostics,
            })
            for day, date in enumerate(dates):
                row: dict[str, object] = {
                    "dataset": dataset, "model": model, "null": null,
                    "date": str(date.date()),
                }
                for block_index, block in enumerate(BLOCKS):
                    for lag_index, window in enumerate(core.WINDOWS):
                        index = block_index * 3 + lag_index
                        row[f"m_{block}_{window}"] = float(m[day, index])
                        row[f"zeta_{block}_{window}"] = float(adjusted_score[day, index])
                daily_scores.append(row)
            print(
                f"{dataset} {model} {null}: joint p={results[-1]['p_chi2']:.5g}"
                f" min_w={weights.min():.3g}", flush=True,
            )
        del phi
    return results, null_fits, daily_scores


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    for dataset in DATASETS:
        dataset_rows, dataset_fits, dataset_scores = run_dataset(dataset)
        rows.extend(dataset_rows)
        fits.extend(dataset_fits)
        scores.extend(dataset_scores)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "lm_results.csv", index=False)
    pd.DataFrame(scores).to_csv(OUTPUT / "daily_lm_scores.csv", index=False)
    (OUTPUT / "restricted_null_estimates.json").write_text(
        json.dumps(fits, indent=2), encoding="utf-8"
    )
    print(frame[["dataset", "model", "null", "block", "LM", "p_chi2"]].to_string(index=False))


if __name__ == "__main__":
    main()
