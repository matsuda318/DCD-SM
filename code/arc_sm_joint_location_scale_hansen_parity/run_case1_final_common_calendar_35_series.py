"""Final locked Case-1 M1--M5 experiment on a common calendar window.

All 35 applications are re-estimated from M1 onward.  The common calendar
window is 2016-07-05 through 2026-07-24, the training sample ends on
2023-12-29, and the OOS sample begins with the first observation on or after
2024-01-02.  The S&P 500 real-estate sector is retained from its first
available date (2016-09-01).

Every dynamic stage uses the same overlapping HAR(1,5,22) regressors.  For a
state dated t+1 these are s_t, mean(s_t,...,s_{t-4}), and
mean(s_t,...,s_{t-21}).  Exactly 22 scores are retained; unavailable
pre-sample scores are zero.  This file deliberately does not reuse any saved
M1 or M2 fit, preventing the historical 21-score market filter from entering
the S&P 500 or Nikkei 225 results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2

import run_direct_dynamic_skewt_shape_test as shape
import run_identity_sm_standard_endpoint_five_models as identity
import run_raw_score_zero_intercept_five_models as raw_old
import run_raw_score_zero_intercept_full22_five_models as full22
import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_standard_even_odd_model5 as endpoint
import run_zero_intercept_endpoint_sp500 as zero
from estimate_score_har import hac_meat, numerical_hessian


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_case1_final_common_calendar_35_series"
START = pd.Timestamp("2016-07-05")
TRAINING_END = pd.Timestamp("2023-12-29")
OOS_START = pd.Timestamp("2024-01-02")
END = pd.Timestamp("2026-07-24")
DATASETS = tuple(identity.ALL_DATASETS)
MODELS = (identity.M1, identity.M2, identity.M3, identity.M4, identity.M5)
MODEL_SHORT = {
    identity.M1: "M1",
    identity.M2: "M2",
    identity.M3: "M3",
    identity.M4: "M4",
    identity.M5: "M5",
}
HIGHER_IS_BETTER = {
    "continuous_log_score",
    "categorical_log_score_50",
    "categorical_log_score_100",
    "categorical_log_score_200",
}


def trim_panel(panel: core.RawPanel) -> core.RawPanel:
    dates = pd.DatetimeIndex(panel.daily["date"]).normalize()
    # Keep the requested common window while explicitly excluding observations
    # between the training cutoff and the requested OOS start.  This matters for
    # FX, where Yahoo records 2024-01-01 although the locked OOS window begins on
    # 2024-01-02.
    keep = (
        (dates >= START)
        & (dates <= END)
        & ((dates <= TRAINING_END) | (dates >= OOS_START))
    )
    positions = np.flatnonzero(keep)
    if positions.size == 0:
        raise RuntimeError("common-calendar trim produced an empty panel")
    counts = panel.offsets[positions + 1] - panel.offsets[positions]
    value_blocks = [
        panel.values[int(panel.offsets[pos]) : int(panel.offsets[pos + 1])]
        for pos in positions
    ]
    values = np.concatenate(value_blocks) if value_blocks else panel.values[:0].copy()
    daily = panel.daily.iloc[positions].reset_index(drop=True).copy()
    daily["date"] = dates[positions]
    return core.RawPanel(
        daily=daily,
        values=values,
        offsets=np.concatenate(([0], np.cumsum(counts))),
    )


def category(dataset: str) -> str:
    if dataset in ("sp500", "n225"):
        return "market"
    return endpoint.category(dataset)


def fit_projected_m3(panel, n_train: int, anchor, m2_train, m2_oos, m2_full):
    train_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    geometry = shape.shape_geometry(anchor)
    old_filter = shape.filter_shape_layer
    shape.filter_shape_layer = raw_old.filter_shape_har22
    try:
        betas, summary, _nested, training = shape.fit_shape_layer(
            panel.values[:train_rows], train_offsets, m2_train, anchor, geometry
        )
        oos, full, continuation = base.identity.continue_shape(
            panel, n_train, m2_oos, m2_full, anchor, betas, geometry, training
        )
    finally:
        shape.filter_shape_layer = old_filter
    return betas, summary, training, oos, full, continuation, geometry


def wald(beta: np.ndarray, covariance: np.ndarray, indices: list[int]) -> dict[str, float]:
    selected = np.asarray(indices, dtype=int)
    values = beta[selected]
    variance = covariance[np.ix_(selected, selected)]
    statistic = float(values @ np.linalg.solve(variance, values))
    return {
        "df": int(selected.size),
        "wald": statistic,
        "p": float(chi2.sf(statistic, selected.size)),
    }


def infer_m2(values, offsets, anchor, betas, summary) -> dict[str, object]:
    """Conditional Godambe-HAC(22) Wald inference for the exact-full22 M2."""
    units = np.asarray(summary["parameter_units"], dtype=float)
    theta = (np.asarray(betas, dtype=float)[:, :2] / units).ravel()

    def run(candidate: np.ndarray):
        candidate_betas = np.zeros((3, 4))
        candidate_betas[:, :2] = np.asarray(candidate).reshape(3, 2) * units
        return full22.filter_reference_har22(values, offsets, anchor, candidate_betas)

    fitted = run(theta)
    likelihood_error = abs(
        fitted.composite_log_likelihood
        - float(summary["dynamic_composite_log_likelihood"])
    )
    if likelihood_error > 1e-8:
        raise RuntimeError(f"M2 exact-full22 likelihood reproduction error: {likelihood_error}")

    def objective(candidate: np.ndarray) -> float:
        try:
            value = -run(candidate).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    hessian = None
    eigenvalues = None
    derivative_step = None
    attempts = []
    for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        candidate = numerical_hessian(objective, theta, step)
        candidate = (candidate + candidate.T) / 2.0
        candidate_eigenvalues = np.linalg.eigvalsh(candidate)
        attempts.append({"step": step, "eigenvalues": candidate_eigenvalues.tolist()})
        if candidate_eigenvalues[0] > 1e-10:
            hessian = candidate
            eigenvalues = candidate_eigenvalues
            derivative_step = step
            break
    if hessian is None:
        raise RuntimeError(f"nonpositive M2 Hessian: {attempts}")

    days = offsets.size - 1
    gradients = np.empty((days, 6))
    for coordinate in range(6):
        direction = np.zeros(6)
        direction[coordinate] = derivative_step
        gradients[:, coordinate] = (
            run(theta + direction).contributions
            - run(theta - direction).contributions
        ) / (2.0 * derivative_step)
    inverse_hessian = np.linalg.inv(hessian)
    covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
    covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
    scaling = np.diag(units.ravel())
    covariance = scaling @ covariance_theta @ scaling
    covariance = (covariance + covariance.T) / 2.0
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    if covariance_eigenvalues[0] <= 0:
        raise RuntimeError(f"nonpositive M2 covariance: {covariance_eigenvalues}")

    beta = np.asarray(betas, dtype=float)[:, :2].ravel()
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(t_values))
    parameters = []
    for index, (horizon, state) in enumerate(
        (horizon, state)
        for horizon in core.HORIZONS
        for state in ("location", "log_scale_squared")
    ):
        parameters.append({
            "state": state,
            "horizon": horizon,
            "window": int(core.WINDOWS[index // 2]),
            "estimate": float(beta[index]),
            "hac22_se_conditional_on_m1": float(standard_errors[index]),
            "t": float(t_values[index]),
            "p_two_sided": float(p_values[index]),
        })
    tests = {
        "location_three_coefficients": wald(beta, covariance, [0, 2, 4]),
        "scale_three_coefficients": wald(beta, covariance, [1, 3, 5]),
        "location_scale_six_coefficients": wald(beta, covariance, list(range(6))),
    }
    return {
        "parameters": parameters,
        "wald_tests": tests,
        "derivative_step": derivative_step,
        "hessian_eigenvalues_scaled": eigenvalues.tolist(),
        "covariance_eigenvalues": covariance_eigenvalues.tolist(),
        "maximum_absolute_mean_day_gradient": float(
            np.max(np.abs(gradients.mean(axis=0)))
        ),
        "likelihood_reproduction_error": float(likelihood_error),
        "score_history": "exactly 22 observations",
    }


def state_recursion_error(states, scores, anchor, betas) -> float:
    errors = []
    for day in range(states.shape[0]):
        expected = np.asarray(anchor, dtype=float).copy()
        history = scores[max(0, day - 22) : day]
        for beta, window in zip(betas, core.WINDOWS):
            expected += beta * history[-window:].sum(axis=0) / window
        errors.append(np.max(np.abs(states[day] - expected)))
    return float(max(errors, default=0.0))


def endpoint_recursion_error(filtered, coefficients) -> float:
    errors = []
    for day in range(filtered.states.shape[0]):
        expected = np.zeros(filtered.states.shape[1])
        history = filtered.component_scores[max(0, day - 22) : day]
        for lag, window in enumerate(core.WINDOWS):
            expected += np.sum(
                coefficients[:, lag, None] * history[-window:].sum(axis=0) / window,
                axis=0,
            )
        errors.append(np.max(np.abs(filtered.states[day] - expected)))
    return float(max(errors, default=0.0))


def run_dataset(dataset: str) -> dict[str, object]:
    raw_panel, info, saved, label = identity.load_dataset(dataset)
    panel = trim_panel(raw_panel)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    if n_train <= 22 or n_train >= dates.size:
        raise RuntimeError(f"{dataset}: invalid train/OOS split")
    train_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]

    print(f"{dataset}: refit M1 on common calendar", flush=True)
    anchor, m1_summary = core.fit_static_reference(
        panel.values[:train_rows], train_offsets
    )
    print(f"{dataset}: refit exact-full22 M2", flush=True)
    try:
        initial_m2 = base.locked.m2_betas(saved)
    except (KeyError, TypeError, ValueError):
        initial_m2 = np.zeros((3, 4))
    m2_betas, m2_summary, m2_train = full22.fit_m2_har22(
        panel.values[:train_rows], train_offsets, anchor, initial_m2
    )
    m2_oos, m2_full, m2_continuation = full22.continue_m2(
        panel, n_train, anchor, m2_betas, m2_train
    )
    print(f"{dataset}: exact-full22 M2 Wald", flush=True)
    m2_inference = infer_m2(
        panel.values[:train_rows], train_offsets, anchor, m2_betas, m2_summary
    )

    print(f"{dataset}: refit exact-full22 projected M3", flush=True)
    (
        m3_betas, m3_summary, m3_train, m3_oos, m3_full,
        m3_continuation, m3_geometry,
    ) = fit_projected_m3(panel, n_train, anchor, m2_train, m2_oos, m2_full)

    knots = core.make_clamped_knots(base.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    directions = np.vstack(endpoint.endpoint_directions(helmert))
    fits, corrections, checks, geometries, endpoint_training = {}, {}, {}, {}, {}
    old_endpoint_filter = split.filter_block_arc_sm
    split.filter_block_arc_sm = zero.filter_no_intercept_har22
    try:
        for model, ref_train, ref_oos, ref_full in (
            ("M4", m2_train, m2_oos, m2_full),
            ("M5", m3_train, m3_oos, m3_full),
        ):
            print(f"{dataset}: refit exact-full22 {model}", flush=True)
            u = core.dynamic_reference_pit(panel.values, panel.offsets, ref_full.states)
            phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
            geometry = scoring.corrected_tangent_geometry(
                anchor, masses, knots, masses, helmert, (0, 1),
                f"{dataset} final Case1 {model}: q score off m/log variance",
            )
            transform = endpoint.four_tangent_endpoint_transform(
                directions, geometry.projection_q_on_hansen
            )
            fit, training, oos, _full, diagnostic = identity.fit_endpoint(
                phi, panel, n_train, ref_train.scores, ref_oos.scores,
                ref_full.scores, masses, helmert, transform,
                f"{dataset}: final Case1 {model}",
            )
            fits[model], corrections[model] = fit, oos
            checks[model], geometries[model] = diagnostic, geometry.diagnostics
            endpoint_training[model] = training
    finally:
        split.filter_block_arc_sm = old_endpoint_filter

    lag_audit = {
        "definition": {
            "state_t_plus_1": "daily s_t; weekly mean(s_t,...,s_t-4); monthly mean(s_t,...,s_t-21)",
            "windows": [1, 5, 22],
            "retained_scores": 22,
            "presample_scores": "zero",
        },
        "m2_maximum_state_recursion_error": state_recursion_error(
            m2_train.states, m2_train.scores, anchor, m2_betas
        ),
        "m3_maximum_state_recursion_error": state_recursion_error(
            m3_train.states[:, 2:], m3_train.efficient_shape_scores,
            anchor[2:], m3_betas,
        ),
        "m4_maximum_state_recursion_error": endpoint_recursion_error(
            endpoint_training["M4"], np.asarray(fits["M4"]["coefficients"])
        ),
        "m5_maximum_state_recursion_error": endpoint_recursion_error(
            endpoint_training["M5"], np.asarray(fits["M5"]["coefficients"])
        ),
        "m2_terminal_history_length": int(m2_train.terminal_score_history.shape[0]),
        "m3_terminal_history_length": int(m3_train.terminal_score_history.shape[0]),
        "m4_terminal_history_length": int(
            endpoint_training["M4"].terminal_component_history.shape[0]
        ),
        "m5_terminal_history_length": int(
            endpoint_training["M5"].terminal_component_history.shape[0]
        ),
    }
    maximum_lag_error = max(
        value for key, value in lag_audit.items() if key.endswith("recursion_error")
    )
    if maximum_lag_error > 1e-10 or any(
        lag_audit[key] != 22 for key in lag_audit if key.endswith("history_length")
    ):
        raise RuntimeError(f"{dataset}: HAR22 audit failed: {lag_audit}")

    scores = identity.score_models(
        panel, dates, n_train, anchor, knots, masses,
        m2_oos, m3_oos, corrections["M4"], corrections["M5"],
    )
    comparisons = identity.compare(scores)
    levels = scores.groupby("model", as_index=False)[list(scoring.SCORE_NAMES)].mean()
    result = {
        "analysis": "final locked Case1 common-calendar exact-full22 M1--M5",
        "dataset": info,
        "dataset_key": dataset,
        "label": label,
        "category": category(dataset),
        "sample": {
            "requested_start": str(START.date()),
            "actual_start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "requested_oos_start": str(OOS_START.date()),
            "actual_oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(dates.size - n_train),
        },
        "specification": {
            "case": "Case1 projected q0=0",
            "reference": "static skew-t MLE refitted on the trimmed training sample",
            "static_sm": "identity h(u)=1; q0=0 fixed",
            "basis_breakpoints": base.BREAKPOINTS.tolist(),
            "windows": [1, 5, 22],
            "score_history": "exactly 22 observations at M2--M5",
            "projection": "M3--M5 scores Fisher-projected off m/log variance",
            "score_whitening": False,
            "M4_parent": "M2",
            "M5_parent": "M3; eta/lambda directions retained in q score",
            "inference": "sandwich Wald conditional on frozen upstream paths",
        },
        "model1": m1_summary,
        "model2": m2_summary,
        "model2_betas": m2_betas.tolist(),
        "model2_inference": m2_inference,
        "model2_continuation": m2_continuation,
        "model3": m3_summary,
        "model3_betas": m3_betas.tolist(),
        "model3_geometry": m3_geometry.diagnostics,
        "model3_continuation": m3_continuation,
        "model4": fits["M4"],
        "model5": fits["M5"],
        "model4_geometry": geometries["M4"],
        "model5_geometry": geometries["M5"],
        "checks": checks,
        "lag_audit": lag_audit,
        "oos_summary": levels.to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
    }
    out = OUTPUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "daily_oos_scores.csv", index=False)
    levels.to_csv(out / "oos_levels.csv", index=False)
    comparisons.to_csv(out / "oos_comparisons.csv", index=False)
    pd.DataFrame(m2_inference["parameters"]).to_csv(
        out / "model2_parameters.csv", index=False
    )
    pd.DataFrame(m3_summary["parameters"]).to_csv(
        out / "model3_parameters.csv", index=False
    )
    pd.DataFrame(fits["M4"]["parameters"]).to_csv(
        out / "model4_parameters.csv", index=False
    )
    pd.DataFrame(fits["M5"]["parameters"]).to_csv(
        out / "model5_parameters.csv", index=False
    )
    (out / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{dataset}: complete; M2={m2_inference['wald_tests']['location_scale_six_coefficients']['p']:.5g}, "
        f"M3={m3_summary['wald_tests']['eta_lambda_six_coefficients']['p']:.5g}, "
        f"M4={fits['M4']['joint_all']['p']:.5g}, M5={fits['M5']['joint_all']['p']:.5g}",
        flush=True,
    )
    return result


def family(dataset: str) -> str:
    if dataset == "sp500":
        return "S&P 500"
    if dataset == "n225":
        return "N225"
    if dataset.startswith("sp500_s"):
        return "S&P 500 sectors"
    if dataset.startswith("n225_n"):
        return "N225 sectors"
    kind = category(dataset)
    return {"stock": "12 stocks", "fx": "4 FX"}[kind]


def summarize() -> None:
    missing = [d for d in DATASETS if not (OUTPUT / d / "estimation.json").exists()]
    if missing:
        raise FileNotFoundError(f"incomplete final Case1 batch: {missing}")
    tests, levels, comparisons, parameters, samples, lag_rows = [], [], [], [], [], []
    for dataset in DATASETS:
        result = json.loads((OUTPUT / dataset / "estimation.json").read_text(encoding="utf-8"))
        common = {
            "dataset": dataset,
            "label": result["label"],
            "category": result["category"],
            "family": family(dataset),
        }
        samples.append({**common, **result["sample"]})
        lag_rows.append({**common, **{
            key: value for key, value in result["lag_audit"].items()
            if key != "definition"
        }})
        m2_tests = result["model2_inference"]["wald_tests"]
        tests.extend([
            {**common, "model": "M2", "block": "location", **m2_tests["location_three_coefficients"]},
            {**common, "model": "M2", "block": "scale", **m2_tests["scale_three_coefficients"]},
            {**common, "model": "M2", "block": "joint", **m2_tests["location_scale_six_coefficients"]},
        ])
        parameters.extend({**common, "model": "M2", **row}
                          for row in result["model2_inference"]["parameters"])
        m3_tests = result["model3"]["wald_tests"]
        tests.extend([
            {**common, "model": "M3", "block": "eta", **m3_tests["eta_three_coefficients"]},
            {**common, "model": "M3", "block": "lambda", **m3_tests["lambda_three_coefficients"]},
            {**common, "model": "M3", "block": "joint", **m3_tests["eta_lambda_six_coefficients"]},
        ])
        parameters.extend({**common, "model": "M3", **row}
                          for row in result["model3"]["parameters"])
        for model in ("M4", "M5"):
            fitted = result[f"model{model[-1]}"]
            tests.extend([
                {**common, "model": model, "block": "even", **fitted["joint_blocks"]["common_even"]},
                {**common, "model": model, "block": "odd", **fitted["joint_blocks"]["asymmetric_odd"]},
                {**common, "model": model, "block": "joint", **fitted["joint_all"]},
            ])
            parameters.extend({**common, "model": model, **row}
                              for row in fitted["parameters"])
        levels.extend({**common, **row} for row in result["oos_summary"])
        comparisons.extend({**common, **row} for row in result["oos_comparisons"])

    test_frame = pd.DataFrame(tests)
    level_frame = pd.DataFrame(levels)
    comparison_frame = pd.DataFrame(comparisons)
    parameter_frame = pd.DataFrame(parameters)
    sample_frame = pd.DataFrame(samples)
    lag_frame = pd.DataFrame(lag_rows)
    test_frame.to_csv(OUTPUT / "case1_wald_tests.csv", index=False)
    level_frame.to_csv(OUTPUT / "case1_oos_levels.csv", index=False)
    comparison_frame.to_csv(OUTPUT / "case1_oos_comparisons.csv", index=False)
    parameter_frame.to_csv(OUTPUT / "case1_parameters.csv", index=False)
    sample_frame.to_csv(OUTPUT / "case1_samples.csv", index=False)
    lag_frame.to_csv(OUTPUT / "case1_har22_audit.csv", index=False)

    groups = (
        "S&P 500", "N225", "S&P 500 sectors", "N225 sectors",
        "12 stocks", "4 FX", "All 35",
    )
    best_rows = []
    for group in groups:
        block = level_frame if group == "All 35" else level_frame.loc[level_frame.family == group]
        for metric in scoring.SCORE_NAMES:
            wide = block.pivot(index="dataset", columns="model", values=metric)
            winners = wide.idxmax(axis=1) if metric in HIGHER_IS_BETTER else wide.idxmin(axis=1)
            counts = winners.value_counts()
            best_rows.append({
                "family": group,
                "metric": metric,
                **{MODEL_SHORT[model]: int(counts.get(model, 0)) for model in MODELS},
                "N": int(len(winners)),
            })
    best_frame = pd.DataFrame(best_rows)
    best_frame.to_csv(OUTPUT / "case1_oos_best_model_counts.csv", index=False)

    lines = [
        "# Final Case 1 common-calendar results",
        "",
        "All 35 datasets were re-estimated from M1 through M5. Calendar bounds are",
        "2016-07-05 to 2026-07-24, training ends on 2023-12-29, and OOS starts",
        "with the first available observation on or after 2024-01-02. The S&P 500",
        "real-estate sector retains its later first available date.",
        "",
        "Every M2--M5 recursion retains exactly 22 scores and uses",
        "s_t, mean(s_t,...,s_t-4), and mean(s_t,...,s_t-21) for the t+1 state.",
        "The machine-readable recursion audit is `case1_har22_audit.csv`.",
        "",
        "## Five-percent Wald rejection counts",
        "",
        "| Family | Model/block | Rejections | N |",
        "|---|---|---:|---:|",
    ]
    for group in groups[:-1]:
        block = test_frame.loc[test_frame.family == group]
        for model, tested_blocks in (
            ("M2", ("location", "scale", "joint")),
            ("M3", ("eta", "lambda", "joint")),
            ("M4", ("even", "odd", "joint")),
            ("M5", ("even", "odd", "joint")),
        ):
            for tested_block in tested_blocks:
                selected = block.loc[(block.model == model) & (block.block == tested_block)]
                lines.append(
                    f"| {group} | {model} {tested_block} | "
                    f"{int((selected.p < .05).sum())} | {len(selected)} |"
                )
    lines += [
        "",
        "## Strict-OOS best-model counts",
        "",
        "Each vector is M1/M2/M3/M4/M5. Log scores are maximized; all losses are minimized.",
        "",
        "| Metric | S&P 500 | N225 | S&P sectors | N225 sectors | 12 stocks | 4 FX | All 35 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    display_names = {
        "continuous_log_score": "Continuous log",
        "crps": "CRPS",
        "categorical_log_score_50": "50-bin log",
        "categorical_log_score_100": "100-bin log",
        "categorical_log_score_200": "200-bin log",
        "q_loss_0.01": "Q-loss 1%",
        "q_loss_0.05": "Q-loss 5%",
        "brier_0.01": "Brier 1%",
        "brier_0.05": "Brier 5%",
    }
    family_columns = groups
    for metric in scoring.SCORE_NAMES:
        values = []
        for group in family_columns:
            row = best_frame.loc[(best_frame.family == group) & (best_frame.metric == metric)].iloc[0]
            values.append("/".join(str(int(row[model])) for model in ("M1", "M2", "M3", "M4", "M5")))
        lines.append(f"| {display_names[metric]} | " + " | ".join(values) + " |")
    totals = best_frame.groupby("family")[["M1", "M2", "M3", "M4", "M5"]].sum()
    values = ["/".join(str(int(totals.loc[group, model])) for model in ("M1", "M2", "M3", "M4", "M5")) for group in family_columns]
    lines.append("| **Nine-metric total** | **" + "** | **".join(values) + "** |")
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return
    if args.dataset and args.datasets:
        parser.error("use either --dataset or --datasets, not both")
    selected = ((args.dataset,) if args.dataset else
                (tuple(args.datasets) if args.datasets else DATASETS))
    for dataset in selected:
        if args.resume and (OUTPUT / dataset / "estimation.json").exists():
            print(f"{dataset}: completed output retained", flush=True)
            continue
        run_dataset(dataset)
    if set(selected) == set(DATASETS):
        summarize()


if __name__ == "__main__":
    main()
