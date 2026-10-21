"""Parsimonious boundary-tail DCD-SM experiments for S&P 500.

Both models keep the fitted static skew-t, fitted static spline weights, frozen
M2 location/scale path, and decile clamped-cubic basis unchanged.  The spline
score is projected off the fixed-anchor location/scale scores before it is
restricted to either one common boundary-logit direction or two separate
left/right boundary-logit directions.  Every scalar state uses HAR(1,5,22).
There is no Fisher whitening.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_skewt_four_models as fitted
import run_static_q0_symmetric_t_four_models as scoring


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_static_q0_skewt_boundary_tail_m4_sp500"
SAVED = fitted.OUTPUT_ROOT / "sp500" / "estimation.json"

M2 = fitted.MODEL_NAMES["m2"]
M3 = fitted.MODEL_NAMES["m3"]
COMMON = "m4_common_boundary_tail_arc_sm"
LEFT_RIGHT = "m4_left_right_boundary_tail_arc_sm"


def endpoint_directions(q_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q-coordinate directions for exact endpoint logit shifts."""
    basis_count = q_matrix.shape[0]
    left_logit = np.zeros(basis_count)
    right_logit = np.zeros(basis_count)
    left_logit[0] = 1.0
    right_logit[-1] = 1.0
    left = q_matrix.T @ left_logit
    right = q_matrix.T @ right_logit
    return left + right, left, right


def boundary_transform(
    directions: np.ndarray,
    projection_ms: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Efficient q score followed by fixed boundary-direction restriction."""
    directions = np.asarray(directions, dtype=float)

    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        efficient = raw_q_score - projection_ms @ reference_score[:2]
        scalar_scores = directions @ efficient
        return directions * scalar_scores[:, None]

    return transform


def model2_betas(saved: dict[str, object]) -> np.ndarray:
    betas = np.zeros((3, 4))
    for index, horizon in enumerate(core.HORIZONS):
        betas[index, 0] = saved["model2"]["betas"][horizon]["location"]
        betas[index, 1] = saved["model2"]["betas"][horizon]["log_scale_squared"]
    return betas


def comparison_rows(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        (COMMON, M2), (LEFT_RIGHT, M2),
        (COMMON, M3), (LEFT_RIGHT, M3),
        (LEFT_RIGHT, COMMON),
    )
    return pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in scoring.SCORE_NAMES
    ])


def write_results(result: dict[str, object], comparisons: pd.DataFrame) -> None:
    common = result["common_tail"]
    left_right = result["left_right_tail"]
    lines = [
        "# S&P 500 fitted-static-SM boundary-tail DCD-SM", "",
        "## Locked specification", "",
        "- Static skew-t, fitted static spline weights, frozen M2, and decile knots are unchanged.",
        "- The spline score is Fisher-projected off location/log-scale-squared at the fixed anchor.",
        "- `common_tail` changes the two endpoint logits together; `left_right` changes them separately.",
        "- Every scalar direction uses HAR(1,5,22); there is no Fisher whitening.", "",
        "## In-sample coefficient tests", "",
        "### Common-tail one-direction model", "",
        "| horizon | estimate | HAC(22) SE | t | p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in common["parameters"]:
        lines.append(
            f"| {row['horizon']} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_reference']:.9g} | "
            f"{row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines.extend([
        "",
        f"- Three-coefficient joint Wald p: {common['joint_all']['p']:.9g}.",
        f"- Training log-likelihood gain over fitted static spline: {common['gain_vs_static']:.9g}.",
        "", "### Separate left/right two-direction model", "",
        "| boundary | horizon | estimate | HAC(22) SE | t | p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in left_right["parameters"]:
        lines.append(
            f"| {row['block']} | {row['horizon']} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_reference']:.9g} | "
            f"{row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines.extend([
        "",
        f"- Left three-coefficient joint p: {left_right['joint_blocks']['left_tail']['p']:.9g}.",
        f"- Right three-coefficient joint p: {left_right['joint_blocks']['right_tail']['p']:.9g}.",
        f"- All-six joint p: {left_right['joint_all']['p']:.9g}.",
        f"- Training log-likelihood gain over fitted static spline: {left_right['gain_vs_static']:.9g}.",
        "", "## Strict OOS comparisons", "",
        "Positive improvement favors the proposed model.", "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ])
    for row in comparisons.to_dict(orient="records"):
        lines.append(
            f"| {row['proposed']} | {row['benchmark']} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    checks = result["checks"]
    lines.extend([
        "", "## Numerical checks", "",
        f"- fitted baseline versus uniform maximum weight difference: {checks['fitted_vs_uniform_max_weight_difference']:.6g}",
        f"- saved M2 training-log reproduction error: {checks['saved_model2_training_log_reproduction_error']:.6g}",
        f"- saved/current M2 daily-OOS score error: {checks['saved_model2_daily_oos_score_error']:.6g}",
        f"- common-tail zero-response fitted-weight error: {checks['common_zero_response_weight_error']:.6g}",
        f"- left/right zero-response fitted-weight error: {checks['left_right_zero_response_weight_error']:.6g}",
        f"- common-tail train/OOS continuation error: {checks['common_continuation_error']:.6g}",
        f"- left/right train/OOS continuation error: {checks['left_right_continuation_error']:.6g}",
    ])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    panel, dataset_info = five.load_dataset("sp500")
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= core.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows

    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)
    anchor = np.asarray(saved["static_skewt"]["state"], dtype=float)
    weights0 = np.asarray(saved["static_q0"]["weights"], dtype=float)
    betas = model2_betas(saved)

    model2_train = fitted.filter_modular_reference(
        panel.values[:training_rows], train_offsets, anchor, betas,
        weights0, knots, masses,
    )
    model2_oos = fitted.continue_modular(
        panel, n_train, anchor, betas, weights0, knots, masses, model2_train,
    )
    model2_error = float(abs(
        model2_train.composite_log_likelihood
        - saved["model2"]["dynamic_log_likelihood"]
    ))
    if model2_error > 1e-10:
        raise RuntimeError(f"saved M2 reproduction failed: {model2_error}")

    full_states = np.vstack((model2_train.states, model2_oos.states))
    full_scores = np.vstack((model2_train.scores, model2_oos.scores))
    full_u = core.dynamic_reference_pit(panel.values, panel.offsets, full_states)
    full_phi = core.normalized_bspline_basis(full_u, knots, core.DEGREE, masses)
    geometry = five.sequential_geometry(anchor, weights0, knots, masses, q_matrix)
    common_direction, left_direction, right_direction = endpoint_directions(q_matrix)
    common_transform = boundary_transform(
        common_direction[None, :], geometry.projection_ms
    )
    left_right_transform = boundary_transform(
        np.vstack((left_direction, right_direction)), geometry.projection_ms
    )

    print("SP500: fitting common boundary-tail HAR(1,5,22)", flush=True)
    common_fit, common_static, common_train = split.fit_block_arc_sm(
        full_phi[:training_rows], train_offsets, model2_train.scores,
        weights0, q_matrix, common_transform, ("common_tail",),
    )
    common_oos = fitted.continue_arc(
        full_phi[training_rows:], oos_offsets, model2_oos.scores,
        np.asarray(common_fit["coefficients"]), weights0, q_matrix,
        common_transform, common_train,
    )

    print("SP500: fitting separate left/right boundary-tail HAR(1,5,22)", flush=True)
    lr_fit, lr_static, lr_train = split.fit_block_arc_sm(
        full_phi[:training_rows], train_offsets, model2_train.scores,
        weights0, q_matrix, left_right_transform, ("left_tail", "right_tail"),
    )
    lr_oos = fitted.continue_arc(
        full_phi[training_rows:], oos_offsets, model2_oos.scores,
        np.asarray(lr_fit["coefficients"]), weights0, q_matrix,
        left_right_transform, lr_train,
    )

    common_full = split.filter_block_arc_sm(
        full_phi, panel.offsets, full_scores,
        np.asarray(common_fit["coefficients"]), weights0, q_matrix,
        common_transform,
    )
    lr_full = split.filter_block_arc_sm(
        full_phi, panel.offsets, full_scores,
        np.asarray(lr_fit["coefficients"]), weights0, q_matrix,
        left_right_transform,
    )
    common_continuation = float(max(
        np.max(np.abs(common_full.states[n_train:] - common_oos.states)),
        np.max(np.abs(common_full.weights[n_train:] - common_oos.weights)),
        np.max(np.abs(common_full.contributions[n_train:] - common_oos.contributions)),
    ))
    lr_continuation = float(max(
        np.max(np.abs(lr_full.states[n_train:] - lr_oos.states)),
        np.max(np.abs(lr_full.weights[n_train:] - lr_oos.weights)),
        np.max(np.abs(lr_full.contributions[n_train:] - lr_oos.contributions)),
    ))

    values_oos = panel.values[training_rows:]
    dates_oos = dates[n_train:]
    common_scores = scoring.score_oos_pair(
        values_oos, oos_offsets, dates_oos, panel.values[:training_rows],
        model2_oos.as_reference(), common_oos, knots, masses, weights0,
        M2, COMMON,
    )
    lr_scores = scoring.score_oos_pair(
        values_oos, oos_offsets, dates_oos, panel.values[:training_rows],
        model2_oos.as_reference(), lr_oos, knots, masses, weights0,
        M2, LEFT_RIGHT,
    )
    scores = pd.concat((
        common_scores.loc[common_scores["model"] == M2],
        common_scores.loc[common_scores["model"] == COMMON],
        lr_scores.loc[lr_scores["model"] == LEFT_RIGHT],
    ), ignore_index=True)
    locked_scores = pd.read_csv(fitted.OUTPUT_ROOT / "sp500" / "daily_oos_scores.csv")
    locked_m2 = locked_scores.loc[locked_scores["model"] == M2].sort_values("date")
    current_m2 = scores.loc[scores["model"] == M2].sort_values("date")
    if locked_m2["date"].tolist() != current_m2["date"].tolist():
        raise RuntimeError("saved/current M2 OOS dates do not align")
    m2_daily_error = float(np.max(np.abs(
        locked_m2[list(scoring.SCORE_NAMES)].to_numpy(dtype=float)
        - current_m2[list(scoring.SCORE_NAMES)].to_numpy(dtype=float)
    )))
    if m2_daily_error > 1e-10:
        raise RuntimeError(f"saved/current M2 OOS scores differ: {m2_daily_error}")
    scores = pd.concat((
        scores,
        locked_scores.loc[locked_scores["model"] == M3],
    ), ignore_index=True)
    comparisons = comparison_rows(scores)

    common_fit["direction"] = {
        "density_effect": "equal logit shift of the first and last clamped-cubic basis weights",
        "q_coordinate": common_direction.tolist(),
    }
    lr_fit["directions"] = {
        "left_q_coordinate": left_direction.tolist(),
        "right_q_coordinate": right_direction.tolist(),
    }
    result = {
        "analysis": "S&P 500 fitted-static-q0 parsimonious boundary-tail DCD-SM",
        "dataset": dataset_info,
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
        },
        "specification": {
            "baseline": "fitted static skew-t and fitted static spline w0",
            "weight_chart": "w(q)=softmax(log(w0)+Qq)",
            "basis_breakpoints": list(core.BREAKPOINTS),
            "reference_path": "frozen fitted-static-q0 M2 location/scale path",
            "score_projection": "fixed-anchor Fisher projection off location/log-scale-squared",
            "score_whitening": False,
            "state_updates": "HAR(1,5,22), diagonal across boundary directions",
        },
        "common_tail": common_fit,
        "left_right_tail": lr_fit,
        "oos_summary": scores.groupby("model", as_index=False)[list(scoring.SCORE_NAMES)].mean().to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": {
            "fitted_vs_uniform_max_weight_difference": float(np.max(np.abs(weights0 - masses))),
            "saved_model2_training_log_reproduction_error": model2_error,
            "saved_model2_daily_oos_score_error": m2_daily_error,
            "common_zero_response_weight_error": float(np.max(np.abs(common_static.weights - weights0[None, :]))),
            "left_right_zero_response_weight_error": float(np.max(np.abs(lr_static.weights - weights0[None, :]))),
            "common_continuation_error": common_continuation,
            "left_right_continuation_error": lr_continuation,
            "common_maximum_training_state_norm": float(np.max(np.linalg.norm(common_train.states, axis=1))),
            "left_right_maximum_training_state_norm": float(np.max(np.linalg.norm(lr_train.states, axis=1))),
            "common_minimum_oos_weight": float(np.min(common_oos.weights)),
            "left_right_minimum_oos_weight": float(np.min(lr_oos.weights)),
        },
    }
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    comparisons.to_csv(OUTPUT / "oos_comparisons.csv", index=False)
    pd.DataFrame(common_fit["parameters"]).to_csv(OUTPUT / "common_tail_parameters.csv", index=False)
    pd.DataFrame(lr_fit["parameters"]).to_csv(OUTPUT / "left_right_tail_parameters.csv", index=False)
    (OUTPUT / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_results(result, comparisons)


if __name__ == "__main__":
    main()
