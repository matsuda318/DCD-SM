"""Common/asymmetric endpoint DCD-SM for S&P 500 and N225.

The fitted static SM uses extreme-tail breakpoints at 0.01/0.99 in addition
to 0.05/0.95 and deciles.  Locked M2/M3 estimates remain reference-only.
The M4 score is projected off location/log-scale-squared and restricted to
the common endpoint direction d_C=d_L+d_R and the asymmetric endpoint
direction d_A=d_R-d_L.  Both use diagonal HAR(1,5,22) responses and no
whitening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_skewt_boundary_tail_m4_sp500 as boundary
import run_static_q0_skewt_four_models as fitted
import run_static_q0_symmetric_t_four_models as q0
import run_static_sm_boundary_tail_har_tail_knots_batch as base


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_sm_common_asymmetric_extreme_knots_markets"
DATASETS = ("sp500", "n225")
BREAKPOINTS = np.array(
    [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
     0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    dtype=float,
)

M2 = base.M2
M3 = base.M3
COMMON_ONLY = "m4_extreme_common_endpoint_arc_sm"
COMMON_ASYMMETRIC = "m4_extreme_common_asymmetric_endpoint_arc_sm"


def location_scale_projection(
    anchor: np.ndarray,
    weights0: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    q_matrix: np.ndarray,
    quadrature_order: int = 128,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute only the q-on-location/scale projection required by this model.

    The general sequential geometry also constructs eta and full non-eta
    complements.  Extreme fitted single-stock weights can make that unused
    complement nearly rank deficient.  This restricted calculation evaluates
    exactly the same location/scale projection without constructing or
    regularizing irrelevant directions.
    """
    nodes, legendre_weights = np.polynomial.legendre.leggauss(quadrature_order)
    q_parts, ms_parts, measure_parts = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2.0 + (right - left) * nodes / 2.0
        du_weight = legendre_weights * (right - left) / 2.0
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        h0 = phi @ weights0
        responsibility = phi * weights0[None, :] / h0[:, None]
        q_parts.append((responsibility - weights0[None, :]) @ q_matrix)
        z = core.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        ms_parts.append(core.reference_logpdf_and_score(y, anchor)[1][:, :2])
        measure_parts.append(du_weight * h0)
    q_score = np.vstack(q_parts)
    ms_score = np.vstack(ms_parts)
    measure = np.concatenate(measure_parts)
    information_ms = (ms_score.T * measure) @ ms_score
    eigenvalues = np.linalg.eigvalsh((information_ms + information_ms.T) / 2.0)
    if eigenvalues[0] <= 1e-12:
        raise RuntimeError("nonpositive location/scale information")
    cross = (q_score.T * measure) @ ms_score
    projection = cross @ np.linalg.inv(information_ms)
    residual = q_score - ms_score @ projection.T
    return projection, {
        "quadrature_mass_error": float(abs(np.sum(measure) - 1.0)),
        "location_scale_information_eigenvalues": eigenvalues.tolist(),
        "maximum_post_location_scale_cross_information": float(
            np.max(np.abs((residual.T * measure) @ ms_score))
        ),
    }


def comparison_rows(
    scores: pd.DataFrame,
    common_only_name: str = COMMON_ONLY,
    common_asymmetric_name: str = COMMON_ASYMMETRIC,
) -> pd.DataFrame:
    pairs = (
        (common_only_name, M2),
        (common_only_name, M3),
        (common_asymmetric_name, M2),
        (common_asymmetric_name, M3),
        (common_asymmetric_name, common_only_name),
    )
    return pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in q0.SCORE_NAMES
    ])


def write_report(result: dict[str, object], comparisons: pd.DataFrame, output: Path) -> None:
    common = result["common_only"]
    ca = result["common_asymmetric"]
    lines = [
        f"# {result['label']}: {result['analysis']}", "",
        "## Specification", "",
        f"- Breakpoints: `{result['specification']['basis_breakpoints']}`.",
        "- Static SM is fitted and frozen; q=0 reproduces its nonuniform weights.",
        "- M2/M3 are the locked modular reference-score estimates, not full-density GAS updates.",
        "- q score is projected off location/log-scale-squared; no whitening.",
        "- d_C=d_L+d_R and d_A=d_R-d_L; each direction uses HAR(1,5,22).", "",
        "## Common-only model", "",
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
        f"- Common joint p: {common['joint_all']['p']:.9g}.",
        f"- Training correction-log gain: {common['gain_vs_static']:.9g}.",
        "", "## Joint common/asymmetric model", "",
        "| direction | horizon | estimate | HAC(22) SE | t | p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in ca["parameters"]:
        lines.append(
            f"| {row['block']} | {row['horizon']} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_reference']:.9g} | "
            f"{row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    lines.extend([
        "",
        f"- Common-direction joint p: {ca['joint_blocks']['common']['p']:.9g}.",
        f"- Asymmetric-direction joint p: {ca['joint_blocks']['asymmetric']['p']:.9g}.",
        f"- All-six joint p: {ca['joint_all']['p']:.9g}.",
        f"- Training correction-log gain: {ca['gain_vs_static']:.9g}.",
        "", "## Strict OOS", "",
        "Positive improvement favors the proposed model.", "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ])
    for row in comparisons.to_dict(orient="records"):
        lines.append(
            f"| {row['proposed']} | {row['benchmark']} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    lines.extend(["", "## Numerical checks", ""])
    for name, value in result["checks"].items():
        lines.append(f"- {name}: {value:.9g}")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(
    dataset: str,
    edge_basis_count: int = 1,
    output_root: Path = OUTPUT_ROOT,
    common_only_name: str = COMMON_ONLY,
    common_asymmetric_name: str = COMMON_ASYMMETRIC,
    breakpoints: np.ndarray = BREAKPOINTS,
    analysis_name: str = "extreme-knot common/asymmetric endpoint DCD-SM",
) -> dict[str, object]:
    output = output_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    panel, info, saved = base.load_inputs(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= base.locked.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_values = panel.values[:training_rows]
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows

    selected_breakpoints = np.asarray(breakpoints, dtype=float)
    knots = core.make_clamped_knots(selected_breakpoints, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)
    anchor = np.asarray(saved["model1"]["state"], dtype=float)

    print(f"{dataset}: fitting extreme-knot static SM", flush=True)
    weights0, static_q0 = q0.fit_static_q0(
        train_values, train_offsets, anchor, knots, masses, q_matrix
    )
    print(f"{dataset}: reproducing reference-only M2/M3", flush=True)
    (
        m2_train, m2_oos, m2_full,
        _m3_train, m3_oos, _m3_full, checks,
    ) = base.reference_paths(panel, n_train, anchor, saved)

    full_u = core.dynamic_reference_pit(panel.values, panel.offsets, m2_full.states)
    full_phi = core.normalized_bspline_basis(full_u, knots, core.DEGREE, masses)
    projection_ms, projection_diagnostics = location_scale_projection(
        anchor, weights0, knots, masses, q_matrix
    )
    if edge_basis_count < 1 or 2 * edge_basis_count > masses.size:
        raise ValueError("edge_basis_count is incompatible with the spline basis")
    e_left = np.zeros(masses.size)
    e_right = np.zeros(masses.size)
    e_left[:edge_basis_count] = 1.0
    e_right[-edge_basis_count:] = 1.0
    d_left = q_matrix.T @ e_left
    d_right = q_matrix.T @ e_right
    d_common = d_left + d_right
    d_asymmetric = d_right - d_left
    euclidean_inner_product = float(d_common @ d_asymmetric)
    if abs(euclidean_inner_product) > 1e-12:
        raise RuntimeError(f"common/asymmetric directions are not orthogonal: {euclidean_inner_product}")

    common_transform = boundary.boundary_transform(
        d_common[None, :], projection_ms
    )
    common_asymmetric_transform = boundary.boundary_transform(
        np.vstack((d_common, d_asymmetric)), projection_ms
    )

    print(f"{dataset}: fitting common-only extreme endpoint", flush=True)
    common_fit, common_static, common_train = split.fit_block_arc_sm(
        full_phi[:training_rows], train_offsets, m2_train.scores,
        weights0, q_matrix, common_transform, ("common",),
    )
    common_oos = fitted.continue_arc(
        full_phi[training_rows:], oos_offsets, m2_oos.scores,
        np.asarray(common_fit["coefficients"]), weights0, q_matrix,
        common_transform, common_train,
    )

    print(f"{dataset}: fitting joint common/asymmetric extreme endpoints", flush=True)
    ca_fit, ca_static, ca_train = split.fit_block_arc_sm(
        full_phi[:training_rows], train_offsets, m2_train.scores,
        weights0, q_matrix, common_asymmetric_transform,
        ("common", "asymmetric"),
    )
    ca_oos = fitted.continue_arc(
        full_phi[training_rows:], oos_offsets, m2_oos.scores,
        np.asarray(ca_fit["coefficients"]), weights0, q_matrix,
        common_asymmetric_transform, ca_train,
    )

    full_scores = np.vstack((m2_train.scores, m2_oos.scores))
    common_full = split.filter_block_arc_sm(
        full_phi, panel.offsets, full_scores,
        np.asarray(common_fit["coefficients"]), weights0, q_matrix,
        common_transform,
    )
    ca_full = split.filter_block_arc_sm(
        full_phi, panel.offsets, full_scores,
        np.asarray(ca_fit["coefficients"]), weights0, q_matrix,
        common_asymmetric_transform,
    )
    checks.update({
        "common_asymmetric_euclidean_inner_product": euclidean_inner_product,
        "common_zero_response_weight_error": float(
            np.max(np.abs(common_static.weights - weights0[None, :]))
        ),
        "common_asymmetric_zero_response_weight_error": float(
            np.max(np.abs(ca_static.weights - weights0[None, :]))
        ),
        "common_continuation_error": float(max(
            np.max(np.abs(common_full.states[n_train:] - common_oos.states)),
            np.max(np.abs(common_full.weights[n_train:] - common_oos.weights)),
            np.max(np.abs(common_full.contributions[n_train:] - common_oos.contributions)),
        )),
        "common_asymmetric_continuation_error": float(max(
            np.max(np.abs(ca_full.states[n_train:] - ca_oos.states)),
            np.max(np.abs(ca_full.weights[n_train:] - ca_oos.weights)),
            np.max(np.abs(ca_full.contributions[n_train:] - ca_oos.contributions)),
        )),
        "common_maximum_absolute_training_state": float(np.max(np.abs(common_train.states))),
        "common_maximum_absolute_oos_state": float(np.max(np.abs(common_oos.states))),
        "common_minimum_training_weight": float(np.min(common_train.weights)),
        "common_minimum_oos_weight": float(np.min(common_oos.weights)),
        "common_asymmetric_maximum_absolute_training_state": float(np.max(np.abs(ca_train.states))),
        "common_asymmetric_maximum_absolute_oos_state": float(np.max(np.abs(ca_oos.states))),
        "common_asymmetric_minimum_training_weight": float(np.min(ca_train.weights)),
        "common_asymmetric_minimum_oos_weight": float(np.min(ca_oos.weights)),
    })

    values_oos = panel.values[training_rows:]
    dates_oos = dates[n_train:]
    common_scores = q0.score_oos_pair(
        values_oos, oos_offsets, dates_oos, train_values, m2_oos,
        common_oos, knots, masses, weights0, M2, common_only_name,
    )
    ca_scores = q0.score_oos_pair(
        values_oos, oos_offsets, dates_oos, train_values, m2_oos,
        ca_oos, knots, masses, weights0, M2, common_asymmetric_name,
    )
    m3_reference = base.identity.shape_as_reference(m3_oos)
    m3_u = core.dynamic_reference_pit(values_oos, oos_offsets, m3_reference.states)
    m3_phi = core.normalized_bspline_basis(m3_u, knots, core.DEGREE, masses)
    m3_static = five.static_correction(m3_phi, oos_offsets, weights0)
    m3_scores = q0.score_oos_pair(
        values_oos, oos_offsets, dates_oos, train_values, m3_reference,
        m3_static, knots, masses, weights0, M3, "_discard",
    )
    scores = pd.concat((
        common_scores.loc[common_scores["model"] == M2],
        common_scores.loc[common_scores["model"] == common_only_name],
        ca_scores.loc[ca_scores["model"] == common_asymmetric_name],
        m3_scores.loc[m3_scores["model"] == M3],
    ), ignore_index=True)
    comparisons = comparison_rows(scores, common_only_name, common_asymmetric_name)

    result = {
        "analysis": analysis_name,
        "dataset": info,
        "label": base.dataset_label(dataset),
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
        },
        "specification": {
            "basis_breakpoints": selected_breakpoints.tolist(),
            "basis_count": int(masses.size),
            "static_sm": "fitted and frozen; q=0 reproduces fitted nonuniform h0",
            "m2_m3_estimation": (
                "modular reference-score stages; exact for the stated modular recursion, "
                "not full corrected-density GAS"
            ),
            "directions": "d_C=d_L+d_R; d_A=d_R-d_L",
            "edge_basis_count_per_side": int(edge_basis_count),
            "score_projection": "fixed-anchor Fisher projection off location/log-scale-squared",
            "score_whitening": False,
            "updates": "diagonal HAR(1,5,22) for each scalar direction",
        },
        "static_q0": static_q0,
        "location_scale_projection": projection_diagnostics,
        "common_only": common_fit,
        "common_asymmetric": ca_fit,
        "oos_summary": scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean().to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": checks,
    }
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    comparisons.to_csv(output / "oos_comparisons.csv", index=False)
    pd.DataFrame(common_fit["parameters"]).to_csv(output / "common_only_parameters.csv", index=False)
    pd.DataFrame(ca_fit["parameters"]).to_csv(output / "common_asymmetric_parameters.csv", index=False)
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(result, comparisons, output)
    print(f"{dataset}: complete", flush=True)
    return result


def write_combined(results: list[dict[str, object]]) -> None:
    rows = []
    knot_rows = []
    for result in results:
        lookup = {
            (row["proposed"], row["benchmark"], row["score"]): row
            for row in result["oos_comparisons"]
        }
        ca = result["common_asymmetric"]
        rows.append({
            "dataset": result["dataset"].get("dataset", result["label"]),
            "label": result["label"],
            "common_only_joint_p": result["common_only"]["joint_all"]["p"],
            "joint_model_common_p": ca["joint_blocks"]["common"]["p"],
            "joint_model_asymmetric_p": ca["joint_blocks"]["asymmetric"]["p"],
            "joint_model_all_p": ca["joint_all"]["p"],
            "common_only_log_gain_vs_m3": lookup[(COMMON_ONLY, M3, "continuous_log_score")]["improvement_mean"],
            "common_only_log_p_vs_m3": lookup[(COMMON_ONLY, M3, "continuous_log_score")]["hac22_two_sided_p"],
            "common_only_crps_gain_vs_m3": lookup[(COMMON_ONLY, M3, "crps")]["improvement_mean"],
            "common_only_crps_p_vs_m3": lookup[(COMMON_ONLY, M3, "crps")]["hac22_two_sided_p"],
            "joint_log_gain_vs_m3": lookup[(COMMON_ASYMMETRIC, M3, "continuous_log_score")]["improvement_mean"],
            "joint_log_p_vs_m3": lookup[(COMMON_ASYMMETRIC, M3, "continuous_log_score")]["hac22_two_sided_p"],
            "joint_crps_gain_vs_m3": lookup[(COMMON_ASYMMETRIC, M3, "crps")]["improvement_mean"],
            "joint_crps_p_vs_m3": lookup[(COMMON_ASYMMETRIC, M3, "crps")]["hac22_two_sided_p"],
        })
        dataset = str(result["dataset"].get("dataset", "")).lower()
        old = json.loads(
            (base.OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8")
        )
        old_lookup = {
            (row["proposed"], row["benchmark"], row["score"]): row
            for row in old["oos_comparisons"]
        }
        _panel, _info, locked_result = base.load_inputs(dataset)
        knot_rows.append({
            "dataset": dataset,
            "label": result["label"],
            "m3_eta_joint_p": locked_result["model3"]["wald_tests"]["eta_three_coefficients"]["p"],
            "five_percent_common_joint_p": old["common_tail"]["joint_all"]["p"],
            "one_percent_common_joint_p": result["common_only"]["joint_all"]["p"],
            "five_percent_log_gain_vs_m3": old_lookup[(base.COMMON, base.M3, "continuous_log_score")]["improvement_mean"],
            "one_percent_log_gain_vs_m3": lookup[(COMMON_ONLY, M3, "continuous_log_score")]["improvement_mean"],
            "five_percent_crps_gain_vs_m3": old_lookup[(base.COMMON, base.M3, "crps")]["improvement_mean"],
            "one_percent_crps_gain_vs_m3": lookup[(COMMON_ONLY, M3, "crps")]["improvement_mean"],
        })
    frame = pd.DataFrame(rows)
    knot_frame = pd.DataFrame(knot_rows)
    frame.to_csv(OUTPUT_ROOT / "combined_summary.csv", index=False)
    knot_frame.to_csv(OUTPUT_ROOT / "knot_comparison.csv", index=False)
    lines = [
        "# Extreme-knot common/asymmetric endpoint DCD-SM", "",
        "Positive OOS gains favor endpoint DCD-SM over M3.", "",
        "| market | common-only p | joint common p | asymmetric p | all-six p | common log gain | p | common CRPS gain | p | C/A log gain | p | C/A CRPS gain | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['common_only_joint_p']:.4g} | "
            f"{row['joint_model_common_p']:.4g} | {row['joint_model_asymmetric_p']:.4g} | "
            f"{row['joint_model_all_p']:.4g} | {row['common_only_log_gain_vs_m3']:.6g} | "
            f"{row['common_only_log_p_vs_m3']:.4g} | {row['common_only_crps_gain_vs_m3']:.6g} | "
            f"{row['common_only_crps_p_vs_m3']:.4g} | {row['joint_log_gain_vs_m3']:.6g} | "
            f"{row['joint_log_p_vs_m3']:.4g} | {row['joint_crps_gain_vs_m3']:.6g} | "
            f"{row['joint_crps_p_vs_m3']:.4g} |"
        )
    lines.extend([
        "", "## Detection and OOS comparison with the preceding knot set", "",
        "The 5% specification uses endpoints 0,.05,.1,...,.9,.95,1; the 1% specification additionally uses .01/.99, so its endpoint direction is more localized.", "",
        "| market | M3 eta p | 5% common p | 1% common p | 5% log gain vs M3 | 1% log gain | 5% CRPS gain | 1% CRPS gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in knot_frame.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['m3_eta_joint_p']:.4g} | "
            f"{row['five_percent_common_joint_p']:.4g} | {row['one_percent_common_joint_p']:.4g} | "
            f"{row['five_percent_log_gain_vs_m3']:.6g} | {row['one_percent_log_gain_vs_m3']:.6g} | "
            f"{row['five_percent_crps_gain_vs_m3']:.6g} | {row['one_percent_crps_gain_vs_m3']:.6g} |"
        )
    lines.extend([
        "", "The 1% endpoint direction greatly strengthens in-sample rejection relative",
        "to both the preceding 5% endpoint model and M3's eta test. It simultaneously",
        "weakens OOS performance, especially for S&P 500. Thus the added knots improve",
        "detection but do not provide evidence of improved predictive regularization.",
    ])
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    selected = DATASETS if args.market == "both" else (args.market,)
    if args.summarize_only:
        results = [
            json.loads((OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8"))
            for dataset in selected
        ]
    else:
        results = [run_dataset(dataset) for dataset in selected]
    if selected == DATASETS:
        write_combined(results)


if __name__ == "__main__":
    main()
