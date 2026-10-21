"""Five-model skew-t/DCD-SM experiment with identity static SM, h0(u)=1.

M1: static skew-t.
M2: dynamic m and log sigma^2 with eta/lambda fixed.
M3: M2 plus dynamic eta/lambda; their reference scores are projected off m/sigma.
M4: M2 plus DCD-SM, split into the eta-aligned and remaining q-score directions.
M5: M3 plus one DCD-SM residual-vector update after projecting q off all four
    reference tangents (m, log sigma^2, eta, lambda).

Because h0 is identically one, all reference update scores are exactly scores
of log f_theta; there is no omitted PIT derivative.  Projection matrices are
fixed at the static skew-t MLE.  There is no Fisher whitening or parity split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_direct_dynamic_skewt_shape_test as direct
import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_single_stocks as stocks
import run_static_q0_symmetric_t_four_models as q0


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_identity_sm_skewt_five_models"
DATASETS = five.DATASETS
MODEL_NAMES = {
    "m1": "m1_static_skewt_identity_sm",
    "m2": "m2_dynamic_location_scale_fixed_eta_lambda_identity_sm",
    "m3": "m3_dynamic_location_scale_eta_lambda_identity_sm",
    "m4": "m4_dynamic_location_scale_fixed_eta_lambda_qeta_others_arc_sm",
    "m5": "m5_dynamic_location_scale_eta_lambda_residual_arc_sm",
}
LABELS = {
    MODEL_NAMES["m1"]: "M1: static skew-t",
    MODEL_NAMES["m2"]: "M2: dynamic m,sigma; fixed eta,lambda",
    MODEL_NAMES["m3"]: "M3: M2 + dynamic eta,lambda",
    MODEL_NAMES["m4"]: "M4: M2 + q_eta/others DCD-SM",
    MODEL_NAMES["m5"]: "M5: M3 + four-tangent residual DCD-SM",
}


def saved_direct(dataset: str) -> dict[str, object]:
    path = direct.OUTPUT_ROOT / dataset / "estimation.json"
    if not path.exists():
        raise FileNotFoundError(f"missing direct M1-M3 result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def reference_betas(saved: dict[str, object]) -> np.ndarray:
    return np.asarray(saved["dynamic_location_scale_betas_matrix"], dtype=float)


def shape_betas(saved: dict[str, object]) -> np.ndarray:
    output = np.empty((3, 2))
    source = saved["dynamic_eta_lambda"]["betas"]
    for index, horizon in enumerate(core.HORIZONS):
        output[index] = (source[horizon]["eta"], source[horizon]["lambda"])
    return output


def continue_reference(
    panel: core.RawPanel, n_train: int, anchor: np.ndarray, betas: np.ndarray
) -> tuple[core.FourStateFilter, core.FourStateFilter, core.FourStateFilter, float]:
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    training = core.filter_reference(panel.values[:training_rows], train_offsets, anchor, betas)
    oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, betas,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full = core.filter_reference(panel.values, panel.offsets, anchor, betas)
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"M2 continuation error: {error}")
    return training, oos, full, error


def continue_shape(
    panel: core.RawPanel,
    n_train: int,
    m2_oos: core.FourStateFilter,
    m2_full: core.FourStateFilter,
    anchor: np.ndarray,
    betas: np.ndarray,
    geometry: direct.ShapeGeometry,
    training: direct.ShapeFilter,
) -> tuple[direct.ShapeFilter, direct.ShapeFilter, float]:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = direct.filter_shape_layer(
        panel.values[training_rows:], oos_offsets, m2_oos, anchor, betas, geometry,
        initial_shape_state=training.terminal_shape_state,
        initial_score_history=training.terminal_score_history,
    )
    full = direct.filter_shape_layer(
        panel.values, panel.offsets, m2_full, anchor, betas, geometry
    )
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"M3 continuation error: {error}")
    return oos, full, error


def shape_as_reference(filtered: direct.ShapeFilter) -> core.FourStateFilter:
    parameters = np.array([
        core.state_to_parameters(state[2:])[:2] for state in filtered.states
    ])
    return core.FourStateFilter(
        states=filtered.states,
        means=filtered.states[:, 0],
        log_variances=filtered.states[:, 1],
        variances=np.exp(filtered.states[:, 1]),
        eta=parameters[:, 0],
        skew=parameters[:, 1],
        scores=filtered.scores,
        contributions=filtered.contributions,
        terminal_state=np.r_[filtered.states[-1, :2], filtered.terminal_shape_state],
        terminal_score_history=np.empty((0, 4)),
    )


def four_tangent_transform(geometry: core.EfficientGeometry):
    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        residual = raw_q_score - geometry.projection_q_on_hansen @ reference_score[:4]
        return residual[None, :]
    return transform


def score_models(
    panel: core.RawPanel,
    dates: pd.DatetimeIndex,
    n_train: int,
    anchor: np.ndarray,
    masses: np.ndarray,
    knots: np.ndarray,
    m2_oos: core.FourStateFilter,
    m3_oos: core.FourStateFilter,
    m4_oos: split.BlockFilter,
    m5_oos: split.BlockFilter,
) -> pd.DataFrame:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    zero_betas = np.zeros((3, 4))
    m1_oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, zero_betas
    )
    m1_u = core.dynamic_reference_pit(
        panel.values[training_rows:], oos_offsets, m1_oos.states
    )
    m1_phi = core.normalized_bspline_basis(m1_u, knots, core.DEGREE, masses)
    identity = five.static_correction(m1_phi, oos_offsets, masses)
    m1_pair = five.score_pair(
        panel, n_train, dates, m1_oos, identity, knots, masses, masses,
        MODEL_NAMES["m1"], "_discard_m1",
    )
    m2_m4 = five.score_pair(
        panel, n_train, dates, m2_oos, m4_oos, knots, masses, masses,
        MODEL_NAMES["m2"], MODEL_NAMES["m4"],
    )
    m3_m5 = five.score_pair(
        panel, n_train, dates, m3_oos, m5_oos, knots, masses, masses,
        MODEL_NAMES["m3"], MODEL_NAMES["m5"],
    )
    scores = pd.concat((
        m1_pair.loc[m1_pair["model"] == MODEL_NAMES["m1"]],
        m2_m4,
        m3_m5,
    ), ignore_index=True)
    if scores.groupby("date")["model"].nunique().min() != 5:
        raise RuntimeError("incomplete five-model strict-OOS panel")
    return scores


def run_dataset(dataset: str) -> dict[str, object]:
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    saved = saved_direct(dataset)
    panel, dataset_info = five.load_dataset(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    training_end = core.TRAINING_END if dataset in ("sp500", "n225") else stocks.TRAINING_END
    n_train = int(np.sum(dates <= training_end))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    anchor = np.asarray(saved["static_reference"]["state"], dtype=float)
    m2_betas = reference_betas(saved)
    m3_betas = shape_betas(saved)

    print(f"{dataset}: reproducing M2 identity-SM reference", flush=True)
    m2_train, m2_oos, m2_full, m2_continuation = continue_reference(
        panel, n_train, anchor, m2_betas
    )
    saved_m2_error = float(abs(
        m2_train.composite_log_likelihood
        - saved["dynamic_location_scale"]["dynamic_composite_log_likelihood"]
    ))
    if saved_m2_error > 1e-2:
        raise RuntimeError(f"material saved/current M2 mismatch: {saved_m2_error}")

    shape_geometry = direct.shape_geometry(anchor)
    print(f"{dataset}: reproducing M3 dynamic eta/lambda", flush=True)
    m3_nested = direct.filter_shape_layer(
        panel.values[:training_rows], train_offsets, m2_train, anchor,
        np.zeros((3, 2)), shape_geometry,
    )
    m3_train = direct.filter_shape_layer(
        panel.values[:training_rows], train_offsets, m2_train, anchor,
        m3_betas, shape_geometry,
    )
    saved_m3_error = float(abs(
        m3_train.composite_log_likelihood
        - saved["dynamic_eta_lambda"]["dynamic_shape_log_likelihood"]
    ))
    if saved_m3_error > 1e-2:
        raise RuntimeError(f"material saved/current M3 mismatch: {saved_m3_error}")
    m3_oos_filter, m3_full_filter, m3_continuation = continue_shape(
        panel, n_train, m2_oos, m2_full, anchor, m3_betas,
        shape_geometry, m3_train,
    )
    m3_train_reference = shape_as_reference(m3_train)
    m3_oos_reference = shape_as_reference(m3_oos_filter)
    m3_full_reference = shape_as_reference(m3_full_filter)

    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)
    m2_u = core.dynamic_reference_pit(panel.values, panel.offsets, m2_full.states)
    m3_u = core.dynamic_reference_pit(panel.values, panel.offsets, m3_full_filter.states)
    m2_phi = core.normalized_bspline_basis(m2_u, knots, core.DEGREE, masses)
    m3_phi = core.normalized_bspline_basis(m3_u, knots, core.DEGREE, masses)

    # Under h0=1 the eta/others geometry is evaluated under du.
    m4_geometry = five.sequential_geometry(anchor, masses, knots, masses, q_matrix)
    m4_geometry.diagnostics.update({
        "anchor_measure": "du because the static SM is h0(u)=1",
        "reference_score": (
            "exact derivative of log f_theta: the derivative of log h0 is "
            "identically zero"
        ),
    })
    print(f"{dataset}: fitting M4 q_eta/others at identity q0", flush=True)
    m4_fit, m4_oos, m4_checks = five.fit_and_continue_arc(
        m2_phi, panel, n_train, m2_train, m2_oos, m2_full,
        masses, q_matrix, five.eta_direction_transform(m4_geometry),
        ("q_eta", "others"),
    )

    # Project the q score off every reference tangent at the static skew-t MLE.
    m5_geometry = q0.corrected_tangent_geometry(
        anchor, masses, knots, masses, q_matrix, (0, 1, 2, 3),
        "identity-SM q score off m/log-sigma2/eta/lambda reference tangents",
    )
    m5_geometry.diagnostics.update({
        "anchor_measure": "du because the static SM is h0(u)=1",
        "q_zero_baseline": "uniform spline weights equal to their masses; h0(u)=1",
        "nuisance_score": (
            "exact reference-only score because the derivative of log h0 is "
            "identically zero"
        ),
        "parity_interpretation": "not used",
    })
    print(f"{dataset}: fitting M5 four-tangent residual DCD-SM", flush=True)
    m5_fit, m5_oos, m5_checks = five.fit_and_continue_arc(
        m3_phi, panel, n_train, m3_train_reference, m3_oos_reference,
        m3_full_reference, masses, q_matrix,
        four_tangent_transform(m5_geometry), ("residual_vector",),
    )

    scores = score_models(
        panel, dates, n_train, anchor, masses, knots, m2_oos,
        m3_oos_reference, m4_oos, m5_oos,
    )
    summary = scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean()
    pairs = (
        (MODEL_NAMES["m2"], MODEL_NAMES["m1"]),
        (MODEL_NAMES["m3"], MODEL_NAMES["m2"]),
        (MODEL_NAMES["m4"], MODEL_NAMES["m2"]),
        (MODEL_NAMES["m5"], MODEL_NAMES["m3"]),
        (MODEL_NAMES["m5"], MODEL_NAMES["m4"]),
    )
    comparisons = pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs for score in q0.SCORE_NAMES
    ])
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "five_model_oos_summary.csv", index=False)
    comparisons.to_csv(output / "five_model_oos_comparisons.csv", index=False)
    pd.DataFrame(saved["dynamic_eta_lambda"]["parameters"]).to_csv(
        output / "model3_eta_lambda_parameters.csv", index=False
    )
    pd.DataFrame(m4_fit["parameters"]).to_csv(output / "model4_arc_sm_parameters.csv", index=False)
    pd.DataFrame(m5_fit["parameters"]).to_csv(output / "model5_arc_sm_parameters.csv", index=False)

    m3_zero_error = float(max(
        np.max(np.abs(m3_nested.states - m2_train.states)),
        np.max(np.abs(m3_nested.contributions - m2_train.contributions)),
    ))
    result = {
        "analysis": f"{dataset} identity-static-SM skew-t five-model comparison",
        "dataset": dataset_info,
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
            "total_observations": int(panel.values.size),
        },
        "specification": {
            "static_sm": "identity h0(u)=1; equivalently uniform q0=0",
            "models": {
                "M1": "static skew-t MLE; eta and lambda fixed",
                "M2": "M1 plus dynamic m/log sigma2; then frozen",
                "M3": "M2 plus joint dynamic eta/lambda with their scores projected off m/log sigma2",
                "M4": "M2 plus q_eta/others DCD-SM after m/log sigma2 projection",
                "M5": "M3 plus one q residual-vector block after projection off m/log sigma2/eta/lambda",
            },
            "reference_score": "exact derivative of log f_skewt; h0 derivative is exactly zero because h0=1",
            "projection_measure": "static skew-t density, equivalently du",
            "projection_anchor": "static skew-t MLE",
            "score_whitening": False,
            "parity_split": False,
            "windows": list(core.WINDOWS),
            "estimation": "sequential/conditional: M2 fixed for M3/M4; M3 fixed for M5",
        },
        "model1": saved["static_reference"],
        "model2": saved["dynamic_location_scale"],
        "model3": saved["dynamic_eta_lambda"],
        "model4": m4_fit,
        "model5": m5_fit,
        "shape_geometry": shape_geometry.diagnostics,
        "model4_geometry": m4_geometry.diagnostics,
        "model5_geometry": m5_geometry.diagnostics,
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": {
            "saved_current_model2_likelihood_error": saved_m2_error,
            "saved_current_model3_likelihood_error": saved_m3_error,
            "model2_continuation_error": m2_continuation,
            "model3_zero_nesting_error": m3_zero_error,
            "model3_continuation_error": m3_continuation,
            "model4": m4_checks,
            "model5": m5_checks,
        },
    }
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output, result, summary, comparisons)
    print(summary.to_string(index=False), flush=True)
    return result


def write_report(
    output: Path,
    result: dict[str, object],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    lines = [
        f"# {result['analysis']}",
        "",
        "The static SM is the identity h0(u)=1. Consequently every reference",
        "update uses the exact score of log f_skewt, with no missing PIT term.",
        "All projections use the static skew-t Fisher measure; there is no",
        "whitening or parity split.",
        "",
        "## In-sample joint tests",
        "",
        "| block | p |",
        "|---|---:|",
        f"| M3 eta | {result['model3']['wald_tests']['eta_three_coefficients']['p']:.6g} |",
        f"| M3 lambda | {result['model3']['wald_tests']['lambda_three_coefficients']['p']:.6g} |",
        f"| M3 eta+lambda | {result['model3']['wald_tests']['eta_lambda_six_coefficients']['p']:.6g} |",
        f"| M4 q_eta | {result['model4']['joint_blocks']['q_eta']['p']:.6g} |",
        f"| M4 others | {result['model4']['joint_blocks']['others']['p']:.6g} |",
        f"| M4 all | {result['model4']['joint_all']['p']:.6g} |",
        f"| M5 residual | {result['model5']['joint_all']['p']:.6g} |",
        "",
        "## Strict-OOS levels",
        "",
        "| model | log score | CRPS |",
        "|---|---:|---:|",
    ]
    for row in summary.sort_values("continuous_log_score", ascending=False).to_dict("records"):
        lines.append(
            f"| {LABELS[row['model']]} | {row['continuous_log_score']:.9f} | {row['crps']:.9f} |"
        )
    lines += [
        "",
        "## Structural strict-OOS comparisons",
        "",
        "Positive improvement favors the proposed model.",
        "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ]
    selected = comparisons.loc[comparisons["score"].isin(("continuous_log_score", "crps"))]
    for row in selected.to_dict("records"):
        lines.append(
            f"| {LABELS[row['proposed']]} | {LABELS[row['benchmark']]} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    lines += [
        "",
        "All nine loss comparisons are in `five_model_oos_comparisons.csv`.",
        "Daily strict-OOS scores and complete coefficient tables are stored beside this report.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined(results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"]["label"],
            "m3_eta_p": result["model3"]["wald_tests"]["eta_three_coefficients"]["p"],
            "m3_lambda_p": result["model3"]["wald_tests"]["lambda_three_coefficients"]["p"],
            "m3_joint_p": result["model3"]["wald_tests"]["eta_lambda_six_coefficients"]["p"],
            "m4_q_eta_p": result["model4"]["joint_blocks"]["q_eta"]["p"],
            "m4_others_p": result["model4"]["joint_blocks"]["others"]["p"],
            "m4_joint_p": result["model4"]["joint_all"]["p"],
            "m5_residual_p": result["model5"]["joint_all"]["p"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "combined_tests.csv", index=False)
    lines = [
        "# Identity-static-SM skew-t five-model results",
        "",
        "Here h0(u)=1 exactly. Thus eta/lambda reference updates use only the",
        "reference score by construction, not by approximation. M3 and M4 are",
        "sibling branches from frozen M2; M5 is layered on frozen M3.",
        "",
        "| dataset | M3 eta | M3 lambda | M3 joint | M4 q_eta | M4 others | M4 joint | M5 residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['m3_eta_p']:.6g} | {row['m3_lambda_p']:.6g} | "
            f"{row['m3_joint_p']:.6g} | {row['m4_q_eta_p']:.6g} | "
            f"{row['m4_others_p']:.6g} | {row['m4_joint_p']:.6g} | "
            f"{row['m5_residual_p']:.6g} |"
        )
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
    if set(datasets) == set(DATASETS):
        write_combined(results)


if __name__ == "__main__":
    main()
