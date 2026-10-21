"""Five-model identity-SM experiment with the locked endpoint q updates.

M1 fixes a static skew-t reference and h=1. M2 updates location/log variance by
the exact reference score. M3 freezes M2 and updates eta/lambda by reference
scores projected off location/log variance. M4 freezes M2 and adds the locked
one-endpoint common/even and asymmetric/odd DCD-SM after projecting q off the
two location/scale tangents. M5 freezes M3 and uses the same endpoint update
after projecting q off location/log variance only; eta/lambda directions are
deliberately retained. There is no static SM fit and no Fisher whitening.
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
import run_static_q0_skewt_four_models as fitted
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_standard_even_odd_model5 as endpoint
import run_static_sm_standard_even_odd_sectors as sectors


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_identity_sm_standard_endpoint_five_models"
M1 = "m1_static_skewt_identity_sm"
M2 = "m2_dynamic_location_scale_identity_sm"
M3 = "m3_dynamic_eta_lambda_identity_sm"
M4 = "m4_identity_sm_standard_endpoint_even_odd_after_m2"
M5 = "m5_identity_sm_standard_endpoint_even_odd_after_m3"
ALL_DATASETS = endpoint.ALL_DATASETS


def load_dataset(dataset: str):
    if dataset in endpoint.SECTORS:
        panel, info, saved = sectors.load_inputs(dataset)
        label = sectors.dataset_label(dataset)
    else:
        panel, info, saved = base.load_inputs(dataset)
        label = base.dataset_label(dataset)
    return panel, info, saved, label


def fit_endpoint(
    phi: np.ndarray,
    panel: core.RawPanel,
    n_train: int,
    train_scores: np.ndarray,
    oos_scores: np.ndarray,
    full_scores: np.ndarray,
    weights0: np.ndarray,
    q_matrix: np.ndarray,
    transform,
    label: str,
):
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    print(f"{label}: fitting", flush=True)
    fit, static, training = split.fit_block_arc_sm(
        phi[:training_rows], train_offsets, train_scores,
        weights0, q_matrix, transform, ("common_even", "asymmetric_odd"),
    )
    oos = fitted.continue_arc(
        phi[training_rows:], oos_offsets, oos_scores,
        np.asarray(fit["coefficients"], dtype=float), weights0, q_matrix,
        transform, training,
    )
    full = split.filter_block_arc_sm(
        phi, panel.offsets, full_scores,
        np.asarray(fit["coefficients"], dtype=float), weights0, q_matrix,
        transform,
    )
    checks = {
        "q_zero_weight_error": float(np.max(np.abs(static.weights - weights0[None, :]))),
        "continuation_error": float(max(
            np.max(np.abs(full.states[n_train:] - oos.states)),
            np.max(np.abs(full.weights[n_train:] - oos.weights)),
            np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
        )),
        "maximum_absolute_training_state": float(np.max(np.abs(training.states))),
        "maximum_absolute_oos_state": float(np.max(np.abs(oos.states))),
        "minimum_training_weight": float(np.min(training.weights)),
        "minimum_oos_weight": float(np.min(oos.weights)),
    }
    return fit, training, oos, full, checks


def score_models(
    panel: core.RawPanel,
    dates: pd.DatetimeIndex,
    n_train: int,
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    m2_oos,
    m3_oos,
    m4_oos,
    m5_oos,
) -> pd.DataFrame:
    training_rows = int(panel.offsets[n_train])
    values = panel.values[training_rows:]
    offsets = panel.offsets[n_train:] - training_rows
    oos_dates = dates[n_train:]
    static = core.filter_reference(values, offsets, anchor, np.zeros((3, 4)))
    static_u = core.dynamic_reference_pit(values, offsets, static.states)
    static_phi = core.normalized_bspline_basis(static_u, knots, core.DEGREE, masses)
    identity = five.static_correction(static_phi, offsets, masses)
    m1 = scoring.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], static,
        identity, knots, masses, masses, M1, "_discard_m1",
    )
    m2_m4 = scoring.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], m2_oos,
        m4_oos, knots, masses, masses, M2, M4,
    )
    m3_reference = base.identity.shape_as_reference(m3_oos)
    m3_m5 = scoring.score_oos_pair(
        values, offsets, oos_dates, panel.values[:training_rows], m3_reference,
        m5_oos, knots, masses, masses, M3, M5,
    )
    scores = pd.concat((
        m1.loc[m1["model"] == M1], m2_m4, m3_m5,
    ), ignore_index=True)
    if scores.groupby("date")["model"].nunique().min() != 5:
        raise RuntimeError("incomplete five-model OOS score panel")
    return scores


def compare(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = ((M2, M1), (M3, M2), (M4, M2), (M5, M3), (M4, M3), (M5, M4))
    return pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in scoring.SCORE_NAMES
    ])


def run_dataset(dataset: str) -> dict[str, object]:
    panel, info, saved, label = load_dataset(dataset)
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= base.locked.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    anchor = np.asarray(saved["model1"]["state"], dtype=float)
    knots = core.make_clamped_knots(base.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)

    print(f"{dataset}: reproducing exact identity-SM M1--M3 paths", flush=True)
    m2_train, m2_oos, m2_full, m3_train, m3_oos, m3_full, reference_checks = (
        base.reference_paths(panel, n_train, anchor, saved)
    )
    d_common, d_asymmetric = endpoint.endpoint_directions(q_matrix)
    directions = np.vstack((d_common, d_asymmetric))

    m2_u = core.dynamic_reference_pit(panel.values, panel.offsets, m2_full.states)
    m2_phi = core.normalized_bspline_basis(m2_u, knots, core.DEGREE, masses)
    geometry2 = scoring.corrected_tangent_geometry(
        anchor, masses, knots, masses, q_matrix, (0, 1),
        "identity-h q score off exact location/log-sigma2 reference scores",
    )
    transform2 = endpoint.four_tangent_endpoint_transform(
        directions, geometry2.projection_q_on_hansen
    )
    m4, _m4_train, m4_oos, _m4_full, checks4 = fit_endpoint(
        m2_phi, panel, n_train, m2_train.scores, m2_oos.scores, m2_full.scores,
        masses, q_matrix, transform2, f"{dataset}: M4 endpoint after M2",
    )

    m3_u = core.dynamic_reference_pit(panel.values, panel.offsets, m3_full.states)
    m3_phi = core.normalized_bspline_basis(m3_u, knots, core.DEGREE, masses)
    geometry3 = scoring.corrected_tangent_geometry(
        anchor, masses, knots, masses, q_matrix, (0, 1),
        "identity-h q score off exact m/log-sigma2 reference scores only; "
        "eta/lambda retained",
    )
    transform3 = endpoint.four_tangent_endpoint_transform(
        directions, geometry3.projection_q_on_hansen
    )
    m5, _m5_train, m5_oos, _m5_full, checks5 = fit_endpoint(
        m3_phi, panel, n_train, m3_train.scores, m3_oos.scores, m3_full.scores,
        masses, q_matrix, transform3, f"{dataset}: M5 endpoint after M3",
    )

    scores = score_models(
        panel, dates, n_train, anchor, knots, masses,
        m2_oos, m3_oos, m4_oos, m5_oos,
    )
    comparisons = compare(scores)
    checks = {
        **reference_checks,
        "identity_density_error": float(np.max(np.abs(
            core.normalized_bspline_basis(
                np.linspace(1e-6, 1 - 1e-6, 1001), knots, core.DEGREE, masses
            ) @ masses - 1.0
        ))),
        "common_asymmetric_euclidean_inner_product": float(d_common @ d_asymmetric),
        "m4": checks4,
        "m5": checks5,
    }
    scalar_errors = [
        checks["identity_density_error"],
        abs(checks["common_asymmetric_euclidean_inner_product"]),
        checks4["q_zero_weight_error"], checks4["continuation_error"],
        checks5["q_zero_weight_error"], checks5["continuation_error"],
    ]
    if max(scalar_errors) > 2e-10:
        raise RuntimeError(f"{dataset}: nesting/continuation failure: {checks}")

    result = {
        "analysis": "identity-SM locked-standard endpoint M1--M5 comparison",
        "dataset": info,
        "label": label,
        "category": endpoint.category(dataset),
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
        },
        "specification": {
            "basis_breakpoints": base.BREAKPOINTS.tolist(),
            "static_sm": "identity h(u)=1; weights equal spline masses; no SM estimation",
            "M1": "static skew-t MLE",
            "M2": "M1 plus exact reference-score m/log-sigma2 HAR(1,5,22)",
            "M3": "M2 frozen; exact reference-score eta/lambda HAR after m/sigma projection",
            "M4": "M2 frozen; endpoint q HAR after m/sigma projection",
            "M5": (
                "M3 frozen; endpoint q HAR after m/sigma projection; "
                "eta/lambda directions deliberately retained"
            ),
            "endpoint_directions": "one endpoint basis per side; common/even and asymmetric/odd",
            "score_whitening": False,
            "estimation": "sequential conditional stages",
        },
        "source_model1": saved["model1"],
        "source_model2": saved["model2"],
        "source_model3": saved["model3"],
        "model4": m4,
        "model5": m5,
        "model4_geometry": geometry2.diagnostics,
        "model5_geometry": geometry3.diagnostics,
        "oos_summary": scores.groupby("model", as_index=False)[list(scoring.SCORE_NAMES)].mean().to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": checks,
    }
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    comparisons.to_csv(output / "oos_comparisons.csv", index=False)
    pd.DataFrame(m4["parameters"]).to_csv(output / "model4_parameters.csv", index=False)
    pd.DataFrame(m5["parameters"]).to_csv(output / "model5_parameters.csv", index=False)
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{dataset}: complete; M4 p={m4['joint_all']['p']:.6g}; "
        f"M5 p={m5['joint_all']['p']:.6g}", flush=True,
    )
    return result


def summarize() -> None:
    results = [
        json.loads((OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8"))
        for dataset in ALL_DATASETS
    ]
    tests, parameters, comparisons, diagnostics, levels = [], [], [], [], []
    for dataset, result in zip(ALL_DATASETS, results):
        for model_key in ("model4", "model5"):
            fit = result[model_key]
            tests.append({
                "dataset": dataset, "label": result["label"], "category": result["category"],
                "model": model_key.upper(),
                "even_p": fit["joint_blocks"]["common_even"]["p"],
                "odd_p": fit["joint_blocks"]["asymmetric_odd"]["p"],
                "all_six_p": fit["joint_all"]["p"],
            })
            parameters.extend([
                {"dataset": dataset, "label": result["label"], "category": result["category"],
                 "model": model_key.upper(), **row}
                for row in fit["parameters"]
            ])
            diagnostics.append({
                "dataset": dataset, "label": result["label"], "category": result["category"],
                "model": model_key.upper(),
                "successful_starts": fit["successful_starts"],
                "total_starts": fit["total_starts"],
                "largest_successful_objective_gap": fit["largest_successful_objective_gap"],
                **result["checks"][{"model4": "m4", "model5": "m5"}[model_key]],
            })
        comparisons.extend([
            {"dataset": dataset, "label": result["label"], "category": result["category"], **row}
            for row in result["oos_comparisons"]
        ])
        levels.extend([
            {"dataset": dataset, "label": result["label"], "category": result["category"], **row}
            for row in result["oos_summary"]
        ])
    frames = {
        "score_tests": pd.DataFrame(tests),
        "coefficients": pd.DataFrame(parameters),
        "oos_comparisons": pd.DataFrame(comparisons),
        "numerical_diagnostics": pd.DataFrame(diagnostics),
        "oos_levels": pd.DataFrame(levels),
    }
    for name, frame in frames.items():
        frame.to_csv(OUTPUT_ROOT / f"identity_endpoint_five_model_{name}.csv", index=False)
    write_summary(frames)


def write_summary(frames: dict[str, pd.DataFrame]) -> None:
    tests = frames["score_tests"]
    comparisons = frames["oos_comparisons"]
    lines = [
        "# Identity-SM standard-endpoint M1--M5 results", "",
        "All models use h=1 at q=0; no static SM is estimated. M2/M3 reference",
        "updates are therefore exact conditional score stages. M4 and M5 use the",
        "locked one-endpoint common/even and asymmetric/odd HAR(1,5,22) q updates.", "",
        "## In-sample rejection counts", "",
        "| category | model | N | even | odd | all six |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in ("sp500", "n225", "sector", "stock", "fx"):
        for model in ("MODEL4", "MODEL5"):
            subset = tests.loc[(tests["category"] == group) & (tests["model"] == model)]
            lines.append(
                f"| {group} | {model} | {len(subset)} | {int((subset['even_p'] < .05).sum())} | "
                f"{int((subset['odd_p'] < .05).sum())} | {int((subset['all_six_p'] < .05).sum())} |"
            )
    lines.extend([
        "", "## OOS point-estimate wins", "",
        "Positive improvement favors the proposed model.", "",
        "| category | comparison | score | wins | N |",
        "|---|---|---|---:|---:|",
    ])
    pairs = ((M3, M2), (M4, M2), (M5, M3), (M5, M4))
    for group in ("sp500", "n225", "sector", "stock", "fx"):
        for proposed, benchmark in pairs:
            for score in ("continuous_log_score", "crps"):
                subset = comparisons.loc[
                    (comparisons["category"] == group)
                    & (comparisons["proposed"] == proposed)
                    & (comparisons["benchmark"] == benchmark)
                    & (comparisons["score"] == score)
                ]
                lines.append(
                    f"| {group} | {proposed} vs {benchmark} | {score} | "
                    f"{int((subset['improvement_mean'] > 0).sum())} | {len(subset)} |"
                )
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for dataset in endpoint.selected_datasets(args.dataset):
            run_dataset(dataset)
    if args.summarize_only or args.dataset == "all":
        summarize()


if __name__ == "__main__":
    main()
