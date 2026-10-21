"""Identity-SM skew-t M1--M5 with parity DCD-SM Models 4 and 5.

The locked Models 1--3 are reused from the final identity-SM experiment.
Only Models 4 and 5 are re-estimated:

M4: M2 plus the spline score projected off location/log-scale and then split
    into even and odd PIT-space components, each with HAR(1,5,22) responses.
M5: M3 plus the spline score projected off location/log-scale/eta/lambda and
    then split into even and odd components, again with six responses.

All Fisher projections are evaluated once at the static skew-t training MLE.
There is no Fisher whitening.  The static spline is the identity h0(u)=1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_direct_dynamic_skewt_shape_test as direct
import run_identity_sm_skewt_eight_stocks as eight
import run_identity_sm_skewt_five_models as identity
import run_identity_sm_skewt_sector_five_models as sectors
import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_single_stocks as stocks
import run_static_q0_symmetric_t_four_models as q0


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_identity_sm_skewt_parity_five_models"
TRAINING_END = pd.Timestamp("2023-12-29")

BASE_DATASETS = ("sp500", "n225", "amzn", "toyota", "aapl", "ntt")
ADDED_STOCKS = tuple(eight.ASSETS)
SECTOR_DATASETS = tuple(sectors.SECTORS)
ALL_DATASETS = BASE_DATASETS + ADDED_STOCKS + SECTOR_DATASETS

MODEL_NAMES = {
    "m1": "m1_static_skewt_identity_sm",
    "m2": "m2_dynamic_location_scale_fixed_eta_lambda_identity_sm",
    "m3": "m3_dynamic_location_scale_eta_lambda_identity_sm",
    "m4": "m4_dynamic_location_scale_fixed_eta_lambda_parity_arc_sm",
    "m5": "m5_dynamic_location_scale_eta_lambda_residual_parity_arc_sm",
}
MODEL_LABELS = {
    "m1": "M1: static skew-t",
    "m2": "M2: dynamic m,sigma",
    "m3": "M3: M2 + dynamic eta,lambda",
    "m4": "M4: M2 + even/odd DCD-SM",
    "m5": "M5: M3 + residual even/odd DCD-SM",
}
SCORE_LABELS = {
    "continuous_log_score": "Log",
    "crps": "CRPS",
    "categorical_log_score_50": "BinLog 50",
    "categorical_log_score_100": "BinLog 100",
    "categorical_log_score_200": "BinLog 200",
    "q_loss_0.01": "Q 1%",
    "q_loss_0.05": "Q 5%",
    "brier_0.01": "Brier 1%",
    "brier_0.05": "Brier 5%",
}


def dataset_group(dataset: str) -> str:
    if dataset in ("sp500", "n225"):
        return "whole_markets"
    if dataset in sectors.SECTORS:
        return f"{sectors.SECTORS[dataset]['market']}_sectors"
    return "single_stocks"


def dataset_label(dataset: str) -> str:
    if dataset in sectors.SECTORS:
        specification = sectors.SECTORS[dataset]
        return f"{str(specification['market']).upper()} {specification['label']}"
    if dataset in eight.ASSETS:
        return str(eight.ASSETS[dataset]["label"])
    labels = {
        "sp500": "S&P 500",
        "n225": "Nikkei 225",
        "amzn": "Amazon",
        "toyota": "Toyota",
        "aapl": "Apple",
        "ntt": "NTT",
    }
    return labels[dataset]


def old_result_path(dataset: str) -> Path:
    if dataset in sectors.SECTORS:
        root = sectors.OUTPUT_ROOT
    elif dataset in eight.ASSETS:
        root = eight.OUTPUT_ROOT
    else:
        root = identity.OUTPUT_ROOT
    return root / dataset / "estimation.json"


def load_old_result(dataset: str) -> dict[str, object]:
    path = old_result_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing locked M1--M3 result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset: str) -> tuple[core.RawPanel, dict[str, object]]:
    if dataset in sectors.SECTORS:
        return sectors.load_sector(dataset)
    if dataset in eight.ASSETS:
        return eight.load_asset(dataset)
    return five.load_dataset(dataset)


def m2_betas(saved: dict[str, object]) -> np.ndarray:
    output = np.zeros((3, 4))
    source = saved["model2"]["betas"]
    for h, horizon in enumerate(core.HORIZONS):
        output[h, 0] = source[horizon]["location"]
        output[h, 1] = source[horizon]["log_scale_squared"]
    return output


def m3_betas(saved: dict[str, object]) -> np.ndarray:
    output = np.empty((3, 2))
    source = saved["model3"]["betas"]
    for h, horizon in enumerate(core.HORIZONS):
        output[h] = (source[horizon]["eta"], source[horizon]["lambda"])
    return output


def parity_transform(
    geometry: core.EfficientGeometry,
    nuisance_dimension: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Project first, then split the efficient score into parity blocks."""

    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        efficient = (
            raw_q_score
            - geometry.projection_q_on_hansen
            @ reference_score[:nuisance_dimension]
        )
        return np.vstack((
            geometry.projector_plus @ efficient,
            geometry.projector_minus @ efficient,
        ))

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
    m1_oos = core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, np.zeros((3, 4))
    )
    m1_u = core.dynamic_reference_pit(
        panel.values[training_rows:], oos_offsets, m1_oos.states
    )
    m1_phi = core.normalized_bspline_basis(m1_u, knots, core.DEGREE, masses)
    identity_correction = five.static_correction(m1_phi, oos_offsets, masses)
    m1_pair = five.score_pair(
        panel, n_train, dates, m1_oos, identity_correction, knots, masses,
        masses, MODEL_NAMES["m1"], "_discard_m1",
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
    saved = load_old_result(dataset)
    panel, info = load_dataset(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    anchor = np.asarray(saved["model1"]["state"], dtype=float)

    print(f"{dataset}: reproducing locked M2", flush=True)
    m2_train, m2_oos, m2_full, m2_continuation = identity.continue_reference(
        panel, n_train, anchor, m2_betas(saved)
    )
    saved_m2_error = float(abs(
        m2_train.composite_log_likelihood
        - saved["model2"]["dynamic_composite_log_likelihood"]
    ))
    if saved_m2_error > 1e-2:
        raise RuntimeError(f"material locked/current M2 mismatch: {saved_m2_error}")

    print(f"{dataset}: reproducing locked M3", flush=True)
    shape_geometry = direct.shape_geometry(anchor)
    shape_coefficients = m3_betas(saved)
    m3_zero = direct.filter_shape_layer(
        panel.values[:training_rows], train_offsets, m2_train, anchor,
        np.zeros((3, 2)), shape_geometry,
    )
    m3_train = direct.filter_shape_layer(
        panel.values[:training_rows], train_offsets, m2_train, anchor,
        shape_coefficients, shape_geometry,
    )
    saved_m3_error = float(abs(
        m3_train.composite_log_likelihood
        - saved["model3"]["dynamic_shape_log_likelihood"]
    ))
    if saved_m3_error > 1e-2:
        raise RuntimeError(f"material locked/current M3 mismatch: {saved_m3_error}")
    m3_oos_filter, m3_full_filter, m3_continuation = identity.continue_shape(
        panel, n_train, m2_oos, m2_full, anchor, shape_coefficients,
        shape_geometry, m3_train,
    )
    m3_train_reference = identity.shape_as_reference(m3_train)
    m3_oos_reference = identity.shape_as_reference(m3_oos_filter)
    m3_full_reference = identity.shape_as_reference(m3_full_filter)

    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    m2_u = core.dynamic_reference_pit(panel.values, panel.offsets, m2_full.states)
    m3_u = core.dynamic_reference_pit(
        panel.values, panel.offsets, m3_full_filter.states
    )
    m2_phi = core.normalized_bspline_basis(m2_u, knots, core.DEGREE, masses)
    m3_phi = core.normalized_bspline_basis(m3_u, knots, core.DEGREE, masses)

    m4_geometry = q0.corrected_tangent_geometry(
        anchor, masses, knots, masses, helmert, (0, 1),
        "identity-SM q score off m/log-sigma2 reference tangents",
    )
    m4_geometry.diagnostics.update({
        "anchor_measure": "du because h0(u)=1",
        "projection_then_parity": True,
        "score_whitening": False,
    })
    print(f"{dataset}: fitting M4 location-scale-orthogonal parity DCD-SM", flush=True)
    m4_fit, m4_oos, m4_checks = five.fit_and_continue_arc(
        m2_phi, panel, n_train, m2_train, m2_oos, m2_full,
        masses, helmert, parity_transform(m4_geometry, 2), ("even", "odd"),
    )

    m5_geometry = q0.corrected_tangent_geometry(
        anchor, masses, knots, masses, helmert, (0, 1, 2, 3),
        "identity-SM q score off m/log-sigma2/eta/lambda reference tangents",
    )
    m5_geometry.diagnostics.update({
        "anchor_measure": "du because h0(u)=1",
        "projection_then_parity": True,
        "score_whitening": False,
    })
    print(f"{dataset}: fitting M5 four-tangent-orthogonal parity DCD-SM", flush=True)
    m5_fit, m5_oos, m5_checks = five.fit_and_continue_arc(
        m3_phi, panel, n_train, m3_train_reference, m3_oos_reference,
        m3_full_reference, masses, helmert,
        parity_transform(m5_geometry, 4), ("even", "odd"),
    )

    scores = score_models(
        panel, dates, n_train, anchor, masses, knots, m2_oos,
        m3_oos_reference, m4_oos, m5_oos,
    )
    summary = scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean()
    pairs = (
        ("m2", "m1"), ("m3", "m2"), ("m4", "m2"),
        ("m4", "m3"), ("m5", "m3"), ("m5", "m4"),
    )
    comparisons = pd.DataFrame([
        core.score_comparison(
            scores, MODEL_NAMES[proposed], MODEL_NAMES[benchmark], score
        )
        for proposed, benchmark in pairs for score in q0.SCORE_NAMES
    ])
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "five_model_oos_summary.csv", index=False)
    comparisons.to_csv(output / "five_model_oos_comparisons.csv", index=False)
    pd.DataFrame(saved["model3"]["parameters"]).to_csv(
        output / "model3_eta_lambda_parameters.csv", index=False
    )
    pd.DataFrame(m4_fit["parameters"]).to_csv(
        output / "model4_parity_parameters.csv", index=False
    )
    pd.DataFrame(m5_fit["parameters"]).to_csv(
        output / "model5_parity_parameters.csv", index=False
    )

    m3_zero_error = float(max(
        np.max(np.abs(m3_zero.states - m2_train.states)),
        np.max(np.abs(m3_zero.contributions - m2_train.contributions)),
    ))
    result = {
        "analysis": f"{dataset} identity-SM skew-t parity M1--M5",
        "dataset": {**info, "group": dataset_group(dataset), "label": dataset_label(dataset)},
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
            "static_sm": "identity h0(u)=1; q0=0",
            "models": {
                "M1": "static skew-t MLE",
                "M2": "M1 plus dynamic m/log sigma2; then frozen",
                "M3": "M2 plus dynamic eta/lambda, projected off m/log sigma2",
                "M4": "M2 plus m/log sigma2-orthogonal even/odd DCD-SM",
                "M5": "M3 plus m/log sigma2/eta/lambda-orthogonal even/odd DCD-SM",
            },
            "projection_anchor": "static skew-t training MLE",
            "projection_measure": "static skew-t density, equivalently du",
            "projection_order": "efficient score first, PIT parity split second",
            "score_whitening": False,
            "windows": list(core.WINDOWS),
            "estimation": "conditional: M2 fixed for M3/M4; M3 fixed for M5",
        },
        "model1": saved["model1"],
        "model2": saved["model2"],
        "model3": saved["model3"],
        "model4": m4_fit,
        "model5": m5_fit,
        "model4_geometry": m4_geometry.diagnostics,
        "model5_geometry": m5_geometry.diagnostics,
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": {
            "locked_current_model2_likelihood_error": saved_m2_error,
            "locked_current_model3_likelihood_error": saved_m3_error,
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
    write_dataset_report(output, result, summary, comparisons)
    return result


def write_dataset_report(
    output: Path,
    result: dict[str, object],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    lines = [
        f"# {result['dataset']['label']}: parity M1--M5",
        "",
        "The static spline is h0(u)=1. M4 and M5 project the q score first,",
        "then split the efficient score into even and odd PIT-space blocks.",
        "No Fisher whitening is used.",
        "",
        "## In-sample joint tests",
        "",
        "| model/block | p |",
        "|---|---:|",
        f"| M3 eta | {result['model3']['wald_tests']['eta_three_coefficients']['p']:.6g} |",
        f"| M3 lambda | {result['model3']['wald_tests']['lambda_three_coefficients']['p']:.6g} |",
        f"| M3 all | {result['model3']['wald_tests']['eta_lambda_six_coefficients']['p']:.6g} |",
        f"| M4 even | {result['model4']['joint_blocks']['even']['p']:.6g} |",
        f"| M4 odd | {result['model4']['joint_blocks']['odd']['p']:.6g} |",
        f"| M4 all | {result['model4']['joint_all']['p']:.6g} |",
        f"| M5 even | {result['model5']['joint_blocks']['even']['p']:.6g} |",
        f"| M5 odd | {result['model5']['joint_blocks']['odd']['p']:.6g} |",
        f"| M5 all | {result['model5']['joint_all']['p']:.6g} |",
        "",
        "## Strict-OOS score levels",
        "",
        "| model | log score | CRPS |",
        "|---|---:|---:|",
    ]
    labels_by_name = {MODEL_NAMES[key]: value for key, value in MODEL_LABELS.items()}
    for row in summary.to_dict("records"):
        lines.append(
            f"| {labels_by_name[row['model']]} | {row['continuous_log_score']:.9f} | "
            f"{row['crps']:.9f} |"
        )
    lines += [
        "",
        "## Selected structural OOS comparisons",
        "",
        "Positive improvement favors the proposed model.",
        "",
        "| proposed | benchmark | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ]
    selected = comparisons.loc[
        comparisons["score"].isin(("continuous_log_score", "crps"))
    ]
    for row in selected.to_dict("records"):
        lines.append(
            f"| {labels_by_name[row['proposed']]} | {labels_by_name[row['benchmark']]} | "
            f"{row['score']} | {row['improvement_mean']:.9g} | "
            f"{row['hac22_two_sided_p']:.6g} |"
        )
    lines += [
        "",
        "All nine losses are in `five_model_oos_comparisons.csv`; complete",
        "coefficient estimates and daily strict-OOS scores are stored alongside it.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def relative_percent(levels: pd.DataFrame, model: str, score: str) -> float:
    indexed = levels.set_index("model")
    proposed = float(indexed.loc[MODEL_NAMES[model], score])
    base = float(indexed.loc[MODEL_NAMES["m2"], score])
    if score in {
        "continuous_log_score", "categorical_log_score_50",
        "categorical_log_score_100", "categorical_log_score_200",
    }:
        return 100.0 * (proposed - base) / abs(base)
    return 100.0 * (base - proposed) / abs(base)


def plot_group_heatmaps(results: dict[str, dict[str, object]], group: str) -> None:
    selected = [dataset for dataset in results if dataset_group(dataset) == group]
    if not selected:
        return
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    oos_rows, oos_labels = [], []
    for dataset in selected:
        levels = pd.read_csv(OUTPUT_ROOT / dataset / "five_model_oos_summary.csv")
        for model in ("m3", "m4", "m5"):
            oos_labels.append(f"{dataset_label(dataset)} | {model.upper()}")
            oos_rows.append([
                relative_percent(levels, model, score) for score in SCORE_LABELS
            ])
    oos = np.asarray(oos_rows)
    pd.DataFrame(
        oos, index=oos_labels, columns=list(SCORE_LABELS.values())
    ).rename_axis("dataset_model").to_csv(
        figures / f"{group}_oos_relative_to_m2.csv"
    )
    bound = max(1e-12, float(np.nanmax(np.abs(oos))))
    fig, axis = plt.subplots(
        figsize=(13, max(4.0, 0.31 * len(oos_labels))), constrained_layout=True
    )
    image = axis.imshow(oos, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_yticks(range(len(oos_labels)), oos_labels, fontsize=7)
    axis.set_xticks(
        range(len(SCORE_LABELS)), SCORE_LABELS.values(), rotation=35, ha="right"
    )
    axis.set_title(f"{group}: strict-OOS improvement relative to M2 (%)")
    for row in range(oos.shape[0]):
        for column in range(oos.shape[1]):
            axis.text(column, row, f"{oos[row, column]:+.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axis, label="improvement relative to M2 (%)")
    fig.savefig(figures / f"{group}_oos_relative_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / f"{group}_oos_relative_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    columns = [
        f"{block} {window}"
        for block in ("M3 eta", "M3 lambda", "M4 even", "M4 odd", "M5 even", "M5 odd")
        for window in (1, 5, 22)
    ]
    values = np.full((len(selected), len(columns)), np.nan)
    for row, dataset in enumerate(selected):
        result = results[dataset]
        m3 = {(item["state"], int(item["window"])): item["t"] for item in result["model3"]["parameters"]}
        m4 = {(item["block"], int(item["horizon"])): item["t"] for item in result["model4"]["parameters"]}
        m5 = {(item["block"], int(item["horizon"])): item["t"] for item in result["model5"]["parameters"]}
        values[row] = [
            *[m3[(block, window)] for block in ("eta", "lambda") for window in (1, 5, 22)],
            *[m4[(block, window)] for block in ("even", "odd") for window in (1, 5, 22)],
            *[m5[(block, window)] for block in ("even", "odd") for window in (1, 5, 22)],
        ]
    pd.DataFrame(
        values,
        index=[dataset_label(d) for d in selected],
        columns=columns,
    ).rename_axis("dataset").to_csv(
        figures / f"{group}_coefficient_t_statistics.csv"
    )
    bound = max(1.96, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(
        figsize=(17, max(4.0, 0.5 * len(selected))), constrained_layout=True
    )
    image = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_yticks(range(len(selected)), [dataset_label(d) for d in selected], fontsize=8)
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    axis.set_title(f"{group}: conditional HAC(22) coefficient t statistics")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            star = "*" if abs(values[row, column]) >= 1.96 else ""
            axis.text(column, row, f"{values[row, column]:+.2f}{star}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(image, ax=axis, label="HAC(22) t statistic")
    fig.savefig(figures / f"{group}_coefficient_t_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / f"{group}_coefficient_t_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(datasets: list[str]) -> None:
    results = {
        dataset: json.loads((OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8"))
        for dataset in datasets
    }
    test_rows = []
    for dataset, result in results.items():
        tests = {
            ("M3", "eta"): result["model3"]["wald_tests"]["eta_three_coefficients"]["p"],
            ("M3", "lambda"): result["model3"]["wald_tests"]["lambda_three_coefficients"]["p"],
            ("M3", "all"): result["model3"]["wald_tests"]["eta_lambda_six_coefficients"]["p"],
            ("M4", "even"): result["model4"]["joint_blocks"]["even"]["p"],
            ("M4", "odd"): result["model4"]["joint_blocks"]["odd"]["p"],
            ("M4", "all"): result["model4"]["joint_all"]["p"],
            ("M5", "even"): result["model5"]["joint_blocks"]["even"]["p"],
            ("M5", "odd"): result["model5"]["joint_blocks"]["odd"]["p"],
            ("M5", "all"): result["model5"]["joint_all"]["p"],
        }
        for (model, block), p_value in tests.items():
            test_rows.append({
                "dataset": dataset, "label": dataset_label(dataset),
                "group": dataset_group(dataset), "model": model,
                "block": block, "p": float(p_value),
            })
    tests = pd.DataFrame(test_rows)
    tests["bh_q_within_group_model_block"] = np.nan
    for _, index in tests.groupby(["group", "model", "block"]).groups.items():
        tests.loc[index, "bh_q_within_group_model_block"] = bh_adjust(tests.loc[index, "p"])
    tests.to_csv(OUTPUT_ROOT / "combined_in_sample_tests.csv", index=False)

    parameter_frames, comparison_frames, level_frames, audit_rows = [], [], [], []
    for dataset, result in results.items():
        for model in ("model3", "model4", "model5"):
            frame = pd.DataFrame(result[model]["parameters"])
            frame.insert(0, "model", model.upper().replace("MODEL", "M"))
            frame.insert(0, "group", dataset_group(dataset))
            frame.insert(0, "label", dataset_label(dataset))
            frame.insert(0, "dataset", dataset)
            parameter_frames.append(frame)
        comparisons = pd.read_csv(OUTPUT_ROOT / dataset / "five_model_oos_comparisons.csv")
        comparisons.insert(0, "group", dataset_group(dataset))
        comparisons.insert(0, "label", dataset_label(dataset))
        comparisons.insert(0, "dataset", dataset)
        comparison_frames.append(comparisons)
        levels = pd.read_csv(OUTPUT_ROOT / dataset / "five_model_oos_summary.csv")
        levels.insert(0, "group", dataset_group(dataset))
        levels.insert(0, "label", dataset_label(dataset))
        levels.insert(0, "dataset", dataset)
        level_frames.append(levels)
        checks = result["checks"]
        audit_rows.append({
            "dataset": dataset,
            "m2_saved_likelihood_error": checks["locked_current_model2_likelihood_error"],
            "m3_saved_likelihood_error": checks["locked_current_model3_likelihood_error"],
            "m2_continuation_error": checks["model2_continuation_error"],
            "m3_zero_nesting_error": checks["model3_zero_nesting_error"],
            "m3_continuation_error": checks["model3_continuation_error"],
            "m4_continuation_error": checks["model4"]["continuation_error"],
            "m4_q0_nesting_error": checks["model4"]["q_zero_nesting_error"],
            "m5_continuation_error": checks["model5"]["continuation_error"],
            "m5_q0_nesting_error": checks["model5"]["q_zero_nesting_error"],
            "m4_successful_starts": result["model4"]["successful_starts"],
            "m5_successful_starts": result["model5"]["successful_starts"],
            "m4_objective_gap": result["model4"]["largest_successful_objective_gap"],
            "m5_objective_gap": result["model5"]["largest_successful_objective_gap"],
        })
    pd.concat(parameter_frames, ignore_index=True, sort=False).to_csv(
        OUTPUT_ROOT / "combined_coefficient_t_statistics.csv", index=False
    )
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    comparisons["bh_q_within_group_model_pair_all_losses"] = np.nan
    for _, index in comparisons.groupby(["group", "proposed", "benchmark"]).groups.items():
        comparisons.loc[index, "bh_q_within_group_model_pair_all_losses"] = bh_adjust(
            comparisons.loc[index, "hac22_two_sided_p"]
        )
    comparisons.to_csv(OUTPUT_ROOT / "combined_oos_comparisons.csv", index=False)
    pd.concat(level_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "all_five_model_oos_levels.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(OUTPUT_ROOT / "numerical_audit.csv", index=False)

    for group in ("whole_markets", "sp500_sectors", "n225_sectors", "single_stocks"):
        plot_group_heatmaps(results, group)

    lines = [
        "# Identity-SM skew-t parity M1--M5: complete results",
        "",
        "M1--M3 are locked from the preceding identity-SM experiment. M4",
        "projects the spline score off m/log-sigma2 and then uses even/odd",
        "HAR(1,5,22) blocks. M5 projects off m/log-sigma2/eta/lambda before",
        "the same parity split. Every dynamic shape model has six coefficients.",
        "There is no Fisher whitening.",
        "",
        "## In-sample joint tests",
        "",
        "| dataset | M3 eta | M3 lambda | M4 even | M4 odd | M5 even | M5 odd |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pivot = tests.set_index(["dataset", "model", "block"])["p"]
    for dataset in datasets:
        lines.append(
            f"| {dataset_label(dataset)} | {pivot[(dataset, 'M3', 'eta')]:.6g} | "
            f"{pivot[(dataset, 'M3', 'lambda')]:.6g} | "
            f"{pivot[(dataset, 'M4', 'even')]:.6g} | {pivot[(dataset, 'M4', 'odd')]:.6g} | "
            f"{pivot[(dataset, 'M5', 'even')]:.6g} | {pivot[(dataset, 'M5', 'odd')]:.6g} |"
        )
    lines += [
        "",
        "BH-adjusted in-sample results are in `combined_in_sample_tests.csv`.",
        "All strict-OOS levels and comparisons are in the combined CSV files.",
        "Heatmaps are in `figures/`, and numerical reproduction/nesting checks",
        "are in `numerical_audit.csv`.",
    ]
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_datasets(selection: str) -> list[str]:
    if selection == "all":
        return list(ALL_DATASETS)
    if selection == "markets":
        return ["sp500", "n225"]
    if selection == "sectors":
        return list(SECTOR_DATASETS)
    if selection == "stocks":
        return ["amzn", "toyota", "aapl", "ntt", *ADDED_STOCKS]
    if selection == "sp500_sectors":
        return [d for d in SECTOR_DATASETS if d.startswith("sp500_")]
    if selection == "n225_sectors":
        return [d for d in SECTOR_DATASETS if d.startswith("n225_")]
    return [selection]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(*ALL_DATASETS, "markets", "sectors", "stocks", "sp500_sectors", "n225_sectors", "all"),
        default="all",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = selected_datasets(args.dataset)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for dataset in datasets:
            if (
                not args.force
                and (OUTPUT_ROOT / dataset / "estimation.json").exists()
            ):
                print(f"{dataset}: existing parity result retained", flush=True)
                continue
            run_dataset(dataset)
    completed = [dataset for dataset in ALL_DATASETS if (OUTPUT_ROOT / dataset / "estimation.json").exists()]
    if set(completed) == set(ALL_DATASETS):
        summarize(list(ALL_DATASETS))
    else:
        print(f"completed {len(completed)}/{len(ALL_DATASETS)} datasets", flush=True)


if __name__ == "__main__":
    main()
