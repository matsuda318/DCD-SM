"""Apply the locked raw-return identity-SM skew-t M1--M5 model by sector.

The estimation code is imported unchanged from
``run_identity_sm_skewt_five_models``.  This wrapper supplies point-in-time
S&P 500 and Nikkei 225 sector panels.  Raw log returns are the model input;
the RM/RV diagnostic columns in the source parquet files are not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_direct_dynamic_skewt_shape_test as direct
import run_identity_sm_skewt_five_models as identity
import run_sp500 as core
import run_static_q0_symmetric_t_four_models as q0


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_ROOT = REPO / "data" / "sp500_n225_sector_daily_2016_2026"
OUTPUT_ROOT = ROOT / "output_identity_sm_skewt_sector_five_models"
DIRECT_OUTPUT_ROOT = OUTPUT_ROOT / "_direct_m1_m3_stage"
TRAINING_END = pd.Timestamp("2023-12-29")

SECTORS = {
    "sp500_s10": {"market": "sp500", "code": "s10", "label": "Energy"},
    "sp500_s15": {"market": "sp500", "code": "s15", "label": "Materials"},
    "sp500_s20": {"market": "sp500", "code": "s20", "label": "Industrials"},
    "sp500_s25": {"market": "sp500", "code": "s25", "label": "Consumer Discretionary"},
    "sp500_s30": {"market": "sp500", "code": "s30", "label": "Consumer Staples"},
    "sp500_s35": {"market": "sp500", "code": "s35", "label": "Health Care"},
    "sp500_s40": {"market": "sp500", "code": "s40", "label": "Financials"},
    "sp500_s45": {"market": "sp500", "code": "s45", "label": "Information Technology"},
    "sp500_s50": {"market": "sp500", "code": "s50", "label": "Communication Services"},
    "sp500_s55": {"market": "sp500", "code": "s55", "label": "Utilities"},
    "sp500_s60": {"market": "sp500", "code": "s60", "label": "Real Estate"},
    "n225_n1": {"market": "n225", "code": "n1", "label": "Technology"},
    "n225_n2": {"market": "n225", "code": "n2", "label": "Financials"},
    "n225_n3": {"market": "n225", "code": "n3", "label": "Consumer Goods"},
    "n225_n4": {"market": "n225", "code": "n4", "label": "Materials"},
    "n225_n5": {"market": "n225", "code": "n5", "label": "Capital Goods/Others"},
    "n225_n6": {"market": "n225", "code": "n6", "label": "Transportation and Utilities"},
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


def load_sector(dataset: str) -> tuple[core.RawPanel, dict[str, object]]:
    specification = SECTORS[dataset]
    path = (
        DATA_ROOT
        / str(specification["market"])
        / "daily_returns"
        / f"{specification['code']}.parquet"
    )
    frame = pd.read_parquet(path, columns=["date", "identifier", "log_return"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[np.isfinite(frame["log_return"])].copy()
    frame = frame.sort_values(["date", "identifier"]).reset_index(drop=True)
    if frame.duplicated(["date", "identifier"]).any():
        raise RuntimeError(f"duplicate sector return rows in {path}")
    counts = frame.groupby("date", sort=True).size()
    dates = pd.DatetimeIndex(counts.index)
    n_train = int(np.sum(dates <= TRAINING_END))
    if n_train <= 22 or n_train == len(dates):
        raise RuntimeError(f"invalid split for {dataset}: {n_train}/{len(dates)}")
    values = frame["log_return"].to_numpy(dtype=float)
    offsets = np.r_[0, counts.cumsum().to_numpy(dtype=int)]
    panel = core.RawPanel(
        daily=pd.DataFrame({
            "date": dates,
            "n_constituents": counts.to_numpy(dtype=int),
        }),
        values=values,
        offsets=offsets,
    )
    return panel, {
        "dataset": dataset,
        "label": f"{str(specification['market']).upper()} {specification['label']}",
        "market": specification["market"],
        "sector_code": specification["code"],
        "sector": specification["label"],
        "source": str(path),
        "model_input": "raw log_return; no RM/RV pre-normalization",
    }


def configure_imported_experiment() -> None:
    direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    direct.data_source.load_dataset = load_sector
    identity.OUTPUT_ROOT = OUTPUT_ROOT
    identity.direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    identity.five.load_dataset = load_sector


def run_sector(dataset: str) -> dict[str, object]:
    print(f"{dataset}: fitting direct M1--M3 stage on raw returns", flush=True)
    direct.run_dataset(dataset)
    print(f"{dataset}: fitting complete identity-SM M1--M5", flush=True)
    return identity.run_dataset(dataset)


def result_path(dataset: str) -> Path:
    return OUTPUT_ROOT / dataset / "estimation.json"


def load_result(dataset: str) -> dict[str, object]:
    return json.loads(result_path(dataset).read_text(encoding="utf-8"))


def structural_comparisons(dataset: str) -> pd.DataFrame:
    scores = pd.read_csv(OUTPUT_ROOT / dataset / "daily_oos_scores.csv")
    pairs = (
        ("m3", "m2"),
        ("m4", "m2"),
        ("m4", "m3"),
        ("m5", "m3"),
        ("m5", "m4"),
    )
    rows = [
        {
            "dataset": dataset,
            "market": SECTORS[dataset]["market"],
            "sector_code": SECTORS[dataset]["code"],
            "sector": SECTORS[dataset]["label"],
            **core.score_comparison(
                scores,
                identity.MODEL_NAMES[proposed],
                identity.MODEL_NAMES[benchmark],
                score,
            ),
        }
        for proposed, benchmark in pairs
        for score in q0.SCORE_NAMES
    ]
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_ROOT / dataset / "sector_structural_oos_comparisons.csv", index=False)
    return output


def relative_percent(levels: pd.DataFrame, model: str, score: str) -> float:
    indexed = levels.set_index("model")
    proposed = float(indexed.loc[model, score])
    base = float(indexed.loc[identity.MODEL_NAMES["m2"], score])
    if score in {
        "continuous_log_score",
        "categorical_log_score_50",
        "categorical_log_score_100",
        "categorical_log_score_200",
    }:
        return 100.0 * (proposed - base) / abs(base)
    return 100.0 * (base - proposed) / abs(base)


def bh_adjust(values: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg q-values for one pre-specified test family."""
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def plot_oos_heatmaps(datasets: list[str]) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for market in ("sp500", "n225"):
        selected = [d for d in datasets if SECTORS[d]["market"] == market]
        if not selected:
            continue
        row_labels, values = [], []
        for dataset in selected:
            levels = pd.read_csv(OUTPUT_ROOT / dataset / "five_model_oos_summary.csv")
            for model in ("m3", "m4", "m5"):
                row_labels.append(f"{SECTORS[dataset]['label']} | {model.upper()}")
                values.append([
                    relative_percent(levels, identity.MODEL_NAMES[model], score)
                    for score in SCORE_LABELS
                ])
        array = np.asarray(values)
        bound = max(1e-12, float(np.nanmax(np.abs(array))))
        height = max(6.0, 0.32 * len(row_labels))
        fig, axis = plt.subplots(figsize=(13, height), constrained_layout=True)
        image = axis.imshow(array, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
        axis.set_xticks(range(len(SCORE_LABELS)), SCORE_LABELS.values(), rotation=35, ha="right")
        axis.set_title(f"{market.upper()} sectors: strict-OOS accuracy relative to M2")
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                axis.text(column, row, f"{array[row, column]:+.3f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axis, label="OOS improvement relative to M2 (%)")
        fig.savefig(figures / f"{market}_sector_oos_relative_heatmap.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures / f"{market}_sector_oos_relative_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_coefficient_heatmaps(results: dict[str, dict[str, object]]) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    columns = (
        "eta D", "eta W", "eta M", "lambda D", "lambda W", "lambda M",
        "q_eta D", "q_eta W", "q_eta M", "others D", "others W", "others M",
        "resid. D", "resid. W", "resid. M",
    )
    for market in ("sp500", "n225"):
        selected = [d for d in results if SECTORS[d]["market"] == market]
        if not selected:
            continue
        values = np.full((len(selected), len(columns)), np.nan)
        for row, dataset in enumerate(selected):
            result = results[dataset]
            m3 = {(item["state"], int(item["window"])): item["t"] for item in result["model3"]["parameters"]}
            values[row, :6] = [m3[(state, horizon)] for state in ("eta", "lambda") for horizon in (1, 5, 22)]
            m4 = {(item["block"], int(item["horizon"])): item["t"] for item in result["model4"]["parameters"]}
            values[row, 6:12] = [m4[(block, horizon)] for block in ("q_eta", "others") for horizon in (1, 5, 22)]
            values[row, 12:15] = [item["t"] for item in result["model5"]["parameters"]]
        bound = max(1.96, float(np.nanmax(np.abs(values))))
        fig, axis = plt.subplots(figsize=(15, max(4.5, 0.55 * len(selected))), constrained_layout=True)
        image = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_yticks(range(len(selected)), [SECTORS[d]["label"] for d in selected])
        axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
        axis.set_title(f"{market.upper()} sectors: HAC(22) coefficient t statistics")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                star = "*" if abs(values[row, column]) >= 1.96 else ""
                axis.text(column, row, f"{values[row, column]:+.2f}{star}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axis, label="HAC(22) t statistic")
        fig.savefig(figures / f"{market}_sector_coefficient_t_heatmap.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures / f"{market}_sector_coefficient_t_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)


def summarize(datasets: list[str]) -> None:
    missing = [dataset for dataset in datasets if not result_path(dataset).exists()]
    if missing:
        raise FileNotFoundError(f"missing sector results: {missing}")
    results = {dataset: load_result(dataset) for dataset in datasets}
    test_rows = []
    level_frames = []
    comparisons = []
    coefficient_rows = []
    audit_rows = []
    for dataset, result in results.items():
        m3 = result["model3"]["wald_tests"]
        test_rows.append({
            "dataset": dataset,
            "market": SECTORS[dataset]["market"],
            "sector_code": SECTORS[dataset]["code"],
            "sector": SECTORS[dataset]["label"],
            "m3_eta_p": m3["eta_three_coefficients"]["p"],
            "m3_lambda_p": m3["lambda_three_coefficients"]["p"],
            "m3_joint_p": m3["eta_lambda_six_coefficients"]["p"],
            "m4_q_eta_p": result["model4"]["joint_blocks"]["q_eta"]["p"],
            "m4_others_p": result["model4"]["joint_blocks"]["others"]["p"],
            "m4_joint_p": result["model4"]["joint_all"]["p"],
            "m5_residual_p": result["model5"]["joint_all"]["p"],
        })
        levels = pd.read_csv(OUTPUT_ROOT / dataset / "five_model_oos_summary.csv")
        levels.insert(0, "sector", SECTORS[dataset]["label"])
        levels.insert(0, "market", SECTORS[dataset]["market"])
        levels.insert(0, "dataset", dataset)
        level_frames.append(levels)
        comparisons.append(structural_comparisons(dataset))
        for model, section, direction_key, horizon_key in (
            ("M3", "model3", "state", "window"),
            ("M4", "model4", "block", "horizon"),
            ("M5", "model5", "block", "horizon"),
        ):
            for item in result[section]["parameters"]:
                coefficient_rows.append({
                    "dataset": dataset,
                    "market": SECTORS[dataset]["market"],
                    "sector_code": SECTORS[dataset]["code"],
                    "sector": SECTORS[dataset]["label"],
                    "model": model,
                    "direction": item[direction_key],
                    "horizon": item[horizon_key],
                    "estimate": item["estimate"],
                    "hac22_se": item.get(
                        "hac22_se_conditional_on_reference",
                        item.get("hac22_se_conditional_on_location_scale"),
                    ),
                    "t": item["t"],
                    "p_two_sided": item["p_two_sided"],
                })
        checks = result["checks"]
        errors = [
            checks["saved_current_model2_likelihood_error"],
            checks["saved_current_model3_likelihood_error"],
            checks["model2_continuation_error"],
            checks["model3_zero_nesting_error"],
            checks["model3_continuation_error"],
            checks["model4"]["continuation_error"],
            checks["model4"]["q_zero_nesting_error"],
            checks["model5"]["continuation_error"],
            checks["model5"]["q_zero_nesting_error"],
        ]
        audit_rows.append({
            "dataset": dataset,
            "model2_successful_starts": result["model2"]["successful_starts"],
            "model2_total_starts": result["model2"]["total_starts"],
            "model3_successful_starts": result["model3"]["successful_starts"],
            "model3_total_starts": result["model3"]["total_starts"],
            "model4_successful_starts": result["model4"]["successful_starts"],
            "model4_total_starts": result["model4"]["total_starts"],
            "model5_successful_starts": result["model5"]["successful_starts"],
            "model5_total_starts": result["model5"]["total_starts"],
            "model2_largest_successful_objective_gap": result["model2"]["largest_successful_objective_gap"],
            "model3_largest_successful_objective_gap": result["model3"]["largest_successful_objective_gap"],
            "model4_largest_successful_objective_gap": result["model4"]["largest_successful_objective_gap"],
            "model5_largest_successful_objective_gap": result["model5"]["largest_successful_objective_gap"],
            "maximum_continuation_or_nesting_error": max(abs(float(value)) for value in errors),
        })
    tests = pd.DataFrame(test_rows)
    p_columns = [column for column in tests if column.endswith("_p")]
    for column in p_columns:
        tests[column.removesuffix("_p") + "_bh_q"] = bh_adjust(tests[column])
    all_levels = pd.concat(level_frames, ignore_index=True)
    all_comparisons = pd.concat(comparisons, ignore_index=True)
    all_comparisons["bh_q_within_model_pair"] = all_comparisons.groupby(
        ["proposed", "benchmark"], sort=False
    )["hac22_two_sided_p"].transform(lambda values: bh_adjust(values))
    tests.to_csv(OUTPUT_ROOT / "combined_in_sample_tests.csv", index=False)
    all_levels.to_csv(OUTPUT_ROOT / "all_five_model_oos_levels.csv", index=False)
    all_comparisons.to_csv(OUTPUT_ROOT / "combined_structural_oos_comparisons.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(
        OUTPUT_ROOT / "combined_coefficient_t_statistics.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(OUTPUT_ROOT / "numerical_audit.csv", index=False)
    significance = {
        "M3 eta": ("m3_eta_p", "m3_eta_bh_q"),
        "M3 lambda": ("m3_lambda_p", "m3_lambda_bh_q"),
        "M4 q_eta": ("m4_q_eta_p", "m4_q_eta_bh_q"),
        "M4 others": ("m4_others_p", "m4_others_bh_q"),
        "M5 residual": ("m5_residual_p", "m5_residual_bh_q"),
    }
    lines = [
        "# Sector-level raw-return identity-SM skew-t M1--M5",
        "",
        "All models use raw point-in-time constituent log returns. No RM/RV",
        "pre-normalization is applied. The M1--M5 specification and estimators",
        "are unchanged from the final market-level identity-SM experiment.",
        "",
        "## Cross-sector finding",
        "",
        "| block | unadjusted p<.05 | BH q<.05 (17 sectors) |",
        "|---|---:|---:|",
    ]
    for label, (p_column, q_column) in significance.items():
        lines.append(
            f"| {label} | {int((tests[p_column] < .05).sum())}/17 | "
            f"{int((tests[q_column] < .05).sum())}/17 |"
        )
    lines += [
        "",
        "BH adjustment is performed separately for each pre-specified block",
        "across the 17 sectors. Individual unadjusted sector p-values remain",
        "exploratory.",
        "",
        "## In-sample joint tests",
        "",
        "| market | sector | M3 eta | M3 lambda | M3 joint | M4 q_eta | M4 others | M4 joint | M5 residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.to_dict("records"):
        lines.append(
            f"| {row['market']} | {row['sector']} | {row['m3_eta_p']:.5g} | "
            f"{row['m3_lambda_p']:.5g} | {row['m3_joint_p']:.5g} | "
            f"{row['m4_q_eta_p']:.5g} | {row['m4_others_p']:.5g} | "
            f"{row['m4_joint_p']:.5g} | {row['m5_residual_p']:.5g} |"
        )
    lines += [
        "",
        "## Strict-OOS M4 versus M3",
        "",
        "Positive improvement favors M4. P-values are paired HAC(22).",
        "",
        "| market | sector | log gain | log p | CRPS gain | CRPS p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selected = all_comparisons.loc[
        (all_comparisons["proposed"] == identity.MODEL_NAMES["m4"])
        & (all_comparisons["benchmark"] == identity.MODEL_NAMES["m3"])
        & all_comparisons["score"].isin(("continuous_log_score", "crps"))
    ]
    for dataset in datasets:
        subset = selected.loc[selected["dataset"] == dataset].set_index("score")
        log = subset.loc["continuous_log_score"]
        crps = subset.loc["crps"]
        lines.append(
            f"| {SECTORS[dataset]['market']} | {SECTORS[dataset]['label']} | "
            f"{log['improvement_mean']:+.7g} | {log['hac22_two_sided_p']:.5g} | "
            f"{crps['improvement_mean']:+.7g} | {crps['hac22_two_sided_p']:.5g} |"
        )
    m4_m3 = all_comparisons.loc[
        (all_comparisons["proposed"] == identity.MODEL_NAMES["m4"])
        & (all_comparisons["benchmark"] == identity.MODEL_NAMES["m3"])
    ]
    m4_m2 = all_comparisons.loc[
        (all_comparisons["proposed"] == identity.MODEL_NAMES["m4"])
        & (all_comparisons["benchmark"] == identity.MODEL_NAMES["m2"])
    ]
    lines += [
        "",
        "Across all 153 sector-score comparisons, M4 versus M3 has "
        f"{int(((m4_m3['hac22_two_sided_p'] < .05) & (m4_m3['improvement_mean'] > 0)).sum())} "
        "significant improvements and "
        f"{int(((m4_m3['hac22_two_sided_p'] < .05) & (m4_m3['improvement_mean'] < 0)).sum())} "
        "significant deteriorations before multiplicity adjustment. M4 versus "
        f"M2 has {int(((m4_m2['hac22_two_sided_p'] < .05) & (m4_m2['improvement_mean'] > 0)).sum())} "
        "significant improvements and "
        f"{int(((m4_m2['hac22_two_sided_p'] < .05) & (m4_m2['improvement_mean'] < 0)).sum())} "
        "significant deteriorations.",
        "After BH adjustment across the 153 scores within each model pair, "
        f"M4 versus M3 retains {int(((m4_m3['bh_q_within_model_pair'] < .05) & (m4_m3['improvement_mean'] > 0)).sum())} "
        "improvements and "
        f"{int(((m4_m3['bh_q_within_model_pair'] < .05) & (m4_m3['improvement_mean'] < 0)).sum())} "
        "deteriorations; M4 versus M2 retains "
        f"{int(((m4_m2['bh_q_within_model_pair'] < .05) & (m4_m2['improvement_mean'] > 0)).sum())} "
        "improvements and "
        f"{int(((m4_m2['bh_q_within_model_pair'] < .05) & (m4_m2['improvement_mean'] < 0)).sum())} "
        "deteriorations.",
        "",
        "Complete nine-score OOS levels and structural comparisons, daily",
        "scores, coefficient tables, numerical checks, and estimation JSON are",
        "stored in the machine-readable output files and sector subdirectories.",
    ]
    (OUTPUT_ROOT / "SECTOR_IDENTITY_SM_SKEWT_FIVE_MODEL_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    plot_oos_heatmaps(datasets)
    plot_coefficient_heatmaps(results)


def selected_datasets(selection: str) -> list[str]:
    if selection == "all":
        return list(SECTORS)
    if selection in ("sp500", "n225"):
        return [dataset for dataset in SECTORS if SECTORS[dataset]["market"] == selection]
    return [selection]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", choices=(*SECTORS, "sp500", "n225", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="skip completed estimation.json files")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    datasets = selected_datasets(args.sector)
    configure_imported_experiment()
    if not args.summarize_only:
        for dataset in datasets:
            if args.resume and result_path(dataset).exists():
                print(f"{dataset}: existing result retained", flush=True)
                continue
            run_sector(dataset)
    if all(result_path(dataset).exists() for dataset in datasets):
        summarize(datasets)


if __name__ == "__main__":
    main()
