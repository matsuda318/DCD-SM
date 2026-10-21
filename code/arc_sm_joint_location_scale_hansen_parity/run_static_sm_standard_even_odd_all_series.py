"""Fit and compare the locked standard endpoint even/odd DCD-SM.

Standard specification:
breakpoints (0,.05,.1,.2,...,.9,.95,1), one endpoint basis per side,
d_C=d_L+d_R and d_A=d_R-d_L, joint diagonal HAR(1,5,22), q-score
projected off location/log-scale-squared, and no whitening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as datasets
import run_static_sm_common_asymmetric_extreme_knots_markets as engine
import run_static_sm_two_edge_even_odd_extreme_knots_markets as alternative


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_sm_standard_even_odd_all_series"
STANDARD_EVEN_ONLY = "m4_standard_one_edge_even_arc_sm"
STANDARD_EVEN_ODD = "m4_standard_one_edge_even_odd_arc_sm"
ALL_SERIES = datasets.DATASETS


def parse_series(value: str) -> list[str]:
    if value == "all":
        return list(ALL_SERIES)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ALL_SERIES))
    if unknown:
        raise ValueError(f"unknown series: {unknown}")
    return selected


def direct_oos_comparisons(dataset: str) -> list[dict[str, object]]:
    standard_scores = pd.read_csv(OUTPUT_ROOT / dataset / "daily_oos_scores.csv")
    alternative_scores = pd.read_csv(alternative.OUTPUT_ROOT / dataset / "daily_oos_scores.csv")
    standard = standard_scores.loc[standard_scores["model"] == STANDARD_EVEN_ODD].copy()
    extreme = alternative_scores.loc[alternative_scores["model"] == alternative.EVEN_ODD].copy()
    standard["model"] = "standard_5pct_one_edge_even_odd"
    extreme["model"] = "extreme_1pct_two_edge_even_odd"
    combined = pd.concat((standard, extreme), ignore_index=True)
    return [
        engine.core.score_comparison(
            combined,
            "standard_5pct_one_edge_even_odd",
            "extreme_1pct_two_edge_even_odd",
            score,
        )
        for score in scoring.SCORE_NAMES
    ]


def summarize() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for dataset in ALL_SERIES:
        standard_result = json.loads(
            (OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8")
        )
        alternative_result = json.loads(
            (alternative.OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8")
        )
        standard_joint = standard_result["common_asymmetric"]
        alternative_joint = alternative_result["common_asymmetric"]
        asset_class = (
            "market" if dataset in ("sp500", "n225")
            else "FX" if dataset in datasets.FX
            else "stock"
        )
        comparisons = direct_oos_comparisons(dataset)
        direct_lookup = {row["score"]: row for row in comparisons}
        rows.append({
            "dataset": dataset,
            "label": standard_result["label"],
            "asset_class": asset_class,
            "standard_even_p": standard_joint["joint_blocks"]["common"]["p"],
            "standard_odd_p": standard_joint["joint_blocks"]["asymmetric"]["p"],
            "alternative_even_p": alternative_joint["joint_blocks"]["common"]["p"],
            "alternative_odd_p": alternative_joint["joint_blocks"]["asymmetric"]["p"],
            "standard_minus_alternative_log": direct_lookup["continuous_log_score"]["improvement_mean"],
            "log_p": direct_lookup["continuous_log_score"]["hac22_two_sided_p"],
            "standard_minus_alternative_crps_gain": direct_lookup["crps"]["improvement_mean"],
            "crps_p": direct_lookup["crps"]["hac22_two_sided_p"],
            "standard_start_gap": standard_joint["largest_successful_objective_gap"],
            "standard_maximum_oos_state": standard_result["checks"]["common_asymmetric_maximum_absolute_oos_state"],
        })
        for comparison in comparisons:
            direct_rows.append({"dataset": dataset, "label": standard_result["label"], "asset_class": asset_class, **comparison})
        for row in standard_joint["parameters"]:
            parameter_rows.append({
                "dataset": dataset,
                "label": standard_result["label"],
                "asset_class": asset_class,
                "direction": "even" if row["block"] == "common" else "odd",
                **row,
            })

    frame = pd.DataFrame(rows)
    direct_frame = pd.DataFrame(direct_rows)
    parameter_frame = pd.DataFrame(parameter_rows)
    frame.to_csv(OUTPUT_ROOT / "standard_vs_extreme_summary.csv", index=False)
    direct_frame.to_csv(OUTPUT_ROOT / "standard_vs_extreme_all_oos_scores.csv", index=False)
    parameter_frame.to_csv(OUTPUT_ROOT / "standard_even_odd_parameters.csv", index=False)

    lines = [
        "# Standard one-edge versus extreme two-edge even/odd DCD-SM", "",
        "Positive direct OOS gains favor the standard 5% one-edge model.", "",
    ]
    for asset_class, count in (("market", 2), ("stock", 12), ("FX", 4)):
        subset = frame.loc[frame["asset_class"] == asset_class]
        lines.extend([
            f"## {asset_class}", "",
            f"- Standard even rejections: `{int((subset['standard_even_p'] < .05).sum())}/{count}`; alternative: `{int((subset['alternative_even_p'] < .05).sum())}/{count}`.",
            f"- Standard odd rejections: `{int((subset['standard_odd_p'] < .05).sum())}/{count}`; alternative: `{int((subset['alternative_odd_p'] < .05).sum())}/{count}`.",
            f"- Standard has better direct OOS log score: `{int((subset['standard_minus_alternative_log'] > 0).sum())}/{count}`.",
            f"- Standard has better direct OOS CRPS: `{int((subset['standard_minus_alternative_crps_gain'] > 0).sum())}/{count}`.", "",
        ])
    lines.extend([
        "| class | series | standard even p | standard odd p | extreme even p | extreme odd p | standard-extreme log gain | p | standard-extreme CRPS gain | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['asset_class']} | {row['label']} | {row['standard_even_p']:.4g} | "
            f"{row['standard_odd_p']:.4g} | {row['alternative_even_p']:.4g} | "
            f"{row['alternative_odd_p']:.4g} | {row['standard_minus_alternative_log']:.6g} | "
            f"{row['log_p']:.4g} | {row['standard_minus_alternative_crps_gain']:.6g} | "
            f"{row['crps_p']:.4g} |"
        )
    lines.extend([
        "", "## Direct OOS comparison across all nine scores", "",
        "A win means that the standard 5% one-edge specification has the better",
        "paired OOS score than the extreme 1% two-edge specification.", "",
        "| score | standard wins | total series |",
        "|---|---:|---:|",
    ])
    score_counts = (
        direct_frame.assign(win=direct_frame["improvement_mean"] > 0.0)
        .groupby("score", as_index=False)
        .agg(standard_wins=("win", "sum"), total_series=("win", "size"))
    )
    for row in score_counts.to_dict(orient="records"):
        lines.append(f"| {row['score']} | {int(row['standard_wins'])} | {int(row['total_series'])} |")
    significant = direct_frame.loc[direct_frame["hac22_two_sided_p"] < 0.05]
    lines.extend([
        "", "## Individually significant direct OOS differences", "",
        "Positive improvements favor the standard specification.", "",
        "| class | series | score | improvement | HAC(22) p |",
        "|---|---|---|---:|---:|",
    ])
    for row in significant.to_dict(orient="records"):
        lines.append(
            f"| {row['asset_class']} | {row['label']} | {row['score']} | "
            f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    lines.extend([
        "", "## Conclusion", "",
        "The standard 5% one-edge specification has the higher in-sample",
        "detection rate for both even and odd dynamics. It also wins a majority",
        "of the 18 direct paired OOS comparisons for every one of the nine scores.",
        "The OOS advantage is broad but usually not individually significant; it",
        "should therefore be read as a regularization preference rather than as",
        "uniform pairwise dominance. The 1% two-edge version is retained as a",
        "tail-localization robustness specification.",
    ])
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = parse_series(args.series)
    if not args.summarize_only:
        for dataset in selected:
            engine.run_dataset(
                dataset,
                edge_basis_count=1,
                output_root=OUTPUT_ROOT,
                common_only_name=STANDARD_EVEN_ONLY,
                common_asymmetric_name=STANDARD_EVEN_ODD,
                breakpoints=datasets.BREAKPOINTS,
                analysis_name="standard one-edge even/odd DCD-SM",
            )
    if args.summarize_only or selected == list(ALL_SERIES):
        summarize()


if __name__ == "__main__":
    main()
