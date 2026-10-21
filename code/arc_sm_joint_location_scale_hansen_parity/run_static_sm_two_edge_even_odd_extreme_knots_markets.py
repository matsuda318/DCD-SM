"""Fit pooled-two-edge even/odd DCD-SM models for S&P 500 and N225.

The fitted static SM and extreme breakpoints are identical to the preceding
extreme-knot experiment.  Each edge selector now pools two spline bases:
e_L=(1,1,0,...,0) and e_R=(0,...,0,1,1).  The joint model uses the even
direction d_C=Q'(e_L+e_R) and odd direction d_A=Q'(e_R-e_L), with separate
HAR(1,5,22) block tests and no score whitening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_static_sm_common_asymmetric_extreme_knots_markets as base


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_sm_two_edge_even_odd_extreme_knots_markets"
EVEN_ONLY = "m4_extreme_two_edge_even_arc_sm"
EVEN_ODD = "m4_extreme_two_edge_even_odd_arc_sm"


def summarize(results: list[dict[str, object]]) -> None:
    edge_count = int(results[0]["specification"]["edge_basis_count_per_side"])
    edge_ones = ",".join(["1"] * edge_count)
    rows: list[dict[str, object]] = []
    parameters: list[dict[str, object]] = []
    for result in results:
        lookup = {
            (row["proposed"], row["benchmark"], row["score"]): row
            for row in result["oos_comparisons"]
        }
        joint = result["common_asymmetric"]
        rows.append({
            "dataset": result["dataset"].get("dataset", result["label"]),
            "label": result["label"],
            "even_joint_p": joint["joint_blocks"]["common"]["p"],
            "odd_joint_p": joint["joint_blocks"]["asymmetric"]["p"],
            "all_six_p": joint["joint_all"]["p"],
            "log_gain_vs_m3": lookup[(EVEN_ODD, base.M3, "continuous_log_score")]["improvement_mean"],
            "log_p_vs_m3": lookup[(EVEN_ODD, base.M3, "continuous_log_score")]["hac22_two_sided_p"],
            "crps_gain_vs_m3": lookup[(EVEN_ODD, base.M3, "crps")]["improvement_mean"],
            "crps_p_vs_m3": lookup[(EVEN_ODD, base.M3, "crps")]["hac22_two_sided_p"],
        })
        for row in joint["parameters"]:
            parameters.append({"dataset": result["label"], **row})

    frame = pd.DataFrame(rows)
    parameter_frame = pd.DataFrame(parameters)
    frame.to_csv(OUTPUT_ROOT / "combined_summary.csv", index=False)
    parameter_frame.to_csv(OUTPUT_ROOT / "even_odd_parameters.csv", index=False)

    lines = [
        f"# Pooled-{edge_count}-edge even/odd DCD-SM: S&P 500 and N225", "",
        "- Extreme breakpoints: `(0,.01,.05,.1,.2,...,.9,.95,.99,1)`.",
        f"- `e_L=({edge_ones},0,...,0)` and `e_R=(0,...,0,{edge_ones})`.",
        "- Even direction: `d_C=Q'(e_L+e_R)`; odd direction: `d_A=Q'(e_R-e_L)`.",
        "- Both directions are fitted jointly with separate HAR(1,5,22) block tests.",
        "- Scores are projected off location/log-scale-squared; no whitening.", "",
        "| market | even joint p | odd joint p | all-six p | log gain vs M3 | p | CRPS gain vs M3 | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['even_joint_p']:.6g} | {row['odd_joint_p']:.6g} | "
            f"{row['all_six_p']:.6g} | {row['log_gain_vs_m3']:.9g} | "
            f"{row['log_p_vs_m3']:.6g} | {row['crps_gain_vs_m3']:.9g} | "
            f"{row['crps_p_vs_m3']:.6g} |"
        )
    lines.extend(["", "## Coefficients", "",
                  "| market | direction | horizon | estimate | HAC(22) SE | t | p |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in parameter_frame.to_dict(orient="records"):
        direction = "even" if row["block"] == "common" else "odd"
        lines.append(
            f"| {row['dataset']} | {direction} | {row['horizon']} | "
            f"{row['estimate']:.9g} | {row['hac22_se_conditional_on_reference']:.9g} | "
            f"{row['t']:.6g} | {row['p_two_sided']:.6g} |"
        )
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = base.DATASETS if args.market == "both" else (args.market,)
    if args.summarize_only:
        results = [
            json.loads((OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8"))
            for dataset in selected
        ]
    else:
        results = [
            base.run_dataset(
                dataset,
                edge_basis_count=2,
                output_root=OUTPUT_ROOT,
                common_only_name=EVEN_ONLY,
                common_asymmetric_name=EVEN_ODD,
            )
            for dataset in selected
        ]
    if selected == base.DATASETS:
        summarize(results)


if __name__ == "__main__":
    main()
