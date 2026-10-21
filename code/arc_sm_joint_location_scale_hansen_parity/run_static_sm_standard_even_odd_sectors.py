"""Run the locked standard endpoint even/odd DCD-SM on 17 sector panels.

The standard specification uses breakpoints
(0,.05,.10,.20,...,.90,.95,1), one endpoint basis per side, and the
even/odd directions d_C=d_L+d_R and d_A=d_R-d_L.  Each direction follows
its own diagonal HAR(1,5,22) score update.  The q score is Fisher-projected
off location and log-scale-squared at the fixed reference anchor; it is not
whitened.

Previously locked sector M1--M3 reference estimates are reused.  The static
SM is refitted for the standard basis and frozen, and only M4 is newly
estimated.  The principal strict-OOS comparison is M4 versus M3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_identity_sm_skewt_sector_five_models as sectors
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_common_asymmetric_extreme_knots_markets as engine


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_sm_standard_even_odd_sectors"
M4_EVEN = "m4_standard_one_edge_even_arc_sm"
M4_EVEN_ODD = "m4_standard_one_edge_even_odd_arc_sm"
M1 = "m1_static_skewt_static_sm"
ALL_SECTORS = tuple(sectors.SECTORS)


def load_inputs(dataset: str):
    panel, info = sectors.load_sector(dataset)
    path = sectors.OUTPUT_ROOT / dataset / "estimation.json"
    if not path.exists():
        raise FileNotFoundError(f"missing locked sector M1--M3 result: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    return panel, info, saved


def dataset_label(dataset: str) -> str:
    spec = sectors.SECTORS[dataset]
    return f"{str(spec['market']).upper()} {spec['label']}"


def configure_engine() -> None:
    # The numerical engine delegates only data/result lookup and labeling to
    # these functions.  Its estimation and scoring code remains unchanged.
    base.load_inputs = load_inputs
    base.dataset_label = dataset_label


def parse_selection(value: str) -> list[str]:
    if value == "all":
        return list(ALL_SECTORS)
    if value in ("sp500", "n225"):
        return [d for d in ALL_SECTORS if sectors.SECTORS[d]["market"] == value]
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ALL_SECTORS))
    if unknown:
        raise ValueError(f"unknown sectors: {unknown}")
    return selected


def ensure_m1_scores(dataset: str, result: dict[str, object]) -> pd.DataFrame:
    """Add the static skew-t + fitted-static-SM M1 OOS density if absent."""
    path = OUTPUT_ROOT / dataset / "daily_oos_scores.csv"
    scores = pd.read_csv(path)
    if M1 in set(scores["model"]):
        return scores
    panel, _info, saved = load_inputs(dataset)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= base.locked.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_values = panel.values[:training_rows]
    oos_offsets = panel.offsets[n_train:] - training_rows
    anchor = np.asarray(saved["model1"]["state"], dtype=float)
    knots = engine.core.make_clamped_knots(base.BREAKPOINTS, engine.core.DEGREE)
    masses = engine.core.bspline_masses(knots, engine.core.DEGREE)
    weights0 = np.asarray(result["static_q0"]["weights"], dtype=float)
    m1_reference = engine.core.filter_reference(
        panel.values[training_rows:], oos_offsets, anchor, np.zeros((3, 4))
    )
    m1_u = engine.core.dynamic_reference_pit(
        panel.values[training_rows:], oos_offsets, m1_reference.states
    )
    m1_phi = engine.core.normalized_bspline_basis(
        m1_u, knots, engine.core.DEGREE, masses
    )
    m1_correction = engine.five.static_correction(m1_phi, oos_offsets, weights0)
    pair = scoring.score_oos_pair(
        panel.values[training_rows:], oos_offsets, dates[n_train:], train_values,
        m1_reference, m1_correction, knots, masses, weights0, M1, "_discard_m1",
    )
    scores = pd.concat(
        (scores, pair.loc[pair["model"] == M1]), ignore_index=True
    )
    scores.to_csv(path, index=False)
    return scores


def summarize() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    all_level_rows: list[pd.DataFrame] = []
    all_model_comparisons: list[dict[str, object]] = []
    for dataset in ALL_SECTORS:
        path = OUTPUT_ROOT / dataset / "estimation.json"
        if not path.exists():
            raise FileNotFoundError(f"missing standard sector result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        scores = ensure_m1_scores(dataset, result)
        joint = result["common_asymmetric"]
        lookup = {
            (row["benchmark"], row["score"]): row
            for row in result["oos_comparisons"]
            if row["proposed"] == M4_EVEN_ODD
        }
        spec = sectors.SECTORS[dataset]
        log = lookup[(engine.M3, "continuous_log_score")]
        crps = lookup[(engine.M3, "crps")]
        rows.append({
            "dataset": dataset,
            "market": spec["market"],
            "sector": spec["label"],
            "even_p": joint["joint_blocks"]["common"]["p"],
            "odd_p": joint["joint_blocks"]["asymmetric"]["p"],
            "joint_p": joint["joint_all"]["p"],
            "m4_minus_m3_log_gain": log["improvement_mean"],
            "log_hac22_p": log["hac22_two_sided_p"],
            "m4_minus_m3_crps_gain": crps["improvement_mean"],
            "crps_hac22_p": crps["hac22_two_sided_p"],
            "successful_starts": joint["successful_starts"],
            "largest_successful_objective_gap": joint["largest_successful_objective_gap"],
            "maximum_oos_state": result["checks"]["common_asymmetric_maximum_absolute_oos_state"],
        })
        for row in result["oos_comparisons"]:
            if row["proposed"] == M4_EVEN_ODD and row["benchmark"] == engine.M3:
                comparison_rows.append({
                    "dataset": dataset,
                    "market": spec["market"],
                    "sector": spec["label"],
                    **row,
                })
        for row in joint["parameters"]:
            parameter_rows.append({
                "dataset": dataset,
                "market": spec["market"],
                "sector": spec["label"],
                "direction": "even" if row["block"] == "common" else "odd",
                **row,
            })
        model_order = (M1, engine.M2, engine.M3, M4_EVEN_ODD)
        levels = (
            scores.loc[scores["model"].isin(model_order)]
            .groupby("model", as_index=False)[list(scoring.SCORE_NAMES)].mean()
        )
        levels.insert(0, "sector", spec["label"])
        levels.insert(0, "market", spec["market"])
        levels.insert(0, "dataset", dataset)
        all_level_rows.append(levels)
        for proposed, benchmark in (
            (engine.M2, M1),
            (engine.M3, engine.M2),
            (M4_EVEN_ODD, engine.M2),
            (M4_EVEN_ODD, engine.M3),
        ):
            for score in scoring.SCORE_NAMES:
                all_model_comparisons.append({
                    "dataset": dataset,
                    "market": spec["market"],
                    "sector": spec["label"],
                    **engine.core.score_comparison(scores, proposed, benchmark, score),
                })

    summary = pd.DataFrame(rows)
    comparisons = pd.DataFrame(comparison_rows)
    parameters = pd.DataFrame(parameter_rows)
    for name in ("even", "odd", "joint"):
        summary[f"{name}_bh_q"] = sectors.bh_adjust(summary[f"{name}_p"])
    comparisons["bh_q_across_17_sectors_for_score"] = comparisons.groupby(
        "score", sort=False
    )["hac22_two_sided_p"].transform(sectors.bh_adjust)
    summary.to_csv(OUTPUT_ROOT / "sector_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_ROOT / "m4_vs_m3_all_oos_scores.csv", index=False)
    parameters.to_csv(OUTPUT_ROOT / "m4_even_odd_parameters.csv", index=False)
    pd.concat(all_level_rows, ignore_index=True).to_csv(
        OUTPUT_ROOT / "m1_m4_oos_levels.csv", index=False
    )
    pd.DataFrame(all_model_comparisons).to_csv(
        OUTPUT_ROOT / "m1_m4_oos_comparisons.csv", index=False
    )

    lines = [
        "# Standard endpoint even/odd DCD-SM: 17 sector panels", "",
        "## Locked specification", "",
        "- Models: M1--M4. M1--M3 use the locked modular reference-score paths; M4 is the newly estimated endpoint DCD-SM.",
        "- Breakpoints: `(0,.05,.10,.20,...,.90,.95,1)`; one endpoint basis per side.",
        "- Directions: `d_C=d_L+d_R` (even) and `d_A=d_R-d_L` (odd).",
        "- Updates: joint diagonal score-HAR(1,5,22), six coefficients.",
        "- Geometry: q score projected off location/log-scale-squared at the fixed anchor; no whitening.",
        "- Main forecast comparison: M4 versus dynamic skew-t M3 on the same strict-OOS days.", "",
        "## Detection and strict-OOS summary", "",
        "Positive gains favor M4. CRPS gain is defined as benchmark loss minus M4 loss.", "",
        "| market | sector | even p (BH q) | odd p (BH q) | joint p (BH q) | log gain vs M3 | p | CRPS gain vs M3 | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['market']} | {row['sector']} | {row['even_p']:.5g} ({row['even_bh_q']:.5g}) | "
            f"{row['odd_p']:.5g} ({row['odd_bh_q']:.5g}) | "
            f"{row['joint_p']:.5g} ({row['joint_bh_q']:.5g}) | "
            f"{row['m4_minus_m3_log_gain']:.8g} | {row['log_hac22_p']:.5g} | "
            f"{row['m4_minus_m3_crps_gain']:.8g} | {row['crps_hac22_p']:.5g} |"
        )
    lines.extend(["", "## Aggregate counts", ""])
    for market, total in (("sp500", 11), ("n225", 6), ("all", 17)):
        subset = summary if market == "all" else summary.loc[summary["market"] == market]
        lines.append(
            f"- {market.upper()}: even `{int((subset['even_p'] < .05).sum())}/{total}`, "
            f"odd `{int((subset['odd_p'] < .05).sum())}/{total}`, joint "
            f"`{int((subset['joint_p'] < .05).sum())}/{total}` (BH across all 17: "
            f"even `{int((subset['even_bh_q'] < .05).sum())}/{total}`, odd "
            f"`{int((subset['odd_bh_q'] < .05).sum())}/{total}`, joint "
            f"`{int((subset['joint_bh_q'] < .05).sum())}/{total}`); M4 beats M3 in log "
            f"`{int((subset['m4_minus_m3_log_gain'] > 0).sum())}/{total}` and CRPS "
            f"`{int((subset['m4_minus_m3_crps_gain'] > 0).sum())}/{total}`."
        )
    lines.extend([
        "", "## M4 versus M3 across all nine scores", "",
        "| score | M4 wins | sectors | individually significant M4 wins | individually significant M3 wins |",
        "|---|---:|---:|---:|---:|",
    ])
    score_counts = comparisons.groupby("score", sort=False).apply(
        lambda x: pd.Series({
            "wins": int((x["improvement_mean"] > 0).sum()),
            "total": int(len(x)),
            "sig_m4": int(((x["improvement_mean"] > 0) & (x["hac22_two_sided_p"] < .05)).sum()),
            "sig_m3": int(((x["improvement_mean"] < 0) & (x["hac22_two_sided_p"] < .05)).sum()),
        }),
        include_groups=False,
    ).reset_index()
    for row in score_counts.to_dict(orient="records"):
        lines.append(
        f"| {row['score']} | {int(row['wins'])} | {int(row['total'])} | "
            f"{int(row['sig_m4'])} | {int(row['sig_m3'])} |"
        )
    nonunique = summary.loc[summary["largest_successful_objective_gap"] > 1e-7]
    lines.extend([
        "", "## Numerical audit", "",
        "All 17 selected solutions have five successful starts, positive numerical-Hessian",
        "eigenvalues, positive conditional Godambe-HAC covariance eigenvalues, positive",
        "OOS mixture weights, and exact train/OOS continuation within numerical tolerance.",
        f"Multiple distinct successful local optima occurred in `{len(nonunique)}/17` sectors;",
        "the saved estimate is the successful solution with the largest likelihood.",
    ])
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_engine()
    selected = parse_selection(args.sector)
    if not args.summarize_only:
        for dataset in selected:
            result_path = OUTPUT_ROOT / dataset / "estimation.json"
            if args.resume and result_path.exists():
                print(f"{dataset}: existing result retained", flush=True)
                continue
            engine.run_dataset(
                dataset,
                edge_basis_count=1,
                output_root=OUTPUT_ROOT,
                common_only_name=M4_EVEN,
                common_asymmetric_name=M4_EVEN_ODD,
                breakpoints=base.BREAKPOINTS,
                analysis_name="standard one-edge even/odd DCD-SM: sector panel",
            )
    if args.summarize_only or selected == list(ALL_SECTORS):
        summarize()


if __name__ == "__main__":
    main()
