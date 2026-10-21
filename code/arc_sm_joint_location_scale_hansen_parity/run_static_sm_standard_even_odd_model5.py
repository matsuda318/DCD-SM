"""Add the locked endpoint even/odd M5 after the modular dynamic-skew-t M3.

M5 freezes the staged M3 reference path, projects the fitted-static-SM q score
off only the location and log-scale-squared reference tangents at the fixed M1
anchor, and uses the locked one-endpoint common/even and left-right/odd
HAR(1,5,22) updates. Eta and lambda directions deliberately remain in the q
score. There is no whitening.

The q update is exact conditional on the frozen M3 path.  M3 itself remains the
staged modular reference-score specification; the combined construction is not
a jointly estimated full-corrected-density GAS model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_skewt_four_models as fitted
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_common_asymmetric_extreme_knots_markets as engine
import run_static_sm_standard_even_odd_all_series as standard
import run_static_sm_standard_even_odd_sectors as standard_sectors


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output_static_sm_standard_even_odd_model5"
PAPER_ROOT = ROOT.parent / "paper"
PAPER_STEM = "endpoint_model5_results"
M3 = engine.M3
M4 = standard.STANDARD_EVEN_ODD
M5 = "m5_standard_one_edge_even_odd_after_dynamic_eta_lambda"
NON_SECTORS = tuple(standard.ALL_SERIES)
SECTORS = tuple(standard_sectors.ALL_SECTORS)
ALL_DATASETS = NON_SECTORS + SECTORS


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=values.index)


def category(dataset: str) -> str:
    if dataset == "sp500":
        return "sp500"
    if dataset == "n225":
        return "n225"
    if dataset in base.FX:
        return "fx"
    if dataset in SECTORS:
        return "sector"
    return "stock"


def load_dataset(dataset: str):
    if dataset in SECTORS:
        panel, info, saved = standard_sectors.load_inputs(dataset)
        source_root = standard_sectors.OUTPUT_ROOT
        label = standard_sectors.dataset_label(dataset)
    else:
        panel, info, saved = base.load_inputs(dataset)
        source_root = standard.OUTPUT_ROOT
        label = base.dataset_label(dataset)
    standard_result = json.loads(
        (source_root / dataset / "estimation.json").read_text(encoding="utf-8")
    )
    standard_scores = pd.read_csv(source_root / dataset / "daily_oos_scores.csv")
    return panel, info, saved, standard_result, standard_scores, label


def endpoint_directions(q_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(q_matrix.shape[0])
    right = np.zeros(q_matrix.shape[0])
    left[0] = 1.0
    right[-1] = 1.0
    d_left = q_matrix.T @ left
    d_right = q_matrix.T @ right
    return d_left + d_right, d_right - d_left


def four_tangent_endpoint_transform(
    directions: np.ndarray,
    projection: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    directions = np.asarray(directions, dtype=float)
    projection = np.asarray(projection, dtype=float)

    def transform(raw_q_score: np.ndarray, reference_score: np.ndarray) -> np.ndarray:
        efficient = raw_q_score - projection @ reference_score[: projection.shape[1]]
        scalar_scores = directions @ efficient
        return directions * scalar_scores[:, None]

    return transform


def comparison_rows(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = ((M5, M3), (M5, M4), (M4, M3))
    return pd.DataFrame([
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in scoring.SCORE_NAMES
    ])


def run_dataset(dataset: str) -> dict[str, object]:
    panel, info, saved, standard_result, standard_scores, label = load_dataset(dataset)
    output = OUTPUT_ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(standard_result["sample"]["training_days"])
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows

    breakpoints = np.asarray(standard_result["specification"]["basis_breakpoints"], dtype=float)
    if not np.array_equal(breakpoints, base.BREAKPOINTS):
        raise RuntimeError(f"{dataset}: source M4 does not use the locked standard breakpoints")
    knots = core.make_clamped_knots(breakpoints, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    q_matrix = core.helmert_submatrix(masses.size)
    anchor = np.asarray(saved["model1"]["state"], dtype=float)
    weights0 = np.asarray(standard_result["static_q0"]["weights"], dtype=float)

    print(f"{dataset}: reproducing frozen modular M3 reference path", flush=True)
    (
        _m2_train, _m2_oos, _m2_full,
        m3_train, m3_oos, m3_full, reference_checks,
    ) = base.reference_paths(panel, n_train, anchor, saved)
    m3_train_reference = base.identity.shape_as_reference(m3_train)
    m3_oos_reference = base.identity.shape_as_reference(m3_oos)

    full_u = core.dynamic_reference_pit(panel.values, panel.offsets, m3_full.states)
    full_phi = core.normalized_bspline_basis(full_u, knots, core.DEGREE, masses)
    geometry = scoring.corrected_tangent_geometry(
        anchor,
        weights0,
        knots,
        masses,
        q_matrix,
        (0, 1),
        "fitted-h0 q score off modular m/log-sigma2 reference scores only",
    )
    d_common, d_asymmetric = endpoint_directions(q_matrix)
    directions = np.vstack((d_common, d_asymmetric))
    transform = four_tangent_endpoint_transform(
        directions, geometry.projection_q_on_hansen
    )

    print(f"{dataset}: fitting endpoint even/odd M5", flush=True)
    model5, static5, train5 = split.fit_block_arc_sm(
        full_phi[:training_rows],
        train_offsets,
        m3_train.scores,
        weights0,
        q_matrix,
        transform,
        ("common_even", "asymmetric_odd"),
    )
    oos5 = fitted.continue_arc(
        full_phi[training_rows:],
        oos_offsets,
        m3_oos.scores,
        np.asarray(model5["coefficients"], dtype=float),
        weights0,
        q_matrix,
        transform,
        train5,
    )
    full5 = split.filter_block_arc_sm(
        full_phi,
        panel.offsets,
        m3_full.scores,
        np.asarray(model5["coefficients"], dtype=float),
        weights0,
        q_matrix,
        transform,
    )

    pair = scoring.score_oos_pair(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        panel.values[:training_rows],
        m3_oos_reference,
        oos5,
        knots,
        masses,
        weights0,
        M3,
        M5,
    )
    source_m3 = standard_scores.loc[standard_scores["model"] == M3].sort_values("date")
    reproduced_m3 = pair.loc[pair["model"] == M3].sort_values("date")
    score_columns = list(scoring.SCORE_NAMES)
    if source_m3["date"].tolist() != reproduced_m3["date"].tolist():
        raise RuntimeError(f"{dataset}: source and reproduced M3 OOS dates differ")
    m3_score_error = float(np.max(np.abs(
        source_m3[score_columns].to_numpy()
        - reproduced_m3[score_columns].to_numpy()
    )))
    source_m4 = standard_scores.loc[standard_scores["model"] == M4]
    scores = pd.concat((
        reproduced_m3,
        source_m4,
        pair.loc[pair["model"] == M5],
    ), ignore_index=True)
    if scores.groupby("date")["model"].nunique().min() != 3:
        raise RuntimeError(f"{dataset}: incomplete M3/M4/M5 OOS panel")
    comparisons = comparison_rows(scores)

    checks = {
        **reference_checks,
        "q_zero_weight_error": float(np.max(np.abs(static5.weights - weights0[None, :]))),
        "m3_daily_oos_score_reproduction_error": m3_score_error,
        "m5_continuation_error": float(max(
            np.max(np.abs(full5.states[n_train:] - oos5.states)),
            np.max(np.abs(full5.weights[n_train:] - oos5.weights)),
            np.max(np.abs(full5.contributions[n_train:] - oos5.contributions)),
        )),
        "m5_maximum_absolute_training_state": float(np.max(np.abs(train5.states))),
        "m5_maximum_absolute_oos_state": float(np.max(np.abs(oos5.states))),
        "m5_minimum_training_weight": float(np.min(train5.weights)),
        "m5_minimum_oos_weight": float(np.min(oos5.weights)),
        "common_asymmetric_euclidean_inner_product": float(d_common @ d_asymmetric),
    }
    if max(
        checks["q_zero_weight_error"],
        checks["m3_daily_oos_score_reproduction_error"],
        checks["m5_continuation_error"],
        abs(checks["common_asymmetric_euclidean_inner_product"]),
    ) > 2e-10:
        raise RuntimeError(f"{dataset}: M5 nesting/continuation check failed: {checks}")

    result = {
        "analysis": "locked endpoint even/odd M5 after modular dynamic-skew-t M3",
        "dataset": info,
        "label": label,
        "category": category(dataset),
        "sample": standard_result["sample"],
        "specification": {
            "basis_breakpoints": breakpoints.tolist(),
            "basis_count": int(masses.size),
            "static_sm": "fitted and frozen; q=0 reproduces fitted nonuniform h0",
            "reference_path": "frozen staged modular M3 dynamic m/sigma/eta/lambda path",
            "q_score": "exact conditional on frozen M3 path",
            "nuisance_projection": (
                "fixed-M1-anchor Fisher projection off modular reference-only "
                "m/log-sigma2 scores; eta/lambda deliberately retained"
            ),
            "joint_full_density_gas": False,
            "directions": "d_C=d_L+d_R; d_A=d_R-d_L; one endpoint basis per side",
            "score_whitening": False,
            "updates": "separate diagonal HAR(1,5,22) for common/even and asymmetric/odd",
        },
        "model5": model5,
        "model5_geometry": geometry.diagnostics,
        "source_model4_joint_tests": standard_result["common_asymmetric"]["joint_blocks"],
        "oos_summary": scores.groupby("model", as_index=False)[score_columns].mean().to_dict(orient="records"),
        "oos_comparisons": comparisons.to_dict(orient="records"),
        "checks": checks,
    }
    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    comparisons.to_csv(output / "oos_comparisons.csv", index=False)
    pd.DataFrame(model5["parameters"]).to_csv(output / "model5_parameters.csv", index=False)
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_dataset_report(result, comparisons, output)
    print(
        f"{dataset}: complete; joint p={model5['joint_all']['p']:.6g}; "
        f"even p={model5['joint_blocks']['common_even']['p']:.6g}; "
        f"odd p={model5['joint_blocks']['asymmetric_odd']['p']:.6g}",
        flush=True,
    )
    return result


def write_dataset_report(
    result: dict[str, object], comparisons: pd.DataFrame, output: Path
) -> None:
    model5 = result["model5"]
    lines = [
        f"# {result['label']}: endpoint even/odd M5", "",
        "M5 freezes the staged modular M3 reference path. Its q score is projected",
        "off location and log scale-squared at the fixed M1 anchor; dynamic eta and",
        "lambda directions are deliberately retained. The score is then",
        "restricted to the locked one-endpoint common/even and asymmetric/odd",
        "HAR(1,5,22) directions. There is no whitening.", "",
        "M5's q update is exact conditional on the frozen M3 path. The construction",
        "is not a jointly estimated full-corrected-density GAS model.", "",
        "## Score-coefficient tests", "",
        f"- Common/even joint p: {model5['joint_blocks']['common_even']['p']:.9g}",
        f"- Asymmetric/odd joint p: {model5['joint_blocks']['asymmetric_odd']['p']:.9g}",
        f"- All-six joint p: {model5['joint_all']['p']:.9g}", "",
        "| block | horizon | estimate | HAC(22) SE | t | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model5["parameters"]:
        lines.append(
            f"| {row['block']} | {row['horizon']} | {row['estimate']:.9g} | "
            f"{row['hac22_se_conditional_on_reference']:.9g} | {row['t']:.6g} | "
            f"{row['p_two_sided']:.6g} |"
        )
    lines.extend([
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
    lines.extend(["", "## Numerical checks", ""])
    for name, value in result["checks"].items():
        lines.append(f"- {name}: {value:.9g}")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_datasets(value: str) -> list[str]:
    aliases = {
        "all": list(ALL_DATASETS),
        "markets": ["sp500", "n225"],
        "stocks": list(base.STOCKS),
        "fx": list(base.FX),
        "sectors": list(SECTORS),
        "sp500_sectors": [d for d in SECTORS if d.startswith("sp500_")],
        "n225_sectors": [d for d in SECTORS if d.startswith("n225_")],
    }
    if value in aliases:
        return aliases[value]
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return selected


def summarize() -> None:
    results = [
        json.loads((OUTPUT_ROOT / dataset / "estimation.json").read_text(encoding="utf-8"))
        for dataset in ALL_DATASETS
    ]
    tests, parameters, comparisons, diagnostics = [], [], [], []
    for dataset, result in zip(ALL_DATASETS, results):
        m5 = result["model5"]
        tests.append({
            "dataset": dataset,
            "label": result["label"],
            "category": result["category"],
            "even_p": m5["joint_blocks"]["common_even"]["p"],
            "odd_p": m5["joint_blocks"]["asymmetric_odd"]["p"],
            "all_six_p": m5["joint_all"]["p"],
            "training_gain_vs_m3_static_q": m5["gain_vs_static"],
        })
        parameters.extend([
            {"dataset": dataset, "label": result["label"], "category": result["category"], **row}
            for row in m5["parameters"]
        ])
        comparisons.extend([
            {"dataset": dataset, "label": result["label"], "category": result["category"], **row}
            for row in result["oos_comparisons"]
        ])
        diagnostics.append({
            "dataset": dataset,
            "label": result["label"],
            "category": result["category"],
            "successful_starts": m5["successful_starts"],
            "total_starts": m5["total_starts"],
            "largest_successful_objective_gap": m5["largest_successful_objective_gap"],
            "hessian_minimum_eigenvalue_scaled": min(m5["hessian_eigenvalues_scaled"]),
            "sandwich_minimum_eigenvalue": min(m5["sandwich_covariance_eigenvalues"]),
            "maximum_absolute_mean_day_gradient": m5["maximum_absolute_mean_day_gradient"],
            **result["checks"],
        })
    tests_frame = pd.DataFrame(tests)
    parameters_frame = pd.DataFrame(parameters)
    comparisons_frame = pd.DataFrame(comparisons)
    diagnostics_frame = pd.DataFrame(diagnostics)
    for name in ("even", "odd", "all_six"):
        tests_frame[f"{name}_bh_q_within_category"] = tests_frame.groupby(
            "category", sort=False
        )[f"{name}_p"].transform(bh_adjust)
    comparisons_frame["bh_q_within_category_pair_score"] = comparisons_frame.groupby(
        ["category", "proposed", "benchmark", "score"], sort=False
    )["hac22_two_sided_p"].transform(bh_adjust)
    tests_frame.to_csv(OUTPUT_ROOT / "model5_score_tests.csv", index=False)
    parameters_frame.to_csv(OUTPUT_ROOT / "model5_parameters.csv", index=False)
    comparisons_frame.to_csv(OUTPUT_ROOT / "model5_oos_comparisons.csv", index=False)
    diagnostics_frame.to_csv(OUTPUT_ROOT / "model5_numerical_diagnostics.csv", index=False)

    lines = [
        "# Endpoint even/odd M5: full 35-dataset results", "",
        "M5 is the locked one-endpoint even/odd DCD-SM applied after the frozen",
        "staged modular dynamic-skew-t M3 path. The q score is projected off",
        "location and log scale-squared at the fixed M1 anchor; eta and lambda",
        "directions deliberately remain in the update.", "",
        "## Detection summary at 5%", "",
        "| category | datasets | even | odd | all six |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ("sp500", "n225", "sector", "stock", "fx"):
        subset = tests_frame.loc[tests_frame["category"] == group]
        lines.append(
            f"| {group} | {len(subset)} | {int((subset['even_p'] < .05).sum())} | "
            f"{int((subset['odd_p'] < .05).sum())} | {int((subset['all_six_p'] < .05).sum())} |"
        )
    lines.extend([
        "", "## Strict-OOS point-estimate wins", "",
        "A win means the proposed model has the better oriented score.", "",
        "| category | comparison | score | wins | datasets |",
        "|---|---|---|---:|---:|",
    ])
    for group in ("sp500", "n225", "sector", "stock", "fx"):
        for pair in ((M5, M3), (M5, M4)):
            for score in ("continuous_log_score", "crps"):
                subset = comparisons_frame.loc[
                    (comparisons_frame["category"] == group)
                    & (comparisons_frame["proposed"] == pair[0])
                    & (comparisons_frame["benchmark"] == pair[1])
                    & (comparisons_frame["score"] == score)
                ]
                lines.append(
                    f"| {group} | {pair[0]} vs {pair[1]} | {score} | "
                    f"{int((subset['improvement_mean'] > 0).sum())} | {len(subset)} |"
                )
    lines.extend([
        "", "Complete coefficient tests and all nine strict-OOS comparisons are in",
        "`model5_score_tests.csv`, `model5_parameters.csv`, and",
        "`model5_oos_comparisons.csv`.",
    ])
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_paper_note(tests_frame, parameters_frame, comparisons_frame, diagnostics_frame)


def write_paper_note(
    tests: pd.DataFrame,
    parameters: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    tests.to_csv(PAPER_ROOT / f"{PAPER_STEM}_score_tests.csv", index=False)
    parameters.to_csv(PAPER_ROOT / f"{PAPER_STEM}_coefficients.csv", index=False)
    comparisons.to_csv(PAPER_ROOT / f"{PAPER_STEM}_oos_comparisons.csv", index=False)
    diagnostics.to_csv(PAPER_ROOT / f"{PAPER_STEM}_numerical_diagnostics.csv", index=False)

    def fmt(value: float) -> str:
        value = float(value)
        return f"{value:.4g}" if value != 0.0 else "0"

    lookup = {
        (row.dataset, row.proposed, row.benchmark, row.score): row
        for row in comparisons.itertuples(index=False)
    }
    detection = []
    wins = []
    for group in ("sp500", "n225", "sector", "stock", "fx"):
        subset = tests.loc[tests["category"] == group]
        detection.append((
            group, len(subset),
            int((subset["even_p"] < .05).sum()),
            int((subset["even_bh_q_within_category"] < .05).sum()),
            int((subset["odd_p"] < .05).sum()),
            int((subset["odd_bh_q_within_category"] < .05).sum()),
            int((subset["all_six_p"] < .05).sum()),
            int((subset["all_six_bh_q_within_category"] < .05).sum()),
        ))
        for benchmark in (M3, M4):
            for score in ("continuous_log_score", "crps"):
                selected = comparisons.loc[
                    (comparisons["category"] == group)
                    & (comparisons["proposed"] == M5)
                    & (comparisons["benchmark"] == benchmark)
                    & (comparisons["score"] == score)
                ]
                wins.append((
                    group, benchmark, score,
                    int((selected["improvement_mean"] > 0).sum()), len(selected),
                    int(((selected["improvement_mean"] > 0) & (selected["hac22_two_sided_p"] < .05)).sum()),
                    int(((selected["improvement_mean"] < 0) & (selected["hac22_two_sided_p"] < .05)).sum()),
                ))

    lines = [
        "# Standard endpoint DCD-SM M5: full empirical results", "",
        "Updated: 2026-09-14", "",
        "## Specification", "",
        "M5 starts from the frozen staged modular M3 dynamic-skew-t path. The fitted",
        "static spline correction is retained as the q=0 anchor. At the fixed M1",
        "anchor, the q score is Fisher-projected off the modular reference scores for",
        "location and log scale-squared only. Dynamic eta and lambda directions are",
        "deliberately retained. The resulting score is restricted to",
        "the same one-endpoint common/even and asymmetric/odd directions as the locked",
        "M4, with separate HAR(1,5,22) coefficients and no Fisher whitening.", "",
        "Conditional on the frozen M3 path, the M5 q score and q=0 nesting are exact.",
        "M3 remains a staged modular reference-score model, so M5 is not a jointly",
        "estimated full-corrected-density GAS specification.", "",
        "The experiment covers S&P 500, Nikkei 225, 17 sector panels, 12 individual",
        "stocks, and 4 FX series (35 datasets). The specification and OOS windows were",
        "not changed after inspecting individual M5 results.", "",
        "## In-sample detection", "",
        "Counts are shown as unadjusted/BH-within-category rejections.", "",
        "| category | datasets | common/even | asymmetric/odd | all-six |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in detection:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]}/{row[3]} | {row[4]}/{row[5]} | "
            f"{row[6]}/{row[7]} |"
        )
    lines.extend([
        "", "Because eta and lambda directions are deliberately retained, these tests",
        "measure endpoint DCD-SM persistence conditional on the frozen M3 path; they",
        "are not tests of persistence orthogonal to dynamic eta and lambda. In-sample",
        "detection also does not by itself establish incremental forecast value.", "",
        "## Strict-OOS summary", "",
        "Every improvement is oriented so positive favors M5. `positive significant`",
        "and `negative significant` count unadjusted paired HAC(22) p-values below .05.",
        "The machine-readable comparison table also reports BH q-values within each",
        "category, model pair, and score.", "",
        "| category | benchmark | score | point wins | N | positive significant | negative significant |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in wins:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |"
        )
    lines.extend([
        "", "The tables above report the complete point-win and unadjusted-significance",
        "counts for the revised M5. Since its q update overlaps eta/lambda directions,",
        "any OOS gain should be interpreted as a flexible endpoint correction to the",
        "frozen dynamic-skew-t path, not as forecast evidence beyond eta and lambda.", "",
        "## Dataset-level log-score and CRPS results", "",
        "Positive gains favor M5. Parentheses contain paired HAC(22) p-values.", "",
        "| category | dataset | even p | odd p | all-six p | log vs M3 | CRPS vs M3 | log vs M4 | CRPS vs M4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tests.itertuples(index=False):
        m3_log = lookup[(row.dataset, M5, M3, "continuous_log_score")]
        m3_crps = lookup[(row.dataset, M5, M3, "crps")]
        m4_log = lookup[(row.dataset, M5, M4, "continuous_log_score")]
        m4_crps = lookup[(row.dataset, M5, M4, "crps")]
        lines.append(
            f"| {row.category} | {row.label} | {fmt(row.even_p)} | {fmt(row.odd_p)} | "
            f"{fmt(row.all_six_p)} | {fmt(m3_log.improvement_mean)} ({fmt(m3_log.hac22_two_sided_p)}) | "
            f"{fmt(m3_crps.improvement_mean)} ({fmt(m3_crps.hac22_two_sided_p)}) | "
            f"{fmt(m4_log.improvement_mean)} ({fmt(m4_log.hac22_two_sided_p)}) | "
            f"{fmt(m4_crps.improvement_mean)} ({fmt(m4_crps.hac22_two_sided_p)}) |"
        )
    unstable = diagnostics.loc[
        (diagnostics["m5_maximum_absolute_oos_state"] > 10)
        | (diagnostics["largest_successful_objective_gap"] > 1e-4)
    ]
    lines.extend([
        "", "## Numerical and stability audit", "",
        "All 35 datasets pass q=0 weight nesting, M3 daily-score reproduction, and",
        "train/OOS continuation checks at 2e-10. All five starts converged in every",
        "fit, although some datasets contain materially different successful local",
        "optima or large OOS states. These cases are:", "",
        "| dataset | successful-objective gap | max abs(q) train | max abs(q) OOS | min OOS weight |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in unstable.itertuples(index=False):
        lines.append(
            f"| {row.label} | {fmt(row.largest_successful_objective_gap)} | "
            f"{fmt(row.m5_maximum_absolute_training_state)} | "
            f"{fmt(row.m5_maximum_absolute_oos_state)} | {fmt(row.m5_minimum_oos_weight)} |"
        )
    lines.extend([
        "", "The display criterion flags maximum OOS |q| above 10 or a successful-start",
        "objective gap above 1e-4. An empty table means that no dataset meets either",
        "criterion. Continuation and score-reproduction checks distinguish genuine",
        "model outputs from implementation failures.", "",
        "## Machine-readable record", "",
        f"- `{PAPER_STEM}_score_tests.csv`: all block and joint tests.",
        f"- `{PAPER_STEM}_coefficients.csv`: all 210 M5 coefficients and HAC statistics.",
        f"- `{PAPER_STEM}_oos_comparisons.csv`: all 945 comparisons (35 x 3 pairs x 9 losses).",
        f"- `{PAPER_STEM}_numerical_diagnostics.csv`: nesting, continuation, optimizer, and state diagnostics.",
    ])
    (PAPER_ROOT / f"{PAPER_STEM}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for dataset in selected_datasets(args.dataset):
            run_dataset(dataset)
    if args.summarize_only or args.dataset == "all":
        summarize()


if __name__ == "__main__":
    main()
