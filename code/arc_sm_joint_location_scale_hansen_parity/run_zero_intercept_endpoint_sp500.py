"""SP500 LM and OOS for q_t = sum HAR-lagged projected endpoint scores.

This is the locked skew-t, identity-h, tail-refined-knot M4/M5 estimator,
rerun into a separate output directory.  There is no static q0 or intercept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import run_endpoint_two_null_lm_markets as lm
import run_identity_sm_standard_endpoint_five_models as identity
import run_score_direction_split as split
import run_sp500 as core
import run_static_q0_five_models as five
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_boundary_tail_har_tail_knots_batch as base
import run_static_sm_standard_even_odd_model5 as endpoint


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_zero_intercept_endpoint_sp500"


def filter_no_intercept_har22(
    phi: np.ndarray, offsets: np.ndarray, reference_scores: np.ndarray,
    coefficients: np.ndarray, masses: np.ndarray, helmert: np.ndarray,
    transform, initial_state: np.ndarray | None = None,
    initial_component_history: np.ndarray | None = None,
) -> split.BlockFilter:
    """Versioned endpoint filter: no intercept and a full 22-score history."""
    days = len(offsets) - 1
    blocks = coefficients.shape[0]
    dimension = masses.size - 1
    if coefficients.shape != (blocks, 3) or reference_scores.shape[0] != days:
        raise ValueError("endpoint coefficient/reference shape mismatch")
    state = (np.zeros(dimension) if initial_state is None
             else np.asarray(initial_state, dtype=float).copy())
    history = (np.empty((0, blocks, dimension)) if initial_component_history is None
               else np.asarray(initial_component_history, dtype=float).copy()[-22:])
    states = np.empty((days, dimension))
    components = np.empty((days, blocks, dimension))
    weights_path = np.empty((days, masses.size))
    contributions = np.empty(days)
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        states[day] = state
        contribution, component_score, weights = split.cluster_density_and_component_score(
            state, phi[start:stop], masses, helmert
        )
        transformed = transform(helmert.T @ component_score, reference_scores[day])
        if transformed.shape != (blocks, dimension):
            raise ValueError("endpoint transform shape mismatch")
        contributions[day] = contribution
        components[day] = transformed
        weights_path[day] = weights
        history = np.concatenate((history, transformed[None, :, :]), axis=0)[-22:]
        state = np.zeros(dimension)
        for lag, window in enumerate(core.WINDOWS):
            state += np.sum(
                coefficients[:, lag, None] * history[-window:].sum(axis=0) / window,
                axis=0,
            )
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("unstable no-intercept endpoint state")
    return split.BlockFilter(
        contributions=contributions, states=states,
        component_scores=components, weights=weights_path,
        terminal_state=state, terminal_component_history=history,
    )


def no_intercept_lm(
    phi: np.ndarray, offsets: np.ndarray, masses: np.ndarray,
    helmert: np.ndarray, directions: np.ndarray,
    reference_scores: np.ndarray, projection: np.ndarray, model: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Kappa-score LM for the six-parameter, no-intercept alternative.

    No static q0 coordinate, derivative, or nuisance correction is introduced.
    """
    days = len(offsets) - 1
    raw = np.empty((days, helmert.shape[1]))
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw[day] = ((phi[start:stop].mean(axis=0) - 1.0) * masses) @ helmert
    current = raw @ directions.T
    driver = (raw - reference_scores[:, :2] @ projection.T) @ directions.T
    moments = np.column_stack([
        current[:, block] * lm.fixed_denominator_past_mean(driver[:, block], window)
        for block in range(2) for window in core.WINDOWS
    ])
    covariance = moments.T @ moments / days
    numerator = np.sqrt(days) * moments.mean(axis=0)
    rows = []
    for block, indices in (("common_even", np.arange(3)),
                           ("asymmetric_odd", np.arange(3, 6)),
                           ("joint_six", np.arange(6))):
        matrix = covariance[np.ix_(indices, indices)]
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        if eigenvalues[0] <= 1e-12:
            raise RuntimeError(f"{model} {block}: singular LM covariance")
        statistic = float(numerator[indices] @ np.linalg.solve(matrix, numerator[indices]))
        rows.append({
            "dataset": "sp500", "model": model, "null": "kappa_zero_no_intercept",
            "block": block, "df": len(indices), "training_days": days,
            "LM": statistic, "p_chi2": float(chi2.sf(statistic, len(indices))),
            "min_variance_eigenvalue": float(eigenvalues[0]),
        })
    return rows, moments


def kappa_gradient_check(
    phi: np.ndarray, offsets: np.ndarray, masses: np.ndarray,
    helmert: np.ndarray, directions: np.ndarray,
    reference_scores: np.ndarray, projection: np.ndarray,
    moments: np.ndarray,
) -> float:
    """Numerically compare the no-intercept likelihood kappa derivative."""
    days = min(70, len(offsets) - 1)
    transform = endpoint.four_tangent_endpoint_transform(directions, projection)
    short_phi = phi[:offsets[days]]
    short_offsets = offsets[:days + 1]
    short_scores = reference_scores[:days]
    step = 1e-5
    errors = []
    for coordinate in range(6):
        coefficients = np.zeros((2, 3))
        coefficients.flat[coordinate] = step
        plus = filter_no_intercept_har22(
            short_phi, short_offsets, short_scores, coefficients,
            masses, helmert, transform,
        ).composite_log_likelihood
        minus = filter_no_intercept_har22(
            short_phi, short_offsets, short_scores, -coefficients,
            masses, helmert, transform,
        ).composite_log_likelihood
        numerical = (plus - minus) / (2 * step)
        analytic = moments[:days, coordinate].sum()
        errors.append(abs(numerical - analytic))
    return float(max(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_output = identity.OUTPUT_ROOT
    original_loader = identity.load_dataset
    original_filter = split.filter_block_arc_sm
    original_estimate = json.loads(
        (original_output / "sp500" / "estimation.json").read_text(encoding="utf-8")
    )
    panel, info = five.load_dataset("sp500")
    saved = {
        "model1": original_estimate["source_model1"],
        "model2": original_estimate["source_model2"],
        "model3": original_estimate["source_model3"],
    }
    if args.summarize_only:
        estimate = json.loads(
            (OUTPUT / "sp500" / "estimation.json").read_text(encoding="utf-8")
        )
    else:
        try:
            identity.OUTPUT_ROOT = OUTPUT
            split.filter_block_arc_sm = filter_no_intercept_har22
            identity.load_dataset = lambda dataset: (
                panel, info, saved, original_estimate["label"]
            ) if dataset == "sp500" else original_loader(dataset)
            estimate = identity.run_dataset("sp500")
        finally:
            identity.OUTPUT_ROOT = original_output
            identity.load_dataset = original_loader
            split.filter_block_arc_sm = original_filter
        estimate["analysis"] = "no-intercept endpoint HAR22 SP500 pilot"
        estimate["specification"]["M4_M5_history"] = (
            "exactly 22 lagged endpoint scores; no static q0 or intercept"
        )
        (OUTPUT / "sp500" / "estimation.json").write_text(
            json.dumps(estimate, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    n_train = int(estimate["sample"]["training_days"])
    train_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[:n_train + 1]
    anchor = np.asarray(saved["model1"]["state"], dtype=float)
    knots = core.make_clamped_knots(base.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(len(masses))
    directions = np.vstack(endpoint.endpoint_directions(helmert))
    geometry = scoring.corrected_tangent_geometry(
        anchor, masses, knots, masses, helmert, (0, 1),
        "q0=0 endpoint score off location/log variance",
    )
    m2_train, _m2_oos, _m2_full, m3_train, _m3_oos, _m3_full, checks = (
        base.reference_paths(panel, n_train, anchor, saved)
    )
    if max(checks.values()) > 1e-2:
        raise RuntimeError(f"reference-path reproduction failed: {checks}")

    lm_rows = []
    daily_rows = []
    gradient_errors = {}
    for name, reference in (("M4", m2_train), ("M5", m3_train)):
        u = core.dynamic_reference_pit(
            panel.values[:train_rows], train_offsets, reference.states
        )
        phi = core.normalized_bspline_basis(u, knots, core.DEGREE, masses)
        rows, moments = no_intercept_lm(
            phi, train_offsets, masses, helmert, directions,
            reference.scores, geometry.projection_q_on_hansen, name,
        )
        gradient_errors[name] = kappa_gradient_check(
            phi, train_offsets, masses, helmert, directions,
            reference.scores, geometry.projection_q_on_hansen, moments,
        )
        if gradient_errors[name] > 1e-5:
            raise RuntimeError(
                f"{name}: no-intercept kappa derivative mismatch "
                f"{gradient_errors[name]:.8g}"
            )
        lm_rows.extend(rows)
        daily = pd.DataFrame(moments, columns=[
            f"{block}_har{window}" for block in ("common_even", "asymmetric_odd")
            for window in core.WINDOWS
        ])
        daily.insert(0, "date", panel.daily.date.iloc[:n_train].astype(str).to_numpy())
        daily.insert(0, "model", name)
        daily_rows.append(daily)

    (OUTPUT / "lm_results.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lm_rows).to_csv(OUTPUT / "lm_results.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(
        OUTPUT / "daily_lm_scores.csv", index=False
    )

    old_scores = pd.read_csv(original_output / "sp500" / "daily_oos_scores.csv")
    new_scores = pd.read_csv(OUTPUT / "sp500" / "daily_oos_scores.csv")
    old_scores = old_scores.sort_values(["date", "model"]).reset_index(drop=True)
    new_scores = new_scores.sort_values(["date", "model"]).reset_index(drop=True)
    score_names = list(scoring.SCORE_NAMES)
    if not old_scores[["date", "model"]].equals(new_scores[["date", "model"]]):
        raise RuntimeError("rerun OOS panel does not align with locked standard")
    score_error = float(np.max(np.abs(
        old_scores[score_names].to_numpy() - new_scores[score_names].to_numpy()
    )))

    old_lm = pd.read_csv(lm.OUTPUT / "lm_results.csv")
    old_lm = old_lm.loc[
        (old_lm.dataset == "sp500") & (old_lm.null == "q0_fixed_zero")
    ].sort_values(["model", "block"])
    new_lm = pd.DataFrame(lm_rows).sort_values(["model", "block"])
    old_lm.model = old_lm.model.map({"M4_after_M2": "M4", "M5_after_M3": "M5"})
    lm_error = float(np.max(np.abs(old_lm.LM.to_numpy() - new_lm.LM.to_numpy())))
    if lm_error > 1e-6:
        raise RuntimeError(f"rerun fixed-q0 LM differs from saved test: {lm_error}")

    levels = {row["model"]: row for row in estimate["oos_summary"]}
    tests = {(row["model"], row["block"]): row for row in lm_rows}
    lines = [
        "# SP500: zero-intercept endpoint DCD-SM", "",
        "M1--M3 retain the locked skew-t, h=1 reference stages. M4 freezes M2;",
        "M5 freezes M3. Both fit only six endpoint HAR coefficients:",
        "two directions (common/even and asymmetric/odd) times windows 1, 5, 22.",
        "The full q state is the sum of coefficient-weighted lagged projected",
        "scores. It starts at zero and contains neither a static q0 nor an",
        "intercept. Five starts are used for each M4/M5 estimation.", "",
        "The LM null fixes all six coefficients to zero, so it is precisely",
        "q_t=0 and h=1. Current-day density derivatives are raw q scores;",
        "predictable lagged drivers are projected off location/log variance.",
        "No q0 nuisance adjustment or Fisher whitening is used.", "",
        "| Model | Even LM(3), p | Odd LM(3), p | Joint LM(6), p |",
        "|---|---:|---:|---:|",
    ]
    for name in ("M4", "M5"):
        values = [tests[(name, block)] for block in
                  ("common_even", "asymmetric_odd", "joint_six")]
        lines.append(
            f"| {name} | " + " | ".join(
                f"{row['LM']:.3f}, {row['p_chi2']:.3g}" for row in values
            ) + " |"
        )
    lines += ["", "| Model | OOS Log score ↑ | CRPS ↓ | Q-loss 1% ↓ | Q-loss 5% ↓ |",
              "|---|---:|---:|---:|---:|"]
    for name, model in (("M2", identity.M2), ("M3", identity.M3),
                        ("M4", identity.M4), ("M5", identity.M5)):
        row = levels[model]
        lines.append(
            f"| {name} | {row['continuous_log_score']:.7f} | "
            f"{row['crps']:.8f} | {row['q_loss_0.01']:.9f} | "
            f"{row['q_loss_0.05']:.9f} |"
        )
    comparisons = {
        (row["proposed"], row["benchmark"], row["score"]): row
        for row in estimate["oos_comparisons"]
    }
    lines += ["", "Paired OOS improvements (positive favors the endpoint model):", "",
              "| Comparison | Log score gain (HAC p) | Q-loss 1% gain (HAC p) |",
              "|---|---:|---:|"]
    for label, proposed, benchmark in (("M4 vs M2", identity.M4, identity.M2),
                                       ("M5 vs M3", identity.M5, identity.M3)):
        log = comparisons[(proposed, benchmark, "continuous_log_score")]
        q1 = comparisons[(proposed, benchmark, "q_loss_0.01")]
        lines.append(
            f"| {label} | {log['improvement_mean']:+.6g} "
            f"({log['hac22_two_sided_p']:.3g}) | "
            f"{q1['improvement_mean']:+.6g} ({q1['hac22_two_sided_p']:.3g}) |"
        )
    lines += ["", "Numerical derivative check: the LM score equals the",
              "six-parameter likelihood derivative at kappa=0, with no",
              "static-q0 coordinate in either calculation. Maximum errors:",
              f"M4 {gradient_errors['M4']:.3g}; M5 {gradient_errors['M5']:.3g}.", "",
              "OOS uses the training terminal state and score history,",
              "without resetting or refitting. This version retains all 22",
              "scores, whereas the prior filter retained only 21. Therefore",
              f"the maximum daily OOS difference from the old standard is {score_error:.3g}.",
              f"The analytic no-intercept LM matches the saved fixed-zero",
              f"formula to {lm_error:.3g}; the new numerical derivative check",
              "establishes that it matches the corrected 22-score estimator.", "",
              "Inference is conditional on the estimated reference path; the",
              "daily-cluster LM covariance omits first-stage error and time-HAC.",
              "Because the null also fixes static h=1, this LM tests the",
              "literal no-intercept model against q=0; static shape departures",
              "can contribute to rejection, so it is not a pure persistence test.", ""]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SP500 complete: old/new OOS max difference={score_error:.3g}; LM max error={lm_error:.3g}")


if __name__ == "__main__":
    main()
