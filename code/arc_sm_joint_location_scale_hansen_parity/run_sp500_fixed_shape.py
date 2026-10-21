"""Fixed Hansen shape, dynamic location-scale, and orthogonal parity DCD-SM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_sp500 as full


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_sp500_fixed_shape"
REFERENCE_NAME = "fixed_shape_dynamic_location_scale_hansen"
PROPOSED_NAME = "fixed_shape_dynamic_location_scale_hansen_orthogonal_parity_arc_sm"


def location_scale_geometry(
    anchor: np.ndarray,
    knots: np.ndarray,
    masses: np.ndarray,
    helmert: np.ndarray,
    quadrature_order: int = 128,
) -> full.EfficientGeometry:
    """Project the spline score off location and log-scale only, without whitening."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    q_parts, tangent_parts, weight_parts = [], [], []
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        u = (left + right) / 2 + (right - left) * nodes / 2
        weight = weights * (right - left) / 2
        phi = full.normalized_bspline_basis(
            u, knots, full.DEGREE, masses
        )
        q_parts.append(
            (phi * masses[None, :] - masses[None, :]) @ helmert
        )
        z = full.hansen_ppf(u, anchor[2:])
        y = anchor[0] + np.exp(0.5 * anchor[1]) * z
        tangent_parts.append(
            full.reference_logpdf_and_score(y, anchor)[1][:, :2]
        )
        weight_parts.append(weight)
    q_score = np.vstack(q_parts)
    tangent = np.vstack(tangent_parts)
    weight = np.concatenate(weight_parts)
    information_q = (q_score.T * weight) @ q_score
    information_tangent = (tangent.T * weight) @ tangent
    cross = (q_score.T * weight) @ tangent
    tangent_sd = np.sqrt(np.diag(information_tangent))
    tangent_scaled = tangent / tangent_sd[None, :]
    information_tangent_scaled = (tangent_scaled.T * weight) @ tangent_scaled
    projection_scaled = (
        (q_score.T * weight) @ tangent_scaled
    ) @ np.linalg.inv(information_tangent_scaled)
    projection = projection_scaled / tangent_sd[None, :]
    efficient = q_score - tangent @ projection.T
    residual_information = (efficient.T * weight) @ efficient
    residual_information = (residual_information + residual_information.T) / 2
    eigenvalues = np.linalg.eigvalsh(residual_information)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(
            f"nonpositive location-scale residual information: {eigenvalues}"
        )
    reversal = np.fliplr(np.eye(masses.size))
    reflection = helmert.T @ reversal @ helmert
    identity = np.eye(masses.size - 1)
    plus = (identity + reflection) / 2
    minus = (identity - reflection) / 2
    post_cross = (efficient.T * weight) @ tangent
    return full.EfficientGeometry(
        projection_q_on_hansen=projection,
        residual_information=residual_information,
        inverse_sqrt=identity,
        projector_plus=plus,
        projector_minus=minus,
        diagnostics={
            "projection": "fixed-anchor location and log-scale-squared tangents only",
            "nuisance_dimension": 2,
            "eta_lambda": "fixed after training static MLE and deliberately retained in the DCD-SM span",
            "residual_score_whitening": False,
            "quadrature_order_per_knot_interval": quadrature_order,
            "tangent_information_eigenvalues_scaled": np.linalg.eigvalsh(
                information_tangent_scaled
            ).tolist(),
            "residual_information_eigenvalues": eigenvalues.tolist(),
            "residual_information_condition_number": float(
                eigenvalues[-1] / eigenvalues[0]
            ),
            "maximum_post_projection_cross_information": float(
                np.max(np.abs(post_cross))
            ),
            "plus_idempotence_error": float(np.max(np.abs(plus @ plus - plus))),
            "minus_idempotence_error": float(np.max(np.abs(minus @ minus - minus))),
            "plus_minus_orthogonality_error": float(np.max(np.abs(plus @ minus))),
            "information_q_condition_number": float(np.linalg.cond(information_q)),
        },
    )


def fit_dynamic_location_scale(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    initial_betas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], full.FourStateFilter]:
    """Fit six diagonal HAR responses while eta and lambda stay fixed."""
    static = full.filter_reference(
        values, offsets, anchor, np.zeros((3, 4))
    )
    averages = full.fixed_denominator_score_averages(
        static.scores[:, :2], full.WINDOWS
    )
    target_steps = np.array([1e-4, 1e-2])
    units = np.array(
        [
            [
                target_steps[coordinate]
                / max(float(np.median(np.abs(average[:, coordinate]))), 1e-12)
                for coordinate in range(2)
            ]
            for average in averages
        ]
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        betas = np.zeros((3, 4))
        betas[:, :2] = np.asarray(theta).reshape(3, 2) * units
        return betas

    def run(theta: np.ndarray) -> full.FourStateFilter:
        return full.filter_reference(values, offsets, anchor, unpack(theta))

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, ValueError, OverflowError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    rng = np.random.default_rng(full.SEED + 1)
    starts = [np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6)]
    if initial_betas is not None:
        starts.insert(0, (np.asarray(initial_betas) / units).ravel())
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500, 500),
            method="L-BFGS-B",
            bounds=[(-500.0, 500.0)] * 6,
            options={
                "maxiter": 4000,
                "ftol": 1e-14,
                "gtol": 1e-8,
                "eps": 1e-5,
                "maxls": 80,
            },
        )
        print(
            f"fixed-shape location-scale start {index + 1}: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all fixed-shape location-scale starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x)
    betas = unpack(theta)
    dynamic = run(theta)
    return betas, {
        "betas": {
            horizon: {
                "location": float(betas[index, 0]),
                "log_scale_squared": float(betas[index, 1]),
            }
            for index, horizon in enumerate(full.HORIZONS)
        },
        "eta_lambda_dynamic_coefficients": "fixed exactly at zero",
        "parameter_units": units.tolist(),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_composite_log_likelihood": dynamic.composite_log_likelihood,
        "gain_vs_static": dynamic.composite_log_likelihood
        - static.composite_log_likelihood,
        "successful_starts": len(successful),
        "total_starts": len(starts),
        "largest_successful_objective_gap": float(
            max(float(fit.fun) for fit in successful) - float(optimum.fun)
        ),
    }, dynamic


def score_names() -> list[str]:
    return [
        "continuous_log_score",
        "crps",
        *(f"categorical_log_score_{count}" for count in full.BIN_COUNTS),
        *(f"q_loss_{alpha:g}" for alpha in full.TAIL_LEVELS),
        *(f"brier_{alpha:g}" for alpha in full.TAIL_LEVELS),
    ]


def write_report(result: dict[str, object]) -> None:
    fit = result["residual_parity_arc_sm"]
    primary = pd.DataFrame(result["primary_oos_comparisons"])
    cross = pd.DataFrame(result["cross_specification_oos_comparisons"])
    lines = [
        "# Fixed Hansen shape + dynamic location-scale + orthogonal parity DCD-SM",
        "",
        f"- Fixed eta: `{result['static_reference']['eta']:.9g}`",
        f"- Fixed lambda: `{result['static_reference']['lambda']:.9g}`",
        "- Dynamic reference states: location and log scale-squared only.",
        "- DCD-SM score: location/scale projection, no Fisher whitening, parity HAR(1,5,22).",
        "",
        "## Location-scale coefficients",
        "",
        "| state | horizon | estimate |",
        "|---|---:|---:|",
    ]
    for horizon in full.HORIZONS:
        for name, estimate in result["dynamic_location_scale"]["betas"][horizon].items():
            lines.append(f"| {name} | {horizon} | {estimate:.9g} |")
    lines.extend(
        [
            "",
            "## DCD-SM coefficients",
            "",
            "| block | horizon | estimate | HAC(22) p |",
            "|---|---:|---:|---:|",
        ]
    )
    for block, horizon, estimate, p_value in zip(
        ["even"] * 3 + ["odd"] * 3,
        [*full.WINDOWS, *full.WINDOWS],
        [*fit["kappa_plus"], *fit["kappa_minus"]],
        fit["p_two_sided_conditional_on_hansen"],
    ):
        lines.append(f"| {block} | {horizon} | {estimate:.9g} | {p_value:.6g} |")
    lines.extend(
        [
            "",
            f"Joint DCD-SM Wald p-value: `{fit['joint_all_p']:.6g}`.",
            f"Joint odd-block Wald p-value: `{fit['joint_odd_p']:.6g}`.",
            "",
            "## Strict-OOS: DCD-SM versus matched fixed-shape location-scale reference",
            "",
            "| score | improvement | HAC(22) p |",
            "|---|---:|---:|",
        ]
    )
    for row in primary.loc[
        primary["score"].isin(("continuous_log_score", "crps"))
    ].to_dict(orient="records"):
        lines.append(
            f"| {row['score']} | {row['improvement_mean']:.9g} | "
            f"{row['hac22_two_sided_p']:.6g} |"
        )
    if not cross.empty:
        lines.extend(
            [
                "",
                "## Strict-OOS cross-specification comparison",
                "",
                "| benchmark | score | improvement | HAC(22) p |",
                "|---|---|---:|---:|",
            ]
        )
        for row in cross.loc[
            cross["score"].isin(("continuous_log_score", "crps"))
        ].to_dict(orient="records"):
            lines.append(
                f"| {row['benchmark']} | {row['score']} | "
                f"{row['improvement_mean']:.9g} | {row['hac22_two_sided_p']:.6g} |"
            )
    log_row = primary.loc[
        primary["score"] == "continuous_log_score"
    ].iloc[0]
    crps_row = primary.loc[primary["score"] == "crps"].iloc[0]
    lines.extend(["", "## Interpretation", ""])
    if fit["joint_all_p"] < 0.05:
        lines.append(
            "Conditional on the fixed-shape dynamic location-scale reference, "
            f"the DCD-SM block is jointly significant (p={fit['joint_all_p']:.6g}). "
            f"The odd-block p-value is {fit['joint_odd_p']:.6g}; the coefficient "
            "table identifies the horizons carrying the rejection."
        )
    else:
        lines.append(
            "Conditional on the fixed-shape dynamic location-scale reference, "
            f"the DCD-SM block is not jointly significant (p={fit['joint_all_p']:.6g}); "
            f"the odd block is also insignificant (p={fit['joint_odd_p']:.6g})."
        )
    lines.extend(
        [
            "",
            "Against the matched fixed-shape location-scale reference, the strict-OOS "
            "mean improvements are "
            f"{log_row['improvement_mean']:.9g} for continuous log score "
            f"(p={log_row['hac22_two_sided_p']:.6g}) and "
            f"{crps_row['improvement_mean']:.9g} for CRPS "
            f"(p={crps_row['hac22_two_sided_p']:.6g}).",
        ]
    )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_result_path = full.OUTPUT / "estimation.json"
    if not full_result_path.exists():
        raise FileNotFoundError(
            "run the unwhitened four-state experiment before this fixed-shape version"
        )
    full_result = json.loads(full_result_path.read_text(encoding="utf-8"))
    static_summary = full_result["static_reference"]
    anchor = np.asarray(static_summary["state"], dtype=float)
    initial_betas = np.array(
        [
            [
                full_result["dynamic_reference"]["betas"][horizon]["location"],
                full_result["dynamic_reference"]["betas"][horizon]["log_scale_squared"],
            ]
            for horizon in full.HORIZONS
        ]
    )
    panel = full.load_raw_panel()
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= full.TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows

    print("fitting fixed-shape dynamic location-scale reference", flush=True)
    betas, reference_summary, reference_train = fit_dynamic_location_scale(
        panel.values[:training_rows], train_offsets, anchor, initial_betas
    )
    reference_oos = full.filter_reference(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        initial_state=reference_train.terminal_state,
        initial_score_history=reference_train.terminal_score_history,
    )
    reference_full = full.filter_reference(panel.values, panel.offsets, anchor, betas)
    reference_continuation_error = float(
        np.max(np.abs(reference_full.states[n_train:] - reference_oos.states))
    )
    eta_lambda_state_error = float(
        np.max(np.abs(reference_full.states[:, 2:] - anchor[None, 2:]))
    )
    if max(reference_continuation_error, eta_lambda_state_error) > 2e-12:
        raise RuntimeError("fixed-shape reference continuation/state check failed")

    u = full.dynamic_reference_pit(panel.values, panel.offsets, reference_full.states)
    knots = full.make_clamped_knots(full.BREAKPOINTS, full.DEGREE)
    masses = full.bspline_masses(knots, full.DEGREE)
    helmert = full.helmert_submatrix(masses.size)
    phi = full.normalized_bspline_basis(u, knots, full.DEGREE, masses)
    geometry = location_scale_geometry(anchor, knots, masses, helmert)
    print(f"location-scale geometry: {geometry.diagnostics}", flush=True)

    print("fitting unwhitened location-scale-orthogonal parity DCD-SM", flush=True)
    fit, q_zero_train, correction_train = full.fit_residual_parity6_dcd(
        phi[:training_rows],
        train_offsets,
        reference_train.scores[:, :2],
        masses,
        helmert,
        geometry,
        nested_even_5_22=np.zeros(2),
        numerical_state_step=1e-2,
        adaptive_hessian=True,
    )
    kappa_plus = np.asarray(fit["kappa_plus"])
    kappa_minus = np.asarray(fit["kappa_minus"])
    correction_oos = full.filter_residual_parity6_dcd(
        phi[training_rows:],
        oos_offsets,
        reference_oos.scores[:, :2],
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
        initial_state=correction_train.terminal_state,
        initial_score_history=correction_train.terminal_score_history,
    )
    correction_full = full.filter_residual_parity6_dcd(
        phi,
        panel.offsets,
        reference_full.scores[:, :2],
        kappa_plus,
        kappa_minus,
        masses,
        helmert,
        geometry,
    )
    correction_continuation_error = float(
        np.max(np.abs(correction_full.states[n_train:] - correction_oos.states))
    )
    q_zero_error = float(np.max(np.abs(q_zero_train.contributions)))
    if max(correction_continuation_error, q_zero_error) > 2e-12:
        raise RuntimeError("fixed-shape DCD-SM continuation/nesting check failed")

    scores, summary, primary_comparisons = full.score_oos(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        panel.values[:training_rows],
        reference_oos,
        correction_oos,
        knots,
        masses,
        reference_model_name=REFERENCE_NAME,
        proposed_model_name=PROPOSED_NAME,
    )
    cross_comparisons: list[dict[str, object]] = []
    full_scores_path = full.OUTPUT / "daily_oos_scores.csv"
    if full_scores_path.exists():
        full_scores = pd.read_csv(full_scores_path)
        combined = pd.concat((scores, full_scores), ignore_index=True)
        for benchmark in (
            "four_state_dynamic_hansen",
            "four_state_dynamic_hansen_orthogonal_parity_arc_sm",
        ):
            for score in score_names():
                cross_comparisons.append(
                    full.score_comparison(combined, PROPOSED_NAME, benchmark, score)
                )

    parameters = pd.DataFrame(
        {
            "parameter": [
                *(f"even_{window}" for window in full.WINDOWS),
                *(f"odd_{window}" for window in full.WINDOWS),
            ],
            "estimate": [*fit["kappa_plus"], *fit["kappa_minus"]],
            "hac22_se_conditional_on_reference": fit[
                "se_godambe_hac22_conditional_on_hansen"
            ],
            "t_conditional_on_reference": fit[
                "t_godambe_hac22_conditional_on_hansen"
            ],
            "p_conditional_on_reference": fit[
                "p_two_sided_conditional_on_hansen"
            ],
        }
    )
    parameters.to_csv(OUTPUT / "arc_sm_parameters.csv", index=False)
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    pd.DataFrame(primary_comparisons).to_csv(
        OUTPUT / "oos_comparisons.csv", index=False
    )
    pd.DataFrame(cross_comparisons).to_csv(
        OUTPUT / "cross_specification_oos_comparisons.csv", index=False
    )
    result = {
        "analysis": "fixed static-MLE eta/lambda, dynamic location-scale, and location-scale-orthogonal unwhitened parity DCD-SM",
        "sample": full_result["sample"],
        "static_reference": static_summary,
        "dynamic_location_scale": reference_summary,
        "residual_specification": {
            "breakpoints": full.BREAKPOINTS.tolist(),
            "windows": list(full.WINDOWS),
            "projection": "location and log-scale-squared only",
            "score_scaling": "none after projection",
            "geometry": geometry.diagnostics,
        },
        "residual_parity_arc_sm": fit,
        "oos_score_summary": summary.to_dict(orient="records"),
        "primary_oos_comparisons": primary_comparisons,
        "cross_specification_oos_comparisons": cross_comparisons,
        "checks": {
            "eta_lambda_state_error": eta_lambda_state_error,
            "q_zero_nesting_error": q_zero_error,
            "reference_continuation_error": reference_continuation_error,
            "arc_sm_continuation_error": correction_continuation_error,
            "oos_minimum_weight": float(correction_oos.weights.min()),
            "oos_maximum_absolute_q": float(np.max(np.abs(correction_oos.states))),
        },
        "inference_scope": "DCD-SM Godambe-HAC(22) conditional on the fixed-shape dynamic location-scale reference",
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
