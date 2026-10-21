"""Compare normalized DCD-HAR with a parametric Hansen skew-t GAS-HAR.

The comparison holds the observation-driven update structure fixed.  Hansen's
standardized skew-t supplies a pre-specified two-parameter shape family,
whereas DCD moves a feature-free spline density.  Both use one common scalar
coefficient at each 1/5/22-day score horizon.

Continuous log densities are not directly comparable because the DCD uses a
training empirical reference without an estimated reference density.  The
cross-model log-score comparison is therefore the proper categorical log
score on 50, 100, and 200 fixed training-quantile bins. CRPS is evaluated
directly in the standardized-return space.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logsumexp, ndtr
from scipy.stats import chi2, t as student_t


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_hansen_skewt_gas_benchmark"
DCD_OUTPUT = ROOT / "output_decile_knot_comparison"
STATE_FILE = ROOT / "output_shape_persistence" / "daily_state_score_weights.csv"

WINDOWS = (1, 5, 22)
HORIZONS = ("daily", "weekly", "monthly")
BREAKPOINTS = np.linspace(0.0, 1.0, 11)
ETA_MIN = 2.05
ETA_MAX = 100.0
LAMBDA_MAX = 0.995
STATE_BOUND = 12.0
THETA_BOUND = 250.0
DIAGONAL_THETA_BOUND = 500.0
QUANTILE_NODES = 8192
CDF_GRID_SIZE = 8193
LOG_SCORE_BIN_COUNTS = (50, 100, 200)
SEED = 20260901

sys.path.insert(0, str(ROOT))
from estimate import DEGREE, TRAINING_END, load_and_standardize  # noqa: E402
sys.path.insert(0, str(ROOT))
from estimate_score_har import (  # noqa: E402
    fixed_denominator_score_averages,
    hac_meat,
    numerical_hessian,
)
from run_empirical_reference import (  # noqa: E402
    compare_log_scores,
    empirical_absolute_difference,
    empirical_mean_absolute_error,
    empirical_quantile,
    fixed_empirical_pit,
    hac_mean_test,
)

sys.path.insert(0, str(REPO / "dcd_amzn_daily_score_arch"))
from dcd_model import (  # noqa: E402
    bspline_masses,
    helmert_submatrix,
    make_clamped_knots,
    normalized_bspline_basis,
)


@dataclass(frozen=True)
class HansenFilter:
    states: np.ndarray
    eta: np.ndarray
    skew: np.ndarray
    scores: np.ndarray
    contributions: np.ndarray
    terminal_state: np.ndarray
    terminal_score_history: np.ndarray

    @property
    def composite_log_likelihood(self) -> float:
        return float(np.sum(self.contributions))


def state_to_parameters(state: np.ndarray) -> tuple[float, float, float, float]:
    state = np.asarray(state, dtype=float)
    eta_probability = float(expit(state[0]))
    eta = ETA_MIN + (ETA_MAX - ETA_MIN) * eta_probability
    skew_tanh = float(np.tanh(state[1]))
    skew = LAMBDA_MAX * skew_tanh
    d_eta = (ETA_MAX - ETA_MIN) * eta_probability * (1.0 - eta_probability)
    d_skew = LAMBDA_MAX * (1.0 - skew_tanh**2)
    return eta, skew, d_eta, d_skew


def parameters_to_state(eta: float, skew: float) -> np.ndarray:
    eta_probability = (float(eta) - ETA_MIN) / (ETA_MAX - ETA_MIN)
    eta_probability = float(np.clip(eta_probability, 1e-10, 1.0 - 1e-10))
    scaled_skew = float(np.clip(skew / LAMBDA_MAX, -1.0 + 1e-10, 1.0 - 1e-10))
    return np.array(
        [
            np.log(eta_probability / (1.0 - eta_probability)),
            np.arctanh(scaled_skew),
        ]
    )


def hansen_constants(eta: float, skew: float) -> tuple[float, float, float]:
    log_c = (
        gammaln((eta + 1.0) / 2.0)
        - gammaln(eta / 2.0)
        - 0.5 * np.log(np.pi * (eta - 2.0))
    )
    c = float(np.exp(log_c))
    a = 4.0 * skew * c * (eta - 2.0) / (eta - 1.0)
    b_squared = 1.0 + 3.0 * skew**2 - a**2
    if not np.isfinite(b_squared) or b_squared <= 0.0:
        raise FloatingPointError("invalid Hansen skew-t normalization")
    return a, float(np.sqrt(b_squared)), log_c


def hansen_logpdf(values: np.ndarray, state: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    eta, skew, _, _ = state_to_parameters(state)
    a, b, log_c = hansen_constants(eta, skew)
    side = np.where(values < -a / b, -1.0, 1.0)
    denominator = 1.0 + side * skew
    transformed = (b * values + a) / denominator
    return (
        np.log(b)
        + log_c
        - 0.5 * (eta + 1.0)
        * np.log1p(transformed**2 / (eta - 2.0))
    )


def hansen_logpdf_and_score(
    values: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Log density and analytic gradient with respect to unconstrained state."""
    values = np.asarray(values, dtype=float)
    eta, skew, d_eta_d_state, d_skew_d_state = state_to_parameters(state)
    a, b, log_c = hansen_constants(eta, skew)
    c = float(np.exp(log_c))
    ratio = (eta - 2.0) / (eta - 1.0)
    d_log_c_eta = (
        0.5 * digamma((eta + 1.0) / 2.0)
        - 0.5 * digamma(eta / 2.0)
        - 0.5 / (eta - 2.0)
    )
    d_log_ratio_eta = 1.0 / (eta - 2.0) - 1.0 / (eta - 1.0)
    d_a_eta = a * (d_log_c_eta + d_log_ratio_eta)
    d_a_skew = 4.0 * c * ratio
    d_b_eta = -a * d_a_eta / b
    d_b_skew = (3.0 * skew - a * d_a_skew) / b

    side = np.where(values < -a / b, -1.0, 1.0)
    denominator = 1.0 + side * skew
    transformed = (b * values + a) / denominator
    d_transformed_eta = (d_b_eta * values + d_a_eta) / denominator
    d_transformed_skew = (
        (d_b_skew * values + d_a_skew) / denominator
        - transformed * side / denominator
    )
    eta_minus_two = eta - 2.0
    quadratic = 1.0 + transformed**2 / eta_minus_two
    d_quadratic_eta = (
        2.0 * transformed * d_transformed_eta / eta_minus_two
        - transformed**2 / eta_minus_two**2
    )
    d_quadratic_skew = (
        2.0 * transformed * d_transformed_skew / eta_minus_two
    )
    log_density = (
        np.log(b)
        + log_c
        - 0.5 * (eta + 1.0) * np.log(quadratic)
    )
    d_log_eta = (
        d_b_eta / b
        + d_log_c_eta
        - 0.5 * np.log(quadratic)
        - 0.5 * (eta + 1.0) * d_quadratic_eta / quadratic
    )
    d_log_skew = (
        d_b_skew / b
        - 0.5 * (eta + 1.0) * d_quadratic_skew / quadratic
    )
    score = np.column_stack(
        (d_log_eta * d_eta_d_state, d_log_skew * d_skew_d_state)
    )
    return log_density, score


def hansen_cdf(values: np.ndarray, state: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    eta, skew, _, _ = state_to_parameters(state)
    a, b, _ = hansen_constants(eta, skew)
    left = values < -a / b
    side = np.where(left, -1.0, 1.0)
    transformed = (b * values + a) / (1.0 + side * skew)
    standardized_t = transformed * np.sqrt(eta / (eta - 2.0))
    t_cdf = student_t.cdf(standardized_t, df=eta)
    output = np.where(
        left,
        (1.0 - skew) * t_cdf,
        (1.0 + skew) * t_cdf - skew,
    )
    return np.clip(output, 0.0, 1.0)


def hansen_ppf(probabilities: np.ndarray, state: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    eta, skew, _, _ = state_to_parameters(state)
    a, b, _ = hansen_constants(eta, skew)
    left = probabilities < (1.0 - skew) / 2.0
    t_probabilities = np.where(
        left,
        probabilities / (1.0 - skew),
        (probabilities + skew) / (1.0 + skew),
    )
    t_quantiles = student_t.ppf(t_probabilities, df=eta)
    transformed = t_quantiles * np.sqrt((eta - 2.0) / eta)
    side_scale = np.where(left, 1.0 - skew, 1.0 + skew)
    return (side_scale * transformed - a) / b


def validate_hansen_implementation() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=257)
    state = parameters_to_state(5.7, -0.18)
    _, analytic = hansen_logpdf_and_score(values, state)
    step = 1e-6
    numerical = np.empty_like(analytic)
    for coordinate in range(2):
        direction = np.zeros(2)
        direction[coordinate] = step
        numerical[:, coordinate] = (
            hansen_logpdf(values, state + direction)
            - hansen_logpdf(values, state - direction)
        ) / (2.0 * step)
    gradient_error = float(np.max(np.abs(analytic - numerical)))
    probabilities = np.linspace(1e-5, 1.0 - 1e-5, 1001)
    inversion_error = float(
        np.max(
            np.abs(
                hansen_cdf(hansen_ppf(probabilities, state), state)
                - probabilities
            )
        )
    )
    if gradient_error > 2e-5 or inversion_error > 2e-8:
        raise ValueError(
            f"Hansen implementation validation failed: {gradient_error}, "
            f"{inversion_error}"
        )
    return {
        "maximum_logpdf_score_gradient_error": gradient_error,
        "maximum_cdf_ppf_inversion_error": inversion_error,
    }


def filter_hansen_score_har(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    betas: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> HansenFilter:
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    betas = np.asarray(betas, dtype=float)
    n_days = offsets.size - 1
    states = np.empty((n_days, 2), dtype=float)
    parameters = np.empty((n_days, 2), dtype=float)
    scores = np.empty((n_days, 2), dtype=float)
    contributions = np.empty(n_days, dtype=float)
    state = anchor.copy() if initial_state is None else np.asarray(initial_state).copy()
    history = []
    if initial_score_history is not None:
        history = [row.copy() for row in np.asarray(initial_score_history, dtype=float)]

    for day in range(n_days):
        states[day] = state
        parameters[day, :2] = state_to_parameters(state)[:2]
        start, stop = offsets[day : day + 2]
        log_density, observation_score = hansen_logpdf_and_score(
            values[start:stop], state
        )
        contributions[day] = float(np.mean(log_density))
        scores[day] = np.mean(observation_score, axis=0)
        history.append(scores[day].copy())
        history = history[-max(WINDOWS) :]
        state = anchor.copy()
        history_array = np.asarray(history)
        for beta, window in zip(betas, WINDOWS):
            available = history_array[-min(window, len(history_array)) :]
            state += beta * np.sum(available, axis=0) / window
        if np.any(~np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("unstable Hansen GAS-HAR state")
    return HansenFilter(
        states=states,
        eta=parameters[:, 0],
        skew=parameters[:, 1],
        scores=scores,
        contributions=contributions,
        terminal_state=state,
        terminal_score_history=np.asarray(history),
    )


def fit_static_hansen(
    values: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    counts = np.diff(offsets)
    observation_weights = np.repeat(1.0 / counts, counts)

    def objective(state: np.ndarray) -> float:
        try:
            value = -float(observation_weights @ hansen_logpdf(values, state))
        except (FloatingPointError, ValueError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    starts = [
        parameters_to_state(eta, skew)
        for eta in (3.5, 5.0, 8.0, 15.0, 30.0)
        for skew in (-0.15, 0.0, 0.15)
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-STATE_BOUND, STATE_BOUND)] * 2,
            options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10},
        )
        for start in starts
    ]
    # L-BFGS-B can report an ABNORMAL line-search termination for one start
    # even when other starts converge to the same regular optimum.  Selecting
    # the smallest objective before checking the convergence flag therefore
    # makes an otherwise estimable multi-start fit fail spuriously.  Restrict
    # the final selection to successful, finite fits.
    successful = [
        fit for fit in fits if fit.success and np.isfinite(float(fit.fun))
    ]
    if not successful:
        messages = "; ".join(sorted({str(fit.message) for fit in fits}))
        raise RuntimeError(f"all static Hansen starts failed: {messages}")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    anchor = np.asarray(optimum.x, dtype=float)
    eta, skew, _, _ = state_to_parameters(anchor)
    summary = {
        "eta": eta,
        "lambda": skew,
        "state_anchor": anchor.tolist(),
        "composite_log_likelihood": float(-optimum.fun),
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "starts": len(starts),
        "successful_starts": len(successful),
    }
    return anchor, summary


def fit_dynamic_hansen(
    values: np.ndarray, offsets: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, dict[str, object], HansenFilter, HansenFilter]:
    n_days = offsets.size - 1
    static = filter_hansen_score_har(values, offsets, anchor, np.zeros(3))
    averages = fixed_denominator_score_averages(static.scores, WINDOWS)
    beta_units = np.array(
        [
            1e-3 / max(float(np.median(np.linalg.norm(item, axis=1))), 1e-12)
            for item in averages
        ]
    )

    def run(theta: np.ndarray) -> HansenFilter:
        return filter_hansen_score_har(
            values, offsets, anchor, np.asarray(theta, dtype=float) * beta_units
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, ValueError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    rng = np.random.default_rng(SEED)
    starts = [
        np.zeros(3),
        np.ones(3),
        -np.ones(3),
        rng.normal(size=3),
        3.0 * rng.normal(size=3),
        np.array([0.0, 1.0, 3.0]),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-THETA_BOUND, THETA_BOUND)] * 3,
            options={
                "maxiter": 2000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        print(
            f"Hansen GAS-HAR start {index + 1}: success={fit.success} "
            f"objective={float(fit.fun):.10g} theta={np.asarray(fit.x)}",
            flush=True,
        )
        fits.append(fit)
    optimum = min(fits, key=lambda fit: float(fit.fun))
    if not optimum.success:
        raise RuntimeError(f"Hansen GAS-HAR fit failed: {optimum.message}")
    theta = np.asarray(optimum.x, dtype=float)
    betas = theta * beta_units
    dynamic = run(theta)

    derivative_step = 1e-2
    hessian = numerical_hessian(objective, theta, derivative_step)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    inference: dict[str, object]
    if hessian_eigenvalues[0] > 1e-10:
        gradients = np.empty((n_days, 3), dtype=float)
        for coordinate in range(3):
            direction = np.zeros(3)
            direction[coordinate] = derivative_step
            gradients[:, coordinate] = (
                run(theta + direction).contributions
                - run(theta - direction).contributions
            ) / (2.0 * derivative_step)
        inverse_hessian = np.linalg.inv(hessian)
        covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
        covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
        scaling = np.diag(beta_units)
        covariance_beta = scaling @ covariance_theta @ scaling
        covariance_beta = (covariance_beta + covariance_beta.T) / 2.0
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_beta), 0.0))
        t_values = betas / standard_errors
        p_values = 2.0 * ndtr(-np.abs(t_values))
        joint_wald = float(betas @ np.linalg.inv(covariance_beta) @ betas)
        inference = {
            "available": True,
            "se_beta_godambe_hac22_conditional": standard_errors.tolist(),
            "t_beta_godambe_hac22_conditional": t_values.tolist(),
            "p_beta_two_sided_normal_conditional": p_values.tolist(),
            "joint_wald_godambe_hac22_conditional": joint_wald,
            "joint_wald_p": float(chi2.sf(joint_wald, df=3)),
        }
    else:
        inference = {
            "available": False,
            "reason": "nonpositive numerical Hessian",
        }
    summary = {
        "betas": dict(zip(HORIZONS, betas.tolist())),
        "beta_units": beta_units.tolist(),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "dynamic_composite_log_likelihood": dynamic.composite_log_likelihood,
        "dynamic_gain_vs_static": (
            dynamic.composite_log_likelihood - static.composite_log_likelihood
        ),
        "composite_lr_statistic_descriptive": float(
            2.0
            * (dynamic.composite_log_likelihood - static.composite_log_likelihood)
        ),
        "composite_lr_chi2_3_p_descriptive": float(
            chi2.sf(
                2.0
                * (dynamic.composite_log_likelihood - static.composite_log_likelihood),
                df=3,
            )
        ),
        "hessian_eigenvalues_theta": hessian_eigenvalues.tolist(),
        "inference": inference,
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "best_of_starts": len(starts),
    }
    return betas, summary, static, dynamic


def fit_diagonal_dynamic_hansen(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    shared_betas: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], HansenFilter, HansenFilter]:
    """Fit horizon-specific diagonal responses for eta and lambda scores."""
    n_days = offsets.size - 1
    static = filter_hansen_score_har(
        values, offsets, anchor, np.zeros((len(WINDOWS), 2))
    )
    averages = fixed_denominator_score_averages(static.scores, WINDOWS)
    beta_units = np.array(
        [
            [
                1e-2
                / max(
                    float(np.median(np.abs(average[:, coordinate]))),
                    1e-12,
                )
                for coordinate in range(2)
            ]
            for average in averages
        ]
    )

    def theta_to_betas(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(len(WINDOWS), 2) * beta_units

    def run(theta: np.ndarray) -> HansenFilter:
        return filter_hansen_score_har(
            values, offsets, anchor, theta_to_betas(theta)
        )

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, ValueError):
            return 1e100
        return value if np.isfinite(value) else 1e100

    shared_betas = np.asarray(shared_betas, dtype=float)
    shared_start = (
        np.repeat(shared_betas[:, None], 2, axis=1) / beta_units
    ).ravel()
    print(f"Diagonal Hansen beta units: {beta_units}", flush=True)
    rng = np.random.default_rng(SEED + 1)
    starts = [
        shared_start,
        np.zeros(6),
        np.ones(6),
        -np.ones(6),
        rng.normal(size=6),
        shared_start + rng.normal(size=6),
    ]
    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -DIAGONAL_THETA_BOUND, DIAGONAL_THETA_BOUND),
            method="L-BFGS-B",
            bounds=[(-DIAGONAL_THETA_BOUND, DIAGONAL_THETA_BOUND)] * 6,
            options={
                "maxiter": 3000,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "eps": 1e-5,
                "maxls": 50,
            },
        )
        print(
            f"Diagonal Hansen GAS-HAR start {index + 1}: "
            f"success={fit.success} objective={float(fit.fun):.10g} "
            f"theta={np.asarray(fit.x)}",
            flush=True,
        )
        fits.append(fit)
    optimum = min(fits, key=lambda fit: float(fit.fun))
    if not optimum.success:
        raise RuntimeError(
            f"diagonal Hansen GAS-HAR fit failed: {optimum.message}"
        )
    theta = np.asarray(optimum.x, dtype=float)
    betas = theta_to_betas(theta)
    dynamic = run(theta)
    nested_tolerance = 1e-7
    shared_nested = filter_hansen_score_har(
        values,
        offsets,
        anchor,
        np.repeat(shared_betas[:, None], 2, axis=1),
    )
    nested_gain = (
        dynamic.composite_log_likelihood
        - shared_nested.composite_log_likelihood
    )
    if nested_gain < -nested_tolerance:
        raise ValueError(
            "diagonal Hansen likelihood is below its shared-coefficient "
            f"nested start by {nested_gain}"
        )

    derivative_step = 1e-2
    hessian = numerical_hessian(objective, theta, derivative_step)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    inference: dict[str, object]
    if hessian_eigenvalues[0] > 1e-10:
        gradients = np.empty((n_days, 6), dtype=float)
        for coordinate in range(6):
            direction = np.zeros(6)
            direction[coordinate] = derivative_step
            gradients[:, coordinate] = (
                run(theta + direction).contributions
                - run(theta - direction).contributions
            ) / (2.0 * derivative_step)
        inverse_hessian = np.linalg.inv(hessian)
        covariance_theta = inverse_hessian @ hac_meat(gradients, 22) @ inverse_hessian
        covariance_theta = (covariance_theta + covariance_theta.T) / 2.0
        scaling = np.diag(beta_units.ravel())
        covariance_beta = scaling @ covariance_theta @ scaling
        covariance_beta = (covariance_beta + covariance_beta.T) / 2.0
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_beta), 0.0))
        t_values = betas.ravel() / standard_errors
        p_values = 2.0 * ndtr(-np.abs(t_values))
        joint_wald = float(
            betas.ravel() @ np.linalg.inv(covariance_beta) @ betas.ravel()
        )
        inference = {
            "available": True,
            "se_beta_godambe_hac22_conditional": standard_errors.reshape(3, 2).tolist(),
            "t_beta_godambe_hac22_conditional": t_values.reshape(3, 2).tolist(),
            "p_beta_two_sided_normal_conditional": p_values.reshape(3, 2).tolist(),
            "joint_wald_godambe_hac22_conditional": joint_wald,
            "joint_wald_p": float(chi2.sf(joint_wald, df=6)),
        }
    else:
        inference = {
            "available": False,
            "reason": "nonpositive numerical Hessian",
        }
    summary = {
        "betas": {
            horizon: {"eta": float(betas[index, 0]), "lambda": float(betas[index, 1])}
            for index, horizon in enumerate(HORIZONS)
        },
        "beta_units": beta_units.tolist(),
        "static_composite_log_likelihood": static.composite_log_likelihood,
        "shared_coefficient_composite_log_likelihood": (
            shared_nested.composite_log_likelihood
        ),
        "diagonal_composite_log_likelihood": dynamic.composite_log_likelihood,
        "diagonal_gain_vs_static": (
            dynamic.composite_log_likelihood - static.composite_log_likelihood
        ),
        "diagonal_gain_vs_shared": nested_gain,
        "nested_likelihood_check_passed": True,
        "hessian_eigenvalues_theta": hessian_eigenvalues.tolist(),
        "inference": inference,
        "optimizer_success": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "best_of_starts": len(starts),
        "maximum_absolute_scaled_parameter_fraction_of_bound": float(
            np.max(np.abs(theta)) / DIAGONAL_THETA_BOUND
        ),
    }
    return betas, summary, static, dynamic


def dcd_component_cdfs(
    knots: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, CDF_GRID_SIZE)
    basis = normalized_bspline_basis(grid, knots, DEGREE, masses)
    cdfs = cumulative_trapezoid(basis, grid, axis=0, initial=0.0)
    cdfs /= cdfs[-1][None, :]
    cdfs[0] = 0.0
    cdfs[-1] = 1.0
    return grid, cdfs


def dcd_states_to_weights(
    metric_states: np.ndarray,
    masses: np.ndarray,
    embedding: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    metric_free = np.asarray(metric_states, dtype=float) @ np.linalg.inv(
        np.linalg.cholesky(metric)
    )
    logits = np.log(masses)[None, :] + metric_free @ embedding.T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def score_ordered_predictive_sample(
    ordered_sample: np.ndarray, observations: np.ndarray
) -> float:
    return float(
        np.mean(empirical_mean_absolute_error(ordered_sample, observations))
        - 0.5 * empirical_absolute_difference(ordered_sample)
    )


def categorical_log_score(
    bin_probabilities: np.ndarray,
    observations: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    probabilities = np.maximum(np.asarray(bin_probabilities, dtype=float), 1e-300)
    probabilities /= np.sum(probabilities)
    observation_bins = np.searchsorted(
        bin_edges[1:-1], observations, side="right"
    )
    return float(np.mean(np.log(probabilities[observation_bins])))


def main() -> None:
    required = (
        STATE_FILE,
        DCD_OUTPUT / "q0.npy",
        DCD_OUTPUT / "metric.npy",
        DCD_OUTPUT / "estimation.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked DCD artifacts: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = validate_hansen_implementation()
    print(f"Hansen density validation: {validation}", flush=True)

    returns, daily, offsets = load_and_standardize()
    dates = pd.DatetimeIndex(daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(offsets[n_train])
    standardized = returns["standardized_return"].to_numpy(dtype=float)
    train_values = standardized[:training_rows]
    train_offsets = offsets[: n_train + 1]

    print("Fitting static Hansen skew-t", flush=True)
    anchor, static_summary = fit_static_hansen(train_values, train_offsets)
    print(f"Static Hansen: {static_summary}", flush=True)
    print("Fitting Hansen skew-t GAS-HAR(1,5,22)", flush=True)
    betas, dynamic_summary, hansen_static_train, hansen_dynamic_train = (
        fit_dynamic_hansen(train_values, train_offsets, anchor)
    )
    print("Fitting diagonal six-coefficient Hansen GAS-HAR", flush=True)
    diagonal_betas, diagonal_summary, _, hansen_diagonal_train = (
        fit_diagonal_dynamic_hansen(
            train_values, train_offsets, anchor, betas
        )
    )

    oos_values = standardized[training_rows:]
    oos_offsets = offsets[n_train:] - offsets[n_train]
    hansen_static_oos = filter_hansen_score_har(
        oos_values, oos_offsets, anchor, np.zeros(3)
    )
    hansen_dynamic_oos = filter_hansen_score_har(
        oos_values,
        oos_offsets,
        anchor,
        betas,
        initial_state=hansen_dynamic_train.terminal_state,
        initial_score_history=hansen_dynamic_train.terminal_score_history,
    )
    hansen_diagonal_oos = filter_hansen_score_har(
        oos_values,
        oos_offsets,
        anchor,
        diagonal_betas,
        initial_state=hansen_diagonal_train.terminal_state,
        initial_score_history=hansen_diagonal_train.terminal_score_history,
    )
    continuous_hansen_oos = compare_log_scores(
        "Hansen GAS-HAR versus static Hansen",
        hansen_dynamic_oos.contributions,
        hansen_static_oos.contributions,
    )
    diagonal_continuous_hansen_oos = [
        compare_log_scores(
            "diagonal Hansen GAS-HAR versus static Hansen",
            hansen_diagonal_oos.contributions,
            hansen_static_oos.contributions,
        ),
        compare_log_scores(
            "diagonal Hansen GAS-HAR versus shared-coefficient Hansen GAS-HAR",
            hansen_diagonal_oos.contributions,
            hansen_dynamic_oos.contributions,
        ),
    ]

    state = pd.read_csv(STATE_FILE, parse_dates=["date"])
    if not np.array_equal(pd.DatetimeIndex(state["date"]).to_numpy(), dates.to_numpy()):
        raise ValueError("DCD state dates and return panel do not align")
    q_columns = sorted(
        (
            column
            for column in state.columns
            if column.startswith("q_") and column.removeprefix("q_").isdigit()
        ),
        key=lambda column: int(column.split("_")[1]),
    )
    q_metric = state[q_columns].to_numpy(dtype=float)
    q0_metric = np.load(DCD_OUTPUT / "q0.npy")
    metric = np.load(DCD_OUTPUT / "metric.npy")
    knots = make_clamped_knots(BREAKPOINTS, DEGREE)
    masses = bspline_masses(knots, DEGREE)
    embedding = helmert_submatrix(masses.size)
    dcd_dynamic_weights = dcd_states_to_weights(q_metric, masses, embedding, metric)
    dcd_static_weights = dcd_states_to_weights(
        q0_metric[None, :], masses, embedding, metric
    )[0]
    u_grid, component_cdfs = dcd_component_cdfs(knots, masses)
    ordered_reference = np.sort(train_values)
    probability_nodes = (np.arange(QUANTILE_NODES) + 0.5) / QUANTILE_NODES
    bin_edges = {
        count: np.concatenate(
            (
                [-np.inf],
                np.quantile(
                    ordered_reference,
                    np.arange(1, count) / count,
                ),
                [np.inf],
            )
        )
        for count in LOG_SCORE_BIN_COUNTS
    }
    bin_u = {
        count: np.concatenate(
            (
                [0.0],
                fixed_empirical_pit(
                    ordered_reference, bin_edges[count][1:-1]
                ),
                [1.0],
            )
        )
        for count in LOG_SCORE_BIN_COUNTS
    }

    rows: list[dict[str, object]] = []
    oos_targets = np.arange(n_train, len(dates))
    for local_day, target in enumerate(oos_targets):
        if local_day % 50 == 0:
            print(
                f"Scoring normalized OOS day {local_day + 1}/{len(oos_targets)}",
                flush=True,
            )
        start, stop = offsets[target : target + 2]
        observations = standardized[start:stop]
        hansen_states = {
            "hansen_static": anchor,
            "hansen_score_har": hansen_dynamic_oos.states[local_day],
            "hansen_diagonal_score_har": hansen_diagonal_oos.states[local_day],
        }
        for model, model_state in hansen_states.items():
            ordered_sample = hansen_ppf(probability_nodes, model_state)
            log_scores = {
                f"categorical_log_score_{count}": categorical_log_score(
                    np.diff(hansen_cdf(bin_edges[count], model_state)),
                    observations,
                    bin_edges[count],
                )
                for count in LOG_SCORE_BIN_COUNTS
            }
            rows.append(
                {
                    "date": str(dates[target].date()),
                    "model": model,
                    "n_constituents": int(stop - start),
                    "crps": score_ordered_predictive_sample(
                        ordered_sample, observations
                    ),
                    **log_scores,
                }
            )
        dcd_weights = {
            "dcd_static": dcd_static_weights,
            "dcd_score_har": dcd_dynamic_weights[target],
        }
        for model, model_weights in dcd_weights.items():
            mixture_cdf = np.maximum.accumulate(component_cdfs @ model_weights)
            mixture_cdf[0] = 0.0
            mixture_cdf[-1] = 1.0
            u_nodes = np.interp(probability_nodes, mixture_cdf, u_grid)
            ordered_sample = empirical_quantile(ordered_reference, u_nodes)
            log_scores = {}
            for count in LOG_SCORE_BIN_COUNTS:
                bin_cdf = np.interp(bin_u[count], u_grid, mixture_cdf)
                bin_probabilities = np.maximum(np.diff(bin_cdf), 0.0)
                log_scores[f"categorical_log_score_{count}"] = (
                    categorical_log_score(
                        bin_probabilities, observations, bin_edges[count]
                    )
                )
            rows.append(
                {
                    "date": str(dates[target].date()),
                    "model": model,
                    "n_constituents": int(stop - start),
                    "crps": score_ordered_predictive_sample(
                        ordered_sample, observations
                    ),
                    **log_scores,
                }
            )

    scores = pd.DataFrame(rows)
    log_score_columns = [
        f"categorical_log_score_{count}" for count in LOG_SCORE_BIN_COUNTS
    ]
    score_summary = scores.groupby("model", as_index=False)[
        ["crps", *log_score_columns]
    ].mean()
    comparisons = []
    pairs = (
        ("hansen_score_har", "hansen_static"),
        ("hansen_diagonal_score_har", "hansen_static"),
        ("hansen_diagonal_score_har", "hansen_score_har"),
        ("dcd_score_har", "dcd_static"),
        ("dcd_score_har", "hansen_diagonal_score_har"),
        ("dcd_score_har", "hansen_score_har"),
        ("dcd_score_har", "hansen_static"),
        ("dcd_static", "hansen_static"),
    )
    for proposed, benchmark in pairs:
        proposed_scores = scores.loc[scores["model"] == proposed].set_index("date")
        benchmark_scores = scores.loc[scores["model"] == benchmark].set_index("date")
        crps_reduction = (
            benchmark_scores["crps"] - proposed_scores["crps"]
        ).to_numpy()
        mean, standard_error, p_value = hac_mean_test(crps_reduction, 22)
        comparisons.append(
            {
                "proposed": proposed,
                "benchmark": benchmark,
                "score": "crps",
                "improvement_mean": mean,
                "hac22_standard_error": standard_error,
                "hac22_two_sided_p": p_value,
                "positive_day_fraction": float(np.mean(crps_reduction > 0.0)),
            }
        )
        for score_name in log_score_columns:
            log_gain = (
                proposed_scores[score_name]
                - benchmark_scores[score_name]
            ).to_numpy()
            mean, standard_error, p_value = hac_mean_test(log_gain, 22)
            comparisons.append(
                {
                    "proposed": proposed,
                    "benchmark": benchmark,
                    "score": score_name,
                    "improvement_mean": mean,
                    "hac22_standard_error": standard_error,
                    "hac22_two_sided_p": p_value,
                    "positive_day_fraction": float(np.mean(log_gain > 0.0)),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)

    parameter_frame = pd.DataFrame(
        {
            "date": dates,
            "sample": np.where(np.arange(len(dates)) < n_train, "training", "oos"),
            "eta": np.concatenate(
                (hansen_dynamic_train.eta, hansen_dynamic_oos.eta)
            ),
            "lambda": np.concatenate(
                (hansen_dynamic_train.skew, hansen_dynamic_oos.skew)
            ),
            "diagonal_eta": np.concatenate(
                (hansen_diagonal_train.eta, hansen_diagonal_oos.eta)
            ),
            "diagonal_lambda": np.concatenate(
                (hansen_diagonal_train.skew, hansen_diagonal_oos.skew)
            ),
        }
    )
    scores.to_csv(OUTPUT / "daily_oos_scores.csv", index=False)
    score_summary.to_csv(OUTPUT / "oos_score_summary.csv", index=False)
    comparison_frame.to_csv(OUTPUT / "oos_score_comparisons.csv", index=False)
    parameter_frame.to_csv(OUTPUT / "hansen_dynamic_parameters.csv", index=False)
    pd.DataFrame(
        {
            "date": dates[n_train:],
            "hansen_static_continuous_log_score": hansen_static_oos.contributions,
            "hansen_score_har_continuous_log_score": hansen_dynamic_oos.contributions,
            "hansen_diagonal_score_har_continuous_log_score": (
                hansen_diagonal_oos.contributions
            ),
        }
    ).to_csv(OUTPUT / "hansen_oos_continuous_log_scores.csv", index=False)

    output = {
        "analysis": "Hansen skew-t GAS-HAR benchmark for normalized S&P 500 DCD",
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_days": n_train,
            "oos_days": int(len(dates) - n_train),
            "training_observations": training_rows,
        },
        "normalization": (
            "realized daily cross-sectional mean zero and population variance one"
        ),
        "hansen_density": {
            "parameterization": (
                "Hansen (1994) standardized skew-t with eta in (2.05,100) "
                "and lambda in (-0.995,0.995)"
            ),
            "static_training_fit": static_summary,
            "implementation_validation": validation,
        },
        "hansen_score_har": {
            "recursion": (
                "x[t+1]=x0+kappa1*s[t]+kappa5*avg5(s)+kappa22*avg22(s); "
                "the three kappas are scalars shared by eta and lambda states"
            ),
            "anchor_fixed_at_static_training_mle": True,
            "training": dynamic_summary,
            "oos_continuous_log_score_vs_static": continuous_hansen_oos,
            "parameter_ranges": {
                sample: {
                    parameter: {
                        "mean": float(group[parameter].mean()),
                        "minimum": float(group[parameter].min()),
                        "maximum": float(group[parameter].max()),
                    }
                    for parameter in ("eta", "lambda")
                }
                for sample, group in parameter_frame.groupby("sample")
            },
        },
        "hansen_diagonal_score_har": {
            "recursion": (
                "x[t+1]=x0+B1*s[t]+B5*avg5(s)+B22*avg22(s); each Bh is "
                "diagonal with separate eta and lambda coefficients"
            ),
            "anchor_fixed_at_static_training_mle": True,
            "training": diagonal_summary,
            "oos_continuous_log_score_comparisons": (
                diagonal_continuous_hansen_oos
            ),
            "parameter_ranges": {
                sample: {
                    parameter.removeprefix("diagonal_"): {
                        "mean": float(group[parameter].mean()),
                        "minimum": float(group[parameter].min()),
                        "maximum": float(group[parameter].max()),
                    }
                    for parameter in ("diagonal_eta", "diagonal_lambda")
                }
                for sample, group in parameter_frame.groupby("sample")
            },
        },
        "cross_model_scoring": {
            "continuous_log_score_comparable": False,
            "reason": (
                "DCD uses a training empirical reference CDF without a fitted "
                "continuous reference density"
            ),
            "log_score": (
                "proper categorical log score on 50, 100, and 200 fixed "
                "training-quantile bins"
            ),
            "crps": "standardized-return CRPS from 8192 deterministic quantile nodes",
            "summary": score_summary.to_dict(orient="records"),
            "comparisons": comparison_frame.to_dict(orient="records"),
        },
        "interpretation": (
            "This is not Hansen's original ARCD transition. It holds the GAS-HAR "
            "update fixed to isolate pre-specified parametric shape versus "
            "feature-free full-density dynamics."
        ),
    }
    (OUTPUT / "estimation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
