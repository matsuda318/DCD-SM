"""Exact-HAR(1,5,22) location-scale helpers for locked Case 1.

This module supplies the three helpers used by
``run_case1_final_common_calendar_35_series.py``.  It preserves the standard
fixed-shape M2 objective while retaining exactly 22 daily reference scores.
Unavailable pre-sample scores enter the fixed-denominator HAR means as zero.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import run_sp500 as core
from estimate_score_har import fixed_denominator_score_averages


def filter_reference_har22(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    betas: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> core.FourStateFilter:
    """Four-state reference filter retaining exactly 22 past scores."""
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if anchor.shape != (4,) or betas.shape != (3, 4):
        raise ValueError("anchor/beta dimensions do not match the four-state model")
    days = offsets.size - 1
    states = np.empty((days, 4))
    means = np.empty(days)
    log_variances = np.empty(days)
    variances = np.empty(days)
    eta_path = np.empty(days)
    skew_path = np.empty(days)
    scores = np.empty((days, 4))
    contributions = np.empty(days)
    state = anchor.copy() if initial_state is None else np.asarray(initial_state).copy()
    history = (
        np.empty((0, 4))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-22:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if stop <= start:
            raise ValueError("empty daily cross section")
        mean, log_variance = map(float, state[:2])
        if (
            np.any(~np.isfinite(state))
            or abs(mean) > 0.5
            or log_variance < -20.0
            or log_variance > 0.0
            or np.max(np.abs(state[2:])) > 20.0
        ):
            raise FloatingPointError("unstable four-state reference")
        log_density, observation_score = core.reference_logpdf_and_score(
            values[start:stop], state
        )
        eta, skew, _, _ = core.state_to_parameters(state[2:])
        states[day] = state
        means[day] = mean
        log_variances[day] = log_variance
        variances[day] = np.exp(log_variance)
        eta_path[day] = eta
        skew_path[day] = skew
        scores[day] = np.mean(observation_score, axis=0)
        contributions[day] = float(np.mean(log_density))
        history = np.vstack((history, scores[day][None, :]))[-22:]
        state = anchor.copy()
        for beta, window in zip(betas, core.WINDOWS):
            state += beta * history[-window:].sum(axis=0) / window

    return core.FourStateFilter(
        states=states,
        means=means,
        log_variances=log_variances,
        variances=variances,
        eta=eta_path,
        skew=skew_path,
        scores=scores,
        contributions=contributions,
        terminal_state=state,
        terminal_score_history=history,
    )


def fit_m2_har22(
    values: np.ndarray,
    offsets: np.ndarray,
    anchor: np.ndarray,
    initial_betas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], core.FourStateFilter]:
    """Fit six diagonal M2 HAR coefficients with eta/lambda fixed."""
    static = filter_reference_har22(values, offsets, anchor, np.zeros((3, 4)))
    averages = fixed_denominator_score_averages(static.scores[:, :2], core.WINDOWS)
    target_steps = np.array([1e-4, 1e-2])
    units = np.array([
        [
            target_steps[coordinate]
            / max(float(np.median(np.abs(average[:, coordinate]))), 1e-12)
            for coordinate in range(2)
        ]
        for average in averages
    ])

    def unpack(theta: np.ndarray) -> np.ndarray:
        output = np.zeros((3, 4))
        output[:, :2] = np.asarray(theta, dtype=float).reshape(3, 2) * units
        return output

    def run(theta: np.ndarray) -> core.FourStateFilter:
        return filter_reference_har22(values, offsets, anchor, unpack(theta))

    def objective(theta: np.ndarray) -> float:
        try:
            value = -run(theta).composite_log_likelihood
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    rng = np.random.default_rng(core.SEED + 1)
    starts = [np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6)]
    if initial_betas is not None:
        candidate = np.asarray(initial_betas, dtype=float)
        if candidate.shape == (3, 4):
            candidate = candidate[:, :2]
        if candidate.shape != (3, 2):
            raise ValueError("initial M2 betas must be 3 x 2 or 3 x 4")
        starts.insert(0, (candidate / units).ravel())

    fits = []
    for index, start in enumerate(starts):
        fit = minimize(
            objective,
            np.clip(start, -500.0, 500.0),
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
            f"exact-full22 M2 start {index + 1}/{len(starts)}: "
            f"success={fit.success} objective={float(fit.fun):.12g}",
            flush=True,
        )
        fits.append(fit)
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("all exact-full22 M2 starts failed")
    optimum = min(successful, key=lambda fit: float(fit.fun))
    theta = np.asarray(optimum.x, dtype=float)
    if np.max(np.abs(theta)) > 499.9:
        raise RuntimeError("exact-full22 M2 optimum hit the scaled bound")
    betas = unpack(theta)
    dynamic = run(theta)
    summary = {
        "betas": {
            horizon: {
                "location": float(betas[index, 0]),
                "log_scale_squared": float(betas[index, 1]),
            }
            for index, horizon in enumerate(core.HORIZONS)
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
        "maximum_scaled_parameter_fraction_of_bound": float(
            np.max(np.abs(theta)) / 500.0
        ),
        "score_history": "exactly 22 observations",
    }
    return betas, summary, dynamic


def continue_m2(
    panel: core.RawPanel,
    n_train: int,
    anchor: np.ndarray,
    betas: np.ndarray,
    training: core.FourStateFilter,
) -> tuple[core.FourStateFilter, core.FourStateFilter, float]:
    """Continue a fitted exact-full22 M2 path into the OOS sample."""
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = filter_reference_har22(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full = filter_reference_har22(panel.values, panel.offsets, anchor, betas)
    error = float(max(
        np.max(np.abs(full.states[n_train:] - oos.states)),
        np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
    ))
    if error > 2e-12:
        raise RuntimeError(f"M2 exact-full22 continuation error: {error}")
    return oos, full, error
