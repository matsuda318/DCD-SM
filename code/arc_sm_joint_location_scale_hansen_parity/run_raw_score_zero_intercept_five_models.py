"""Compatibility helpers for the locked Case-1 exact-HAR(1,5,22) run.

The final common-calendar runner imports this module only for the M3 shape
filter.  It is the projected Hansen-shape filter from
``run_direct_dynamic_skewt_shape_test`` with one deliberate difference: all
22 past daily scores are retained, so the monthly regressor is exactly
``mean(s_t, ..., s_{t-21})``.
"""

from __future__ import annotations

import numpy as np

import run_direct_dynamic_skewt_shape_test as shape
import run_sp500 as core


def filter_shape_har22(
    values: np.ndarray,
    offsets: np.ndarray,
    frozen_location_scale: core.FourStateFilter,
    anchor: np.ndarray,
    betas: np.ndarray,
    geometry: shape.ShapeGeometry,
    *,
    initial_shape_state: np.ndarray | None = None,
    initial_score_history: np.ndarray | None = None,
) -> shape.ShapeFilter:
    """Filter projected eta/lambda states while retaining 22 scores."""
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    anchor = np.asarray(anchor, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if betas.shape != (3, 2):
        raise ValueError("shape betas must have dimension 3 x 2")
    days = offsets.size - 1
    if frozen_location_scale.states.shape[0] != days:
        raise ValueError("frozen location-scale path has the wrong number of days")

    states = np.empty((days, 4))
    scores = np.empty((days, 4))
    efficient_scores = np.empty((days, 2))
    contributions = np.empty(days)
    shape_state = (
        anchor[2:].copy()
        if initial_shape_state is None
        else np.asarray(initial_shape_state, dtype=float).copy()
    )
    history = (
        np.empty((0, 2))
        if initial_score_history is None
        else np.asarray(initial_score_history, dtype=float).copy()[-22:]
    )
    for day, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if stop <= start:
            raise ValueError("empty daily cross section")
        state = frozen_location_scale.states[day].copy()
        state[2:] = shape_state
        if np.any(~np.isfinite(state)) or np.max(np.abs(shape_state)) > 20.0:
            raise FloatingPointError("unstable eta/lambda state")
        log_density, observation_score = core.reference_logpdf_and_score(
            values[start:stop], state
        )
        daily_score = np.mean(observation_score, axis=0)
        efficient = (
            daily_score[2:]
            - geometry.projection_shape_on_ms @ daily_score[:2]
        )
        states[day] = state
        scores[day] = daily_score
        efficient_scores[day] = efficient
        contributions[day] = float(np.mean(log_density))
        history = np.vstack((history, efficient[None, :]))[-22:]
        shape_state = anchor[2:].copy()
        for beta, window in zip(betas, core.WINDOWS):
            shape_state += beta * history[-window:].sum(axis=0) / window

    return shape.ShapeFilter(
        states=states,
        scores=scores,
        efficient_shape_scores=efficient_scores,
        contributions=contributions,
        terminal_shape_state=shape_state,
        terminal_score_history=history,
    )
