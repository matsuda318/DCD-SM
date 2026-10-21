"""Feasible, audited BFGS for the three scalar-filter coefficients.

Only the estimation objective is normalized. Invalid filter evaluations are
rejected by backtracking; they never produce invented gradients or objective
penalties. A small objective change alone is not a convergence criterion.
"""

import numpy as np
from scipy.optimize import minimize


GRADIENT_TOLERANCE = 1e-8
NEIGHBOR_GAIN_TOLERANCE = 1e-10
BOUND = 1000.0


def projected_gradient(theta, gradient):
    result = gradient.copy()
    result[(theta <= -BOUND + 1e-10) & (gradient > 0)] = 0
    result[(theta >= BOUND - 1e-10) & (gradient < 0)] = 0
    return result


def maximize_scalar_filter(run, units, days, *, maxiter=300):
    """Start at zero, preserve legacy bounds, reject infeasible trial points.

    `run(theta, gradient)` returns the same unmodified scalar filter as before.
    Its objective is the sum of daily cross-sectional mean log densities.
    We minimize its negative *time average*, centered at the null for numerical
    accuracy. Raw likelihoods remain available for LR and sandwich inference.
    """
    units = np.asarray(units, dtype=float)
    null = run(np.zeros(3), True)
    baseline = np.asarray(null.contributions)
    counters = {"evaluations": 0, "rejected_trials": 0, "curvature_resets": 0,
                "neighbor_restarts": 0, "primary_iterations": 0}

    def evaluate(theta, gradient=True):
        counters["evaluations"] += 1
        try:
            fitted = run(theta, gradient)
            value = -float(np.mean(fitted.contributions - baseline))
            derivative = -np.asarray(fitted.gradient) * units / days if gradient else None
            if not np.isfinite(value) or (gradient and not np.all(np.isfinite(derivative))):
                raise FloatingPointError("nonfinite objective or gradient")
            return value, derivative, fitted
        except (FloatingPointError, ValueError, OverflowError):
            counters["rejected_trials"] += 1
            return None

    theta = np.zeros(3)
    current = evaluate(theta)
    if current is None:
        raise FloatingPointError("null scalar filter is infeasible")
    value, gradient, fitted = current
    inverse_hessian = np.eye(3)
    primary_message = "not run"
    if maxiter > 0:
        # Keep the established L-BFGS-B search where it works, but use the
        # mean objective. Abort an infeasible search rather than inventing
        # an objective/gradient; the explicit backtracking below takes over.
        best_theta, best_current = theta.copy(), current

        class InfeasibleTrial(Exception):
            pass

        def primary_objective(candidate):
            nonlocal best_theta, best_current
            trial = evaluate(candidate)
            if trial is None:
                raise InfeasibleTrial()
            if trial[0] < best_current[0]:
                best_theta, best_current = candidate.copy(), trial
            return trial[0], trial[1]

        try:
            optimum = minimize(primary_objective, theta, method="L-BFGS-B", jac=True,
                               bounds=[(-BOUND, BOUND)] * 3,
                               options={"maxiter": min(150, maxiter), "ftol": 1e-12,
                                        "gtol": GRADIENT_TOLERANCE, "maxls": 30})
            primary_message = str(optimum.message)
            counters["primary_iterations"] = int(optimum.nit)
            proposed_inverse = np.asarray(optimum.hess_inv.todense())
            if np.all(np.isfinite(proposed_inverse)) and np.linalg.eigvalsh(proposed_inverse).min() > 0:
                inverse_hessian = proposed_inverse
        except InfeasibleTrial:
            primary_message = "infeasible primary trial; switched to explicit backtracking"
        theta = best_theta
        value, gradient, fitted = best_current
    success = False
    message = "maximum iterations reached"
    neighbor_gain = float("nan")
    iteration = 0
    for iteration in range(maxiter + 1):
        projected = projected_gradient(theta, gradient)
        if np.max(np.abs(projected)) <= GRADIENT_TOLERANCE:
            # A stationary point can be a saddle. Check actual nearby values,
            # without using true coefficients or an empirical starting value.
            best_theta, best_value = None, value
            for j in range(3):
                for sign in (-1., 1.):
                    candidate = theta.copy()
                    candidate[j] += sign * 1e-3 * max(1., abs(theta[j]))
                    candidate = np.clip(candidate, -BOUND, BOUND)
                    trial = evaluate(candidate, False)
                    if trial is not None and trial[0] < best_value:
                        best_theta, best_value = candidate, trial[0]
            neighbor_gain = value - best_value
            if neighbor_gain <= NEIGHBOR_GAIN_TOLERANCE:
                success, message = True, "projected gradient and local improvement checks passed"
                break
            theta = best_theta
            value, gradient, fitted = evaluate(theta)
            inverse_hessian = np.eye(3)
            counters["neighbor_restarts"] += 1
            continue
        if iteration == maxiter:
            break

        accepted = None
        for fallback in (False, True):
            direction = -projected if fallback else -inverse_hessian @ projected
            direction[(theta <= -BOUND + 1e-10) & (direction < 0)] = 0
            direction[(theta >= BOUND - 1e-10) & (direction > 0)] = 0
            if gradient @ direction >= 0 or not np.all(np.isfinite(direction)):
                direction = -projected
            # A numerical search safeguard, not a constraint on final kappas.
            direction *= min(1., 25. / max(np.max(np.abs(direction)), 1e-30))
            step = 1.
            for _ in range(60):
                candidate = np.clip(theta + step * direction, -BOUND, BOUND)
                displacement = candidate - theta
                if np.max(np.abs(displacement)) < 1e-14 * max(1., np.max(np.abs(theta))):
                    break
                if gradient @ displacement >= 0:
                    step *= .5
                    continue
                trial = evaluate(candidate)
                if trial is not None and trial[0] <= value + 1e-4 * (gradient @ displacement):
                    accepted = candidate, trial, displacement
                    break
                step *= .5
            if accepted is not None:
                if fallback:
                    inverse_hessian = np.eye(3)
                    counters["curvature_resets"] += 1
                break
        if accepted is None:
            message = "no feasible descent step; convergence NOT certified"
            break
        candidate, trial, displacement = accepted
        next_value, next_gradient, next_fitted = trial
        change = next_gradient - gradient
        curvature = float(change @ displacement)
        if curvature > 1e-12 * np.linalg.norm(change) * np.linalg.norm(displacement) and curvature > 0:
            transform = np.eye(3) - np.outer(displacement, change) / curvature
            inverse_hessian = transform @ inverse_hessian @ transform.T + np.outer(displacement, displacement) / curvature
            inverse_hessian = (inverse_hessian + inverse_hessian.T) / 2
        else:
            inverse_hessian = np.eye(3)
            counters["curvature_resets"] += 1
        theta, value, gradient, fitted = candidate, next_value, next_gradient, next_fitted

    final_gradient = float(np.max(np.abs(projected_gradient(theta, gradient))))
    # Never turn a small step, invalid evaluation, or solver status into success.
    success = success and final_gradient <= GRADIENT_TOLERANCE and value <= 1e-12
    return theta * units, fitted, {
        "success": float(success), "iterations": float(iteration + counters["primary_iterations"]),
        "fallback_iterations": float(iteration), "primary_message": primary_message,
        "method": "mean-objective L-BFGS-B with feasible BFGS fallback and convergence audit",
        "message": message, "projected_gradient_mean": final_gradient,
        "neighbor_gain_mean": neighbor_gain, "mean_objective_gain": -value,
        "at_boundary": float(np.any(np.abs(theta) >= BOUND - 1e-8)),
        "objective_gap_between_starts": 0.0,
        **{k: float(v) for k, v in counters.items()},
    }
