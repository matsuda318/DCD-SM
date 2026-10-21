"""Compiled versions of the existing Case-1 recursions, with parameter scores.

The model is unchanged: day-average driver scores, zero presample history,
exactly 22 retained scores, fixed-anchor Fisher projection and no whitening.
Likelihood aggregation is deliberately left to the caller (observation sum).
Reference driver Jacobians use centered differences in the two active states;
endpoint driver Jacobians are analytic. Both include recursion derivatives.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _digamma(x):
    value = 0.0
    while x < 12.0:
        value -= 1.0 / x
        x += 1.0
    z = 1.0 / (x*x)
    return value + math.log(x) - 0.5/x - z*(1/12 - z*(1/120 - z*(1/252 - z*(1/240 - z*(1/132 - z*691/32760)))))


@njit(cache=True)
def reference_day(values, start, stop, state):
    """Hansen raw-return mean log density and all four state scores."""
    mean, lv, aeta, alam = state
    sigma = math.exp(0.5*lv)
    probability = 1.0/(1.0+math.exp(-aeta))
    eta = 2.05 + 97.95*probability
    tanh = math.tanh(alam)
    skew = 0.995*tanh
    deta = 97.95*probability*(1-probability)
    dskew = 0.995*(1-tanh*tanh)
    logc = math.lgamma((eta+1)/2)-math.lgamma(eta/2)-0.5*math.log(math.pi*(eta-2))
    c = math.exp(logc)
    a = 4*skew*c*(eta-2)/(eta-1)
    b = math.sqrt(1+3*skew*skew-a*a)
    dlc = 0.5*_digamma((eta+1)/2)-0.5*_digamma(eta/2)-0.5/(eta-2)
    dae = a*(dlc+1/(eta-2)-1/(eta-1))
    dal = 4*c*(eta-2)/(eta-1)
    dbe = -a*dae/b
    dbl = (3*skew-a*dal)/b
    score = np.zeros(4)
    logtotal = 0.0
    for i in range(start, stop):
        z = (values[i]-mean)/sigma
        side = -1.0 if z < -a/b else 1.0
        denominator = 1+side*skew
        w = (b*z+a)/denominator
        de = (dbe*z+dae)/denominator
        dl = (dbl*z+dal)/denominator-w*side/denominator
        quadratic = 1+w*w/(eta-2)
        logq = math.log(quadratic)
        logtotal += math.log(b)+logc-0.5*(eta+1)*logq-0.5*lv
        dz = -(eta+1)*w*b/denominator/(eta-2+w*w)
        score[0] -= dz/sigma
        score[1] += -0.5-0.5*z*dz
        score[2] += (dbe/b+dlc-0.5*logq-0.5*(eta+1)*(2*w*de/(eta-2)-w*w/(eta-2)**2)/quadratic)*deta
        score[3] += (dbl/b-0.5*(eta+1)*(2*w*dl/(eta-2))/quadratic)*dskew
    n = stop-start
    return logtotal/n, score/n


@njit(cache=True)
def reference_filter(values, offsets, anchor, coefficients, units, frozen, projection, shape_layer, gradient):
    days = len(offsets)-1
    states = np.empty((days,4))
    scores = np.empty((days,4))
    drivers = np.empty((days,2))
    contributions = np.empty(days)
    parameter_scores = np.zeros((days,6))
    active = 2 if shape_layer else 0
    moving = anchor[active:active+2].copy()
    derivative = np.zeros((2,6))
    history = np.zeros((22,2))
    derivative_history = np.zeros((22,2,6))
    windows = (1,5,22)
    for t in range(days):
        state = anchor.copy()
        if shape_layer:
            state[:2] = frozen[t,:2]
        state[active:active+2] = moving
        if abs(state[0]) > 0.5 or state[1] < -20 or state[1] > 0 or abs(state[2]) > 20 or abs(state[3]) > 20 or not np.all(np.isfinite(state)):
            raise FloatingPointError("unstable Case1 reference filter")
        contribution, score = reference_day(values, offsets[t], offsets[t+1], state)
        driver = score[active:active+2].copy()
        if shape_layer:
            driver -= projection @ score[:2]
        states[t] = state
        scores[t] = score
        drivers[t] = driver
        contributions[t] = contribution
        driver_derivative = np.zeros((2,6))
        if gradient:
            parameter_scores[t] = score[active:active+2] @ derivative
            jac = np.zeros((2,2))
            for j in range(2):
                step = 1e-5
                if not shape_layer and j == 0:
                    step *= math.exp(0.5*state[1])
                plus, minus = state.copy(), state.copy()
                plus[active+j] += step
                minus[active+j] -= step
                _, sp = reference_day(values, offsets[t], offsets[t+1], plus)
                _, sm = reference_day(values, offsets[t], offsets[t+1], minus)
                ds = (sp-sm)/(2*step)
                jac[:,j] = ds[active:active+2]
                if shape_layer:
                    jac[:,j] -= projection @ ds[:2]
            driver_derivative = jac @ derivative
        history[:-1] = history[1:].copy()
        history[-1] = driver
        derivative_history[:-1] = derivative_history[1:].copy()
        derivative_history[-1] = driver_derivative
        moving = anchor[active:active+2].copy()
        derivative = np.zeros((2,6))
        for h in range(3):
            window = windows[h]
            average = np.zeros(2)
            avg_derivative = np.zeros((2,6))
            for k in range(22-window,22):
                average += history[k]/window
                if gradient:
                    avg_derivative += derivative_history[k]/window
            moving += coefficients[h]*average
            if gradient:
                for j in range(2):
                    derivative[j] += coefficients[h,j]*avg_derivative[j]
                    derivative[j,2*h+j] += units[h,j]*average[j]
    return states, scores, drivers, contributions, parameter_scores, moving, history


@njit(cache=True)
def endpoint_filter(phi, offsets, reference_scores, masses, loadings, projection, coefficients, units, gradient):
    """Two scalar endpoint directions, exactly equivalent to full q recursion."""
    days = len(offsets)-1
    components = len(masses)
    states = np.zeros((days,2))
    drivers = np.zeros((days,2))
    weights = np.empty((days,components))
    contributions = np.zeros(days)
    parameter_scores = np.zeros((days,6))
    moving = np.zeros(2)
    derivative = np.zeros((2,6))
    history = np.zeros((22,2))
    derivative_history = np.zeros((22,2,6))
    windows = (1,5,22)
    for t in range(days):
        logits = np.log(masses) + moving @ loadings
        w = np.exp(logits-np.max(logits))
        w /= w.sum()
        expected = loadings @ w
        posterior_mean = np.zeros(2)
        posterior_covariance = np.zeros((2,2))
        count = offsets[t+1]-offsets[t]
        for i in range(offsets[t],offsets[t+1]):
            resp = phi[i]*w
            density = resp.sum()
            if density <= 0 or not math.isfinite(density):
                raise FloatingPointError("invalid endpoint density")
            contributions[t] += math.log(density)/count
            resp /= density
            conditional = loadings @ resp
            posterior_mean += conditional/count
            if gradient:
                for j in range(components):
                    for a in range(2):
                        for b in range(2):
                            posterior_covariance[a,b] += resp[j]*loadings[a,j]*loadings[b,j]/count
                for a in range(2):
                    for b in range(2):
                        posterior_covariance[a,b] -= conditional[a]*conditional[b]/count
        raw = posterior_mean-expected
        driver = raw-projection @ reference_scores[t,:2]
        states[t] = moving
        drivers[t] = driver
        weights[t] = w
        driver_derivative = np.zeros((2,6))
        if gradient:
            parameter_scores[t] = raw @ derivative
            covariance = np.zeros((2,2))
            for j in range(components):
                for a in range(2):
                    for b in range(2):
                        covariance[a,b] += w[j]*loadings[a,j]*loadings[b,j]
            for a in range(2):
                for b in range(2):
                    covariance[a,b] -= expected[a]*expected[b]
            driver_derivative = (posterior_covariance-covariance) @ derivative
        history[:-1] = history[1:].copy()
        history[-1] = driver
        derivative_history[:-1] = derivative_history[1:].copy()
        derivative_history[-1] = driver_derivative
        moving = np.zeros(2)
        derivative = np.zeros((2,6))
        for h in range(3):
            window = windows[h]
            average = np.zeros(2)
            avg_derivative = np.zeros((2,6))
            for k in range(22-window,22):
                average += history[k]/window
                if gradient:
                    avg_derivative += derivative_history[k]/window
            moving += coefficients[:,h]*average
            if gradient:
                for b in range(2):
                    derivative[b] += coefficients[b,h]*avg_derivative[b]
                    derivative[b,b*3+h] += units[b,h]*average[b]
        if not np.all(np.isfinite(moving)) or np.max(np.abs(moving)) > 1e6:
            raise FloatingPointError("unstable endpoint filter")
    return states, drivers, weights, contributions, parameter_scores, moving, history
