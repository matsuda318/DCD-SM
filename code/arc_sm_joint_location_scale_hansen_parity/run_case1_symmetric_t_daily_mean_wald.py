"""Symmetric-t M1--M5 using equal dates in every likelihood and Wald layer.

The earlier observation-sum runner and saved results are retained. Run this
entry point for the corrected empirical criterion. Do not import it in the
same process as archived experiment runners: configuration is process-local.
"""
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

import run_case1_symmetric_t_wald as base

ROOT, REPO = base.ROOT, base.REPO
OUTPUT = ROOT/'output_symmetric_t_m1_m5_equal_date_wald'
SCHEMA = 'symmetric-t-frozen-upstream-daily-mean-wald0-v1'
archived = base.archived
BASE_SOURCE = base.source_hash()
ORIGINAL_CONFIGURE = base.configure
ORIGINAL_RUN_DATASET = base.run_dataset
ORIGINAL_ARCHIVED_RUN_DATASET = archived.run_dataset
reference_day = archived.kernels.reference_day
TRAINING_OFFSETS = None


def source_hash():
    digest = hashlib.sha256(BASE_SOURCE.encode())
    for path in (Path(__file__),ROOT/'test_case1_symmetric_t_daily_mean_wald.py'):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


@njit(cache=True)
def static_daily_mean(values, offsets, state):
    """Total date-mean log likelihood and matching three state scores."""
    likelihood = 0.
    score = np.zeros(4)
    for t in range(len(offsets)-1):
        value, derivative = reference_day(values, offsets[t], offsets[t+1], state)
        likelihood += value
        score += derivative
    return likelihood, score


def fit_static_daily_mean(values, warm):
    assert TRAINING_OFFSETS is not None
    offsets = TRAINING_OFFSETS
    assert offsets[-1]==len(values) and np.all(np.diff(offsets)>0)
    days = len(offsets)-1
    weights = np.repeat(1/np.diff(offsets),np.diff(offsets))/days
    center = float(weights@values)
    sigma = max(float(np.sqrt(weights@((values-center)**2))),1e-6)
    units = np.array([sigma,1.,1.])

    def objective(theta):
        state = np.r_[theta*units,0.]
        try:
            likelihood, scores = static_daily_mean(values,offsets,state)
            value, gradient = -likelihood/days, -scores[:3]*units/days
            if np.isfinite(value) and np.isfinite(gradient).all():
                return value,gradient
        except (FloatingPointError,ValueError,OverflowError):
            pass
        return 1e100,np.zeros(3)

    starts = [np.r_[center,np.log(sigma*sigma),archived.core.parameters_to_state(eta,0.)[0]]/units
              for eta in (3.5,5.,8.,15.)]
    if warm:
        starts.insert(0,np.asarray(warm['model1']['state'])[:3]/units)
    fit = archived.maximize(objective,starts,
        [(-.1/sigma,.1/sigma),(-20.,0.),(-12.,12.)],'M1 symmetric t daily mean')
    anchor = np.r_[np.asarray(fit['theta'])*units,0.]
    eta,skew,_,_ = archived.core.state_to_parameters(anchor[2:])
    assert skew==0.
    return {**fit,'state':anchor.tolist(),'mean':float(anchor[0]),
        'scale':float(np.exp(anchor[1]/2)),'eta':float(eta),'lambda':0.,
        'parameter_units':units.tolist(),'log_likelihood':float(-fit['mean_nll']*days),
        'shape_anchor_bound_reached':bool(abs(anchor[2])>=11.999)}


def wald_covariance(theta,run,actual_counts,blocks):
    """Date-mean score/Hessian; zero counts may mark an excluded warmup."""
    weights = (np.asarray(actual_counts)>0).astype(float)
    return base.wald_covariance(theta,run,weights,blocks)


def configure(output,inference):
    ORIGINAL_CONFIGURE(output,inference)
    archived.fit_static = fit_static_daily_mean
    archived.run_dataset = partial(ORIGINAL_ARCHIVED_RUN_DATASET,daily_mean=True)
    archived.SCHEMA = SCHEMA
    archived.source_hash = source_hash


def run_dataset(entry,output):
    global TRAINING_OFFSETS
    with np.load(REPO/entry['input']) as raw:
        n_train = int((raw['dates']<=np.datetime64('2023-12-29')).sum())
        TRAINING_OFFSETS = raw['offsets'][:n_train+1].copy()
    result = ORIGINAL_RUN_DATASET(entry,output)
    path = Path(output)/entry['dataset']/'estimation.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved['specification']['likelihood'] = 'sum of daily cross-sectional mean log densities; equal date weights'
    saved['specification']['wald_weights'] = 'daily mean parameter scores and matching daily mean Hessian; lag 0'
    archived.dump(path,saved)
    return result


if __name__ == 'run_case1_symmetric_t_daily_mean_wald':
    base.configure = configure
    base.run_dataset = run_dataset
    base.source_hash = source_hash
    base.OUTPUT = OUTPUT
    base.SCHEMA = SCHEMA


if __name__=='__main__':
    # Workers and Numba's disk cache must resolve the same canonical module.
    # A second import of this script otherwise resets TRAINING_OFFSETS and
    # captures already-patched functions when a worker takes its next panel.
    import run_case1_symmetric_t_daily_mean_wald as entrypoint
    entrypoint.base.main()
