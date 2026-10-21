"""Own-N truth/filter consistency, warmup exclusion and daily-mean weights."""
from dataclasses import replace
import numpy as np
import pytest

import run_symmetric_t_observed_N_scaled as new


def test_scaling_and_no_duplicate_univariate_cell():
    design = new.sim.Design()
    combinations = new.combinations(design)
    assert len(combinations)==35 and len(set(combinations))==35
    assert all(c=='iid' or n!=1 for d,c,n in combinations)
    for dgp in new.sim.DGPS:
        original = new.BASE_CALIBRATION(dgp)
        for n in design.sample_sizes:
            scaled = new.scaled_calibration(n,dgp)
            for key in ('eta_kappa','endpoint_kappa'):
                np.testing.assert_allclose(scaled[key],original[key]*np.sqrt(n/500))
            np.testing.assert_array_equal(scaled['anchor'],original['anchor'])


@pytest.mark.parametrize('n',[1,20,50,100])
@pytest.mark.parametrize('dgp',new.sim.DGPS)
def test_true_recursions_match_observed_N_filters_with_warmup(tmp_path,n,dgp):
    design = new.sim.Design(training=35,oos=3,burn=25)
    entry = new.generate(0,dgp,'iid',n,design,tmp_path)
    runs,units,weights,null = new.training_runs(entry,{})
    truth = new.scaled_calibration(n,dgp)
    theta = np.zeros((3,2)); theta[:,0] = truth['eta_kappa']/units['M3'][:,0]
    checkpoint = {'M3':{'parameter_units':units['M3'].tolist(),
                       'unrestricted':{'theta':theta.ravel().tolist()}}}
    runs,units,_,_ = new.training_runs(entry,checkpoint)
    reference = runs['M3'](theta.ravel(),False)
    endpoint = runs['M5']((truth['endpoint_kappa']/units['M5']).ravel(),False)
    with np.load(new.REPO/entry['truth']) as master:
        stop = design.burn+design.training
        np.testing.assert_allclose(reference[0],master['reference_states'][:stop],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(endpoint[0],master['endpoint_states'][:stop],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(reference[2][:,0],master['drivers'][:stop,0],rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose(endpoint[1],master['drivers'][:stop,1:],rtol=1e-7,atol=1e-9)
    np.testing.assert_array_equal(weights[:25],0.)
    np.testing.assert_array_equal(weights[25:],1.)


def test_factor_calibrates_actual_N_and_preserves_marginals():
    master,audit,seed = new.generate_panel(0,new.sim.DGPS[0],'heterogeneous_factor',20,
                                         new.sim.Design(training=5,oos=2,burn=5))
    assert master['values'].shape==(12,20) and audit['mean_share_over_N']==20
    np.testing.assert_allclose(master['factor_return_shares'].mean(axis=1),.3,atol=3e-4,rtol=0)
    beta,e = new.factor_tools.latent_coefficients(master['factor_multipliers'][:,None],master['factor_loadings'])
    np.testing.assert_allclose(beta**2+e**2,1.,atol=1e-14,rtol=0)
    assert len(seed)==5


def test_warmup_derivatives_and_future_exclusion(tmp_path):
    design = new.sim.Design(training=35,oos=3,burn=25)
    entry = new.generate(0,new.sim.DGPS[4],'iid',20,design,tmp_path)
    runs,units,weights,_ = new.training_runs(entry,{})
    checkpoint = {'M3':{'parameter_units':units['M3'].tolist(),
                        'unrestricted':{'theta':[.1,0.,-.05,0.,.03,0.]}}}
    runs,_,_,_ = new.training_runs(entry,checkpoint)
    snapshots = {}
    for model in new.sim.MODELS:
        run = new.sim.estimator.active_run(runs[model],[0,2,4]) if model=='M3' else runs[model]
        theta = np.array([.05,-.03,.01]) if model=='M3' else np.array([.05,-.03,.01,.02,-.01,.02])
        fit = run(theta,True); snapshots[model] = runs[model](np.zeros(6),True)
        for j in range(len(theta)):
            step = np.zeros_like(theta); step[j] = 1e-5
            numeric = (weights@run(theta+step,False)[3]-weights@run(theta-step,False)[3])/(2e-5)
            np.testing.assert_allclose(weights@fit[4][:,j],numeric,rtol=3e-4,atol=3e-7)
    path = new.REPO/entry['input']
    with np.load(path) as raw: saved = {k:raw[k].copy() for k in raw.files}
    saved['values'][saved['offsets'][int(saved['training'])]:] += 1e6
    np.savez_compressed(path,**saved)
    changed,_,_,_ = new.training_runs(entry,checkpoint)
    for model in new.sim.MODELS:
        for before,after in zip(snapshots[model],changed[model](np.zeros(6),True)):
            np.testing.assert_array_equal(before,after)


def test_wald_meat_excludes_warmup_but_keeps_recursion():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(30,3)); y = rng.normal(size=30)
    weights = np.r_[np.zeros(8),np.ones(22)]
    def run(theta,gradient):
        error = y-x@theta
        return None,None,None,-error**2/2,error[:,None]*x
    theta = np.linalg.solve(x[8:].T@x[8:],x[8:].T@y[8:])
    tests,diag = new.wald_covariance(theta,run,weights,[('joint',[0,1,2])])
    scores = run(theta,True)[4][8:]; scores -= scores.mean(axis=0)
    np.testing.assert_allclose(diag['meat'],scores.T@scores,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(diag['hessian'],x[8:].T@x[8:],rtol=1e-9,atol=1e-9)
    assert tests[0]['wald_valid']
