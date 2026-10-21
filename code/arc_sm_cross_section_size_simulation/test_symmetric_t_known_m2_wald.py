"""Independent checks of symmetric-t truth, joint odd dynamics and staging."""
import numpy as np
import pytest

pytest.importorskip('numba')
import run_symmetric_t_known_m2_wald as new
from test_full_stage_copula_lr_wald import quadratic_example


@pytest.mark.parametrize('dgp',new.DGPS)
@pytest.mark.parametrize('case',new.CASES)
def test_five_dgps_fixed_M2_correct_active_blocks_and_common_share(dgp,case):
    master,factor = new.generate_master(0,dgp,case,new.Design(training=45,oos=3,burn=30))
    audit = new.truth_audit(master,dgp)
    np.testing.assert_array_equal(master['reference_states'][:,[0,1,3]],0.)
    assert audit['true_recursion_maximum_error']<1e-10
    for block,active in [('eta',dgp in (new.DGPS[1],new.DGPS[3],new.DGPS[4])),
                         ('even',dgp in (new.DGPS[2],new.DGPS[3],new.DGPS[4])),
                         ('odd',dgp==new.DGPS[4])]:
        assert (audit[block+'_state_std']>1e-12)==active
    if case=='shared_factor':
        assert factor['days']==78
        assert factor['maximum_common_share_error']<=new.legacy.copula.SHARE_TOLERANCE
    else: assert factor is None


def entry(tmp_path,dgp=new.DGPS[4],burn=0):
    output = tmp_path/(tmp_path.parent.name+'_'+tmp_path.name)
    entries = new.generate(0,dgp,'iid',new.Design(training=65,oos=3,burn=burn),output)
    return next(e for e in entries if e['N']==100)


def test_joint_true_generation_matches_estimation_recursions(tmp_path):
    e = entry(tmp_path)
    runs,units,_,_ = new.training_runs(e,{})
    true = new.calibration()
    theta = np.zeros((3,2)); theta[:,0] = true['eta_kappa']/units['M3'][:,0]
    checkpoint = {'M3':{'parameter_units':units['M3'].tolist(),'unrestricted':{'theta':theta.ravel().tolist()}}}
    runs,units,_,_ = new.training_runs(e,checkpoint)
    fitted = runs['M3'](theta.ravel(),False)
    correction = runs['M5']((true['endpoint_kappa']/units['M5']).ravel(),False)
    path = new.REPO/e['input']
    with np.load(path.parent/f"{e['dgp']}_{e['case']}_master.npz") as master:
        np.testing.assert_allclose(fitted[0],master['reference_states'][:65],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(correction[0],master['endpoint_states'][:65],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(fitted[2][:,0],master['drivers'][:65,0],rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose(correction[1],master['drivers'][:65,1:],rtol=1e-7,atol=1e-9)


def test_fitted_M3_freezes_M5_and_OOS_cannot_change_training(tmp_path):
    e = entry(tmp_path,burn=30)
    runs,units,counts,_ = new.training_runs(e,{})
    theta = np.array([.1,0.,-.05,0.,.03,0.])
    checkpoint = {'M3':{'parameter_units':units['M3'].tolist(),'unrestricted':{'theta':theta.tolist()}}}
    runs,_,_,_ = new.training_runs(e,checkpoint)
    # With no SM correction, M5's saved weights are exactly the static identity.
    zero = runs['M5'](np.zeros(6),False)
    masses = new.geometry()[1]
    np.testing.assert_allclose(zero[2],np.tile(masses,(65,1)),rtol=1e-12,atol=1e-12)
    baseline = {model:runs[model](theta,True) for model in new.MODELS}
    path = new.REPO/e['input']
    with np.load(path) as raw: saved = {key:raw[key].copy() for key in raw.files}
    saved['values'][counts.sum():] += 1e6
    np.savez_compressed(path,**saved)
    changed,_,_,_ = new.training_runs(e,checkpoint)
    for model in new.MODELS:
        for a,b in zip(baseline[model],changed[model](theta,True)):
            np.testing.assert_array_equal(a,b)


def test_reduced_M3_and_even_odd_daily_derivatives(tmp_path):
    e = entry(tmp_path)
    runs,units,_,_ = new.training_runs(e,{})
    checkpoint = {'M3':{'parameter_units':units['M3'].tolist(),
                        'unrestricted':{'theta':[.1,0.,-.05,0.,.03,0.]}}}
    runs,_,_,_ = new.training_runs(e,checkpoint)
    for model in new.MODELS:
        run = new.estimator.active_run(runs[model],[0,2,4]) if model=='M3' else runs[model]
        theta = np.array([.05,-.03,.01]) if model=='M3' else np.array([.05,-.03,.01,.02,-.01,.02])
        fitted = run(theta,True)
        for j in range(len(theta)):
            delta = np.zeros(len(theta)); delta[j] = 1e-5
            difference = (run(theta+delta,False)[3]-run(theta-delta,False)[3])/(2e-5)
            np.testing.assert_allclose(fitted[4][:,j],difference,rtol=3e-4,atol=2e-7)


def test_observation_sum_fit_and_daily_cluster_covariance_with_reduced_eta():
    _,run,counts,X = quadratic_example()
    ids = np.array([0,2,4])
    reduced_X = X[:,ids]
    bread = reduced_X.T@reduced_X
    rhs = (counts@run(np.zeros(6),True)[4])[ids]
    expected = np.linalg.solve(bread,rhs)
    fit,_ = new.estimator.fit_layer_wald(run,np.ones((3,2)),counts,[],None,'M3')
    theta = np.asarray(fit['unrestricted']['active_theta'])
    np.testing.assert_allclose(theta,expected,rtol=1e-6,atol=1e-7)
    full = np.zeros(6); full[ids] = theta
    np.testing.assert_array_equal(np.asarray(fit['coefficients'])[:,1],0.)
    score = counts[:,None]*run(full,True)[4][:,ids]
    centered = score-score.mean(axis=0)
    inverse = np.linalg.inv(bread)
    np.testing.assert_allclose(fit['wald_diagnostics']['covariance'],inverse@(centered.T@centered)@inverse.T,rtol=1e-8,atol=1e-10)
    assert fit['wald_tests'][0]['df']==3 and len(fit['wald_tests'])==1


def test_low_df_state_extends_hermite_degree_without_relaxing_share_tolerance():
    # Actual pilot day 2030, where degree-40's conservative rho**41 bound failed.
    state = np.array([0.,0.,-8.588189036277114,0.])
    q_state = np.array([.44831159566862067,.25883282044010886,.183022442526841,
        .14176857437773183,.11575355626241607,.09782961057513767,.084722928000408,
        .07471859927810345,.0668303468677624,.060450323228092136,.055183342734174294,
        .0507612923916297,.04699582478660227,-.5687595145916843])
    _,masses,helmert,*_ = new.geometry()
    grid,_,cdf,inverse = new.component_sampler()
    sampler = new.CommonFactorSampler(grid,cdf)
    u = sampler(100,q_state,masses,helmert,inverse,np.random.default_rng(9),reference_state=state)
    diagnostic = sampler.audit[-1]
    assert diagnostic['hermite_degree']>40
    assert diagnostic['hermite_tail_bound']<1e-7
    assert abs(diagnostic['share_error'])<=new.legacy.copula.SHARE_TOLERANCE
    logits = np.log(masses)+helmert@q_state
    weights = np.exp(logits-new.logsumexp(logits))
    mixture = np.maximum.accumulate(cdf@weights); mixture[0],mixture[-1] = 0,1
    def quantile(v):
        reference_u = np.clip(np.interp(v,mixture,grid),1e-12,1-1e-12)
        return new.legacy.sim.core.hansen_ppf(reference_u,state[2:])
    # Verify with both higher integration order and twice the accepted expansion degree.
    energy,_,_ = new.hermite_moments(quantile,6144,2*diagnostic['hermite_degree'])
    checked = new.legacy.copula.correlation_from_moments(diagnostic['latent_rho'],energy)
    assert abs(checked-.3)<new.legacy.copula.SHARE_TOLERANCE
    assert np.isfinite(u).all() and np.all((u>0)&(u<1))


def test_case4_skewed_state_fallback_with_independent_composite_order():
    # Pilot case4 day 204: whole-domain Hermite variance integration oscillated.
    state = np.array([0.,0.,-.6395060263018939,0.])
    q = np.array([2.95731279120554,1.7074053360804426,1.2073178913765772,
        .9351844173759475,.7635748793243486,.645338558064787,.5588795853257245,
        .49288546520092347,.44085016212434863,.3987639754183124,.3640200407617421,
        .3348497356264085,.31001061564645405,-9.122220627513892])
    _,masses,helmert,*_ = new.geometry()
    grid,_,cdf,inverse = new.component_sampler()
    sampler = new.CommonFactorSampler(grid,cdf)
    sampler(100,q,masses,helmert,inverse,np.random.default_rng(8),reference_state=state)
    diagnostic = sampler.audit[-1]
    assert diagnostic['integration_method']=='composite_gauss_legendre'
    logits = np.log(masses)+helmert@q
    weights = np.exp(logits-new.logsumexp(logits))
    mixture = np.maximum.accumulate(cdf@weights); mixture[0],mixture[-1] = 0,1
    def quantile(v):
        return new.legacy.sim.core.hansen_ppf(np.clip(np.interp(v,mixture,grid),1e-12,1-1e-12),state[2:])
    clipped = np.interp([1e-12,1-1e-12],grid,mixture)
    points = new.ndtri(np.clip(clipped,1e-300,1-np.finfo(float).eps))
    energy,_,_ = new.composite_normal_moments(quantile,512,2*diagnostic['hermite_degree'],points)
    checked = new.legacy.copula.correlation_from_moments(diagnostic['latent_rho'],energy)
    assert abs(checked-.3)<new.legacy.copula.SHARE_TOLERANCE
