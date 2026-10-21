"""Distribution preservation, signed dependence, calibration and nesting."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

import run_symmetric_t_heterogeneous_factor as new


def test_latent_normalization_and_signed_dependence():
    a = np.array([-.1, 0., .3, 1.1])
    beta, individual = new.latent_coefficients(2., a)
    np.testing.assert_allclose(beta**2+individual**2, 1., atol=3e-16)
    assert beta[0]*beta[-1] < 0
    rng = np.random.default_rng(25)
    z = rng.standard_normal((250000,1))*beta + rng.standard_normal((250000,4))*individual
    assert abs(z.mean(axis=0)).max() < .008
    assert abs(z.var(axis=0)-1).max() < .015
    np.testing.assert_allclose(np.quantile(z,[.1,.5,.9],axis=0),
                               np.tile(norm.ppf([.1,.5,.9])[:,None],(1,4)),atol=.02)
    expected = beta[:,None]*beta[None,:]
    np.fill_diagonal(expected, 1.)
    np.testing.assert_allclose(np.cov(z,rowvar=False), expected, atol=.015)


def test_linear_return_mean_share_calibration_is_exact():
    a = new.loadings(0)
    c, audit = new.calibrate(lambda u: ndtri(np.clip(u,1e-12,1-1e-12)), a)
    beta, _ = new.latent_coefficients(c, a)
    assert abs(np.mean(beta**2)-.3) < 1e-8
    np.testing.assert_allclose(audit['individual_return_shares'],beta**2,atol=1e-8)
    assert audit['hermite_tail_bound'] < 1e-7


def test_nonlinear_share_uses_conditional_variance_not_pair_correlation():
    # Y=Z^2-1 has only second Hermite energy: its individual common share
    # is beta^4 and does not cancel under negative loadings.
    a = new.loadings(1)
    c, audit = new.calibrate(lambda u: ndtri(np.clip(u,1e-12,1-1e-12))**2-1, a)
    beta, _ = new.latent_coefficients(c,a)
    assert abs(np.mean(beta**4)-.3) < 1e-8
    np.testing.assert_allclose(audit['individual_return_shares'],beta**4,atol=1e-8)
    rng = np.random.default_rng(1)
    f = rng.standard_normal(500000)
    estimated_share = np.mean(beta**4)*np.var(f**2-1)/2
    assert abs(estimated_share-.3) < .006


def test_loading_stream_fixed_and_not_reused_for_innovations():
    np.testing.assert_array_equal(new.loadings(0),new.loadings(0))
    assert not np.array_equal(new.loadings(0),new.loadings(1))
    assert np.all((new.loadings(0)>=-.1)&(new.loadings(0)<=1.1))


def test_static_generation_preserves_M2_and_fixed_heterogeneous_loadings():
    master,factor = new.generate_master(0,new.sim.DGPS[0],new.CASE,
                                      new.sim.Design(training=45,oos=6,burn=30))
    audit = new.sim.truth_audit(master,new.sim.DGPS[0])
    assert audit['true_recursion_maximum_error'] < 1e-10
    np.testing.assert_array_equal(master['factor_loadings'],new.loadings(0))
    np.testing.assert_allclose(master['factor_return_shares'].mean(axis=1),.3,atol=3e-4)
    assert np.ptp(master['factor_return_shares'][0]) > .1
    assert factor['maximum_common_share_error'] <= 3e-4
    assert factor['maximum_hermite_tail_bound'] < 1e-7
    assert master['values'].shape == (51,100)


def test_nested_observations_and_training_reference_unchanged(tmp_path):
    entries = new.sim.generate(0,new.sim.DGPS[0],new.CASE,
                               new.sim.Design(training=45,oos=6,burn=30),tmp_path)
    a = None
    full = next(e for e in entries if e['N']==100)
    with np.load(new.REPO/full['input']) as raw:
        a = raw['values'].reshape(-1,100).copy()
    for entry in entries:
        with np.load(new.REPO/entry['input']) as raw:
            np.testing.assert_array_equal(raw['values'].reshape(-1,entry['N']),a[:,:entry['N']])
