"""Unequal N must change date weighting; constant N must preserve inference."""
import numpy as np
import pytest

pytest.importorskip('numba')
import run_case1_symmetric_t_daily_mean_wald as daily


def test_static_likelihood_averages_within_each_unequal_date():
    values = np.array([-.015,.006,.003,.04,-.03,.012])
    offsets = np.array([0,1,3,6])
    state = np.r_[.001,np.log(.02**2),daily.archived.core.parameters_to_state(4.2,0.)]
    got,score = daily.static_daily_mean(values,offsets,state)
    likelihood,scores = daily.archived.core.reference_logpdf_and_score(values,state)
    expected = sum(likelihood[offsets[t]:offsets[t+1]].mean() for t in range(3))
    expected_score = sum(scores[offsets[t]:offsets[t+1]].mean(axis=0) for t in range(3))
    np.testing.assert_allclose(got,expected,atol=2e-12)
    np.testing.assert_allclose(score,expected_score,atol=2e-9)
    assert not np.isclose(got,likelihood.sum())
    for j in range(3):
        step = np.zeros(4); step[j] = 1e-6 if j==0 else 1e-5
        numerical = (daily.static_daily_mean(values,offsets,state+step)[0]
                    -daily.static_daily_mean(values,offsets,state-step)[0])/(2*step[j])
        np.testing.assert_allclose(score[j],numerical,rtol=2e-5,atol=2e-7)


def quadratic():
    observations = np.array([-.3,.5,-.8,.9,.1,-.4])
    def run(theta,gradient):
        score = (observations-theta[0])[:,None]
        return None,None,None,-.5*score[:,0]**2,score
    return observations,run


def test_variable_N_wald_uses_equal_dates():
    obs,run = quadratic()
    counts = np.array([2,7,3,9,4,6])
    theta = np.array([obs.mean()])
    _,diag = daily.wald_covariance(theta,run,counts,[('joint',[0])])
    expected = np.sum((obs-obs.mean())**2)/len(obs)**2
    np.testing.assert_allclose(diag['covariance'][0][0],expected,rtol=1e-10)
    np.testing.assert_allclose(diag['hessian'],[[len(obs)]],rtol=1e-10)
    assert not np.isclose(obs.mean(),np.average(obs,weights=counts))


def test_constant_N_sum_and_daily_mean_have_same_wald():
    obs,run = quadratic(); theta = np.array([obs.mean()])
    counts = np.full(len(obs),20)
    test_sum,diag_sum = daily.base.wald_covariance(theta,run,counts,[('joint',[0])])
    test_mean,diag_mean = daily.wald_covariance(theta,run,counts,[('joint',[0])])
    np.testing.assert_allclose(diag_sum['covariance'],diag_mean['covariance'],rtol=1e-10)
    np.testing.assert_allclose(test_sum[0]['wald'],test_mean[0]['wald'],rtol=1e-10)


def test_daily_static_fit_reproduces_daily_criterion():
    rng = np.random.default_rng(417)
    counts = np.tile([1,4,9],15)
    offsets = np.r_[0,counts.cumsum()]
    state = np.r_[.002,np.log(.02**2),daily.archived.core.parameters_to_state(5.,0.)]
    values = state[0]+.02*daily.archived.core.hansen_ppf(rng.uniform(.005,.995,offsets[-1]),state[2:])
    daily.TRAINING_OFFSETS = offsets
    fit = daily.fit_static_daily_mean(values,{})
    likelihood,_ = daily.static_daily_mean(values,offsets,np.asarray(fit['state']))
    np.testing.assert_allclose(fit['log_likelihood'],likelihood,rtol=1e-11)
    assert fit['lambda']==0.


def test_unequal_N_full_empirical_pipeline_and_independent_hessian(tmp_path):
    """Exercise the real M1--M5 orchestration, not just helper functions."""
    import hashlib
    import json
    import pandas as pd
    import audit_case1_symmetric_t_daily_mean_wald as audit
    rng = np.random.default_rng(701)
    counts = np.tile([7,20,33],80)
    offsets = np.r_[0,counts.cumsum()].astype(np.int64)
    state = daily.archived.core.parameters_to_state(5.,0.)
    values = .002+.02*daily.archived.core.hansen_ppf(rng.uniform(.001,.999,offsets[-1]),state)
    dates = np.r_[pd.bdate_range(end='2023-12-29',periods=210).values,
                  pd.bdate_range(start='2024-01-02',periods=30).values]
    folder = daily.REPO/'data'/'daily_mean_pipeline_tests'/tmp_path.name
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'input.npz'
    np.savez_compressed(path,values=values,offsets=offsets,dates=dates)
    entry = dict(dataset='unequal_N_test',input=str(path.relative_to(daily.REPO)),
                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 role='main',label='Unequal N verification',family='test',oos_days=30)
    daily.run_dataset(entry,tmp_path)
    result = json.loads((tmp_path/entry['dataset']/'estimation.json').read_text(encoding='utf-8'))
    runs,weights = audit.training_runs(result)
    np.testing.assert_array_equal(weights,1.)
    assert 'mean log densities' in result['specification']['likelihood']
    for model in ('M2','M3'):
        fit = result['models'][model]
        filtered = runs[model](np.asarray(fit['unrestricted']['theta']),False)
        np.testing.assert_allclose(fit['layer_log_likelihood'],filtered[3].sum(),rtol=1e-10)
        assert not np.isclose(fit['layer_log_likelihood'],counts[:210]@filtered[3])
    for model in ('M4','M5'):
        fit = result['models'][model]
        filtered = runs[model](np.asarray(fit['unrestricted']['theta']),False)
        np.testing.assert_allclose(fit['layer_log_likelihood'],filtered[3].sum(),atol=1e-10)
    checks = audit.audit_tools.selected_derivatives(result)
    assert len(checks)==4


def test_cli_worker_reuse_keeps_each_panels_date_offsets(tmp_path):
    """Run two real panels in one spawned worker through the script entry point."""
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import pandas as pd
    rng = np.random.default_rng(719)
    folder = daily.REPO/'data'/'daily_mean_pipeline_tests'/tmp_path.name
    folder.mkdir(parents=True, exist_ok=True)
    dates = np.r_[pd.bdate_range(end='2023-12-29', periods=100).values,
                  pd.bdate_range(start='2024-01-02', periods=10).values]
    entries = []
    for index in range(2):
        counts = np.resize([3+index, 8, 14-index], len(dates))
        offsets = np.r_[0, counts.cumsum()].astype(np.int64)
        values = .02*rng.standard_t(5., offsets[-1])*np.sqrt(3/5)
        path = folder/f'panel{index}.npz'
        np.savez_compressed(path, values=values, offsets=offsets, dates=dates)
        entries.append(dict(dataset=f'panel{index}', input=str(path.relative_to(daily.REPO)),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), target='raw',
            label=f'Panel {index}', family='S&P 500', oos_days=10))
    manifest = tmp_path/'source.json'
    manifest.write_text(json.dumps({'entries':entries+[
        {**entries[0], 'dataset':f'unused{i}'} for i in range(17)]}), encoding='utf-8')
    output = tmp_path/'output'
    program = (
        'import runpy,sys; '
        f'sys.path.insert(0,{str(daily.ROOT)!r}); '
        'import run_case1_symmetric_t_wald as base; '
        f'base.INPUT_MANIFEST=base.Path({str(manifest)!r}); '
        f'sys.argv=["runner","--output",{str(output)!r},"--datasets","panel0,panel1","--workers","1"]; '
        f'runpy.run_path({str(daily.ROOT/"run_case1_symmetric_t_daily_mean_wald.py")!r},run_name="__main__")'
    )
    env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
    finished = subprocess.run([sys.executable, '-c', program], capture_output=True,
                              text=True, env=env, timeout=180)
    assert finished.returncode == 0, finished.stdout+'\n'+finished.stderr
    status = json.loads((output/'run_status.json').read_text(encoding='utf-8'))
    assert status['completed'] == status['requested'] == 2
    for entry in entries:
        saved = json.loads((output/entry['dataset']/'estimation.json').read_text(encoding='utf-8'))
        assert 'mean log densities' in saved['specification']['likelihood']
        assert saved['audit']['M1_likelihood_reproduction_error'] < 1e-9
