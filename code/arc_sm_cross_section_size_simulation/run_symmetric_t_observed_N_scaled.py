"""Known M2, SP500 coefficients times sqrt(N/500), own-N score recursion.

Equal-date likelihood/Wald throughout. Burn-in observations warm each
candidate filter but contribute no likelihood, parameter score or meat.
N=1 is run once (iid); heterogeneous-factor N=1 is distributionally identical.
All configuration changes affect only this separate Python process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import chi2

import run_symmetric_t_heterogeneous_factor as factor_tools

sim = factor_tools.sim
scoring, oos = factor_tools.scoring, factor_tools.oos
ROOT, REPO = sim.ROOT, sim.REPO
OUTPUT = ROOT/'output_symmetric_t_observed_N_scaled'
PREFIX = 'arc_sm_symmetric_t_observed_N_scaled'
CASES = ('iid','heterogeneous_factor')
REFERENCE_N = 500
BASE_SOURCE = factor_tools.fingerprint()
BASE_CALIBRATION = sim.calibration
BASE_SUMMARIZE = sim.summarize
BASE_WALD_COVARIANCE = sim.estimator.wald_covariance


@sim.lru_cache(maxsize=1)
def fingerprint():
    # Audit-only corrections do not change the frozen numerical experiment.
    import ast
    frozen = ROOT/'snapshots/symmetric_t_observed_N_scaled_numerical_source.txt'
    def numerical_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        tree.body = [node for node in tree.body if not (
            isinstance(node,ast.FunctionDef) and node.name in ('fingerprint','audit'))]
        return ast.dump(tree,include_attributes=False)
    assert numerical_ast(Path(__file__))==numerical_ast(frozen), 'numerical code changed after frozen run'
    digest = hashlib.sha256(BASE_SOURCE.encode())
    for path in (Path(__file__),ROOT/'test_symmetric_t_observed_N_scaled.py'):
        digest.update(path.name.encode())
        source = frozen if path==Path(__file__) else path
        digest.update(source.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def scaled_calibration(n,dgp):
    original = BASE_CALIBRATION(dgp)
    multiplier = float(np.sqrt(n/REFERENCE_N))
    return dict(anchor=original['anchor'].copy(),
        eta_kappa=original['eta_kappa']*multiplier,
        endpoint_kappa=original['endpoint_kappa']*multiplier,
        coefficient_multiplier=multiplier)


def combinations(design):
    return [(d,c,n) for d in sim.DGPS for c in CASES for n in design.sample_sizes
            if c=='iid' or n!=1]


class FactorSampler(factor_tools.HeterogeneousSampler):
    def __init__(self,grid,cdf,a,target):
        sim.legacy.copula.CommonFactorSampler.__init__(self,grid,cdf,target)
        self.loadings = np.asarray(a).copy()

    def __call__(self,count,q_state,masses,helmert,inverse,rng,*,reference_state):
        assert count==len(self.loadings)
        logits = np.log(masses)+helmert@q_state
        weights = np.exp(logits-logsumexp(logits))
        mixture = np.maximum.accumulate(self.component_cdf@weights)
        mixture[0],mixture[-1] = 0.,1.
        def reference_u(v):
            return np.clip(np.interp(v,mixture,self.grid),1e-12,1-1e-12)
        def quantile(v):
            return sim.legacy.sim.core.hansen_ppf(reference_u(v),reference_state[2:])
        key = (reference_state[2:].tobytes(),q_state.tobytes())
        if key==self._cache_key:
            multiplier,audit = self._cache_calibration
        else:
            clipped = np.interp([1e-12,1-1e-12],self.grid,mixture)
            points = ndtri(np.clip(clipped,1e-300,1-np.finfo(float).eps))
            multiplier,audit = factor_tools.calibrate(quantile,self.loadings,self.target,points)
            self._cache_key,self._cache_calibration = key,(multiplier,audit)
        self.audit.append(dict(audit,generated_day=len(self.audit)))
        beta,individual = factor_tools.latent_coefficients(multiplier,self.loadings)
        common = rng.standard_normal(); epsilon = rng.standard_normal(count)
        return reference_u(ndtr(beta*common+individual*epsilon))


def generate_panel(replication,dgp,case,n,design):
    s = sim.legacy.sim
    knots,masses,helmert,directions,anchor,shape,q = sim.geometry()
    grid,_,cdf,inverse = sim.component_sampler()
    truth = scaled_calibration(n,dgp)
    # N has its own density trajectory; no unobserved master drives the state.
    seed = [design.seed,replication,sim.DGPS.index(dgp),n,701]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    a = factor_tools.loadings(replication,design.seed)[:n]
    sampler = (sim.legacy.IndependentNormalSampler(grid,cdf) if case=='iid'
               else FactorSampler(grid,cdf,a,design.common_share))
    total = design.burn+design.training+design.oos
    values = np.empty((total,n)); states = np.empty((total,4))
    moving_states = np.empty((total,2)); drivers = np.empty((total,3))
    history = np.zeros((22,3)); eta = anchor[2]; moving = np.zeros(2)
    loading = directions@helmert.T; projection = directions@q.projection_q_on_hansen
    for t in range(total):
        state = anchor.copy(); state[2] = eta
        q_state = moving@directions
        u = sampler(n,q_state,masses,helmert,inverse,rng,reference_state=state)
        y = s.core.hansen_ppf(u,state[2:])
        _,scores = s.core.reference_logpdf_and_score(y,state)
        daily = scores.mean(axis=0)
        eta_driver = daily[2]-shape.projection_shape_on_ms[0]@daily[:2]
        phi = s.core.normalized_bspline_basis(u,knots,3,masses)
        logits = np.log(masses)+moving@loading
        weights = np.exp(logits-logsumexp(logits))
        posterior = phi*weights/(phi@weights)[:,None]
        endpoint_driver = loading@(posterior.mean(axis=0)-weights)-projection@daily[:2]
        values[t],states[t],moving_states[t] = y,state,moving
        drivers[t] = np.r_[eta_driver,endpoint_driver]
        history = np.vstack((history[1:],drivers[t]))
        averages = np.array([history[-w:].mean(axis=0) for w in (1,5,22)])
        eta = anchor[2]+truth['eta_kappa']@averages[:,0]
        moving = np.sum(truth['endpoint_kappa']*averages[:,1:].T,axis=1)
        if (not np.isfinite(eta) or abs(eta)>20 or not np.isfinite(moving).all()
                or abs(q_state).max()>100):
            raise FloatingPointError(f'unstable true recursion: day={t}, eta={eta}, endpoint={moving}')
    master = dict(values=values,reference_states=states,endpoint_states=moving_states,
                  drivers=drivers,burn=design.burn,fit_training_days=design.training)
    factor = None
    if case!='iid':
        daily = sampler.audit
        shares = np.asarray([r['individual_return_shares'] for r in daily])
        multipliers = np.asarray([r['multiplier'] for r in daily])
        beta,_ = factor_tools.latent_coefficients(multipliers[:,None],a)
        master.update(factor_loadings=a,factor_return_shares=shares,
                      factor_multipliers=multipliers,factor_latent_loading=beta)
        factor = dict(days=total,mean_share_over_N=n,loadings=a.tolist(),
            loading_distribution='iid Uniform[-0.1,1.1], fixed over time and all five DGPs',
            loading_stream=[design.seed,replication,417],
            maximum_common_share_error=max(abs(r['share_error']) for r in daily),
            maximum_hermite_tail_bound=max(r['hermite_tail_bound'] for r in daily),
            maximum_quadrature_order=max(r['quadrature_order'] for r in daily),
            maximum_hermite_degree=max(r['hermite_degree'] for r in daily),
            composite_quadrature_days=sum(r['integration_method']=='composite_gauss_legendre' for r in daily))
    return master,factor,seed


def truth_audit(master,n,dgp):
    truth = scaled_calibration(n,dgp)
    states, moving = master['reference_states'],master['endpoint_states']
    np.testing.assert_array_equal(states[:,[0,1,3]],0.)
    history = np.zeros((22,3)); maximum = 0.
    for t,driver in enumerate(master['drivers']):
        averages = np.array([history[-w:].mean(axis=0) for w in (1,5,22)])
        expected_eta = truth['anchor'][2]+truth['eta_kappa']@averages[:,0]
        expected_q = np.sum(truth['endpoint_kappa']*averages[:,1:].T,axis=1)
        maximum = max(maximum,abs(states[t,2]-expected_eta),float(abs(moving[t]-expected_q).max()))
        history = np.vstack((history[1:],driver))
    assert maximum<1e-10
    assert master['values'].shape[1]==n and np.isfinite(master['values']).all()
    return dict(true_recursion_maximum_error=maximum,observed_and_driving_N=n,
        coefficient_multiplier=truth['coefficient_multiplier'],
        eta_state_std=float(states[master['burn']:,2].std()),
        even_state_std=float(moving[master['burn']:,0].std()),
        odd_state_std=float(moving[master['burn']:,1].std()),
        maximum_endpoint_state=float(abs(moving).max()))


def generate(replication,dgp,case,n,design,output):
    name = f'rep{replication:04d}_{dgp}_{case}_N{n}'
    folder = REPO/'data'/output.name/f'replication_{replication:04d}'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'{dgp}_{case}_N{n}.npz'
    master_path = folder/f'{dgp}_{case}_N{n}_truth.npz'
    audit_path = output/'generation'/(name+'.json')
    if path.exists() and master_path.exists() and audit_path.exists():
        audit = sim.load(audit_path)
        assert hashlib.sha256(master_path.read_bytes()).hexdigest()==audit['master_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['input_sha256']
    else:
        master,factor,seed = generate_panel(replication,dgp,case,n,design)
        np.savez_compressed(master_path,**master)
        np.savez_compressed(path,values=np.ascontiguousarray(master['values']).ravel(),
            offsets=np.arange(len(master['values'])+1,dtype=np.int64)*n,
            training=design.burn+design.training,burn=design.burn,fit_training_days=design.training)
        audit = dict(replication=replication,dgp=dgp,case=case,N=n,seed=seed,
            master_input=str(master_path.relative_to(REPO)),input=str(path.relative_to(REPO)),
            truth=truth_audit(master,n,dgp),
            master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if factor: audit['common_factor'] = factor
        sim.dump(audit_path,audit)
    return dict(dataset=name,replication=replication,dgp=dgp,case=case,N=n,
        input=str(path.relative_to(REPO)),input_sha256=audit['input_sha256'],
        burn=design.burn,fit_training_days=design.training,truth=str(master_path.relative_to(REPO)))


def training_runs(entry,checkpoint):
    e = sim.estimator.archived
    with np.load(REPO/entry['input']) as raw:
        train = int(raw['training']); burn = int(raw['burn'])
        offsets = raw['offsets'][:train+1].copy(); values = raw['values'][:offsets[-1]].copy()
    knots,masses,helmert,directions,anchor,shape,q = sim.geometry()
    frozen = np.tile(anchor,(train,1))
    null = e.kernels.reference_filter(values,offsets,anchor,np.zeros((3,2)),np.ones((3,2)),
        frozen,shape.projection_shape_on_ms,True,False)
    averages = e.fixed_denominator_score_averages(null[2],(1,5,22))
    units3 = np.array([[1e-2/max(float(np.median(abs(a[burn:,j]))),1e-12)
                       for j in range(2)] for a in averages])
    run3 = lambda theta,gradient: e.kernels.reference_filter(values,offsets,anchor,
        theta.reshape(3,2)*units3,units3,frozen,shape.projection_shape_on_ms,True,gradient)
    loading,projection = directions@helmert.T,directions@q.projection_q_on_hansen
    def endpoint_run(reference):
        pit = e.core.dynamic_reference_pit(values,offsets,reference[0])
        phi = np.ascontiguousarray(e.core.normalized_bspline_basis(pit,knots,3,masses))
        zero = e.kernels.endpoint_filter(phi,offsets,reference[1],masses,loading,projection,
                                        np.zeros((2,3)),np.ones((2,3)),False)
        averages = e.fixed_denominator_score_averages(zero[1],(1,5,22))
        units = np.array([[1e-2/max(float(np.median(abs(averages[h][burn:,b])))*np.linalg.norm(directions[b]),1e-12)
                           for h in range(3)] for b in range(2)])
        run = lambda theta,gradient: e.kernels.endpoint_filter(phi,offsets,reference[1],masses,
            loading,projection,theta.reshape(2,3)*units,units,gradient)
        return run,units
    run4,units4 = endpoint_run(null)
    runs,units = {'M3':run3,'M4':run4},{'M3':units3,'M4':units4}
    if 'M3' in checkpoint:
        fit = checkpoint['M3']; np.testing.assert_array_equal(units3,fit['parameter_units'])
        runs['M5'],units['M5'] = endpoint_run(run3(np.asarray(fit['unrestricted']['theta']),False))
    weights = np.ones(train); weights[:burn] = 0.
    return runs,units,weights,null


def one_replication(r,design,output,source):
    started = time.monotonic(); output = Path(output)
    for dgp,case,n in combinations(design):
        path = sim.case_path(output,r,dgp,case,n)
        if path.exists(): continue
        try:
            entry = generate(r,dgp,case,n,design,output)
        except Exception as error:
            entry = dict(dataset=path.stem,replication=r,dgp=dgp,case=case,N=n)
            sim.dump(path,dict(entry=entry,source_sha256=source,models={},tests=[],diagnostics=[],
                success=False,failures=[dict(model='generation',stage='generation',error=repr(error),traceback=traceback.format_exc())]))
            print('GENERATION FAILED',path.stem,repr(error),flush=True)
            continue
        sim.fit_case(entry,output,source)
    return dict(replication=r,seconds=time.monotonic()-started)


def records(output,requested,design):
    paths = [sim.case_path(output,r,d,c,n) for r in requested for d,c,n in combinations(design)]
    return [sim.load(p) for p in paths if p.exists()],paths


def summarize(output,requested,design,source,state,worker_failures):
    status = BASE_SUMMARIZE(output,requested,design,source,state,worker_failures)
    summary_path = output/'rejection_summary.csv'
    rows = pd.read_csv(summary_path)
    rows = rows[~((rows.case=='heterogeneous_factor')&(rows.N==1))]
    rows.to_csv(summary_path,index=False)
    status['completed_count'] = sum(all(sim.case_path(output,r,d,c,n).exists()
        for d,c,n in combinations(design)) for r in requested)
    status['N1_factor_case'] = 'not rerun; same univariate law as iid'
    sim.dump(output/'run_status.json',status)
    return status


def mute_warmup(result,burn):
    result = list(result)
    result[3] = result[3].copy(); result[3][:burn] = 0.
    return tuple(result)


def references(values,offsets,models):
    # Called only by OOS orchestration; full artificial warmup is observed past.
    refs,errors = ORIGINAL_REFERENCES(values,offsets,models)
    return {k:mute_warmup(v,CURRENT_BURN) for k,v in refs.items()},errors


def correction(values,offsets,reference,fit):
    return mute_warmup(ORIGINAL_CORRECTION(values,offsets,reference,fit),CURRENT_BURN)


def score_pair(values,offsets,training,reference,corrected,parent,child):
    # Exclude warmup from training-derived category boundaries and tail cutoffs.
    e = sim.estimator.archived
    knots,masses,*_ = sim.geometry()
    split = int(offsets[training]); start = int(offsets[CURRENT_BURN])
    days = len(offsets)-1-training
    from types import SimpleNamespace
    from contextlib import redirect_stdout
    import io
    corr = (SimpleNamespace(weights=np.tile(masses,(days,1)),contributions=np.zeros(days))
            if corrected is None else SimpleNamespace(weights=corrected[2][training:],contributions=corrected[3][training:]))
    with redirect_stdout(io.StringIO()):
        scores = e.scoring.score_oos_pair(values[split:],offsets[training:]-split,
            pd.date_range('2000-01-01',periods=days,freq='D'),values[start:split],
            e.as_reference(reference,training),corr,knots,masses,masses,parent,child)
    scores['oos_day'] = np.repeat(np.arange(days),2)
    return scores.drop(columns='date')


def wald_covariance(theta,run,weights,blocks):
    # Warmup influences the recursive derivative, but is not a meat sample.
    active = np.asarray(weights)>0
    def retained(theta,gradient):
        result = list(run(theta,gradient))
        result[3] = result[3][active]
        if gradient: result[4] = result[4][active]
        return tuple(result)
    return BASE_WALD_COVARIANCE(theta,retained,np.asarray(weights)[active],blocks)


def evaluate(record, output, fingerprint):
    global CURRENT_BURN
    CURRENT_BURN = record["entry"].get("burn", 0)
    entry = record['entry']; name = entry['dataset']
    path = output/'cases'/(name+'.json')
    if path.exists():
        cached = sim.load(path)
        assert cached['source_sha256']==fingerprint
        return cached
    started = time.monotonic()
    assert record['source_sha256']==sim.fingerprint()
    result = {'entry':entry,'source_sha256':fingerprint,'fit_source_sha256':record['source_sha256'],
        'burn':CURRENT_BURN,'fit_training_days':entry.get('fit_training_days',1886),
        'training':1886,'oos_days':642,'audit':{},'levels':[],'comparisons':[],
        'failures':[],'daily_sha256':None,'generation_failed':False}
    if 'input' not in entry:
        assert any(f['stage']=='generation' for f in record['failures'])
        result['generation_failed'] = True
        result['failures'] = [{'model':'generation','stage':'generation','error':record['failures'][0]['error']}]
        sim.dump(path,result)
        print(name,'OOS UNAVAILABLE: saved generation failure',flush=True)
        return result
    input_path = REPO/entry['input']
    assert hashlib.sha256(input_path.read_bytes()).hexdigest()==entry['input_sha256']
    with np.load(input_path) as raw:
        values,offsets = raw['values'].copy(),raw['offsets'].copy()
        training = int(raw['training'])
    result['training'],result['oos_days'] = training,len(offsets)-1-training
    models = record['models']; split = int(offsets[training]); counts = np.ones(training); counts[:CURRENT_BURN] = 0.
    failures,audits = result['failures'],result['audit']
    for model in sim.MODELS:
        if model not in models:
            old = next((f for f in record['failures'] if f['model']==model),{})
            failures.append({'model':model,'stage':'training unavailable','error':old.get('error','not fitted')})
    refs,errors = scoring.references(values,offsets,models)
    train_refs = None
    if 'M3' in errors:
        failures.append({'model':'M3','stage':'OOS reference','error':errors['M3']})
        # Confirm a full-path failure occurred after the already-fitted sample.
        train_refs,train_errors = scoring.references(values[:split],offsets[:training+1],models)
        assert not train_errors,train_errors
    if 'M3' in models:
        fitted_ref = refs.get('M3') if 'M3' in refs else train_refs['M3']
        audits['M3_training_loglik_error'] = abs(float(counts@fitted_ref[3][:training])-models['M3']['layer_log_likelihood'])
    corrected = {}
    for parent,child in (('M2','M4'),('M3','M5')):
        if child not in models:
            continue
        if parent not in refs:
            failures.append({'model':child,'stage':'OOS parent unavailable','error':'M3 OOS reference failed'})
            trained = scoring.correction(values[:split],offsets[:training+1],train_refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@trained[3])-models[child]['layer_log_likelihood'])
            continue
        try:
            corrected[child] = scoring.correction(values,offsets,refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@corrected[child][3][:training])-models[child]['layer_log_likelihood'])
        except (FloatingPointError,ValueError,OverflowError) as error:
            failures.append({'model':child,'stage':'OOS correction','error':repr(error)})
            if train_refs is None:
                train_refs,train_errors = scoring.references(values[:split],offsets[:training+1],models)
                assert not train_errors,train_errors
            trained = scoring.correction(values[:split],offsets[:training+1],train_refs[parent],models[child])
            audits[child+'_training_loglik_error'] = abs(float(counts@trained[3])-models[child]['layer_log_likelihood'])
    assert max(audits.values(),default=0.)<1e-7,audits
    if entry['replication']==0 and entry['N'] in (1,100):
        days = training+5
        prefix_refs,_ = scoring.references(values[:offsets[days]],offsets[:days+1],models)
        for model in refs:
            np.testing.assert_array_equal(prefix_refs[model][0],refs[model][0][:days])
        for parent,child in (('M2','M4'),('M3','M5')):
            if child in corrected:
                prefix = scoring.correction(values[:offsets[days]],offsets[:days+1],prefix_refs[parent],models[child])
                np.testing.assert_array_equal(prefix[0],corrected[child][0][:days])
        audits['causal_prefix_checked'] = True
    frames = []
    for parent,child in (('M2','M4'),('M3','M5')):
        if parent not in refs:
            continue
        try:
            frame = scoring.score_pair(values,offsets,training,refs[parent],corrected.get(child),parent,child)
        except (FloatingPointError,ValueError,OverflowError) as error:
            if child not in corrected:
                raise
            failures.append({'model':child,'stage':'OOS scoring','error':repr(error)})
            corrected.pop(child)
            frame = scoring.score_pair(values,offsets,training,refs[parent],None,parent,child)
        if child not in corrected:
            frame = frame[frame.model.eq(parent)]
        for model,group in frame.groupby('model'):
            if not np.isfinite(group[list(scoring.METRICS)].to_numpy()).all():
                failures.append({'model':model,'stage':'OOS scoring','error':'nonfinite OOS metric'})
                continue
            expected = refs[parent][3][training:]
            if model==child:
                expected = expected+corrected[child][3][training:]
            np.testing.assert_allclose(group.continuous_log_score,expected,rtol=1e-10,atol=1e-10)
            frames.append(group.copy())
    assert frames,'no baseline forecast scores'
    daily = pd.concat(frames,ignore_index=True)
    assert daily.groupby('model').size().eq(result['oos_days']).all()
    daily_path = output/'daily'/(name+'.csv.gz'); daily_path.parent.mkdir(parents=True,exist_ok=True)
    daily.to_csv(daily_path,index=False,compression='gzip')
    levels = daily.groupby('model')[list(scoring.METRICS)].mean()
    for parent,child in scoring.PAIRS:
        if parent in levels.index and child in levels.index:
            for metric in scoring.METRICS:
                sign = 1 if metric=='continuous_log_score' or metric.startswith('categorical_log_score_') else -1
                result['comparisons'].append({'parent':parent,'child':child,'metric':metric,
                    'improvement':float(sign*(levels.loc[child,metric]-levels.loc[parent,metric]))})
    result['levels'] = levels.reset_index().to_dict('records')
    result['daily_sha256'] = hashlib.sha256(daily_path.read_bytes()).hexdigest()
    sim.dump(path,result)
    print(name,'OOS COMPLETE',f'{time.monotonic()-started:.1f}s',len(failures),'failures',flush=True)
    return result


ORIGINAL_REFERENCES = scoring.references
ORIGINAL_CORRECTION = scoring.correction
CURRENT_BURN = 0


def configure():
    sim.CASES = CASES; sim.OUTPUT = OUTPUT; sim.PREFIX = PREFIX
    sim.fingerprint = fingerprint; sim.training_runs = training_runs
    sim.records = records; sim.summarize = summarize
    sim.estimator.wald_covariance = wald_covariance
    scoring.references = references; scoring.correction = correction; scoring.score_pair = score_pair
    oos.source_hash = fingerprint


configure()


def audit(output,requested,design,source):
    saved,paths = records(output,requested,design)
    assert len(saved)==len(paths)
    generation = []; checks = []; valid_tests = 0
    for path in sorted((output/'generation').glob('*.json')):
        item = sim.load(path)
        master_path = REPO/item['master_input']; input_path = REPO/item['input']
        assert hashlib.sha256(master_path.read_bytes()).hexdigest()==item['master_sha256']
        assert hashlib.sha256(input_path.read_bytes()).hexdigest()==item['input_sha256']
        with np.load(master_path) as raw, np.load(input_path) as obs:
            master = {k:raw[k] for k in raw.files}
            truth_audit(master,item['N'],item['dgp'])
            np.testing.assert_array_equal(obs['values'],raw['values'].ravel())
            assert np.all(np.diff(obs['offsets'])==item['N'])
            if item['case']!='iid':
                a = factor_tools.loadings(item['replication'],design.seed)[:item['N']]
                np.testing.assert_array_equal(a,raw['factor_loadings'])
                np.testing.assert_allclose(raw['factor_return_shares'].mean(axis=1),design.common_share,atol=3e-4,rtol=0)
                beta,individual = factor_tools.latent_coefficients(raw['factor_multipliers'][:,None],a)
                np.testing.assert_allclose(beta**2+individual**2,1.,atol=1e-14,rtol=0)
                np.testing.assert_array_equal(beta,raw['factor_latent_loading'])
                assert item['common_factor']['maximum_hermite_tail_bound']<1e-7
        generation.append(item)
    for record in saved:
        assert record['source_sha256']==source
        assert not(record['entry']['case']=='heterogeneous_factor' and record['entry']['N']==1)
        for diag in record['diagnostics']:
            theta = np.asarray(diag['normalized_theta']); H = np.asarray(diag['hessian'])
            tests = [t for t in record['tests'] if t['model']==diag['model']]
            assert diag['bandwidth']==0 and diag['recursion_error']<1e-9
            if diag['valid']:
                B,V = np.asarray(diag['meat']),np.asarray(diag['covariance'])
                inverse = np.linalg.inv(H)
                np.testing.assert_allclose(V,inverse@B@inverse.T,rtol=2e-9,atol=1e-10)
                for test,(_,ids) in zip(tests,sim.BLOCKS[diag['model']]):
                    ids = np.asarray(ids); value = theta[ids]@np.linalg.solve(V[np.ix_(ids,ids)],theta[ids])
                    np.testing.assert_allclose(value,test['wald'],rtol=1e-9,atol=1e-9)
                    assert abs(chi2.sf(value,len(ids))-test['wald_p'])<1e-10
                    valid_tests += 1
            else:
                assert all(not t['wald_valid'] and t['wald_p'] is None for t in tests)
            if record['entry']['replication']==requested[0] and record['entry']['N'] in (1,100):
                runs,_,weights,_ = training_runs(record['entry'],record['models'])
                run = sim.estimator.active_run(runs[diag['model']],[0,2,4]) if diag['model']=='M3' else runs[diag['model']]
                direct = run(theta,True); numerical = np.empty_like(direct[4]); numerical_H = np.empty_like(H)
                for j in range(len(theta)):
                    step = np.zeros_like(theta); step[j] = 2e-5*max(1.,abs(theta[j]))
                    numerical[:,j] = (run(theta+step,False)[3]-run(theta-step,False)[3])/(2*step[j])
                    numerical_H[:,j] = -(weights@run(theta+step,True)[4]-weights@run(theta-step,True)[4])/(2*step[j])
                numerical_H = (numerical_H+numerical_H.T)/2
                score_error = float(np.linalg.norm(numerical-direct[4])/max(1.,np.linalg.norm(direct[4])))
                H_error = float(np.linalg.norm(numerical_H-H)/max(1.,np.linalg.norm(H)))
                derivative_valid = score_error<.001 and H_error<.02
                if diag['valid']:
                    assert derivative_valid,(record['entry']['dataset'],diag['model'],score_error,H_error)
                checks.append(dict(dataset=record['entry']['dataset'],model=diag['model'],
                    score_error=score_error,H_error=H_error,derivative_valid=derivative_valid,
                    wald_valid=diag['valid'],
                    interpretation=('independent derivatives agree' if derivative_valid else
                        'unstable derivative; Wald already invalid, not certified as a reliable variance')))
    summary = pd.read_csv(output/'rejection_summary.csv'); tests = pd.read_csv(output/'tests.csv')
    assert len(summary)==len(combinations(design))*7
    for row in summary.itertuples():
        selected = tests[(tests.dgp==row.dgp)&(tests.case==row.case)&(tests.N==row.N)&(tests.model==row.model)&(tests.block==row.block)]
        valid = selected[selected.wald_valid.eq(True)]
        assert len(valid)==row.valid and int((valid.wald_p<.05).sum())==row.rejected
        assert row.valid+row.invalid+row.failed_or_not_fitted==len(requested)
    report = dict(passed=True,cases=len(saved),generation_paths=len(generation),valid_tests_checked=valid_tests,
        generation=generation,independent_derivatives=checks,likelihood='daily cross-sectional means',
        warmup_observed_but_not_in_objective=True,N1_factor_not_repeated=True,source_sha256=source,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unstable_invalid_derivative_cases=[c for c in checks if not c['derivative_valid']])
    sim.dump(output/'validation.json',report)
    return report


def audit_oos(output,results,design):
    for result in results:
        assert result['source_sha256']==fingerprint() and result['fit_source_sha256']==fingerprint()
        if result['generation_failed']: continue
        assert result['training']==design.burn+design.training and result['oos_days']==design.oos
        path = output/'daily'/(result['entry']['dataset']+'.csv.gz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==result['daily_sha256']
        daily = pd.read_csv(path); assert daily.groupby('model').size().eq(design.oos).all()
        levels = daily.groupby('model')[list(scoring.METRICS)].mean()
        for row in result['levels']:
            np.testing.assert_allclose(levels.loc[row['model']],[row[m] for m in scoring.METRICS],rtol=1e-10,atol=1e-12)
        for row in result['comparisons']:
            sign = 1 if 'log_score' in row['metric'] else -1
            np.testing.assert_allclose(sign*(levels.loc[row['child'],row['metric']]-levels.loc[row['parent'],row['metric']]),row['improvement'],rtol=1e-9,atol=1e-12)
    errors = [v for r in results for k,v in r['audit'].items() if k.endswith('error')]
    report = dict(passed=True,cases=len(results),scored_models=sum(len(r['levels']) for r in results),
        causal_prefix_checks=sum(bool(r['audit'].get('causal_prefix_checked')) for r in results),
        maximum_training_loglik_error=max(errors,default=0.),burn_excluded_from_scoring_thresholds=True)
    sim.dump(output/'validation.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications',type=int,default=100)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--stage',choices=('fit','oos','all'),default='all')
    parser.add_argument('--finalize-only',action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    design = sim.Design(); source = fingerprint(); requested = list(range(args.replications))
    original = BASE_CALIBRATION()
    metadata = dict(source_sha256=source,design=sim.asdict(design),reference_N=REFERENCE_N,
        coefficient_rule='sqrt(N/500) times SP500 M3 eta and M5 even/odd; same in both dependence settings',
        original_calibration={k:v.tolist() for k,v in original.items()},
        effective_calibration={str(n):{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in scaled_calibration(n,sim.DGPS[4]).items()} for n in design.sample_sizes},
        empirical_calibration_file=str(sim.CALIBRATION.relative_to(REPO)),
        empirical_calibration_sha256=hashlib.sha256(sim.CALIBRATION.read_bytes()).hexdigest(),
        combinations=combinations(design),N1_factor_case='same law as iid, not rerun',
        generating_and_observed_N_equal=True,shared_master_density=False,
        likelihood='sum of daily cross-sectional means; unit date weights, burn-in weights zero',
        inference='daily mean score/Hessian sandwich Wald, lag 0',
        warmup='all 500 artificial warmup observations filter each candidate; no warmup likelihood or meat',
        common_share='mean individual transformed-return explained variance 30% over actual N, calibrated daily',
        loading_distribution='iid Uniform[-0.1,1.1], fixed within replication, prefix of same 100-vector across N',
        common_case_extra_score_variance_rescaling=False,
        scaling_reference='iid theoretical N0=500, not actual correlated SP500 score variance')
    metadata = json.loads(json.dumps(metadata))
    if (output/'design.json').exists(): assert sim.load(output/'design.json')==metadata
    else: sim.dump(output/'design.json',metadata)
    if args.stage in ('fit','all'):
        failures = {}
        if not args.finalize_only:
            pending = [r for r in requested if not all(sim.case_path(output,r,d,c,n).exists() for d,c,n in combinations(design))]
            summarize(output,requested,design,source,'running',failures)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                tasks = {pool.submit(one_replication,r,design,output,source):r for r in pending}
                for future in as_completed(tasks):
                    try: print('REPLICATION COMPLETE',future.result(),flush=True)
                    except Exception: failures[str(tasks[future])] = traceback.format_exc()
                    summarize(output,requested,design,source,'running',failures)
        status = summarize(output,requested,design,source,'complete',failures)
        assert status['finished'] and not failures,status
        audit(output,requested,design,source)
        print('FIT ALL COMPLETE',status,flush=True)
    if args.stage in ('oos','all'):
        saved,_ = records(output,requested,design); results = []
        target = output/'oos'; target.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(evaluate,r,target,source) for r in saved]
            for future in as_completed(tasks):
                results.append(future.result())
                if len(results)%24==0 or len(results)==len(saved): _,status = oos.summarize(target,results,len(saved))
        validation = audit_oos(target,results,design)
        print('OOS ALL COMPLETE',status,validation,flush=True)


if __name__=='__main__':
    main()
