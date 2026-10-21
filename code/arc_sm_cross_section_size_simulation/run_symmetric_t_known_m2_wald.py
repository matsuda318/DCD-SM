"""Five symmetric-t DGPs; known M2; fitted M3 -> M5; observation-sum Wald(0)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr, ndtri, roots_legendre
from scipy.optimize import brentq
from scipy.stats import chi2

import run_full_stage_copula_lr_wald as legacy
import run_case1_symmetric_t_wald as estimator

ROOT, REPO = legacy.ROOT, legacy.REPO
OUTPUT = ROOT/'output_symmetric_t_known_m2_wald'
CALIBRATION = estimator.OUTPUT/'sp500/estimation.json'
DGPS = ('case0_static', 'case1_eta', 'case2_even', 'case3_eta_even', 'case4_eta_even_odd')
CASES = ('iid', 'shared_factor')
MODELS = ('M3', 'M4', 'M5')
BLOCKS = {'M3': (('eta', (0,1,2)),),
          'M4': (('even', (0,1,2)), ('odd', (3,4,5)), ('joint', tuple(range(6)))),
          'M5': (('even', (0,1,2)), ('odd', (3,4,5)), ('joint', tuple(range(6))))}
PREFIX = 'arc_sm_symmetric_t_known_m2_wald'
dump = legacy.dump


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=None)
def weighted_hermite_quadrature(order,degree):
    """Weight the recurrence itself to avoid 0*inf at large nodes/degrees."""
    nodes,weights,_ = legacy.copula.normal_quadrature(order)
    basis = np.empty((degree+1,order))
    basis[0],basis[1] = weights,nodes*weights
    for k in range(1,degree):
        basis[k+1] = (nodes*basis[k]-np.sqrt(k)*basis[k-1])/np.sqrt(k+1)
    return nodes,weights,basis


def hermite_moments(quantile,order,degree):
    if degree==40:
        return legacy.copula.hermite_moments(quantile,order)
    nodes,weights,basis = weighted_hermite_quadrature(order,degree)
    values = quantile(ndtr(nodes))
    mean = float(weights@values)
    centered = values-mean
    variance = float(weights@(centered*centered))
    if not np.isfinite(variance) or variance<=0: raise FloatingPointError('invalid common-factor marginal variance')
    coefficients = basis[1:]@centered
    return coefficients**2/variance,mean,variance


def composite_normal_moments(quantile,order,degree,breakpoints=()):
    # Split the finite normal domain and all PIT clipping boundaries.  Its
    # omitted tails have negligible mass for the bounded sampler quantile.
    edges = np.unique(np.r_[np.linspace(-12.,12.,49),
                           [x for x in breakpoints if np.isfinite(x) and -12<x<12]])
    nodes,weights = roots_legendre(order)
    left,right = edges[:-1,None],edges[1:,None]
    z = ((left+right)/2+(right-left)/2*nodes).ravel()
    w = ((right-left)/2*weights*np.exp(-z.reshape(-1,order)**2/2)/np.sqrt(2*np.pi)).ravel()
    values = quantile(ndtr(z))
    mean = float(w@values); centered = values-mean
    variance = float(w@(centered*centered))
    if not np.isfinite(variance) or variance<=0: raise FloatingPointError('invalid composite marginal variance')
    previous,current = w,z*w
    coefficients = [float(current@centered)]
    for k in range(1,degree):
        previous,current = current,(z*current-np.sqrt(k)*previous)/np.sqrt(k+1)
        coefficients.append(float(current@centered))
    return np.asarray(coefficients)**2/variance,mean,variance


def composite_common_share(quantile,target,breakpoints):
    correlation = legacy.copula.correlation_from_moments
    for degree in (40,80,160,320):
        previous = None; stable = 0
        for order in (16,32,64,128,256,512):
            energy,mean,variance = composite_normal_moments(quantile,order,degree,breakpoints)
            if previous is not None:
                checked = correlation(previous,energy)
                stable = stable+1 if abs(checked-target)<=legacy.copula.SHARE_TOLERANCE/2 else 0
                if stable>=2:
                    bound = previous**(degree+1)
                    if bound<1e-7:
                        return previous,{'latent_rho':previous,'checked_common_share':checked,
                            'share_error':checked-target,'quadrature_order':order,'hermite_degree':degree,
                            'hermite_tail_bound':bound,'integration_method':'composite_gauss_legendre',
                            'marginal_mean_standardized':mean,'marginal_variance_standardized':variance}
                    break
            if correlation(.95,energy)<=target: previous = None; continue
            previous = brentq(lambda rho:correlation(rho,energy)-target,0.,.95,xtol=1e-11)
    raise FloatingPointError('composite common-share quadrature did not converge within unchanged tolerance')


def calibrate_common_share(quantile,target=.30,breakpoints=()):
    """Keep the previous tolerance; extend Hermite degree only if necessary."""
    correlation = legacy.copula.correlation_from_moments
    for degree in (40,80,160,320):
        previous = None
        stable_checks = 0
        for order in (*legacy.copula.QUADRATURE_ORDERS,6144,12288):
            energy,mean,variance = hermite_moments(quantile,order,degree)
            if previous is not None:
                checked = correlation(previous,energy)
                if abs(checked-target)<=legacy.copula.SHARE_TOLERANCE/2:
                    stable_checks += 1
                    tail_bound = previous**(degree+1)
                    if tail_bound<1e-7 and stable_checks>=2:
                        return previous,{'latent_rho':previous,'checked_common_share':checked,
                            'share_error':checked-target,'quadrature_order':order,'hermite_degree':degree,
                            'hermite_tail_bound':tail_bound,'integration_method':'gauss_hermite',
                            'marginal_mean_standardized':mean,
                            'marginal_variance_standardized':variance}
                    # Integration has converged; resolve the truncation bound at higher degree.
                    if tail_bound>=1e-7: break
                else: stable_checks = 0
            if correlation(.95,energy)<=target:
                previous = None
                continue
            previous = brentq(lambda rho:correlation(rho,energy)-target,0.,.95,xtol=1e-11)
        else:
            # Raising expansion degree cannot cure oscillation of variance
            # integration. Switch quadrature while preserving this same G_t.
            return composite_common_share(quantile,target,breakpoints)
    return composite_common_share(quantile,target,breakpoints)


class CommonFactorSampler(legacy.copula.CommonFactorSampler):
    """Same copula/CDF/random draws; adaptive integration for low-df t states."""
    def __call__(self,count,q_state,masses,helmert,component_inverse,rng,*,reference_state):
        logits = np.log(masses)+helmert@q_state
        weights = np.exp(logits-logsumexp(logits))
        mixture = np.maximum.accumulate(self.component_cdf@weights)
        mixture[0],mixture[-1] = 0.,1.
        def reference_u(v):
            return np.clip(np.interp(v,mixture,self.grid),1e-12,1-1e-12)
        def quantile(v):
            return legacy.sim.core.hansen_ppf(reference_u(v),reference_state[2:])
        key = (reference_state[2:].tobytes(),q_state.tobytes())
        if key==self._cache_key: rho,audit = self._cache_calibration
        else:
            clip_probability = np.interp([1e-12,1-1e-12],self.grid,mixture)
            breakpoints = ndtri(np.clip(clip_probability,1e-300,1-np.finfo(float).eps))
            rho,audit = calibrate_common_share(quantile,self.target,breakpoints)
            self._cache_key,self._cache_calibration = key,(rho,audit)
        self.audit.append(dict(audit,generated_day=len(self.audit),eta_state=float(reference_state[2]),
                               q_norm=float(np.linalg.norm(q_state))))
        common = rng.standard_normal(); individual = rng.standard_normal(count)
        return reference_u(ndtr(np.sqrt(rho)*common+np.sqrt(1-rho)*individual))


@dataclass(frozen=True)
class Design:
    training: int = 1886
    oos: int = 642
    burn: int = 500
    seed: int = 20261006
    common_share: float = .30
    sample_sizes: tuple[int, ...] = (1,20,50,100)


def fingerprint():
    digest = hashlib.sha256()
    paths = [Path(__file__), CALIBRATION, Path(legacy.__file__), Path(legacy.sim.__file__),
             Path(legacy.copula.__file__), ROOT/'test_symmetric_t_known_m2_wald.py',
             REPO/'dcd_sp500_daily_composite_score_arch/estimate_score_har.py']
    for path in sorted(set(paths)):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    digest.update(estimator.source_hash().encode())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def calibration(dgp=DGPS[4]):
    saved = load(CALIBRATION)
    anchor = np.array([0.,0.,saved['models']['M1']['state'][2],0.])
    eta = np.asarray(saved['models']['M3']['coefficients'])[:,0].copy()
    endpoint = np.asarray(saved['models']['M5']['coefficients']).copy()
    if dgp not in (DGPS[1],DGPS[3],DGPS[4]):
        eta[:] = 0.
    if dgp not in (DGPS[2],DGPS[3],DGPS[4]):
        endpoint[0] = 0.
    if dgp != DGPS[4]:
        endpoint[1] = 0.
    return {'anchor':anchor, 'eta_kappa':eta, 'endpoint_kappa':endpoint}


@lru_cache(maxsize=1)
def geometry():
    s = legacy.sim
    knots = s.core.make_clamped_knots(s.base.BREAKPOINTS,3)
    masses = s.core.bspline_masses(knots,3)
    helmert = s.core.helmert_submatrix(len(masses))
    directions = np.vstack(s.endpoint.endpoint_directions(helmert))
    anchor = calibration()['anchor']
    shape = s.shape.shape_geometry(anchor)
    q = s.scoring.corrected_tangent_geometry(anchor,masses,knots,masses,helmert,(0,1),'known symmetric t M2')
    return knots,masses,helmert,directions,anchor,shape,q


@lru_cache(maxsize=1)
def component_sampler():
    knots,masses,*_ = geometry()
    return legacy.sim.prepare_component_sampler(knots,masses)


def generate_master(replication,dgp,case,design):
    """True state is driven by all 100 names; observed nested N never alters it."""
    s = legacy.sim
    knots,masses,helmert,directions,anchor,shape,q = geometry()
    grid,_,cdf,inverse = component_sampler()
    calibrated = calibration(dgp)
    rng = np.random.default_rng(design.seed+replication+100000*DGPS.index(dgp))
    sampler = (legacy.IndependentNormalSampler(grid,cdf) if case=='iid' else
               CommonFactorSampler(grid,cdf,design.common_share))
    total = design.burn+design.training+design.oos
    values = np.empty((total,100))
    states = np.empty((total,4))
    moving_states = np.empty((total,2))
    drivers = np.empty((total,3))
    eta_history = np.zeros((22,1))
    endpoint_history = np.zeros((22,2))
    eta = anchor[2]
    moving = np.zeros(2)
    loading = directions@helmert.T
    projection = directions@q.projection_q_on_hansen
    for t in range(total):
        state = anchor.copy(); state[2] = eta
        q_state = moving@directions
        u = sampler(100,q_state,masses,helmert,inverse,rng,reference_state=state)
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
        eta_history = np.vstack((eta_history[1:],[[eta_driver]]))
        endpoint_history = np.vstack((endpoint_history[1:],endpoint_driver))
        eta = anchor[2]+sum(calibrated['eta_kappa'][h]*eta_history[-w:,0].sum()/w
                            for h,w in enumerate((1,5,22)))
        moving = sum(calibrated['endpoint_kappa'][:,h]*endpoint_history[-w:].sum(axis=0)/w
                     for h,w in enumerate((1,5,22)))
        if not np.isfinite(eta) or abs(eta)>20 or not np.isfinite(moving).all() or abs(q_state).max()>100:
            raise FloatingPointError(f'unstable true recursion: day={t}, eta={eta}, endpoint={moving}')
    keep = slice(design.burn,None)
    master = {'values':values[keep], 'reference_states':states[keep],
              'endpoint_states':moving_states[keep], 'drivers':drivers[keep],
              'presample_drivers':drivers[max(0,design.burn-22):design.burn]}
    factor = None
    if case=='shared_factor':
        audit = pd.DataFrame(sampler.audit)
        factor = {'days':len(audit), 'maximum_common_share_error':float(audit.share_error.abs().max()),
                  'latent_rho_min':float(audit.latent_rho.min()), 'latent_rho_max':float(audit.latent_rho.max()),
                  'maximum_hermite_degree':int(audit.hermite_degree.max()),
                  'maximum_quadrature_order':int(audit.quadrature_order.max()),
                  'maximum_hermite_tail_bound':float(audit.hermite_tail_bound.max()),
                  'composite_quadrature_days':int(audit.integration_method.eq('composite_gauss_legendre').sum())}
        assert factor['maximum_common_share_error'] <= legacy.copula.SHARE_TOLERANCE
        assert factor['maximum_hermite_tail_bound'] < 1e-7
    return master,factor


def truth_audit(master,dgp):
    calibrated = calibration(dgp)
    states,moving = master['reference_states'],master['endpoint_states']
    assert np.isfinite(master['values']).all()
    np.testing.assert_array_equal(states[:,[0,1,3]],0.)
    history = np.zeros((22,3))
    prior = master['presample_drivers']
    if len(prior): history[-len(prior):] = prior
    error = 0.
    for t,driver in enumerate(master['drivers']):
        averages = np.array([history[-w:].sum(axis=0)/w for w in (1,5,22)])
        expected_eta = calibrated['anchor'][2]+calibrated['eta_kappa']@averages[:,0]
        expected_endpoint = np.sum(calibrated['endpoint_kappa']*averages[:,1:].T,axis=1)
        error = max(error,abs(states[t,2]-expected_eta),float(abs(moving[t]-expected_endpoint).max()))
        history = np.vstack((history[1:],driver))
    assert error < 1e-10
    if not np.any(calibrated['eta_kappa']):
        np.testing.assert_array_equal(states[:,2],calibrated['anchor'][2])
    for b in range(2):
        if not np.any(calibrated['endpoint_kappa'][b]): np.testing.assert_array_equal(moving[:,b],0.)
    return {'true_recursion_maximum_error':float(error),'eta_state_std':float(states[:,2].std()),
            'even_state_std':float(moving[:,0].std()),'odd_state_std':float(moving[:,1].std()),
            'maximum_endpoint_state':float(abs(moving).max()),
            'location_log_variance_lambda_maximum':float(abs(states[:,[0,1,3]]).max())}


def generate(replication,dgp,case,design,output):
    cache = REPO/'data'/output.name/f'replication_{replication:04d}'
    cache.mkdir(parents=True,exist_ok=True)
    path = cache/f'{dgp}_{case}_master.npz'
    audit_path = output/'generation'/f'rep{replication:04d}_{dgp}_{case}.json'
    if path.exists() and audit_path.exists():
        audit = load(audit_path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['master_sha256']
        with np.load(path) as saved: values = saved['values']
    else:
        master,factor = generate_master(replication,dgp,case,design)
        audit = {'replication':replication,'dgp':dgp,'case':case,
                 'seed':design.seed+replication+100000*DGPS.index(dgp),
                 'master_input':str(path.relative_to(REPO)), 'truth':truth_audit(master,dgp)}
        if factor: audit['common_factor'] = factor
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**master); temporary.replace(path)
        audit['master_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        dump(audit_path,audit)
        values = master['values']
    entries = []
    for n in design.sample_sizes:
        target = cache/f'{dgp}_{case}_N{n}.npz'
        if not target.exists():
            np.savez_compressed(target,values=np.ascontiguousarray(values[:,:n]).ravel(),
                offsets=np.arange(len(values)+1,dtype=np.int64)*n,training=design.training)
        entries.append({'dataset':f'rep{replication:04d}_{dgp}_{case}_N{n}',
            'replication':replication,'dgp':dgp,'case':case,'N':n,
            'input':str(target.relative_to(REPO)), 'input_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    return entries


def training_runs(entry,checkpoint):
    """Only training observations and the known static anchor enter estimation."""
    e = estimator.archived
    with np.load(REPO/entry['input']) as raw:
        train = int(raw['training']); offsets = raw['offsets'][:train+1].copy()
        values = raw['values'][:offsets[-1]].copy()
    knots,masses,helmert,directions,anchor,shape,q = geometry()
    frozen = np.tile(anchor,(train,1))
    null = e.kernels.reference_filter(values,offsets,anchor,np.zeros((3,2)),np.ones((3,2)),
        frozen,shape.projection_shape_on_ms,True,False)
    averages = e.fixed_denominator_score_averages(null[2],(1,5,22))
    units3 = np.array([[1e-2/max(float(np.median(abs(a[:,j]))),1e-12) for j in range(2)] for a in averages])
    run3 = lambda theta,gradient: e.kernels.reference_filter(values,offsets,anchor,
        theta.reshape(3,2)*units3,units3,frozen,shape.projection_shape_on_ms,True,gradient)
    loading,projection = directions@helmert.T,directions@q.projection_q_on_hansen
    def endpoint_run(reference):
        pit = e.core.dynamic_reference_pit(values,offsets,reference[0])
        phi = np.ascontiguousarray(e.core.normalized_bspline_basis(pit,knots,3,masses))
        zero = e.kernels.endpoint_filter(phi,offsets,reference[1],masses,loading,projection,
                                         np.zeros((2,3)),np.ones((2,3)),False)
        averages = e.fixed_denominator_score_averages(zero[1],(1,5,22))
        units = np.array([[1e-2/max(float(np.median(abs(averages[h][:,b])))*np.linalg.norm(directions[b]),1e-12)
                           for h in range(3)] for b in range(2)])
        run = lambda theta,gradient: e.kernels.endpoint_filter(phi,offsets,reference[1],masses,
            loading,projection,theta.reshape(2,3)*units,units,gradient)
        return run,units
    run4,units4 = endpoint_run(null)
    runs,units = {'M3':run3,'M4':run4},{'M3':units3,'M4':units4}
    if 'M3' in checkpoint:
        fit3 = checkpoint['M3']
        np.testing.assert_array_equal(units3,fit3['parameter_units'])
        f3 = run3(np.asarray(fit3['unrestricted']['theta']),False)
        runs['M5'],units['M5'] = endpoint_run(f3)
    return runs,units,np.diff(offsets),null


def fit_case(entry,output,source):
    path = output/'cases'/(entry['dataset']+'.json')
    if path.exists():
        result = load(path); assert result['source_sha256']==source
        return result
    started = time.monotonic()
    checkpoint_path = output/'fits'/entry['dataset']/'checkpoint.json'
    checkpoint = load(checkpoint_path) if checkpoint_path.exists() else {}
    signature = {'source_sha256':source,'input_sha256':entry['input_sha256']}
    if checkpoint: assert all(checkpoint[k]==v for k,v in signature.items())
    checkpoint.update(signature)
    record = {'entry':entry,'source_sha256':source,'models':{},'tests':[],'diagnostics':[],
              'failures':[],'success':False}
    runs,units,counts,null = training_runs(entry,checkpoint)
    for model in MODELS:
        try:
            if model=='M5':
                if 'M3' not in checkpoint: raise RuntimeError('M5 requires successful fitted M3')
                runs,units,counts,null = training_runs(entry,checkpoint)
            if model not in checkpoint:
                fit,_ = estimator.fit_layer_wald(runs[model],units[model],counts,
                    [('even',[0,1,2]),('odd',[3,4,5])],None,model)
                checkpoint[model] = fit; dump(checkpoint_path,checkpoint)
            fit = checkpoint[model]
            full_theta = np.asarray(fit['unrestricted']['theta'])
            filtered = runs[model](full_theta,False)
            if model=='M3':
                np.testing.assert_array_equal(filtered[0][:,[0,1,3]],0.)
                np.testing.assert_array_equal(np.asarray(fit['coefficients'])[:,1],0.)
                recursion = estimator.archived.recursion_error(filtered[0][:,2:],filtered[2],
                    calibration()['anchor'][2:],np.asarray(fit['coefficients']))
            else:
                recursion = estimator.archived.recursion_error(filtered[0],filtered[1],np.zeros(2),
                    np.asarray(fit['coefficients']),True)
            assert recursion < 1e-9
            assert np.isclose(counts@filtered[3],fit['layer_log_likelihood'],atol=1e-7,rtol=1e-12)
            record['models'][model] = fit
            record['tests'].extend({'model':model,'bandwidth':0,**test} for test in fit['wald_tests'])
            record['diagnostics'].append({'model':model,'normalized_theta':fit['unrestricted']['active_theta'],
                'parameter_units':fit['parameter_units'],'recursion_error':recursion,**fit['wald_diagnostics']})
        except Exception as error:
            record['failures'].append({'model':model,'stage':'estimation/Wald','error':repr(error),
                                      'traceback':traceback.format_exc()})
    record['success'] = len(record['models'])==3
    record['seconds'] = time.monotonic()-started
    dump(path,record)
    print(entry['dataset'],'COMPLETE',record['success'],f"{record['seconds']:.1f}s",flush=True)
    return record


def case_path(output,r,d,c,n):
    return output/'cases'/f'rep{r:04d}_{d}_{c}_N{n}.json'


def one_replication(r,design,output,source):
    started = time.monotonic(); output = Path(output)
    for dgp in DGPS:
        for case in CASES:
            if all(case_path(output,r,dgp,case,n).exists() for n in design.sample_sizes): continue
            try:
                entries = generate(r,dgp,case,design,output)
            except Exception as error:
                for n in design.sample_sizes:
                    path = case_path(output,r,dgp,case,n)
                    if not path.exists():
                        entry = {'dataset':path.stem,'replication':r,'dgp':dgp,'case':case,'N':n}
                        dump(path,{'entry':entry,'source_sha256':source,'models':{},'tests':[],
                            'diagnostics':[],'success':False,'failures':[{'model':'generation','stage':'generation',
                            'error':repr(error),'traceback':traceback.format_exc()}]})
                print('GENERATION FAILED',r,dgp,case,repr(error),flush=True)
                continue
            for entry in entries: fit_case(entry,output,source)
    return {'replication':r,'seconds':time.monotonic()-started}


def records(output,requested,design):
    paths = [case_path(output,r,d,c,n) for r in requested for d in DGPS for c in CASES for n in design.sample_sizes]
    return [load(path) for path in paths if path.exists()],paths


def summarize(output,requested,design,source,state,worker_failures):
    saved,paths = records(output,requested,design)
    rows,params,errors = [],[],[]
    for record in saved:
        assert record['source_sha256']==source
        common = {k:record['entry'][k] for k in ('dataset','replication','dgp','case','N')}
        rows.extend({**common,**test} for test in record['tests'])
        for model,fit in record['models'].items():
            params.append({**common,'model':model,'coefficients':json.dumps(fit['coefficients']),
                'parameter_units':json.dumps(fit['parameter_units']),
                'normalized_theta':json.dumps(fit['unrestricted']['active_theta']),
                'training_log_likelihood':fit['layer_log_likelihood'],
                'projected_gradient_max':fit['unrestricted']['projected_gradient_max']})
        errors.extend({**common,**{k:v for k,v in item.items() if k!='traceback'}} for item in record['failures'])
        errors.extend({**common,'model':diag['model'],'stage':'Wald variance','error':diag['invalid_reason']}
                      for diag in record['diagnostics'] if not diag['valid'])
    columns = ['dataset','replication','dgp','case','N','model','bandwidth','block','df','wald_valid','wald','wald_p']
    frame = pd.DataFrame(rows,columns=columns)
    summary = []
    for dgp in DGPS:
        for case in CASES:
            for n in design.sample_sizes:
                for model in MODELS:
                    for block,_ in BLOCKS[model]:
                        subset = frame[(frame.dgp==dgp)&(frame.case==case)&(frame.N==n)&(frame.model==model)&(frame.block==block)]
                        valid = subset[subset.wald_valid.eq(True)]
                        count,rejected = len(valid),int((valid.wald_p<.05).sum())
                        low,high = legacy.wilson(rejected,count)
                        summary.append({'dgp':dgp,'case':case,'N':n,'model':model,'block':block,
                            'requested':len(requested),'fits_succeeded':len(subset),'valid':count,
                            'invalid':len(subset)-count,'failed_or_not_fitted':len(requested)-len(subset),
                            'rejected':rejected,'rate':rejected/count if count else None,
                            'wilson_low':low,'wilson_high':high})
    frame.to_csv(output/'tests.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'rejection_summary.csv',index=False)
    pd.DataFrame(params).to_csv(output/'estimated_parameters.csv',index=False)
    pd.DataFrame(errors,columns=['dataset','replication','dgp','case','N','model','stage','error']).to_csv(output/'failures.csv',index=False)
    status = {'state':state,'finished':len(saved)==len(paths),'requested_count':len(requested),
        'completed_count':sum(all(case_path(output,r,d,c,n).exists() for d in DGPS for c in CASES for n in design.sample_sizes) for r in requested),
        'completed_cases':len(saved),'requested_cases':len(paths),'successful_cases':sum(r['success'] for r in saved),
        'successful_models':len(params),'invalid_wald_tests':int(frame.wald_valid.eq(False).sum()),
        'worker_failures':worker_failures,'source_sha256':source,
        'checked_jst':datetime.now(timezone(timedelta(hours=9))).isoformat()}
    dump(output/'run_status.json',status)
    return status


def audit(output,requested,design,source):
    saved,paths = records(output,requested,design)
    assert len(saved)==len(paths) and fingerprint()==source
    checked,derivatives = 0,[]
    generation = [load(path) for path in sorted((output/'generation').glob('*.json'))]
    for item in generation:
        path = REPO/item['master_input']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['master_sha256']
        with np.load(path) as raw: truth_audit({k:raw[k] for k in raw.files},item['dgp'])
        if item['case']=='shared_factor':
            assert item['common_factor']['maximum_common_share_error']<=legacy.copula.SHARE_TOLERANCE
            assert item['common_factor']['maximum_hermite_tail_bound']<1e-7
    for record in saved:
        assert record['source_sha256']==source
        entry = record['entry']
        if record['models']:
            path = REPO/entry['input']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['input_sha256']
            with np.load(path) as raw, np.load(path.parent/f"{entry['dgp']}_{entry['case']}_master.npz") as master:
                np.testing.assert_array_equal(raw['values'].reshape(-1,entry['N']),master['values'][:,:entry['N']])
        for diag in record['diagnostics']:
            model = diag['model']; theta = np.asarray(diag['normalized_theta'])
            H = np.asarray(diag['hessian'])
            tests = [t for t in record['tests'] if t['model']==model]
            assert len(tests)==len(BLOCKS[model]) and diag['bandwidth']==0
            assert diag['recursion_error']<1e-9
            if diag['valid']:
                B,V = np.asarray(diag['meat']),np.asarray(diag['covariance'])
                inverse = np.linalg.inv(H)
                np.testing.assert_allclose(V,inverse@B@inverse.T,rtol=2e-9,atol=1e-10)
                for test,(_,ids) in zip(tests,BLOCKS[model]):
                    ids = np.asarray(ids)
                    value = theta[ids]@np.linalg.solve(V[np.ix_(ids,ids)],theta[ids])
                    np.testing.assert_allclose(value,test['wald'],rtol=1e-9,atol=1e-9)
                    assert abs(chi2.sf(value,len(ids))-test['wald_p'])<1e-10
                    checked += 1
            else:
                assert all(not test['wald_valid'] and test['wald_p'] is None for test in tests)
            if entry['replication']==requested[0] and entry['N'] in (1,100):
                runs,_,counts,_ = training_runs(entry,record['models'])
                run = estimator.active_run(runs[model],[0,2,4]) if model=='M3' else runs[model]
                direct = run(theta,True)
                numerical = np.empty_like(direct[4]); expected_H = np.empty_like(H)
                for j in range(len(theta)):
                    delta = np.zeros(len(theta)); delta[j] = 2e-5*max(1.,abs(theta[j]))
                    numerical[:,j] = (run(theta+delta,False)[3]-run(theta-delta,False)[3])/(2*delta[j])
                    expected_H[:,j] = -(counts@run(theta+delta,True)[4]-counts@run(theta-delta,True)[4])/(2*delta[j])
                expected_H = (expected_H+expected_H.T)/2
                score_error = float(np.linalg.norm(numerical-direct[4])/max(1.,np.linalg.norm(direct[4])))
                H_error = float(np.linalg.norm(expected_H-H)/max(1.,np.linalg.norm(H)))
                assert score_error<.001 and H_error<.02,(entry,model,score_error,H_error)
                derivatives.append({'dataset':entry['dataset'],'model':model,'score_relative_error':score_error,'H_relative_error':H_error})
    summary,tests = pd.read_csv(output/'rejection_summary.csv'),pd.read_csv(output/'tests.csv')
    assert len(summary)==len(DGPS)*len(CASES)*len(design.sample_sizes)*sum(len(b) for b in BLOCKS.values())
    for row in summary.itertuples():
        subset = tests[(tests.dgp==row.dgp)&(tests.case==row.case)&(tests.N==row.N)&(tests.model==row.model)&(tests.block==row.block)]
        valid = subset[subset.wald_valid.eq(True)]
        assert len(valid)==row.valid and int((valid.wald_p<.05).sum())==row.rejected
        assert row.valid+row.invalid+row.failed_or_not_fitted==len(requested)
    validation = {'all_checks_passed':True,'source_sha256':source,'cases':len(saved),
                  'valid_tests_checked':checked,'generation_audits':generation,'independent_numerical_checks':derivatives}
    dump(output/'validation.json',validation)
    return validation


def report(output,requested,design,validation):
    saved,_ = records(output,requested,design)
    summary = pd.read_csv(output/'rejection_summary.csv')
    meta,status = load(output/'design.json'),load(output/'run_status.json')
    folder = REPO/'paper'
    lines = ['# 対称t・既知M2：M3–M5のHACなしWaldシミュレーション','',
        f"作成：{datetime.now(timezone(timedelta(hours=9))).isoformat()}。各設定{len(requested)}反復。",'',
        'M2までknown：location=0、scale=1、静的η=SP500対称t実証のM1値、λ=0。M1・M2は推定しない。M3はηの3 HAR係数、M4はknown M2をreferenceとしてeven・odd各3係数、M5は各反復で推定したM3を固定したreferenceとしてeven・odd各3係数を推定する。','',
        '| ケース | η | even | odd |','|---|---|---|---|',
        '| 0 | 静的 | 静的 | 静的 |','| 1 | 持続 | 静的 | 静的 |','| 2 | 静的 | 持続 | 静的 |',
        '| 3 | 持続 | 持続 | 静的 |','| 4 | 持続 | 持続 | 持続 |','',
        f'横断iid／共通因子30%、N={list(design.sample_sizes)}、学習{design.training}日、burn-in {design.burn}日、将来{design.oos}日。将来データは生成のみで、推定・検定に使わない。','',
        '共通因子は正規コピュラ：Zᵢₜ=√ρₜ Fₜ+√(1−ρₜ)εᵢₜ、vᵢₜ=Φ(Zᵢₜ)、yᵢₜ=Gₜ⁻¹(vᵢₜ)。F・εは独立標準正規。ρₜは変換後リターンの共通分散寄与率30%になるよう日別に較正する。latent Gaussianのρを一律0.3と置く意味ではない。各反復・DGP・依存設定ごとに100銘柄masterでtrue densityを駆動し、先頭Nの入れ子標本を観測する。Nを変えてもtrue pathは変えない。小Nではtrue pathを駆動する残りの観測は推定側から見えない。','',
        '真のη係数は今回の対称t実証SP500のM3、even・odd係数はM5からそのまま採用する。旧skew-t実験の√(100/491)倍率は今回適用せず、N・依存設定に応じた係数再調整もしない。','',
        '```json',json.dumps(meta['calibration'],ensure_ascii=False,indent=2),'```','',
        'HAR(1,5,22)の日別横断平均driverを用い、location・log variance方向だけの固定Fisher射影、identity static SM、ゼロpresampleを維持する。DGPは500日burn-in後を保存する一方、推定は観測開始時のdriver履歴をゼロとして開始する。','',
        '目的はΣₜΣⱼ log fₜ。H・日別scoreにも実際のNを掛け、V=H⁻¹[Σₜ(Sₜ−S̄)(Sₜ−S̄)′]H⁻ᵀ。HACラグ0、追加のN・T倍率なし。3係数ブロックはχ²₃、M4・M5 jointはχ²₆、5%水準。M3のλは推定せず常に0。実証と同じ推定・Wald関数、5初期値・境界・収束条件を使う。LRやbootstrapは実行しない。','',
        'ケース0はsize診断。M3の純粋なη powerはcase1、M4のeven powerはcase2で評価できる。even/oddを含むDGPでM3が棄却されてもηだけへの帰属を意味しない。ηが動くDGPでM4が棄却されてもeven/oddだけへの帰属を意味しない。M5は上流M3を推定して固定する段階推定手順全体の診断。数学的識別と有限標本の成分帰属を区別する。','',
        '各セルは棄却数／有効検定数（%）。失敗・無効分散は分母から除き、summary.csvにrequested・valid・invalid・failedとWilson 95%区間を保存する。検定不能を非棄却と扱わない。M4はM3の成否と独立に推定する。','']
    def cell(dgp,case,n,model,block):
        row = summary[(summary.dgp==dgp)&(summary.case==case)&(summary.N==n)&(summary.model==model)&(summary.block==block)].iloc[0]
        return f'{row.rejected}/{row.valid}（{100*row.rate:.1f}%）' if row.valid else '0/0（検定不能）'
    for i,dgp in enumerate(DGPS):
        lines += [f'## ケース{i}：{dgp}','','| 設定 | N | M3 η | M4 even | M4 odd | M5 even | M5 odd | M4 joint | M5 joint |',
                  '|---|---|---|---|---|---|---|---|---|']
        for case in CASES:
            for n in design.sample_sizes:
                cells = [cell(dgp,case,n,m,b) for m,b in (('M3','eta'),('M4','even'),('M4','odd'),('M5','even'),('M5','odd'),('M4','joint'),('M5','joint'))]
                lines.append('| '+('iid' if case=='iid' else '共通30%')+f' | {n} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 数値監査・保存','',
        f"全{status['requested_cases']}ケースが記録済み。3モデルすべての推定成功は{status['successful_cases']}ケース、成功モデル{status['successful_models']}件、無効Wald {status['invalid_wald_tests']}件。",'',
        'データhash・Nの入れ子・true再帰・location/scale/λ固定・共通30%較正・推定再帰・H/B/VとWald・p値・集計分母を監査。最初の反復のN=1・100で日別尤度の有限差分scoreと異なるstepのHessianを照合する。無効分散は擬似逆行列で救済しない。','',
        '```powershell','python arc_sm_cross_section_size_simulation/run_symmetric_t_known_m2_wald.py --replications 100 --workers 4',
        'python arc_sm_cross_section_size_simulation/run_symmetric_t_known_m2_wald.py --finalize-only','```','']
    files = {'report':folder/(PREFIX+'.md')}
    for name,original in (('summary','rejection_summary.csv'),('tests','tests.csv'),('parameters','estimated_parameters.csv'),('failures','failures.csv')):
        files[name] = folder/(PREFIX+'_'+name+'.csv')
        files[name].write_bytes((output/original).read_bytes())
    files['diagnostics'] = folder/(PREFIX+'_diagnostics.jsonl.gz')
    with files['diagnostics'].open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
        for record in saved:
            for diag in record['diagnostics']:
                compressed.write((json.dumps({**record['entry'],**diag},allow_nan=False)+'\n').encode())
    files['audit'] = folder/(PREFIX+'_audit.json')
    dump(files['audit'],{'design':meta,'status':status,'validation':validation})
    lines += ['- ['+path.name+']('+path.name+')' for name,path in files.items() if name!='report']
    files['report'].write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return files


def manifest(design,source):
    calibrated = calibration()
    return {'design':asdict(design),'source_sha256':source,'dgps':DGPS,'cases':CASES,
        'calibration':{key:val.tolist() for key,val in calibrated.items()},
        'empirical_calibration_file':CALIBRATION.relative_to(REPO).as_posix(),
        'empirical_calibration_sha256':hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
        'estimator_source_sha256':estimator.source_hash(),'coefficient_multiplier':1.,
        'master_N':100,'M2':'known location 0, scale 1, symmetric t static eta',
        'M3':'3 eta coefficients; lambda 0','M4':'known M2 plus even/odd',
        'M5':'fitted M3 frozen plus even/odd','criterion':'individual observation sum',
        'inference':'daily cluster sandwich Wald, bandwidth 0',
        'common_factor':'30% transformed-return variance contribution, calibrated Gaussian copula',
        'numeric_calibration':{'share_tolerance':legacy.copula.SHARE_TOLERANCE,
            'hermite_degrees':[40,80,160,320],'tail_bound_tolerance':1e-7,
            'quadrature_orders':[*legacy.copula.QUADRATURE_ORDERS,6144,12288],
            'fallback':'composite Gauss-Legendre on split normal domain [-12,12]',
            'composite_orders':[16,32,64,128,256,512]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--training',type=int,default=1886)
    parser.add_argument('--oos',type=int,default=642)
    parser.add_argument('--burn',type=int,default=500)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--finalize-only',action='store_true')
    args = parser.parse_args()
    assert args.replications>0 and args.workers>0 and args.training>22 and args.oos>0 and args.burn>=0
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    source = fingerprint(); design = Design(training=args.training,oos=args.oos,burn=args.burn)
    metadata = json.loads(json.dumps(manifest(design,source)))
    if (output/'design.json').exists(): assert load(output/'design.json')==metadata,'changed design/source; use new output'
    else: dump(output/'design.json',metadata)
    requested = list(range(args.replications)); worker_failures = {}
    if not args.finalize_only:
        pending = [r for r in requested if not all(case_path(output,r,d,c,n).exists() for d in DGPS for c in CASES for n in design.sample_sizes)]
        summarize(output,requested,design,source,'running',worker_failures)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_replication,r,design,output,source):r for r in pending}
            for future in as_completed(futures):
                try: print('REPLICATION COMPLETE',future.result(),flush=True)
                except Exception: worker_failures[str(futures[future])] = traceback.format_exc()
                summarize(output,requested,design,source,'running',worker_failures)
    status = summarize(output,requested,design,source,'complete',worker_failures)
    assert status['finished'],'incomplete experiment; saved fits permit resume'
    validation = audit(output,requested,design,source)
    # A pilot is audited but never overwrites production paper results.
    if args.replications==100 and design==Design() and output==OUTPUT.resolve():
        files = report(output,requested,design,validation)
        dump(output/'report_status.json',{'audited':True,'files':[str(p.relative_to(REPO)) for p in files.values()]})
    print('ALL COMPLETE',json.dumps(status),flush=True)


if __name__=='__main__': main()
