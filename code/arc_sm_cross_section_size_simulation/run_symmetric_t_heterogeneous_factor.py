"""Known-M2 experiment with fixed U[-.1,1.1] loadings and 30% mean shares.

Run in a separate Python process: configuring the imported experiment changes
only this process's module globals, never the previous source or saved results.
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
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr, ndtri

import run_symmetric_t_known_m2_wald as sim
import run_symmetric_t_case0_oos as scoring
import run_symmetric_t_cases1_4_oos as oos

ROOT, REPO = sim.ROOT, sim.REPO
PREFIX = 'arc_sm_symmetric_t_heterogeneous_factor'
OUTPUT = ROOT / ('output_symmetric_t_heterogeneous_factor')
CASE = 'heterogeneous_factor'
BASE_SOURCE = sim.fingerprint()
ORIGINAL_GENERATOR = sim.generate_master
ACTIVE_LOADINGS = None
LAST_SAMPLER = None


@sim.lru_cache(maxsize=1)
def fingerprint():
    # Preserve the frozen numerical run's hash after an audit-only roundoff
    # correction. The AST assertion forbids any generation/fit/OOS/CLI change.
    import ast
    frozen = ROOT/'snapshots/symmetric_t_heterogeneous_factor_numerical_source.txt'
    def numerical_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name in ('fingerprint','audit_generation'))]
        return ast.dump(tree, include_attributes=False)
    assert numerical_ast(Path(__file__)) == numerical_ast(frozen), 'numerical code changed after frozen run'
    digest = hashlib.sha256(BASE_SOURCE.encode())
    for name, path in ((Path(__file__).name, frozen),
                       (Path(scoring.__file__).name, Path(scoring.__file__)),
                       (Path(oos.__file__).name, Path(oos.__file__)),
                       ('test_symmetric_t_heterogeneous_factor.py', ROOT/'test_symmetric_t_heterogeneous_factor.py')):
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def loadings(replication, seed=20261006):
    # A separate stream leaves the previous common/idiosyncratic innovations
    # unchanged. All five DGPs and nested N use this same 100-name vector.
    rng = np.random.default_rng(np.random.SeedSequence([seed, replication, 417]))
    return rng.uniform(-.1, 1.1, 100)


def latent_coefficients(multiplier, a):
    common = multiplier * np.asarray(a)
    denominator = np.hypot(1., common)
    return common / denominator, 1. / denominator


def return_shares(multiplier, a, energy):
    beta, _ = latent_coefficients(multiplier, a)
    return np.polynomial.polynomial.polyval(beta**2, np.r_[0., energy])


def solve_multiplier(a, energy, target):
    high = 1.
    while return_shares(high, a, energy).mean() <= target:
        high *= 2.
        if high > 1e6:
            raise FloatingPointError('mean common share could not be bracketed')
    return brentq(lambda c: float(return_shares(c, a, energy).mean())-target,
                  0., high, xtol=1e-11)


def calibrate(quantile, a, target=.30, breakpoints=()):
    """Hermite coefficients identify Var(E[Y_i|F])/Var(Y_i) per name.

    The sign disappears from individual explained variances, but remains in
    cross-name covariances. Calibrate the empirical mean over the 100 loadings.
    Tail bound is the mean omitted Hermite energy bound, not a correlation.
    """
    tolerance = sim.legacy.copula.SHARE_TOLERANCE
    for method in ('gauss_hermite', 'composite_gauss_legendre'):
        orders = ((*sim.legacy.copula.QUADRATURE_ORDERS, 6144, 12288)
                  if method == 'gauss_hermite' else (16,32,64,128,256,512))
        for degree in (40,80,160,320):
            previous, stable = None, 0
            for order in orders:
                if method == 'gauss_hermite':
                    energy, mean, variance = sim.hermite_moments(quantile, order, degree)
                else:
                    energy, mean, variance = sim.composite_normal_moments(
                        quantile, order, degree, breakpoints)
                if previous is not None:
                    shares = return_shares(previous, a, energy)
                    checked = float(shares.mean())
                    stable = stable+1 if abs(checked-target) <= tolerance/2 else 0
                    beta, _ = latent_coefficients(previous, a)
                    bound = float(np.mean((beta**2)**(degree+1)))
                    if stable >= 2 and bound < 1e-7:
                        # Original audit's latent_rho fields now mean the mean
                        # individual squared Gaussian loading; labels documented.
                        return previous, dict(
                            multiplier=previous, latent_rho=float(np.mean(beta**2)),
                            checked_common_share=checked, share_error=checked-target,
                            individual_return_shares=shares.tolist(),
                            minimum_return_share=float(shares.min()),
                            maximum_return_share=float(shares.max()),
                            minimum_signed_loading=float(beta.min()),
                            maximum_signed_loading=float(beta.max()),
                            quadrature_order=order, hermite_degree=degree,
                            hermite_tail_bound=bound, integration_method=method,
                            marginal_mean_standardized=mean,
                            marginal_variance_standardized=variance,
                        )
                    if stable >= 2 and bound >= 1e-7:
                        break
                previous = solve_multiplier(a, energy, target)
            else:
                # A higher Hermite degree cannot cure oscillatory integration.
                if method == 'gauss_hermite':
                    break
    raise FloatingPointError('heterogeneous mean common-share integration did not converge')


class HeterogeneousSampler(sim.legacy.copula.CommonFactorSampler):
    def __init__(self, grid, component_cdf, target):
        global LAST_SAMPLER
        super().__init__(grid, component_cdf, target)
        assert ACTIVE_LOADINGS is not None
        self.loadings = ACTIVE_LOADINGS.copy()
        LAST_SAMPLER = self

    def __call__(self, count, q_state, masses, helmert, component_inverse, rng,
                 *, reference_state):
        assert count == len(self.loadings) == 100
        logits = np.log(masses)+helmert@q_state
        weights = np.exp(logits-logsumexp(logits))
        mixture = np.maximum.accumulate(self.component_cdf@weights)
        mixture[0], mixture[-1] = 0., 1.

        def reference_u(v):
            return np.clip(np.interp(v, mixture, self.grid), 1e-12, 1-1e-12)

        def quantile(v):
            return sim.legacy.sim.core.hansen_ppf(reference_u(v), reference_state[2:])

        key = (reference_state[2:].tobytes(), q_state.tobytes())
        if key == self._cache_key:
            multiplier, audit = self._cache_calibration
        else:
            probability = np.interp([1e-12, 1-1e-12], self.grid, mixture)
            breakpoints = ndtri(np.clip(probability, 1e-300, 1-np.finfo(float).eps))
            multiplier, audit = calibrate(quantile, self.loadings, self.target, breakpoints)
            self._cache_key, self._cache_calibration = key, (multiplier, audit)
        self.audit.append(dict(audit, generated_day=len(self.audit),
                               eta_state=float(reference_state[2]),
                               q_norm=float(np.linalg.norm(q_state))))
        beta, individual = latent_coefficients(multiplier, self.loadings)
        common = rng.standard_normal()
        epsilon = rng.standard_normal(count)
        return reference_u(ndtr(beta*common+individual*epsilon))


def generate_master(replication, dgp, case, design):
    global ACTIVE_LOADINGS
    assert case == CASE
    ACTIVE_LOADINGS = loadings(replication, design.seed)
    master, factor = ORIGINAL_GENERATOR(replication, dgp, 'shared_factor', design)
    master['factor_loadings'] = ACTIVE_LOADINGS.copy()
    daily = LAST_SAMPLER.audit
    shares = np.asarray([r['individual_return_shares'] for r in daily])
    multipliers = np.asarray([r['multiplier'] for r in daily])
    master['factor_multipliers'] = multipliers[design.burn:]
    master['factor_return_shares'] = shares[design.burn:]
    master['factor_latent_loading'] = np.asarray([
        latent_coefficients(c, ACTIVE_LOADINGS)[0] for c in multipliers[design.burn:]])
    factor.update(
        loadings=ACTIVE_LOADINGS.tolist(), loading_distribution='iid Uniform[-0.1,1.1]',
        loading_stream=[design.seed, replication, 417],
        negative_loadings=int((ACTIVE_LOADINGS<0).sum()),
        multiplier_min=float(multipliers.min()), multiplier_max=float(multipliers.max()),
        minimum_individual_return_share=float(shares.min()),
        maximum_individual_return_share=float(shares.max()),
        latent_rho_definition='mean individual squared Gaussian loading',
        nested_N_mean_return_shares={str(n):float(shares[design.burn:,:n].mean())
                                    for n in design.sample_sizes},
        target_definition='mean individual transformed-return common variance shares over master N=100',
    )
    return master, factor


def configure():
    sim.CASES = (CASE,)
    sim.OUTPUT = OUTPUT
    sim.PREFIX = PREFIX
    sim.fingerprint = fingerprint
    sim.CommonFactorSampler = HeterogeneousSampler
    sim.generate_master = generate_master
    oos.source_hash = fingerprint


configure()


def audit_generation(output):
    rows = []
    maximum_roundoff = 0.
    for path in sorted((output/'generation').glob('*.json')):
        item = sim.load(path)
        assert item['case'] == CASE
        factor = item['common_factor']
        assert factor['maximum_common_share_error'] <= sim.legacy.copula.SHARE_TOLERANCE
        assert factor['maximum_hermite_tail_bound'] < 1e-7
        with np.load(REPO/item['master_input']) as raw:
            a = loadings(item['replication'])
            np.testing.assert_array_equal(a, raw['factor_loadings'])
            assert np.all((a>=-.1)&(a<=1.1))
            np.testing.assert_allclose(raw['factor_return_shares'].mean(axis=1), .3, atol=3e-4, rtol=0)
            beta, individual = latent_coefficients(raw['factor_multipliers'][:,None], a)
            maximum_roundoff = max(maximum_roundoff, float(abs(beta**2+individual**2-1).max()))
            np.testing.assert_allclose(beta**2+individual**2, 1., atol=1e-14, rtol=0)
            np.testing.assert_array_equal(beta, raw['factor_latent_loading'])
            for i, value in enumerate(a):
                rows.append(dict(replication=item['replication'], dgp=item['dgp'],
                                 asset=i, a=value,
                                 mean_return_share=float(raw['factor_return_shares'][:,i].mean())))
    pd.DataFrame(rows).to_csv(output/'loading_diagnostics.csv', index=False)
    result = dict(passed=True, generation_paths=len(rows)//100, loading_rows=len(rows),
                  common_share_target=.3, mean_share_tolerance=3e-4,
                  marginal_standard_normal_identity_checked=True,
                  marginal_variance_identity_atol=1e-14,
                  maximum_marginal_variance_roundoff=maximum_roundoff,
                  numerical_code_matches_frozen_snapshot=True,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                  audit_roundoff_correction='atol 5e-16 -> 1e-14; no generation/fit/OOS code changed')
    sim.dump(output/'loading_validation.json', result)
    return result


def audit_oos(output, results, expected):
    assert len(results) == expected
    for result in results:
        assert result['source_sha256'] == fingerprint()
        assert result['fit_source_sha256'] == fingerprint()
        assert result['training'] == 1886 and result['oos_days'] == 642
        if result['generation_failed']:
            assert not result['levels'] and not result['comparisons']
            continue
        path = output/'daily'/(result['entry']['dataset']+'.csv.gz')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result['daily_sha256']
        daily = pd.read_csv(path)
        assert daily.groupby('model').size().eq(642).all()
        levels = daily.groupby('model')[list(scoring.METRICS)].mean()
        for row in result['levels']:
            np.testing.assert_allclose(levels.loc[row['model']], [row[m] for m in scoring.METRICS],
                                       rtol=1e-10, atol=1e-12)
        for row in result['comparisons']:
            sign = 1 if 'log_score' in row['metric'] else -1
            gain = sign*(levels.loc[row['child'],row['metric']]-levels.loc[row['parent'],row['metric']])
            np.testing.assert_allclose(gain, row['improvement'], rtol=1e-9, atol=1e-12)
    errors = [v for r in results for k,v in r['audit'].items() if k.endswith('error')]
    validation = dict(passed=True, cases=len(results),
                      causal_prefix_checks=sum(bool(r['audit'].get('causal_prefix_checked')) for r in results),
                      maximum_training_loglik_error=max(errors, default=0.),
                      scored_models=sum(len(r['levels']) for r in results))
    sim.dump(output/'validation.json', validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications', type=int, default=100)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--stage', choices=('fit','oos','all'), default='all')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    design = sim.Design(); source = fingerprint()
    requested = list(range(args.replications))
    metadata = sim.manifest(design, source)
    metadata.update(
        base_source_sha256=BASE_SOURCE, cases=[CASE],
        common_factor='fixed heterogeneous signed loadings, mean individual return shares 30%',
        loading_distribution='iid Uniform[-0.1,1.1], once per replication, same vector in all five DGPs',
        latent_equation='Z_it=(c_t*a_i*F_t+epsilon_it)/sqrt(1+c_t^2*a_i^2)',
        share_equation='mean_i Var(E[G_t^-1(Phi(Z_it))|F_t,past])/Var(G_t^-1(Phi(Z_it))|past)=0.30',
        share_master_N=100, nested_subsamples_recalibrated=False,
        loading_seed_sequence='[20261006, replication, 417]',
        existing_controls_reused=True,
    )
    metadata = json.loads(json.dumps(metadata))
    if (output/'design.json').exists():
        assert sim.load(output/'design.json') == metadata, 'changed source/design: choose new output'
    else:
        sim.dump(output/'design.json', metadata)
    if args.stage in ('fit','all'):
        worker_failures = {}
        if not args.finalize_only:
            pending = [r for r in requested if not all(sim.case_path(output,r,d,CASE,n).exists()
                       for d in sim.DGPS for n in design.sample_sizes)]
            sim.summarize(output, requested, design, source, 'running', worker_failures)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(sim.one_replication,r,design,output,source):r for r in pending}
                for future in as_completed(futures):
                    try:
                        print('REPLICATION COMPLETE', future.result(), flush=True)
                    except Exception:
                        worker_failures[str(futures[future])] = traceback.format_exc()
                    sim.summarize(output, requested, design, source, 'running', worker_failures)
        status = sim.summarize(output, requested, design, source, 'complete', worker_failures)
        assert status['finished'] and not worker_failures, status
        sim.audit(output, requested, design, source)
        audit_generation(output)
        print('FIT ALL COMPLETE', status, flush=True)
    if args.stage in ('oos','all'):
        records, paths = sim.records(output, requested, design)
        assert len(records) == len(paths)
        oos_output = output/'oos'; oos_output.mkdir(exist_ok=True)
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(oos.evaluate,record,oos_output,source) for record in records]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results)%24 == 0 or len(results)==len(records):
                    _, status = oos.summarize(oos_output, results, len(records))
        validation = audit_oos(oos_output, results, len(records))
        print('OOS ALL COMPLETE', status, validation, flush=True)


if __name__ == '__main__':
    main()
