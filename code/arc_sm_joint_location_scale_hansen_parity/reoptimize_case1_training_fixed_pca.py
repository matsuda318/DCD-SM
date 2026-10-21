"""Reoptimize one saved M3, then its dependent M5 and OOS comparisons.

The criterion, observations, parameterization and bounds are unchanged.
Centering and positive objective scaling improve optimizer stopping behavior.
Accepted interior candidates must also have a positive-definite Hessian.
Previously published estimates are never overwritten by this search script.
"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import time

import numpy as np
from scipy.optimize import minimize

import run_case1_training_fixed_pca_wald as experiment

ROOT = experiment.ROOT
REPO = experiment.corrected.REPO
DATASET = 'sp500_s50_fixed_pca1_residual'
PUBLISHED = REPO/'paper/results/sec5_empirical_raw_pc1'
OUTPUT = REPO/'work/communication_services_pc1_m3_m5_reoptimization'


def curvature(objective, theta):
    hessian = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        step = np.zeros(len(theta))
        step[j] = 1e-6*max(1., abs(theta[j]))
        hessian[:, j] = (objective(theta+step)[1]-objective(theta-step)[1])/(2*step[j])
    return (hessian+hessian.T)/2


def search(objective, starts, bounds, label, previous, destination):
    """Same objective with explicit first- and second-order diagnostics."""
    if label != 'M3':
        raise ValueError(label)
    center = np.asarray(previous['models']['M3']['unrestricted']['active_theta'])
    reference = float(objective(center)[0])
    eigenvalues, vectors = np.linalg.eigh(curvature(objective, center))
    seeds = [center, np.zeros_like(center), *starts]
    for j in np.flatnonzero(eigenvalues < 0):
        for radius in (.1, 1., 5.):
            seeds.extend(center+sign*radius*vectors[:, j] for sign in (-1, 1))
    rng = np.random.default_rng(20261020)
    seeds.extend(radius*rng.normal(size=len(center)) for radius in (.01, .1, 1., 5.) for _ in range(2))
    records, candidates = [], []
    scale = 1e6

    def scaled(theta):
        if np.any(abs(theta) >= 500.):
            return 1e100, np.zeros_like(theta)
        value, gradient = objective(theta)
        if value > 1e50:
            return 1e100, np.zeros_like(theta)
        return (value-reference)*scale, gradient*scale

    for number, seed in enumerate(seeds):
        if objective(seed)[0] > 1e50:
            continue
        started = time.monotonic()
        def stop_slow(theta):
            return time.monotonic()-started > 25.
        fit = minimize(scaled, seed, jac=True,
                       hess=lambda theta:scale*curvature(objective, theta),
                       method='trust-exact', callback=stop_slow,
                       options={'maxiter':120, 'gtol':1e-5,
                                'initial_trust_radius':.001, 'max_trust_radius':.05})
        value, gradient = objective(fit.x)
        eig = np.linalg.eigvalsh(curvature(objective, fit.x)) if value < 1e50 else np.zeros(len(center))
        residual = float(abs(gradient).max())
        interior = bool(np.all(abs(fit.x) < 499.9))
        positive = bool(eig[0] > 1e-10*max(eig[-1], 1e-6))
        accepted = bool(value <= reference and residual <= 2e-8 and interior and positive)
        record = dict(start=number, theta=fit.x.tolist(), mean_nll=float(value),
                      projected_gradient_max=residual, hessian_eigenvalues=eig.tolist(),
                      success=bool(fit.success), accepted=accepted, message=str(fit.message),
                      iterations=int(fit.nit), evaluations=int(fit.nfev))
        records.append(record)
        experiment.corrected.archived.dump(destination/'M3_search.json',
            {'old_mean_nll':reference, 'objective_scale':scale, 'bounds':bounds,
             'old_mean_hessian_eigenvalues':eigenvalues.tolist(), 'starts':records})
        print('M3 SEARCH', json.dumps(record), flush=True)
        if accepted:
            candidates.append(record)
        # Several independent searches reaching an interior minimum suffice.
        if len(candidates) >= 4 and number >= 12:
            break
    if not candidates:
        raise RuntimeError('No M3 candidate passed gradient and positive-curvature checks')
    best = min(candidates, key=lambda r:r['mean_nll'])
    return {**best, 'total_starts':len(records), 'successful_starts':len(candidates),
            'largest_successful_objective_gap':max(r['mean_nll'] for r in candidates)-best['mean_nll'],
            'starts':records, 'objective_scale':scale, 'reoptimization':'communication_services_pc1_m3_m5'}


def run(output, initial_estimate=None, manifest_path=None):
    output = Path(output).resolve()
    baseline = PUBLISHED/'communication_services_pc1_m3_m5_reoptimization/before_estimation.json'
    original_path = (Path(initial_estimate) if initial_estimate else
                     baseline if baseline.exists() else PUBLISHED/'estimates'/DATASET/'estimation.json')
    previous = json.loads(original_path.read_text(encoding='utf-8'))
    entry = dict(previous['dataset'])
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        selected = [row for row in manifest['entries'] if row['dataset'] == DATASET]
        if len(selected) != 1:
            raise ValueError('Expected one Communication Services PC1 residual input')
        entry = selected[0]
    relative = PureWindowsPath(entry['input'])
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Expected a relative prepared input path')
    entry = dict(entry, input=Path(*relative.parts).as_posix())
    if previous['input_sha256'] != entry['input_sha256']:
        raise ValueError('Initial fit and prepared input hashes differ')
    previous['dataset'] = entry
    if hashlib.sha256((REPO/entry['input']).read_bytes()).hexdigest() != entry['input_sha256']:
        raise ValueError('Prepared input changed after the initial fit')
    folder = output/DATASET
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'estimation.json').exists():
        provenance = json.loads((output/'reoptimization_provenance.json').read_text(encoding='utf-8'))
        current = json.loads((folder/'estimation.json').read_text(encoding='utf-8'))
        if (provenance['original_sha256'] != hashlib.sha256(original_path.read_bytes()).hexdigest()
                or current['input_sha256'] != entry['input_sha256']
                or current['source_hash'] != experiment.corrected.source_hash()
                or provenance['script_sha256'] != hashlib.sha256(
                    Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()):
            raise ValueError('Changed initial fit, input or code. Use a new reoptimization directory.')
        print('ALREADY COMPLETE', str(folder), flush=True)
        return
    experiment.corrected.archived.dump(output/'before_estimation.json', previous)
    checkpoint_path = folder/'fit_checkpoint.json'
    if not checkpoint_path.exists():
        experiment.corrected.archived.dump(checkpoint_path, {
            'source_hash':experiment.corrected.source_hash(),
            'input_sha256':entry['input_sha256'],
            **{model:previous['models'][model] for model in ('M1','M2','M4')}})
    original_maximize = experiment.corrected.archived.maximize
    original_configure = experiment.corrected.base.configure
    original_fit = experiment.corrected.base.fit_layer_wald

    def maximize(objective, starts, bounds, label):
        if label == 'M3':
            return search(objective, starts, bounds, label, previous, output)
        return original_maximize(objective, starts, bounds, label)

    def fit_layer(run, units, counts, blocks, warm, label):
        warm = np.asarray(previous['models'][label]['coefficients'])
        return original_fit(run, units, counts, blocks, warm, label)

    def configure(folder, inference):
        original_configure(folder, inference)
        experiment.corrected.archived.fit_layer = fit_layer

    experiment.corrected.base.configure = configure
    experiment.corrected.archived.maximize = maximize
    result = experiment.corrected.run_dataset(entry, output)
    experiment.corrected.archived.dump(output/'reoptimization_provenance.json', {
        'dataset':DATASET, 'original_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'unchanged_models':['M1','M2','M4'], 'refitted_models':['M3','M5'],
        'input_sha256':entry['input_sha256'], 'criterion_unchanged':True})
    print('COMPLETE', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--initial-estimate', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if bool(args.initial_estimate) != bool(args.manifest):
        parser.error('--initial-estimate and --manifest must be supplied together')
    run(args.output, args.initial_estimate, args.manifest)


if __name__ == '__main__':
    main()
