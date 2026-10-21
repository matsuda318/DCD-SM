"""Retry failed fixed-PC1 fits with extra starts and unchanged acceptance rules."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import run_case1_training_fixed_pca_wald as experiment


original_maximize = experiment.corrected.archived.maximize
RECORDS = []


def maximize_with_extra_starts(objective, starts, bounds, label):
    try:
        return original_maximize(objective, starts, bounds, label)
    except RuntimeError as error:
        if 'no converged optimizer start' not in str(error):
            raise
        failure = str(error)
    extra = []
    # Restart at the finite results of the original searches, then search
    # locally around zero. The objective, bounds, tolerances and acceptance
    # checks remain those of the original estimator.
    options = {'maxiter':2000, 'ftol':1e-14, 'gtol':2e-8, 'maxls':70, 'maxcor':15}
    for start in starts:
        fit = minimize(objective, np.asarray(start, dtype=float), jac=True,
                       method='L-BFGS-B', bounds=bounds, options=options)
        if np.isfinite(fit.fun) and fit.fun < 1e50:
            extra.append(np.asarray(fit.x))
    rng = np.random.default_rng(20261019)
    for scale in (1e-6, 1e-4, 1e-2, .1):
        direction = rng.normal(size=len(bounds))
        extra.extend((scale*direction, -scale*direction))
    recovered = original_maximize(objective, extra, bounds, label)
    record = {'model':label, 'original_failure':failure,
              'additional_starts':[start.tolist() for start in extra],
              'recovered_diagnostics':recovered,
              'optimizer_options':options,
              'projected_gradient_acceptance':2e-6,
              'bounds':bounds}
    RECORDS.append(record)
    recovered['additional_start_recovery'] = True
    recovered['original_start_failure'] = failure
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output', type=Path, default=experiment.corrected.REPO/'work/empirical')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output/'input_manifest.json').read_text(encoding='utf-8'))
    if manifest['source_hash'] != experiment.corrected.source_hash():
        raise ValueError('Changed numerical source')
    keys = args.datasets.split(',')
    entries = [entry for entry in manifest['entries'] if entry['dataset'] in keys]
    if len(entries) != len(keys):
        raise ValueError('Unknown or duplicate dataset')
    experiment.corrected.archived.maximize = maximize_with_extra_starts
    path = output/'optimization_retries.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    for entry in entries:
        if (output/entry['dataset']/'estimation.json').exists():
            raise ValueError('Retry is only for incomplete panels: ' + entry['dataset'])
        if hashlib.sha256((experiment.corrected.REPO/entry['input']).read_bytes()).hexdigest() != entry['input_sha256']:
            raise ValueError('Changed input')
        RECORDS.clear()
        result = experiment.corrected.run_dataset(entry, output)
        saved.append({'dataset':entry['dataset'], 'recovered':True, 'searches':list(RECORDS),
                      'retry_source_sha256':hashlib.sha256(
                          Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()})
        experiment.corrected.archived.dump(path, saved)
        print('RECOVERED', json.dumps(result), flush=True)
    status = experiment.corrected.base.aggregate(manifest, output)
    print('STATUS', status['completed'], status['requested'], flush=True)


if __name__ == '__main__':
    main()
