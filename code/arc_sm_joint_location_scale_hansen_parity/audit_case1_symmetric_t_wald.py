"""Audit fixed-lambda empirical results and selected likelihood derivatives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import run_case1_symmetric_t_wald as experiment

sys.path.insert(0, str(experiment.REPO/'arc_sm_cross_section_size_simulation'))
import run_full_stage_copula_lr_wald as saved_filters


def selected_derivatives(result, step=1e-3):
    runs, counts = saved_filters.training_runs(result)
    checks = []
    for model in ('M2', 'M3', 'M4', 'M5'):
        fit = result['models'][model]
        indices = np.asarray(fit['active_indices'])
        theta = np.asarray(fit['unrestricted']['active_theta'])
        run = experiment.active_run(runs[model], indices)
        analytic = run(theta, True)[4]
        if model in ('M2', 'M3'):
            np.testing.assert_array_equal(run(theta, False)[0][:, 3], 0.)
        numerical = np.empty_like(analytic)
        base = run(theta, False)[3]
        hessian = np.empty((len(theta), len(theta)))
        for i in range(len(theta)):
            delta = np.zeros_like(theta)
            delta[i] = step
            plus = run(theta+delta, False)[3]
            minus = run(theta-delta, False)[3]
            numerical[:, i] = (plus-minus)/(2*step)
            hessian[i, i] = -counts @ ((plus-2*base+minus)/step**2)
            for j in range(i):
                other = np.zeros_like(theta)
                other[j] = step
                cross = (run(theta+delta+other, False)[3]-run(theta+delta-other, False)[3]
                         -run(theta-delta+other, False)[3]+run(theta-delta-other, False)[3])/(4*step**2)
                hessian[i, j] = hessian[j, i] = -counts @ cross
        score_error = float(np.linalg.norm(analytic-numerical)/max(np.linalg.norm(numerical), 1e-12))
        archived_hessian = np.asarray(fit['wald_diagnostics']['hessian'])
        hessian_error = float(np.linalg.norm(hessian-archived_hessian)/max(np.linalg.norm(archived_hessian), 1e-12))
        assert score_error < 1e-4, (result['dataset']['dataset'], model, score_error)
        assert hessian_error < .02, (result['dataset']['dataset'], model, hessian_error)
        checks.append({'model': model, 'score_relative_error': score_error,
                       'likelihood_hessian_relative_error': hessian_error,
                       'finite_difference_step':float(step)})
    return checks


def audit(output, selected=()):
    output = Path(output)
    manifest = json.loads((output/'input_manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_hash'] == experiment.source_hash()
    checks, missing = [], []
    for entry in manifest['entries']:
        path = output/entry['dataset']/'estimation.json'
        if not path.exists():
            missing.append(entry['dataset'])
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['source_hash'] == manifest['source_hash']
        assert result['input_sha256'] == entry['input_sha256']
        assert hashlib.sha256((experiment.REPO/entry['input']).read_bytes()).hexdigest() == entry['input_sha256']
        assert result['models']['M1']['lambda'] == 0.
        np.testing.assert_array_equal(np.asarray(result['models']['M3']['coefficients'])[:, 1], 0.)
        assert result['models']['M3']['free_dimension'] == 3
        scores = pd.read_csv(output/entry['dataset']/'daily_oos_scores.csv')
        assert set(scores.model) == {'M1','M2','M3','M4','M5'}
        assert len(scores) == 5*entry['oos_days']
        assert np.isfinite(scores[list(experiment.archived.scoring.SCORE_NAMES)].to_numpy()).all()
        for model in ('M2','M3','M4','M5'):
            fit = result['models'][model]
            theta = np.asarray(fit['unrestricted']['active_theta'])
            diagnostic = fit['wald_diagnostics']
            if entry['role'] != 'main':
                assert not fit['wald_tests']
                continue
            assert diagnostic['bandwidth'] == 0
            if not diagnostic['valid']:
                assert not any(t['wald_valid'] for t in fit['wald_tests'])
                continue
            H = np.asarray(diagnostic['hessian'])
            B = np.asarray(diagnostic['meat'])
            V = np.asarray(diagnostic['covariance'])
            inverse = np.linalg.inv(H)
            np.testing.assert_allclose(V, inverse@B@inverse.T, rtol=1e-10, atol=1e-12)
            blocks = {'eta': [0,1,2]} if model == 'M3' else {
                'joint': list(range(6)), 'location': [0,2,4], 'scale': [1,3,5],
                'even': [0,1,2], 'odd': [3,4,5]}
            for test in fit['wald_tests']:
                ids = np.asarray(blocks[test['block']])
                value = theta[ids] @ np.linalg.solve(V[np.ix_(ids,ids)], theta[ids])
                np.testing.assert_allclose(value, test['wald'], rtol=1e-10, atol=1e-10)
        record = {'dataset': entry['dataset'], 'checks_passed': True}
        if entry['dataset'] in selected:
            record['independent_derivatives'] = selected_derivatives(result)
        checks.append(record)
    report = {'all_available_checks_passed': True, 'requested': len(manifest['entries']),
              'checked': len(checks), 'missing': missing, 'checks': checks}
    experiment.archived.dump(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=experiment.OUTPUT)
    parser.add_argument('--selected', default='')
    args = parser.parse_args()
    print(json.dumps(audit(args.output, tuple(filter(None,args.selected.split(',')))), indent=2))
