"""Audit equal-date empirical fits using matching Hessian date weights."""
import argparse
import json
from pathlib import Path
import numpy as np

import run_case1_symmetric_t_daily_mean_wald as corrected
import audit_case1_symmetric_t_wald as audit_tools

original_training_runs = audit_tools.saved_filters.training_runs
original_selected_derivatives = audit_tools.selected_derivatives


def training_runs(result):
    runs,counts = original_training_runs(result)
    return runs,np.ones_like(counts,dtype=float)


audit_tools.saved_filters.training_runs = training_runs


def selected_derivatives(result):
    try:
        return original_selected_derivatives(result)
    except AssertionError as coarse_error:
        # A steep recursive filter needs a smaller perturbation. Verify at
        # two successive step sizes without changing the saved estimator/H.
        checks = original_selected_derivatives(result, step=1e-4)
        finer = original_selected_derivatives(result, step=5e-5)
        for check, fine in zip(checks, finer):
            check['coarse_step_failure'] = str(coarse_error)
            check['smaller_step_confirmation'] = fine
        return checks


audit_tools.selected_derivatives = selected_derivatives


def audit(output, selected=()):
    report = audit_tools.audit(output, selected)
    output = Path(output)
    for check in report['checks']:
        result = json.loads((output/check['dataset']/'estimation.json').read_text(encoding='utf-8'))
        assert result['schema'] == corrected.SCHEMA
        assert 'mean log densities' in result['specification']['likelihood']
        assert result['audit']['M1_likelihood_reproduction_error'] < 1e-8
        runs, weights = training_runs(result)
        np.testing.assert_array_equal(weights, 1.)
        with np.load(corrected.REPO/result['dataset']['input']) as raw:
            offsets = raw['offsets'][:len(weights)+1]
            static_ll, _ = corrected.static_daily_mean(raw['values'][:offsets[-1]],
                offsets, np.asarray(result['models']['M1']['state']))
        np.testing.assert_allclose(static_ll, result['models']['M1']['log_likelihood'],
                                   rtol=1e-11, atol=1e-9)
        likelihoods = {'M1':float(static_ll)}
        check['static_daily_mean_likelihood_error'] = abs(
            static_ll-result['models']['M1']['log_likelihood'])
        records = []
        for model in ('M2','M3','M4','M5'):
            fit = result['models'][model]
            filtered = runs[model](np.asarray(fit['unrestricted']['theta']), True)
            ll = float(filtered[3].sum())
            np.testing.assert_allclose(ll, fit['layer_log_likelihood'], rtol=1e-11, atol=1e-9)
            likelihoods[model] = ll if model in ('M2','M3') else likelihoods[
                'M2' if model=='M4' else 'M3']+ll
            score = filtered[4][:,np.asarray(fit['active_indices'])]
            centered = score-score.mean(axis=0, keepdims=True)
            diagnostic = fit['wald_diagnostics']
            np.testing.assert_allclose(centered.T@centered, diagnostic['meat'],
                                       rtol=1e-10, atol=1e-9)
            records.append({'model':model, 'daily_mean_likelihood_error':
                abs(ll-fit['layer_log_likelihood']), 'daily_mean_score_meat_checked':True})
        for model, value in likelihoods.items():
            np.testing.assert_allclose(value, result['training_log_likelihoods'][model],
                                       rtol=1e-11, atol=1e-9)
        check['equal_date_criterion'] = records
    report['likelihood'] = 'sum of daily cross-sectional mean log densities; equal date weights'
    corrected.archived.dump(output/'audit.json', report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=corrected.OUTPUT)
    parser.add_argument('--selected',default='')
    args = parser.parse_args()
    print(json.dumps(audit(args.output,tuple(filter(None,args.selected.split(',')))),indent=2))
