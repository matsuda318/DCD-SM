"""Audit a reoptimized M3/M5 fit without changing the published results."""
import argparse
import json
from pathlib import Path

import numpy as np

import audit_case1_symmetric_t_daily_mean_wald as audit_tools
import run_case1_training_fixed_pca_wald as experiment

OUTPUT = experiment.corrected.REPO/'work/communication_services_pc1_m3_m5_reoptimization'
KEY = 'sp500_s50_fixed_pca1_residual'


def audit(output):
    output = Path(output).resolve()
    old = json.loads((output/'before_estimation.json').read_text(encoding='utf-8'))
    new = json.loads((output/KEY/'estimation.json').read_text(encoding='utf-8'))
    experiment.corrected.archived.dump(output/'input_manifest.json', {
        'source_hash': experiment.corrected.source_hash(), 'entries': [new['dataset']]})
    original = audit_tools.original_selected_derivatives

    def precise_derivatives(result):
        attempts = []
        for step in (1e-4, 1e-5, 1e-6):
            try:
                checks = original(result, step=step)
                for check in checks:
                    check['prior_step_failures'] = attempts
                return checks
            except (AssertionError, FloatingPointError) as error:
                attempts.append({'step': step, 'reason': str(error)})
        raise AssertionError(attempts)

    audit_tools.audit_tools.selected_derivatives = precise_derivatives
    report = audit_tools.audit(output, (KEY,))
    assert not report['missing'] and report['checked'] == 1
    for model in ('M1', 'M2', 'M4'):
        assert old['models'][model] == new['models'][model]
        assert old['training_log_likelihoods'][model] == new['training_log_likelihoods'][model]
        a = next(row for row in old['oos_summary'] if row['model'] == model)
        b = next(row for row in new['oos_summary'] if row['model'] == model)
        for metric in experiment.corrected.archived.scoring.SCORE_NAMES:
            np.testing.assert_allclose(a[metric], b[metric], rtol=1e-13, atol=1e-15)
    for model in ('M3', 'M5'):
        assert new['models'][model]['wald_diagnostics']['valid']
        assert new['models'][model]['unrestricted']['projected_gradient_max'] < 2e-8
    gain = new['training_log_likelihoods']['M3'] - old['training_log_likelihoods']['M3']
    assert gain >= -1e-9
    report.update(unchanged_models_verified=['M1', 'M2', 'M4'], M3_likelihood_gain=gain)
    experiment.corrected.archived.dump(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
