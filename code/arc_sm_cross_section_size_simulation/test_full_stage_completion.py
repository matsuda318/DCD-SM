"""Written output must not turn failed fits or invalid inference into success."""
import copy

import pytest

import run_symmetric_t_full_stage_m1 as experiment


def complete_case():
    blocks = {'M2': ('location', 'scale', 'joint'), 'M3': ('eta',),
              'M4': ('even', 'odd', 'joint'), 'M5': ('even', 'odd', 'joint')}
    return {
        'success': True, 'models': dict.fromkeys(experiment.MODELS, {}),
        'tests': [dict(model=model, block=block, wald_valid=True, wald=1., wald_p=.5)
                  for model, names in blocks.items() for block in names],
        'failures': [], 'oos': {
            'levels': [dict(model=model, **dict.fromkeys(experiment.e.scoring.SCORE_NAMES, 1.))
                       for model in experiment.MODELS], 'failures': []}}


def test_complete_case_and_intentional_fit_only_run():
    result = complete_case()
    assert experiment.case_validation(result) == []
    result['oos'] = {}
    assert experiment.case_validation(result, score_oos=False) == []
    assert 'missing or nonfinite OOS model scores' in experiment.case_validation(result)


@pytest.mark.parametrize('failure', ('model', 'missing_test', 'invalid_test', 'nonfinite_test',
                                     'duplicate_block', 'missing_oos', 'nonfinite_oos', 'recorded'))
def test_partial_or_invalid_case_cannot_pass(failure):
    result = copy.deepcopy(complete_case())
    if failure == 'model':
        del result['models']['M3']
    elif failure == 'missing_test':
        result['tests'].pop()
    elif failure == 'invalid_test':
        result['tests'][0]['wald_valid'] = False
    elif failure == 'nonfinite_test':
        result['tests'][0]['wald_p'] = float('nan')
    elif failure == 'duplicate_block':
        result['tests'][1] = dict(result['tests'][0])
    elif failure == 'missing_oos':
        result['oos']['levels'].pop()
    elif failure == 'nonfinite_oos':
        result['oos']['levels'][0]['continuous_log_score'] = float('inf')
    else:
        result['failures'].append({'stage': 'generation', 'error': 'failed'})
    assert experiment.case_validation(result)
