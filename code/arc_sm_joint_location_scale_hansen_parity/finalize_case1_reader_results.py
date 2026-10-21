"""Reoptimize, audit and aggregate reader fits, without editing frozen results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

import run_case1_symmetric_t_daily_mean_wald as corrected

ROOT = Path(__file__).resolve().parent
KEY = 'sp500_s50_fixed_pca1_residual'
RESULT_FILES = ('estimation.json', 'daily_oos_scores.csv', 'wald_tests.csv',
                'oos_levels.csv', 'oos_comparisons.csv')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT/script), *map(str, args)], check=True)


def finalize(output):
    output = Path(output).resolve()
    if output.is_relative_to(corrected.REPO/'paper/results'):
        raise ValueError('Reader results must not overwrite the frozen paper results')
    manifest = load(output/'input_manifest.json')
    if manifest['source_hash'] != corrected.source_hash():
        raise ValueError('Reader fits were produced by different numerical sources')
    if any(entry['dataset'] == KEY for entry in manifest['entries']):
        search = output/'reoptimization'
        applied = search/'applied.json'
        destination = output/KEY
        sources = {name: digest(ROOT/name) for name in (
            Path(__file__).name, 'reoptimize_case1_training_fixed_pca.py',
            'audit_case1_training_fixed_pca_reoptimization.py')}
        if applied.exists():
            record = load(applied)
            if record['sources'] != sources:
                raise ValueError('Reoptimization code changed. Use a new output directory.')
            for name in RESULT_FILES:
                if digest(destination/name) != record['files'][name]:
                    raise ValueError('Finalized result changed. Use a new output directory.')
            if record['input_sha256'] != load(destination/'estimation.json')['input_sha256']:
                raise ValueError('Finalized input changed')
        else:
            initial = search/'initial'
            initial.mkdir(parents=True, exist_ok=True)
            for name in RESULT_FILES:
                backup = initial/name
                if not backup.exists():
                    shutil.copyfile(destination/name, backup)
                elif digest(backup) != digest(destination/name):
                    refined = search/KEY/name
                    if not refined.exists() or digest(refined) != digest(destination/name):
                        raise ValueError('Initial results changed during reoptimization')
            run('reoptimize_case1_training_fixed_pca.py', '--output', search,
                '--initial-estimate', initial/'estimation.json',
                '--manifest', output/'input_manifest.json')
            run('audit_case1_training_fixed_pca_reoptimization.py', '--output', search)
            audit = load(search/'audit.json')
            if not audit['all_available_checks_passed'] or audit['missing']:
                raise RuntimeError('Reoptimization audit failed')
            for name in RESULT_FILES:
                shutil.copyfile(search/KEY/name, destination/name)
            corrected.archived.dump(applied, {
                'input_sha256': load(destination/'estimation.json')['input_sha256'],
                'files': {name: digest(destination/name) for name in RESULT_FILES},
                'sources': sources,
                'audit': 'audit.json', 'unchanged_models': ['M1', 'M2', 'M4'],
                'refitted_models': ['M3', 'M5']})
    run('audit_case1_symmetric_t_daily_mean_wald.py', '--output', output)
    audit = load(output/'audit.json')
    status = corrected.base.aggregate(manifest, output)
    valid = (audit['all_available_checks_passed'] and not audit['missing']
             and status['completed'] == len(manifest['entries'])
             and status['invalid_wald'] == 0)
    for entry in manifest['entries']:
        result = load(output/entry['dataset']/'estimation.json')
        valid = valid and len(result['wald_tests']) == 10
        valid = valid and {row['model'] for row in result['oos_summary']} == {'M1','M2','M3','M4','M5'}
    status['validation_passed'] = bool(valid)
    corrected.archived.dump(output/'run_status.json', status)
    if not valid:
        raise RuntimeError('Empirical validation failed. Inspect audit.json and run_status.json.')
    # Publish only the reader's aggregates, with the same variant labels as the paper.
    tables = output/'tables'
    tables.mkdir(exist_ok=True)
    for name in ('samples', 'parameters', 'wald_tests', 'oos_levels',
                 'oos_comparisons', 'training_log_likelihoods'):
        frame = pd.read_csv(output/('case1_'+name+'.csv'))
        entries = {entry['dataset']: entry for entry in manifest['entries']}
        frame['base_dataset'] = frame.dataset.map(lambda key: entries[key]['parent_dataset'])
        frame['variant'] = frame.dataset.map(lambda key: {
            'raw_matched': 'matched_raw', 'residual': 'pc1_residual'}[entries[key]['target']])
        frame.to_csv(tables/(name+'.csv'), index=False)
    pca = {entry['parent_dataset']: entry['pca'] for entry in manifest['entries'] if 'pca' in entry}
    pd.DataFrame([dict(base_dataset=key, **value) for key, value in pca.items()]).to_csv(
        tables/'pca_diagnostics.csv', index=False)
    corrected.archived.dump(output/'validation.json', {
        'all_available_checks_passed': True, 'panels': len(manifest['entries']),
        'valid_parameter_tests': 10*len(manifest['entries']), 'invalid_parameter_tests': 0,
        'reoptimization_applied': (output/'reoptimization/applied.json').exists()})
    print('VALIDATED READER RESULTS', json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    finalize(parser.parse_args().output)
