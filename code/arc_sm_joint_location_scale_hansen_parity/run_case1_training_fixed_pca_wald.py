"""Fit M1--M5 to prepared panels using training-fixed PC1 loadings.

The criterion and Wald(0) estimators are unchanged. Failed initial starts
can use the original experiment's additional-start retry procedure.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import traceback

import run_case1_symmetric_t_daily_mean_wald as corrected


ROOT = corrected.ROOT
OUTPUT = ROOT / 'output_sec5_raw_pc1_m1_m5'
PCA_VERSION = 'training-fixed-covariance-pc1-v1'


def fit_dataset(entry, output):
    try:
        return corrected.run_dataset(entry, output)
    except RuntimeError as error:
        if 'no converged optimizer start' not in str(error):
            raise
        # Recover incomplete fits with the original experiment's extra-start
        # procedure. Already completed stages stay in their checkpoints.
        import retry_case1_training_fixed_pca as retry
        original = corrected.archived.maximize
        retry.original_maximize = original
        retry.RECORDS.clear()
        corrected.archived.maximize = retry.maximize_with_extra_starts
        try:
            return corrected.run_dataset(entry, output)
        finally:
            corrected.archived.maximize = original
            corrected.archived.dump(Path(output)/entry['dataset']/'optimization_retry.json', {
                'searches': list(retry.RECORDS),
                'retry_source_sha256': hashlib.sha256(
                    Path(retry.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()})


def run(manifest_path, output, workers=3, datasets=None, summarize_only=False):
    manifest_path, output = Path(manifest_path), Path(output).resolve()
    prepared = json.loads(manifest_path.read_text(encoding='utf-8'))
    if prepared['pca_version'] != PCA_VERSION:
        raise ValueError('Expected training-fixed PCA inputs')
    entries = prepared['entries']
    if datasets:
        selected = set(datasets)
        entries = [entry for entry in entries if entry['dataset'] in selected]
        if len(entries) != len(selected):
            raise ValueError('Unknown or duplicate dataset selection')
    for entry in entries:
        data_path = corrected.REPO / entry['input']
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != entry['input_sha256']:
            raise ValueError('Changed input: ' + entry['dataset'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **prepared, 'schema': corrected.SCHEMA,
        'source_hash': corrected.source_hash(), 'entries': entries,
        'likelihood': 'sum of daily cross-sectional mean log densities',
        'preparation_manifest': str(manifest_path.resolve().relative_to(corrected.REPO)),
        'preparation_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
    }
    saved_manifest = output / 'input_manifest.json'
    if saved_manifest.exists():
        if json.loads(saved_manifest.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Changed inputs, code, or scope. Use a new output directory.')
    else:
        corrected.archived.dump(saved_manifest, manifest)
    failures = []
    if not summarize_only:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = {
                pool.submit(fit_dataset, entry, output): entry['dataset']
                for entry in entries
            }
            for future in as_completed(tasks):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception:
                    failure = {'dataset': tasks[future], 'traceback': traceback.format_exc()}
                    failures.append(failure)
                    corrected.archived.dump(output / 'failed_datasets.json', failures)
                    print('FAILED', failure['dataset'], failure['traceback'], flush=True)
                corrected.base.aggregate(manifest, output)
    status = corrected.base.aggregate(manifest, output)
    if failures or status['completed'] != len(entries):
        raise RuntimeError('Incomplete results. Failure records and checkpoints are retained.')
    print('COMPLETE', json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--datasets')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    run(args.manifest, args.output, args.workers,
        args.datasets.split(',') if args.datasets else None, args.summarize_only)


if __name__ == '__main__':
    main()
