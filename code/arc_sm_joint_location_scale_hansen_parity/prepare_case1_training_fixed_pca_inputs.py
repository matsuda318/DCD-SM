"""Prepare PC1 residuals with loadings estimated once on training returns.

For PCA only, missing training returns are replaced by each stock's
training mean. Model inputs always contain observed returns or their
residuals. Neither means nor loadings use OOS returns.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

import prepare_case1_symmetric_t_pca_inputs as original_inputs


model = original_inputs.model
ROOT, REPO = model.ROOT, model.REPO
OUTPUT = ROOT / 'output_sec5_raw_pc1_inputs'
CACHE = REPO / 'data/sec5_raw_pc1_inputs'
VERSION = 'training-fixed-covariance-pc1-v1'


def fixed_pc1(values, training_mask):
    values = np.asarray(values, dtype=float)
    training_mask = np.asarray(training_mask, dtype=bool)
    if values.ndim != 2 or training_mask.shape != (len(values),) or training_mask.sum() < 2:
        raise ValueError('Expected returns by date and stock, with at least two training dates')
    training = values[training_mask]
    counts = np.isfinite(training).sum(axis=0)
    means = np.divide(np.nansum(training, axis=0), counts,
                      out=np.full(values.shape[1], np.nan), where=counts > 0)
    centered = np.where(np.isfinite(training), training - means, 0.)
    energy = np.sum(centered * centered, axis=0)
    eligible = (counts > 0) & np.isfinite(means) & (energy > 0.)
    ids = np.flatnonzero(eligible)
    if len(ids) < 2:
        raise ValueError('Fewer than two nonconstant stocks observed in training')
    centered = centered[:, ids]
    covariance = centered.T @ centered / (len(training) - 1)
    eigenvalues, eigenvectors = eigh(covariance, subset_by_index=[len(ids)-1, len(ids)-1])
    loading = eigenvectors[:, 0]
    loading /= np.linalg.norm(loading)
    if loading.sum() < 0:
        loading = -loading
    residual = np.full_like(values, np.nan)
    records = []
    for t in range(len(values)):
        observed = np.isfinite(values[t, ids])
        observed_ids = ids[observed]
        observed_loading = loading[observed]
        norm = float(observed_loading @ observed_loading)
        if len(observed_ids) < 2 or norm < 1e-12:
            continue
        current = values[t, observed_ids] - means[observed_ids]
        factor = float(observed_loading @ current / norm)
        residual[t, observed_ids] = current - observed_loading * factor
        records.append({'row':t, 'observed_stocks':len(observed_ids), 'pc1_score':factor,
                        'projection_error':float(abs(observed_loading @ residual[t, observed_ids]))})
    loadings = np.full(values.shape[1], np.nan)
    loadings[ids] = loading
    details = {
        'training_days':int(len(training)), 'eligible_stocks':int(len(ids)),
        'stocks_without_training_observations':int((counts == 0).sum()),
        'stocks_with_constant_training_returns':int(((counts > 0) & ~eligible).sum()),
        'training_missing_cells_imputed_for_pca':int(np.isnan(training[:, ids]).sum()),
        'pc1_explained_fraction':float(eigenvalues[0] / np.trace(covariance)),
        'maximum_projection_error':float(max(r['projection_error'] for r in records)),
    }
    return residual, loadings, means, pd.DataFrame(records), details


def prepare(entry):
    frame, source = original_inputs.source_frame(entry['dataset'])
    wide = frame.pivot(index='date', columns='firm_id', values='log_return').sort_index()
    training_mask = wide.index <= model.archived.old.TRAINING_END
    source_signature = {
        'version':VERSION, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
    }
    folder = CACHE / entry['dataset']
    metadata_path = folder / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['source_signature'] != source_signature:
            raise ValueError('Changed PCA code or source: ' + entry['dataset'])
        for target in metadata['entries']:
            if hashlib.sha256((REPO/target['input']).read_bytes()).hexdigest() != target['input_sha256']:
                raise ValueError('Changed prepared input: ' + target['dataset'])
        return metadata
    original = wide.to_numpy()
    residual, loadings, means, diagnostics, details = fixed_pc1(original, training_mask)
    row, column = np.nonzero(np.isfinite(residual))
    matched = pd.DataFrame({'date':wide.index.to_numpy()[row],
        'firm_id':wide.columns.to_numpy()[column], 'raw_matched':original[row,column],
        'residual':residual[row,column]})
    if matched.duplicated(['date','firm_id']).any() or not np.isfinite(matched.raw_matched).all():
        raise ValueError('Duplicated or unobserved model input')
    if details['maximum_projection_error'] >= 1e-10:
        raise ValueError('PC1 residuals fail orthogonality check')
    folder.mkdir(parents=True, exist_ok=True)
    loading_frame = pd.DataFrame({'firm_id':wide.columns, 'training_mean':means,
                                  'pc1_loading':loadings,
                                  'training_observations':np.isfinite(original[training_mask]).sum(axis=0)})
    loading_frame.to_csv(folder/'training_loadings.csv', index=False)
    diagnostics['date'] = wide.index.to_numpy()[diagnostics.row.to_numpy()]
    diagnostics.to_csv(folder/'daily_projection_diagnostics.csv', index=False)
    train = matched.loc[matched.date <= model.archived.old.TRAINING_END]
    details.update(source_signature, factors_removed=1,
        loadings='fixed leading covariance eigenvector estimated from training dates only',
        missing_training_returns='stock training mean imputation for PCA only',
        factor_score='current observed eligible stocks, least-squares projection',
        OOS_target='residual distribution; not unconditional raw-return forecast',
        matched_asset_days=len(matched),
        training_mean_pair_correlation_raw=original_inputs.mean_pair_correlation(
            train.pivot(index='date', columns='firm_id', values='raw_matched')),
        training_mean_pair_correlation_residual=original_inputs.mean_pair_correlation(
            train.pivot(index='date', columns='firm_id', values='residual')))
    entries = []
    for target in ('raw_matched','residual'):
        item = original_inputs.export_pair(matched, target, entry, folder, details)
        item['dataset'] = entry['dataset'] + '_fixed_pca1_' + target
        item['label'] = entry['label'] + ' training-fixed PC1 ' + target
        entries.append(item)
    metadata = {'source_signature':source_signature, 'entries':entries, 'pca':details}
    model.archived.dump(metadata_path, metadata)
    print(entry['dataset'], 'INPUTS COMPLETE', json.dumps(details), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', default='all')
    args = parser.parse_args()
    original = json.loads(model.INPUT_MANIFEST.read_text(encoding='utf-8'))
    entries = [{**entry, 'role':'main'} for entry in original['entries']
               if entry['target']=='raw' and entry['family'] in model.MAIN_FAMILIES]
    if len(entries) != 19:
        raise ValueError('Expected the existing 19 cross sections')
    if args.datasets != 'all':
        selected = set(args.datasets.split(','))
        entries = [e for e in entries if e['dataset'] in selected]
        if len(entries) != len(selected):
            raise ValueError('Unknown or duplicate dataset selection')
    prepared, diagnostics = [], []
    for entry in entries:
        metadata = prepare(entry)
        prepared.extend(metadata['entries'])
        diagnostics.append({'base_dataset':entry['dataset'], 'label':entry['label'], **metadata['pca']})
        model.archived.dump(OUTPUT/'preparation_status.json',
                            {'requested':len(entries), 'completed':len(diagnostics)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(OUTPUT/'pca_diagnostics.csv', index=False)
    model.archived.dump(OUTPUT/'input_manifest.json', {
        'pca_version':VERSION, 'training_end':str(model.archived.old.TRAINING_END.date()),
        'missing_policy':'training mean imputation for PCA only; observed model inputs only',
        'entries':prepared})
    print('COMPLETE', len(prepared), 'prepared panels', flush=True)


if __name__ == '__main__':
    main()
