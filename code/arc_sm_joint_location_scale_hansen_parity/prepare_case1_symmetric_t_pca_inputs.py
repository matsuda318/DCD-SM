"""One past-only covariance PC per panel; matched raw/residual inputs.

Use the approved existing PCA rule: preceding 126--252 dates, complete
past-window eligibility, no volatility normalization, and current-date
least-squares scores solely to define the residual response.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_case1_symmetric_t_wald as model
import run_case1_pca_robustness_markets as pca
import run_identity_sm_skewt_sector_five_models as sectors

OUTPUT = model.ROOT/'output_symmetric_t_pc1_wald'
CACHE = model.REPO/'data/symmetric_t_pc1_inputs'
VERSION = 'per-panel-rolling-covariance-PC1-252-min126-v1'


def source_frame(dataset):
    if dataset in ('sp500','n225'):
        path = pca.market_factor.MARKETS[dataset]['source']
        identifier = 'firm_id'
    else:
        info = sectors.SECTORS[dataset]
        path = sectors.DATA_ROOT/info['market']/'daily_returns'/(info['code']+'.parquet')
        identifier = 'identifier'
    frame = pd.read_parquet(path, columns=['date',identifier,'log_return'])
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.rename(columns={identifier:'firm_id'})
    frame['firm_id'] = frame.firm_id.astype(str)
    frame = frame.loc[frame.date.between(model.archived.old.START,model.archived.old.END)
                      & np.isfinite(frame.log_return)].copy()
    frame = frame.sort_values(['date','firm_id']).reset_index(drop=True)
    assert not frame.duplicated(['date','firm_id']).any()
    return frame, Path(path)


def verify_original_observations(frame, entry):
    """Compare each date's multiset, since panel row orders may differ."""
    with np.load(model.REPO/entry['input']) as saved:
        dates = pd.DatetimeIndex(saved['dates'])
        offsets, values = saved['offsets'], saved['values']
        groups = frame.groupby('date',sort=True).log_return
        assert len(groups) == len(dates), (entry['dataset'],len(groups),len(dates))
        for index, (date, group) in enumerate(groups):
            assert date == dates[index]
            np.testing.assert_array_equal(np.sort(group.to_numpy()),
                                          np.sort(values[offsets[index]:offsets[index+1]]))


def export_pair(frame, value_column, entry, folder, details):
    key = entry['dataset']+'_pca1_'+value_column
    values = frame[value_column].to_numpy()
    counts = frame.groupby('date',sort=True).size().to_numpy()
    dates = np.array(sorted(frame.date.unique()),dtype='datetime64[ns]')
    offsets = np.r_[0,counts.cumsum()]
    assert len(values) == offsets[-1] and np.isfinite(values).all()
    path = folder/(value_column+'.npz')
    np.savez_compressed(path, values=values, offsets=offsets, dates=dates)
    training = int((dates <= np.datetime64(model.archived.old.TRAINING_END)).sum())
    assert training > 22 and training < len(dates)
    return {**entry, 'dataset':key, 'label':entry['label']+' PC1 '+value_column,
            'parent_dataset':entry['dataset'], 'target':value_column, 'role':'main',
            'sample_variant':'PC1 residual' if value_column=='residual' else 'PC1 matched raw',
            'input':str(path.relative_to(model.REPO)),
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'actual_start':str(pd.Timestamp(dates[0]).date()),
            'training_end':str(pd.Timestamp(dates[training-1]).date()),
            'actual_oos_start':str(pd.Timestamp(dates[training]).date()),
            'end':str(pd.Timestamp(dates[-1]).date()),
            'training_days':training, 'oos_days':len(dates)-training,
            'training_observations':int(offsets[training]),'observations':len(values),
            'N_min':int(counts.min()),'N_median':float(np.median(counts)),
            'N_max':int(counts.max()),'pca':details}


def mean_pair_correlation(wide):
    correlation = wide.corr(min_periods=126).to_numpy()
    pairs = correlation[np.triu_indices(len(correlation),1)]
    return float(np.nanmean(pairs)) if np.isfinite(pairs).any() else None


def prepare(entry):
    dataset = entry['dataset']
    frame, source = source_frame(dataset)
    source_signature = {'version':VERSION,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_input_sha256':entry['input_sha256'],
        'pca_code_sha256':hashlib.sha256(Path(pca.__file__).read_bytes()).hexdigest(),
        'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    folder = CACHE/dataset
    metadata_path = folder/'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        assert metadata['source_signature'] == source_signature, 'changed PCA inputs/source: '+dataset
        for target in metadata['entries']:
            assert hashlib.sha256((model.REPO/target['input']).read_bytes()).hexdigest() == target['input_sha256']
        print(dataset,'PCA inputs reused',flush=True)
        return metadata
    verify_original_observations(frame, entry)
    wide = frame.pivot(index='date',columns='firm_id',values='log_return').sort_index()
    original = wide.to_numpy()
    residual, _, diagnostics = pca.rolling_pc1(original)
    assert len(diagnostics), 'no eligible PCA dates: '+dataset
    row, column = np.nonzero(np.isfinite(residual))
    matched = pd.DataFrame({'date':wide.index.to_numpy()[row],
        'firm_id':wide.columns.to_numpy()[column],
        'raw_matched':original[row,column], 'residual':residual[row,column]})
    assert not matched.duplicated(['date','firm_id']).any()
    assert diagnostics.projection_error.max() < 1e-10
    assert np.all(diagnostics.history_end_row < diagnostics.row)
    diagnostics['date'] = wide.index.to_numpy()[diagnostics.row.to_numpy()]
    diagnostics['history_end_date'] = wide.index.to_numpy()[diagnostics.history_end_row.to_numpy()]
    folder.mkdir(parents=True,exist_ok=True)
    diagnostic_path = OUTPUT/'pca_diagnostics'/(dataset+'.csv')
    diagnostic_path.parent.mkdir(parents=True,exist_ok=True)
    diagnostics.to_csv(diagnostic_path,index=False)
    training_rows = matched.date <= model.archived.old.TRAINING_END
    train = matched.loc[training_rows]
    raw_wide = train.pivot(index='date',columns='firm_id',values='raw_matched')
    residual_wide = train.pivot(index='date',columns='firm_id',values='residual')
    details = {**source_signature,'factors_removed':1,'window':252,'minimum_history':126,
        'loadings':'centered, unscaled prior-window returns; current and future excluded',
        'factor_score':'current observed eligible stocks, least-squares projection',
        'OOS_target':'residual distribution; not unconditional raw-return forecast',
        'matched_asset_days':len(matched),'pc1_explained_fraction_mean':float(diagnostics.pc1_explained_fraction.mean()),
        'maximum_projection_error':float(diagnostics.projection_error.max()),
        'training_mean_pair_correlation_raw':mean_pair_correlation(raw_wide),
        'training_mean_pair_correlation_residual':mean_pair_correlation(residual_wide)}
    entries = [export_pair(matched, target, entry, folder, details) for target in ('raw_matched','residual')]
    with np.load(model.REPO/entries[0]['input']) as raw, np.load(model.REPO/entries[1]['input']) as res:
        np.testing.assert_array_equal(raw['dates'],res['dates'])
        np.testing.assert_array_equal(raw['offsets'],res['offsets'])
    metadata = {'source_signature':source_signature,'entries':entries,'pca':details}
    model.archived.dump(metadata_path,metadata)
    print(dataset,'PCA inputs COMPLETE','asset-days',len(matched),'days',matched.date.nunique(),flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',default='all')
    args = parser.parse_args()
    source = json.loads(model.INPUT_MANIFEST.read_text(encoding='utf-8'))
    entries = [entry for entry in source['entries'] if entry['target']=='raw' and entry['family'] in model.MAIN_FAMILIES]
    assert len(entries)==19
    if args.datasets!='all':
        selected=args.datasets.split(',')
        entries=[entry for entry in entries if entry['dataset'] in selected]
        assert len(entries)==len(selected)
    prepared=[]
    for entry in entries:
        prepared.extend(prepare(entry)['entries'])
        model.archived.dump(OUTPUT/'preparation_status.json',{'requested':len(entries),'completed':len(prepared)//2})
    manifest={'schema':model.SCHEMA,'source_hash':model.source_hash(),'entries':prepared,
              'pca_version':VERSION,'inference':'same symmetric-t Wald(0), fixed upstream'}
    model.archived.dump(OUTPUT/'input_manifest.json',manifest)
    print('COMPLETE','prepared panels',len(prepared),flush=True)


if __name__=='__main__':
    main()
