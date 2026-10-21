"""Collect the paper's equity panels and prepare training-fixed PC1 inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent
CATALOG = PACKAGE / 'data_sources.json'
MODEL = CODE / 'arc_sm_joint_location_scale_hansen_parity'
NIKKEI = CODE / 'dcd_nikkei225_daily_composite_score_arch'
sys.path.insert(0, str(NIKKEI))
sys.path.insert(0, str(MODEL))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def acquire(path, url, supplied=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if supplied:
        supplied = supplied.resolve()
        if path.exists() and digest(path) != digest(supplied):
            raise ValueError('A different source is already cached; choose a new workspace: ' + str(path))
        if supplied != path.resolve():
            shutil.copyfile(supplied, path)
    elif not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (academic replication)'})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    return path


def collect_sp500(args, workspace, catalog):
    import pandas as pd
    import build_weekly_em_data as original
    source = acquire(workspace / 'sources/sp500_membership.csv',
                     catalog['sp500_membership']['download_url'], args.membership_csv)
    content = source.read_bytes()
    if content.startswith(b'\x1f\x8b'):
        content = gzip.decompress(content)
    membership = pd.read_csv(io.BytesIO(content))
    membership['date'] = pd.to_datetime(membership.date).dt.date
    membership = membership.loc[membership.date <= original.END].sort_values('date').drop_duplicates('date', keep='last')
    dates = membership.date.tolist()
    sets = [{v.strip().upper() for v in str(row).split(',') if v.strip()} for row in membership.tickers]
    if not dates or dates[0] > original.START:
        raise ValueError('Membership history does not cover the sample start')
    universe = set(original.membership_on(original.START, dates, sets))
    for date, members in zip(dates, sets):
        if original.START <= date <= original.END:
            universe.update(members)
    original.OUTPUT_DIR = workspace / 'data/weekly_density_em'
    original.DAILY_DIR = original.OUTPUT_DIR / 'daily_returns'
    original.CACHE_DIR = original.OUTPUT_DIR / '_yahoo_cache'
    original.DAILY_DIR.mkdir(parents=True, exist_ok=True)
    original.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    histories = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(original.cached_yahoo_history, ticker): ticker for ticker in sorted(universe)}
        for future in as_completed(tasks):
            ticker = tasks[future]
            histories[ticker] = future.result()
            print('SP500 fetched', len(histories), '/', len(tasks), ticker, flush=True)
    records, status = {}, []
    for ticker in sorted(histories):
        history = histories[ticker]
        status.append({k: history.get(k, '') for k in ('ticker', 'error', 'price_start', 'price_end', 'daily_prices')})
        for _, date, value, dollars, close, volume in history.get('records', []):
            if not original.START <= date <= original.END or ticker not in original.membership_on(date, dates, sets):
                continue
            firm = 'usa|sp500_ticker|' + ticker
            records[(firm, date)] = ('usa', 'United States', original.week_start(date), date,
                firm, ticker, '', 'point-in-time S&P 500 constituent', value, close, volume, dollars, None, None)
    if not records:
        raise RuntimeError('No returns downloaded; inspect the provider response and fetch status')
    summary = original.write_country('usa', records)
    pd.DataFrame(status).to_csv(original.OUTPUT_DIR / 'ticker_fetch_status.csv', index=False)
    dump(original.OUTPUT_DIR / 'reader_collection.json', {
        'summary': summary, 'membership_source_sha256': digest(source),
        'reference_membership_records_match': hashlib.sha256(json.dumps([
            {'date': date.isoformat(), 'tickers': sorted(members)} for date, members in zip(dates, sets)
        ], separators=(',', ':')).encode()).hexdigest() == catalog['sp500_membership']['retained_membership_records_sha256'],
        'last_membership_date': dates[-1].isoformat(), 'requested_tickers': len(universe),
        'fetch_errors': sum(bool(h.get('error')) for h in histories.values()),
        'collected_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    })


def nikkei_sources(args, workspace, catalog):
    import build_nikkei225_daily_data as original
    spec = catalog['nikkei_membership']
    current = acquire(workspace / 'sources/current_component_weights.csv', spec['components_url'], args.components_csv)
    history = acquire(workspace / 'sources/component_changes.pdf', spec['changes_url'], args.history_pdf)
    date, _, _ = original.load_current_components(current)
    if date.isoformat() != spec['retained_components_date']:
        raise ValueError('This ledger requires the component snapshot dated ' + spec['retained_components_date'] +
                         '; the downloaded file is dated ' + date.isoformat() +
                         '. Supply the matching historical file with --components-csv; see README.md.')
    dump(workspace / 'sources/nikkei_source_check.json', {
        'components_date': date.isoformat(), 'components_sha256': digest(current),
        'changes_sha256': digest(history),
        'components_match_retained': digest(current) == spec['retained_components_sha256'],
        'changes_match_retained': digest(history) == spec['retained_changes_sha256'],
    })
    return current, history


def collect_nikkei(args, workspace, catalog):
    import build_nikkei225_daily_data as original
    current, history = nikkei_sources(args, workspace, catalog)
    original.CURRENT_COMPONENTS, original.HISTORY_PDF = current, history
    original.SOURCE_CACHE = current.parent
    original.OUTPUT = workspace / 'data/nikkei225_daily_rebuilt'
    original.PRICE_CACHE = original.OUTPUT / '_yahoo_cache'
    loader = original.load_current_components
    original.load_current_components = lambda: loader(current)
    original.parse_args = lambda: argparse.Namespace(refresh_official=False, workers=args.workers)
    original.main()


def collect_sectors(args, workspace, catalog):
    import build_sp500_n225_sector_daily_2016_2026 as original
    current, history = nikkei_sources(args, workspace, catalog)
    repo = (args.classification_repo or workspace / 'data/_external/s-and-p-500-companies').resolve()
    if not (repo / '.git').exists():
        raise ValueError('Clone the classification repository with its Git history; see README.md: ' + str(repo))
    head = original._run_git(repo, ['rev-parse', 'HEAD']).strip()
    if head != catalog['sp500_classification']['retained_git_head']:
        raise ValueError('Check out the recorded classification revision before building sectors: ' +
                         catalog['sp500_classification']['retained_git_head'])
    original.SP500_RETURNS = workspace / 'data/weekly_density_em/daily_returns/usa.parquet'
    original.N225_RETURNS = workspace / 'data/nikkei225_daily_rebuilt/daily_returns.parquet'
    original.SP500_CLASSIFICATION_REPO = repo
    original.N225_CURRENT, original.N225_HISTORY = current, history
    output = workspace / 'data/sp500_n225_sector_daily_2016_2026'
    output.mkdir(parents=True, exist_ok=True)
    frames = {'sp500': original.build_sp500(output), 'n225': original.build_n225(output)}
    original.write_manifest(output, frames)
    print('Built 11 US and six Japanese sector panels.', flush=True)


def panel_path(root, dataset):
    if dataset == 'sp500':
        return root / 'data/weekly_density_em/daily_returns/usa.parquet', 'firm_id'
    if dataset == 'n225':
        return root / 'data/nikkei225_daily_rebuilt/daily_returns.parquet', 'firm_id'
    market, sector = dataset.split('_')
    return root / 'data/sp500_n225_sector_daily_2016_2026' / market / 'daily_returns' / (sector + '.parquet'), 'identifier'


def prepare_pca(args, workspace, catalog):
    import numpy as np
    import pandas as pd
    import prepare_case1_training_fixed_pca_inputs as original
    baseline = load(CODE / 'paper/results/sec5_empirical_raw_pc1/input_manifest.json')
    source_root = args.panel_root.resolve() if args.panel_root else workspace
    entries, diagnostics = [], []
    by_parent = {e['parent_dataset']: e for e in baseline['entries'] if e['target'] == 'raw_matched'}
    references = {e['dataset']: e['input_sha256'] for e in baseline['entries']}
    for dataset, base in sorted(by_parent.items()):
        path, identifier = panel_path(source_root, dataset)
        frame = pd.read_parquet(path, columns=['date', identifier, 'log_return'])
        frame['date'] = pd.to_datetime(frame.date).dt.normalize()
        frame = frame.rename(columns={identifier: 'firm_id'})
        frame['firm_id'] = frame.firm_id.astype(str)
        frame = frame.loc[frame.date.between('2016-07-05', '2026-07-24') & np.isfinite(frame.log_return)]
        frame = frame.sort_values(['date', 'firm_id']).reset_index(drop=True)
        if frame.duplicated(['date', 'firm_id']).any():
            raise ValueError('Duplicated constituent/date: ' + dataset)
        wide = frame.pivot(index='date', columns='firm_id', values='log_return').sort_index()
        values = wide.to_numpy()
        training = wide.index <= pd.Timestamp('2023-12-29')
        residual, loadings, means, daily, details = original.fixed_pc1(values, training)
        row, column = np.nonzero(np.isfinite(residual))
        matched = pd.DataFrame({'date': wide.index.to_numpy()[row], 'firm_id': wide.columns.to_numpy()[column],
                               'raw_matched': values[row, column], 'residual': residual[row, column]})
        folder = workspace / 'prepared' / dataset
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'firm_id': wide.columns, 'training_mean': means, 'pc1_loading': loadings}).to_csv(folder / 'training_loadings.csv', index=False)
        daily['date'] = wide.index.to_numpy()[daily.row.to_numpy()]
        daily.to_csv(folder / 'daily_projection_diagnostics.csv', index=False)
        details.update(version=original.VERSION, source_sha256=digest(path),
                       preparer_sha256=digest(Path(__file__)), factors_removed=1)
        base = {**base, 'dataset': dataset, 'parent_dataset': dataset,
                'label': base['label'].split(' training-fixed PC1 ')[0],
                'original_source': {'source_panel': str(path.relative_to(source_root)).replace('\\', '/')},
                'role': 'main'}
        for target in ('raw_matched', 'residual'):
            item = original.original_inputs.export_pair(matched, target, base, folder, details)
            item['dataset'] = dataset + '_fixed_pca1_' + target
            item['label'] = base['label'] + ' training-fixed PC1 ' + target
            item['reference_input_sha256'] = references[item['dataset']]
            item['matches_original_input_sha256'] = item['input_sha256'] == references[item['dataset']]
            entries.append(item)
        diagnostics.append({'base_dataset': dataset, **details})
        print('PCA prepared', dataset, flush=True)
    if len(entries) != 38:
        raise ValueError('Expected 38 paired estimation inputs')
    dump(workspace / 'input_manifest.json', {'pca_version': original.VERSION, 'training_end': '2023-12-29', 'entries': entries})
    pd.DataFrame(diagnostics).to_csv(workspace / 'pca_diagnostics.csv', index=False)
    matched_hashes = sum(e['matches_original_input_sha256'] for e in entries)
    print(f'Prepared 38 inputs; {matched_hashes}/38 byte hashes match the paper.', flush=True)
    print('Estimation manifest:', workspace / 'input_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('sources', 'sp500', 'nikkei', 'sectors', 'pca'))
    parser.add_argument('--workspace', type=Path, default=CODE / 'data/reader')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--membership-csv', type=Path)
    parser.add_argument('--components-csv', type=Path)
    parser.add_argument('--history-pdf', type=Path)
    parser.add_argument('--classification-repo', type=Path)
    parser.add_argument('--panel-root', type=Path, help='Optional existing source-panel root for PCA preparation')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    workspace = args.workspace.resolve()
    if not workspace.is_relative_to(CODE / 'data') or workspace == CODE / 'data':
        parser.error('workspace must be a directory inside code/data/ (normally code/data/reader)')
    catalog = load(CATALOG)
    if args.stage == 'sources':
        print(json.dumps(catalog, indent=2))
        return
    workspace.mkdir(parents=True, exist_ok=True)
    functions = {'sp500': collect_sp500, 'nikkei': collect_nikkei, 'sectors': collect_sectors, 'pca': prepare_pca}
    functions[args.stage](args, workspace, catalog)


if __name__ == '__main__':
    main()
