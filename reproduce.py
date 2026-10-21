"""Reproduce the shared simulation and empirical results of both manuscripts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
DATA = CODE / 'data'
MODEL = CODE / 'arc_sm_joint_location_scale_hansen_parity'
SIM = CODE / 'arc_sm_cross_section_size_simulation'
WORK = CODE / 'work'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_path(value):
    """Interpret the original Windows manifest on Windows and POSIX hosts."""
    relative = PureWindowsPath(value)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Expected a relative input path: ' + value)
    return Path(*relative.parts)


def verify():
    manifest = load(ROOT / 'manifest.json')
    failed = []
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != record['sha256']:
            failed.append(name)
    if failed:
        raise RuntimeError('Missing or changed frozen files:\n' + '\n'.join(failed))
    print(f"Verified {len(manifest['files'])} frozen files.", flush=True)
    return manifest


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=CODE, check=True)


def tables():
    manifest = verify()
    run(CODE / 'paper/build_sec4_sec5_tables.py')
    expected = manifest['table_checks']
    audit = load(CODE / 'paper/tables/sec4_sec5_table_audit.json')
    # Counts and source hashes are checked separately from table byte hashes.
    # Normalize generated line endings so Windows and POSIX use one record.
    if ({k: v for k, v in audit.items() if k != 'artifact_sha256'} !=
            {k: v for k, v in expected.items() if k != 'artifact_sha256'} or
            set(audit['artifact_sha256']) != set(expected['artifact_sha256'])):
        raise RuntimeError('Table inputs or validation counts differ from the manuscript.')
    failed = []
    for name, reference_hash in expected['artifact_sha256'].items():
        path = CODE / 'paper' / name
        if (not path.is_file() or hashlib.sha256(
                path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != reference_hash):
            failed.append(name)
    if failed:
        raise RuntimeError('Tables differ from the manuscript:\n' + '\n'.join(failed))
    print(f"Matched all {len(expected['artifact_sha256'])} manuscript/supplement table hashes and validation counts.")
    print('Output:', CODE / 'paper/tables')


def empirical(args):
    verify()
    if args.output.resolve().is_relative_to((CODE / 'paper/results').resolve()):
        raise ValueError('Use a reader output directory, not the frozen paper results')
    if not args.manifest.is_file():
        raise ValueError('Prepared data manifest not found: ' + str(args.manifest)
                         + '\nFollow the data collection steps in README.md, ending with '
                           'python reproduce.py collect pca.')
    manifest = load(args.manifest)
    requested = args.datasets.split(',') if args.datasets else None
    selected = set(requested) if requested else None
    if requested and len(selected) != len(requested):
        raise ValueError('Duplicate dataset selection')
    entries = [e for e in manifest['entries'] if selected is None or e['dataset'] in selected]
    if selected is not None and len(entries) != len(selected):
        raise ValueError('Unknown or duplicate dataset selection')
    if not entries:
        raise ValueError('No empirical datasets selected')
    for entry in entries:
        relative = input_path(entry['input'])
        path = CODE / relative
        if not path.is_file() or sha256(path) != entry['input_sha256']:
            raise ValueError('Missing or changed input; prepare it with python reproduce.py collect pca: ' + str(path))
    # Only normalize path separators. Model inputs and metadata stay unchanged.
    for entry in manifest['entries']:
        entry['input'] = input_path(entry['input']).as_posix()
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / 'empirical_input_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    options = ['--manifest', path, '--output', args.output.resolve(), '--workers', args.workers]
    if args.datasets:
        options += ['--datasets', args.datasets]
    if args.finalize_only:
        previous = load(args.output.resolve() / 'input_manifest.json')
        if previous['entries'] != entries:
            raise ValueError('Prepared inputs or selected datasets differ from the initial fits')
    if not args.finalize_only:
        run(MODEL / 'run_case1_training_fixed_pca_wald.py', *options)
    run(MODEL / 'finalize_case1_reader_results.py', '--output', args.output.resolve())
    print('Empirical fits, reoptimization, audit and aggregates complete:', args.output.resolve())


def simulation(args):
    verify()
    name = 'simulation_smoke' if args.smoke else 'simulation'
    options = ['--output', WORK / name / 'checkpoints', '--report', WORK / name / 'report',
               '--workers', args.workers, '--sample-sizes', 20, 50, 100,
               '--replications', args.replications]
    if args.smoke:
        options = ['--output', WORK / name / 'checkpoints', '--report', WORK / name / 'report',
                   '--workers', 1, '--replications', 1, '--training', 80, '--oos', 10,
                   '--burn', 50, '--sample-sizes', 20, '--dgps', 'case0_static', '--cases', 'iid']
    run(SIM / 'run_symmetric_t_full_stage_m1.py', *options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'tables'):
        commands.add_parser(name)
    collection = commands.add_parser('collect', help='Collect public data and prepare PC1 inputs; see README.md')
    collection.add_argument('stage', choices=('sources', 'sp500', 'nikkei', 'sectors', 'pca'))
    collection.add_argument('options', nargs=argparse.REMAINDER)
    sim = commands.add_parser('simulation', help='Reestimate the main-paper simulation')
    sim.add_argument('--workers', type=int, default=3)
    sim.add_argument('--replications', type=int, default=100)
    sim.add_argument('--smoke', action='store_true', help='Small execution check, not manuscript results')
    emp = commands.add_parser('empirical', help='Fit the prepared empirical panels')
    emp.add_argument('--workers', type=int, default=3)
    emp.add_argument('--datasets', help='Comma-separated dataset names from the manifest')
    emp.add_argument('--manifest', type=Path, default=DATA / 'reader/input_manifest.json')
    emp.add_argument('--output', type=Path, default=WORK / 'empirical')
    emp.add_argument('--finalize-only', action='store_true',
                     help='Reoptimize and audit existing initial fits without repeating the initial estimation')
    args = parser.parse_args()
    if getattr(args, 'workers', 1) < 1 or getattr(args, 'replications', 1) < 1:
        parser.error('workers and replications must be positive')
    if args.command == 'verify':
        verify()
    elif args.command == 'collect':
        verify()
        subprocess.run([sys.executable, str(CODE / 'analysis/collect_paper_data.py'), args.stage,
                        *args.options], cwd=ROOT, check=True)
    elif args.command == 'tables':
        tables()
    elif args.command == 'simulation':
        simulation(args)
    elif args.command == 'empirical':
        empirical(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
