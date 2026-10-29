"""Verify or use a portable simulator study; needs only Python to verify/list."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def contained(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Bundle path must be relative: {name}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Bundle path leaves its directory: {name}')
    return resolved


def verify(root):
    root = Path(root).resolve()
    inventory = json.loads((root / 'inventory.json').read_text())
    if inventory['format_version'] != 1:
        raise ValueError('Unsupported bundle format')
    for name, expected in inventory['files'].items():
        path = contained(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File changed: {name}')
    for required in ('manifest.json', 'bundle.py', 'evaluate_simulator.py', 'LICENSE', 'README.md'):
        if required not in inventory['files']:
            raise ValueError(f'Inventory is missing {required}')
    manifest = json.loads((root / 'manifest.json').read_text())
    source = root / 'source/bta_repro'
    actual = {p.name: hashlib.sha256(contained(root, p.relative_to(root)).read_bytes()).hexdigest()
              for p in source.glob('*.py')}
    if actual != manifest['source_sha256']:
        raise ValueError('Frozen source file set or hashes changed')
    for entry in manifest['runs']:
        paths = [entry['config_path']] + [entry['output_path'] + '/' + name for name in
                 ('config.json', 'metrics.jsonl', 'summary.json', 'checkpoint.pt')]
        if any(name not in inventory['files'] for name in paths):
            raise ValueError('Inventory does not cover every declared run artifact')
        for name in paths:
            contained(root, name)
    return manifest


def command(root, operation, index, output):
    root = Path(root).resolve()
    manifest = verify(root)
    if not 0 <= index < len(manifest['runs']):
        raise ValueError('Run index is outside the manifest')
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Output must be a new path outside the evidence bundle')
    entry = manifest['runs'][index]
    if operation == 'train':
        arguments = [sys.executable, '-m', 'bta_repro.simulator_train', '--config',
                     str(contained(root, entry['config_path'])), '--output', str(output)]
    elif operation == 'evaluate':
        arguments = [sys.executable, str(root / 'evaluate_simulator.py'),
                     str(contained(root, entry['output_path'])), '--output', str(output)]
    else:
        raise ValueError('Expected train or evaluate')
    environment = {**os.environ, 'PYTHONPATH': str(root / 'source'), 'PYTHONNOUSERSITE': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    return arguments, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['verify', 'list', 'train', 'evaluate'])
    parser.add_argument('index', nargs='?', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        manifest = verify(root)
        if args.operation == 'verify':
            print(f"Verified {len(manifest['runs'])} complete runs and their frozen source.")
        elif args.operation == 'list':
            for index, entry in enumerate(manifest['runs']):
                print(f"{index}\t{entry['variant']}\tseed={entry['seed']}")
        else:
            if args.index is None or args.output is None:
                parser.error('train/evaluate require a run index and --output')
            arguments, environment = command(root, args.operation, args.index, args.output)
            subprocess.run(arguments, cwd=root, env=environment, check=True)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
