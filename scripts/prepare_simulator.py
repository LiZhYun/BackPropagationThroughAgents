"""Freeze an explicit simulator plan into at most two serial CPU workers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from bta_repro.simulator_train import Config

def source_snapshot(root, source_manifest=None):
    """Read and verify source bytes before creating any experiment files."""
    root = root.resolve()
    if source_manifest is None:
        source = root / 'src' / 'bta_repro'
        provenance = {'kind': 'working_tree', 'path': 'src/bta_repro'}
        expected = None
    else:
        source_manifest = Path(source_manifest).resolve()
        if not source_manifest.is_relative_to(root):
            raise ValueError('source manifest must belong to this repository')
        manifest_bytes = source_manifest.read_bytes()
        expected = json.loads(manifest_bytes)['source_sha256']
        source = source_manifest.parent / 'source' / 'bta_repro'
        provenance = {'kind': 'frozen_manifest', 'manifest': str(source_manifest.relative_to(root)),
                      'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    if not source.resolve().is_relative_to(root):
        raise ValueError('source directory must belong to this repository')
    files = {}
    for path in sorted(source.glob('*.py')):
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('source file points outside the snapshot')
        files[path.name] = path.read_bytes()
    if '__init__.py' not in files or 'simulator_train.py' not in files:
        raise ValueError('source snapshot must contain the simulator package')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if expected is not None and hashes != expected:
        raise ValueError('frozen source hashes differ from the selected manifest')
    return files, hashes, provenance


def prepare(plan, root, *, source_manifest=None, stamp=None):
    root = Path(root).resolve()
    wall_minutes = plan.get('wall_time_minutes', 120)
    if isinstance(wall_minutes, bool) or not isinstance(wall_minutes, int) or not 1 <= wall_minutes <= 1440:
        raise ValueError('wall_time_minutes must be an integer between1 and1440')
    label = plan['label']
    if not label.replace('_', '').isalnum():
        raise ValueError('label must be alphanumeric with optional underscores')
    seeds = plan['seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('training seeds must be nonempty and unique')
    entries = []
    for variant, overrides in plan['variants'].items():
        if not variant.replace('_', '').isalnum():
            raise ValueError('variant names must be alphanumeric with optional underscores')
        for seed in seeds:
            config = {**plan['base_config'], **overrides, 'seed': seed}
            Config(**config).validate()
            entries.append((variant, seed, config))
    if not entries:
        raise ValueError('the plan must declare at least one run')
    files, hashes, provenance = source_snapshot(root, source_manifest)
    stamp = stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = root / 'results' / f'{label}_{stamp}'
    snapshot = directory / 'source' / 'bta_repro'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (snapshot / name).write_bytes(data)
    runs = []
    for variant, seed, config in entries:
        name = f'{variant}_seed{seed}'
        config_path = directory / f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        runs.append({'environment': config['environment'], 'algorithm': config['algorithm'],
                     'variant': variant, 'seed': seed, 'config_path': str(config_path),
                     'output_path': str(directory / name)})
    (directory / 'manifest.json').write_text(json.dumps({
        'scope': plan['scope'], 'source_sha256': hashes, 'source_provenance': provenance,
        'plan': plan, 'runs': runs}, indent=2) + '\n')
    for name in ('run_simulator_array.py', 'run_simulator_worker.py'):
        shutil.copy2(root / 'scripts' / name, directory / name)
    workers = min(2, len(runs))
    quote = lambda path: shlex.quote(str(path))
    (directory / 'submit.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-simulator-pilot
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time={wall_minutes // 60:02d}:{wall_minutes % 60:02d}:00
#SBATCH --array=0-{workers-1}%2
#SBATCH --output={directory}/slurm_%A_%a.out
#SBATCH --error={directory}/slurm_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={quote(directory / 'source')}
cd {quote(root)}
exec {quote(root / '.venv/bin/python')} {quote(directory / 'run_simulator_worker.py')} {quote(directory / 'manifest.json')} {workers}
''')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--source-manifest', type=Path,
                        help='reuse the verified source of a previous local frozen experiment')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    print(prepare(plan, Path(__file__).resolve().parents[1], source_manifest=args.source_manifest))


if __name__ == '__main__':
    main()
