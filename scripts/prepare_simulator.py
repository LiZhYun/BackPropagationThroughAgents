"""Freeze a simulator plan with at most two concurrent CPU tasks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
from bta_repro.simulator_train import Config
from bta_repro.experiment_source import source_snapshot


def prepare(plan, root, *, source_manifest=None, stamp=None):
    root = Path(root).resolve()
    wall_minutes = plan.get('wall_time_minutes', 120)
    if isinstance(wall_minutes, bool) or not isinstance(wall_minutes, int) or not 1 <= wall_minutes <= 7200:
        raise ValueError('wall_time_minutes must be an integer between 1 and 7200')
    dispatch = plan.get('dispatch_mode', 'serial_workers')
    if dispatch not in ('serial_workers', 'one_run_per_task'):
        raise ValueError('dispatch_mode must be serial_workers or one_run_per_task')
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
    tasks = workers if dispatch == 'serial_workers' else len(runs)
    days, minutes = divmod(wall_minutes, 1440)
    wall_time = f'{minutes // 60:02d}:{minutes % 60:02d}:00'
    if days:
        wall_time = f'{days}-{wall_time}'
    quote = lambda path: shlex.quote(str(path))
    if dispatch == 'serial_workers':
        command = f'{quote(directory / "run_simulator_worker.py")} {quote(directory / "manifest.json")} {workers}'
    else:
        command = f'{quote(directory / "run_simulator_array.py")} {quote(directory / "manifest.json")}'
    (directory / 'submit.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-simulator-pilot
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time={wall_time}
#SBATCH --array=0-{tasks-1}%2
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm_%A_%a.out
#SBATCH --error={directory}/slurm_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={quote(directory / 'source')}
cd {quote(root)}
exec {quote(root / '.venv/bin/python')} {command}
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
