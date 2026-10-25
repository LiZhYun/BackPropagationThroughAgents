"""Freeze an explicit simulator plan into at most two serial CPU workers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from bta_repro.simulator_train import Config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('plan', type=Path)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
wall_minutes = plan.get('wall_time_minutes', 120)
if isinstance(wall_minutes, bool) or not isinstance(wall_minutes, int) or not 1 <= wall_minutes <= 1440:
    raise ValueError('wall_time_minutes must be an integer between1 and1440')
label = plan['label']
if not label.replace('_', '').isalnum():
    raise ValueError('label must be alphanumeric with optional underscores')
root = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
directory = root / 'results' / f'{label}_{stamp}'
snapshot = directory / 'source' / 'bta_repro'
snapshot.mkdir(parents=True, exist_ok=False)
hashes = {}
for source in (root / 'src' / 'bta_repro').glob('*.py'):
    shutil.copy2(source, snapshot / source.name)
    hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
runs = []
for variant, overrides in plan['variants'].items():
    if not variant.replace('_', '').isalnum():
        raise ValueError('variant names must be alphanumeric with optional underscores')
    for seed in plan['seeds']:
        config = {**plan['base_config'], **overrides, 'seed': seed}
        Config(**config).validate()
        name = f'{variant}_seed{seed}'
        config_path = directory / f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        runs.append({'environment': config['environment'], 'algorithm': config['algorithm'],
                     'variant': variant, 'seed': seed, 'config_path': str(config_path),
                     'output_path': str(directory / name)})
if not runs:
    raise ValueError('the plan must declare at least one run')
(directory / 'manifest.json').write_text(json.dumps({
    'scope': plan['scope'], 'source_sha256': hashes, 'plan': plan, 'runs': runs}, indent=2) + '\n')
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
print(directory)
