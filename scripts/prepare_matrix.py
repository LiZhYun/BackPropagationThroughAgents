"""Freeze code/configs for a bounded matrix development array; never submits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from bta_repro.train import Config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', default='matrix_dev')
parser.add_argument('--overrides', type=Path, help='JSON Config overrides applied to every run')
parser.add_argument('--algorithms', nargs='+', default=['mappo', 'arppo', 'bppo', 'agentmixer'])
parser.add_argument('--games', nargs='+', default=['climbing', 'penalty'])
parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--workers', type=int, default=2, help='Bounded CPU workers, each runs several configurations')
args = parser.parse_args()
if not 1 <= args.workers <= 2:
    raise ValueError('development sweeps support one or two workers')
if not args.label.replace('_', '').isalnum():
    raise ValueError('label must contain only letters, digits, and underscores')
root = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
directory = root/'results'/f'{args.label}_{stamp}'
directory.mkdir(parents=True, exist_ok=False)
snapshot = directory/'source'/'bta_repro'
snapshot.mkdir(parents=True)
names = ['__init__.py','envs.py','core.py','bppo.py','agentmixer.py','agentmixer_joint.py','train.py','checkpoint.py']
hashes = {}
for name in names:
    source = root/'src'/'bta_repro'/name
    shutil.copy2(source, snapshot/name)
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
runs = []
for algorithm in args.algorithms:
    for game in args.games:
        for seed in args.seeds:
            name = f'{algorithm}_{game}_seed{seed}'
            config = {'steps':args.steps, 'batch_size':1024, 'epochs':4, 'threads':1, **overrides,
                      'algorithm':algorithm, 'game':game, 'seed':seed}
            Config(**config).validate()
            config_path = directory/(name+'.json')
            config_path.write_text(json.dumps(config,indent=2)+'\n')
            runs.append({'algorithm':algorithm,'game':game,'seed':seed,
                         'config_path':str(config_path),'output_path':str(directory/name)})
plan = {'scope':'development only; one-step matrix tasks; no final performance claim',
        'source_sha256':hashes, 'runs':runs}
(directory/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
shutil.copy2(root/'scripts'/'run_matrix_array.py', directory/'run_array.py')
script = f'''#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-matrix-dev
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-{min(args.workers, len(runs))-1}%2
#SBATCH --output={directory}/slurm_%A_%a.out
#SBATCH --error={directory}/slurm_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BTA_MATRIX_WORKERS={args.workers}
export PYTHONPATH={directory}/source
cd {root}
exec {root}/.venv/bin/python {directory}/run_array.py {directory}/manifest.json
'''
(directory/'submit.sbatch').write_text(script)
print(directory)
