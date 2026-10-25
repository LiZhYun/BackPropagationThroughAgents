"""Freeze eight tiny simulator integration runs; no jobs are submitted here."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
directory = root / 'results' / f'simulator_smoke_{stamp}'
snapshot = directory / 'source' / 'bta_repro'
snapshot.mkdir(parents=True, exist_ok=False)
hashes = {}
for source in (root / 'src' / 'bta_repro').glob('*.py'):
    if source.name in ('agentmixer_joint.py', 'mujoco_partitions.py'):
        continue  # Independent experimental extensions are not used by this smoke.
    shutil.copy2(source, snapshot / source.name)
    hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
runs = []
for environment in ('smac', 'mujoco'):
    for algorithm in ('mappo', 'arppo', 'bppo', 'agentmixer'):
        name = f'{environment}_{algorithm}'
        env_config = ({'scenario': 'terran_5v5'} if environment == 'smac' else {
            'scenario': 'HalfCheetah-v2', 'agent_conf': '2x3', 'agent_obsk': 0, 'episode_limit': 32})
        config = {'environment': environment, 'env_config': env_config, 'algorithm': algorithm,
                  'seed': 0, 'eval_seed': 1000000, 'steps': 128, 'num_envs': 1,
                  'rollout_steps': 64, 'epochs': 2, 'eval_episodes': 2,
                  'checkpoint_every_updates': 0, 'threads': 1}
        config_path = directory / f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        runs.append({'environment': environment, 'algorithm': algorithm,
                     'config_path': str(config_path), 'output_path': str(directory / name)})
manifest = {'scope': '128-step integration smoke only, not a performance result',
            'source_sha256': hashes, 'runs': runs}
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
shutil.copy2(root / 'scripts' / 'run_simulator_array.py', directory / 'run_array.py')
(directory / 'submit.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-sim-integration
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-7%2
#SBATCH --output={directory}/slurm_%A_%a.out
#SBATCH --error={directory}/slurm_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={directory}/source
cd {root}
exec {root}/.venv/bin/python {directory}/run_array.py {directory}/manifest.json
''')
print(directory)
