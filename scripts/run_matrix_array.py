"""Run exactly one frozen matrix configuration from a Slurm array manifest."""
import json
import os
from pathlib import Path
import subprocess
import sys

plan = json.loads(Path(sys.argv[1]).read_text())
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
workers = int(os.environ.get('BTA_MATRIX_WORKERS', '0'))
entries = plan['runs'][index::workers] if workers else [plan['runs'][index]]
for entry in entries:
    subprocess.run([
        sys.executable, '-m', 'bta_repro.train', '--config', entry['config_path'],
        '--output', entry['output_path'],
    ], check=True, env=os.environ)
