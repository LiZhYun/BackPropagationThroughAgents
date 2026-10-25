"""Run a deterministic slice of a frozen plan; each run keeps its own log."""
import json
import os
from pathlib import Path
import subprocess
import sys

manifest = Path(sys.argv[1])
workers = int(sys.argv[2])
worker = int(os.environ['SLURM_ARRAY_TASK_ID'])
runs = json.loads(manifest.read_text())['runs']
for index in range(worker, len(runs), workers):
    entry = runs[index]
    environment = {**os.environ, 'SLURM_ARRAY_TASK_ID': str(index)}
    output = Path(entry['output_path'])
    with output.with_suffix('.out').open('x') as stdout, output.with_suffix('.err').open('x') as stderr:
        subprocess.run([sys.executable, str(manifest.parent / 'run_simulator_array.py'), str(manifest)],
                       check=True, env=environment, stdout=stdout, stderr=stderr)
