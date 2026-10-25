"""Execute one frozen simulator integration configuration in its private prefix."""
import json
import os
from pathlib import Path
import subprocess
import sys

manifest = Path(sys.argv[1])
root = manifest.parents[2]
entry = json.loads(manifest.read_text())['runs'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
environment = os.environ.copy()
environment['PYTHONNOUSERSITE'] = '1'
if entry['environment'] == 'smac':
    interpreter = root / '.venv-smac' / 'bin' / 'python'
    environment.setdefault('SC2PATH', '/scratch/work/liz23/3rdparty/StarCraftII')
elif entry['environment'] == 'mujoco':
    prefix = root / '.venv-mujoco'
    engine = root / '.simulators' / 'mujoco210'
    interpreter = prefix / 'bin' / 'python'
    environment.update({'MUJOCO_PY_MUJOCO_PATH': str(engine), 'MUJOCO_PY_FORCE_CPU': '1',
                        'SUPPRESS_GR_PROMPT': '1'})
    for key, value in {'PATH': prefix / 'bin', 'CPATH': prefix / 'include',
                       'LIBRARY_PATH': prefix / 'lib'}.items():
        environment[key] = str(value) + os.pathsep + environment.get(key, '')
    environment['LD_LIBRARY_PATH'] = (str(engine / 'bin') + os.pathsep + str(prefix / 'lib')
                                       + os.pathsep + environment.get('LD_LIBRARY_PATH', ''))
elif entry['environment'] == 'prey':
    interpreter = root / '.venv' / 'bin' / 'python'
elif entry['environment'] == 'football':
    prefix = root / '.venv-football'
    interpreter = prefix / 'bin' / 'python'
    environment['LD_LIBRARY_PATH'] = str(prefix / 'lib') + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
    environment['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
else:
    raise ValueError('No private interpreter configured for this environment')
subprocess.run([str(interpreter), '-m', 'bta_repro.simulator_train',
                '--config', entry['config_path'], '--output', entry['output_path']],
               check=True, env=environment)
