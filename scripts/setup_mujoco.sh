#!/usr/bin/env bash
# Install the legacy simulator in private prefixes; never alter the user environment.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_prefix="$project_dir/.venv-mujoco"
engine_dir="$project_dir/.simulators/mujoco210"
native_dir="$project_dir/.native"
micromamba_bin="${BTA_MICROMAMBA:-/appl/manual_installations/software/micromamba/2.3.0/bin/micromamba}"
export MAMBA_ROOT_PREFIX="$project_dir/.cache/mamba"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
export MAMBA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export CONDA_ENVS_PATH="$MAMBA_ROOT_PREFIX/envs"
export MAMBA_ENVS_DIRS="$CONDA_ENVS_PATH"
export PIP_CACHE_DIR="$native_dir/pip-cache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export SUPPRESS_GR_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
mkdir -p "$sim_prefix/conda-meta" "$MAMBA_ROOT_PREFIX/conda-meta" "$native_dir" "$engine_dir"
# micromamba create registers ~/.conda/environments.txt. Installing into an
# initialized prefix avoids that registration (verified in micromamba 2.3.0).
touch "$sim_prefix/conda-meta/history" "$MAMBA_ROOT_PREFIX/conda-meta/history"
registry_before="$(python3 - <<'PY'
from pathlib import Path
import hashlib
p = Path.home() / '.conda/environments.txt'
print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'absent')
PY
)"
"$micromamba_bin" install --no-rc --override-channels -c conda-forge \
  --strict-channel-priority --yes --root-prefix "$MAMBA_ROOT_PREFIX" --prefix "$sim_prefix" \
  python=3.10 numpy=1.23.5 cython=0.29.36 scipy=1.10.1 \
  mesalib=25.0.5 libgl-devel libglx-devel glew glfw patchelf \
  pip=23.3.2 setuptools=65.5.0 wheel=0.38.4
registry_after="$(python3 - <<'PY'
from pathlib import Path
import hashlib
p = Path.home() / '.conda/environments.txt'
print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'absent')
PY
)"
test "$registry_before" = "$registry_after"

export MUJOCO_PY_MUJOCO_PATH="$engine_dir"
export MUJOCO_PY_FORCE_CPU=1
export PATH="$sim_prefix/bin:$PATH"
export LD_LIBRARY_PATH="$engine_dir/bin:$sim_prefix/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$sim_prefix/include:${CPATH:-}"
export LIBRARY_PATH="$sim_prefix/lib:${LIBRARY_PATH:-}"

"$sim_prefix/bin/python" - "$engine_dir" <<'PY'
from pathlib import Path
import hashlib, json, sys, tarfile, urllib.request
root = Path(sys.argv[1])
url = 'https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz'
archive = root / 'mujoco210-linux-x86_64.tar.gz'
if not archive.exists():
    temporary = archive.with_suffix('.download')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(archive)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
expected = 'a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012'
if digest != expected:
    raise RuntimeError('Official MuJoCo 2.1 archive differs from the audited checksum.')
record = root / 'SOURCE.json'
if record.exists() and json.loads(record.read_text())['sha256'] != digest:
    raise RuntimeError('MuJoCo archive checksum changed; inspect before proceeding.')
if not (root / 'bin/libmujoco210.so').exists():
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != 'mujoco210':
                raise RuntimeError(f'Unexpected archive path: {member.name}')
        bundle.extractall(root.parent, filter='data')
record.write_text(json.dumps({'url': url, 'sha256': digest, 'engine_version': '2.1.0'}, indent=2) + '\n')
print('MuJoCo archive SHA256:', digest)
PY

test -f "$sim_prefix/include/GL/osmesa.h"
test -f "$sim_prefix/include/GL/gl.h"
test -f "$sim_prefix/lib/libOSMesa.so"
cat > "$native_dir/mujoco-constraints.txt" <<'EOF'
numpy==1.23.5
Cython==0.29.36
scipy==1.10.1
gym==0.10.8
mujoco-py==2.1.2.14
setuptools==65.5.0
wheel==0.38.4
pyglet==1.5.27
EOF
"$sim_prefix/bin/python" -m pip install --constraint "$native_dir/mujoco-constraints.txt" \
  gym==0.10.8 mujoco-py==2.1.2.14 pyglet==1.5.27 glfw==2.7.0 \
  imageio==2.34.2 cffi==1.16.0 fasteners==0.19 \
  'MAMujoco @ git+https://github.com/schroederdewitt/multiagent_mujoco.git@b212ddd74b258e7cea006ff1d642b5ffada4b99d'
"$sim_prefix/bin/python" -m pip check
"$micromamba_bin" list --no-rc --prefix "$sim_prefix" --explicit > "$native_dir/mujoco-conda-explicit.txt"
"$sim_prefix/bin/python" -m pip freeze > "$native_dir/mujoco-pip-freeze.txt"
"$sim_prefix/bin/python" - "$native_dir/mujoco-smoke.json" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json, sys
import numpy as np
import mujoco_py
from multiagent_mujoco.mujoco_multi import MujocoMulti
env = MujocoMulti(env_args={'scenario': 'HalfCheetah-v2', 'agent_conf': '2x3', 'agent_obsk': 0, 'episode_limit': 3})
env.env.seed(123)
env.reset()
print('Simulator info:', env.get_env_info())
records = []
for step in range(3):
    reward, done, info = env.step([[0.0] * env.n_actions for _ in range(env.n_agents)])
    assert np.isfinite(reward) and np.isfinite(env.get_obs()).all() and np.isfinite(env.get_state()).all()
    assert np.asarray(env.get_obs()).shape == (env.n_agents, env.get_obs_size())
    assert bool(done) == (step == 2)
    if done:
        assert info['episode_limit'] is True
    records.append({'step': step + 1, 'reward': float(reward), 'done': bool(done), 'timeout': bool(info.get('episode_limit', False))})
    print('Zero-action step:', float(reward), bool(done), info)
env.env.close()  # Upstream MujocoMulti.close is unimplemented.
Path(sys.argv[1]).write_text(json.dumps({'status': 'passed', 'purpose': 'simulator smoke test, not learning', 'versions': {p: version(p) for p in ['gym', 'mujoco-py', 'numpy', 'Cython', 'MAMujoco']}, 'scenario': 'HalfCheetah-v2', 'agent_conf': '2x3', 'agent_obsk': 0, 'seed': 123, 'steps': records}, indent=2) + '\n')
PY
