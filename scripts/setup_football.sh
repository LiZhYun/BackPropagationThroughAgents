#!/usr/bin/env bash
# Rebuild the verified Linux x86_64 CPU profile in a project-private prefix.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_name="${BTA_FOOTBALL_ENV_NAME:-.venv-football}"
if [[ ! "$env_name" =~ ^\.venv-football(-[A-Za-z0-9_-]+)?$ ]]; then
  echo 'BTA_FOOTBALL_ENV_NAME must be .venv-football or .venv-football-SUFFIX' >&2
  exit 2
fi
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
  echo 'This verified lock supports Linux x86_64 only.' >&2
  exit 2
fi
micromamba_bin="${BTA_MICROMAMBA:-$(command -v micromamba || true)}"
uv_bin="${BTA_UV:-$(command -v uv || true)}"
test -x "$micromamba_bin" || { echo 'Install micromamba or set BTA_MICROMAMBA.' >&2; exit 2; }
test -x "$uv_bin" || { echo 'Install uv or set BTA_UV.' >&2; exit 2; }
export BTA_FOOTBALL_PREFIX="$project_dir/$env_name"
test ! -L "$BTA_FOOTBALL_PREFIX" || { echo 'Refusing a symlinked environment prefix.' >&2; exit 2; }
build_dir="$project_dir/.native/football-build/$env_name"
lock_dir="$project_dir/configs/environment_locks"
export MAMBA_ROOT_PREFIX="$project_dir/.cache/football-mamba"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs" MAMBA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
export CONDA_ENVS_PATH="$MAMBA_ROOT_PREFIX/envs" MAMBA_ENVS_DIRS="$MAMBA_ROOT_PREFIX/envs"
export PIP_CACHE_DIR="$project_dir/.cache/football-pip" UV_CACHE_DIR="$project_dir/.cache/uv"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
mkdir -p "$BTA_FOOTBALL_PREFIX/conda-meta" "$MAMBA_ROOT_PREFIX/conda-meta" "$build_dir"
# Installing into an initialized prefix avoids registering a new environment
# in ~/.conda/environments.txt (micromamba2.3.0 behavior).
touch "$BTA_FOOTBALL_PREFIX/conda-meta/history" "$MAMBA_ROOT_PREFIX/conda-meta/history"
"$micromamba_bin" install --no-rc --yes --root-prefix "$MAMBA_ROOT_PREFIX" \
  --prefix "$BTA_FOOTBALL_PREFIX" --file "$lock_dir/football-conda-linux-64.txt"
export PATH="$BTA_FOOTBALL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$BTA_FOOTBALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CC="$BTA_FOOTBALL_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$BTA_FOOTBALL_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="$BTA_FOOTBALL_PREFIX"
export PKG_CONFIG_PATH="$BTA_FOOTBALL_PREFIX/lib/pkgconfig"
"$uv_bin" pip install --python "$BTA_FOOTBALL_PREFIX/bin/python" --no-deps -r "$lock_dir/football-python.txt"
"$uv_bin" pip install --python "$BTA_FOOTBALL_PREFIX/bin/python" --no-deps \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'

"$BTA_FOOTBALL_PREFIX/bin/python" - "$build_dir" <<'PY'
from pathlib import Path
import hashlib, json, sys, tarfile, urllib.request
root = Path(sys.argv[1])
source = root / 'gfootball-2.10.2'
archive = root / 'gfootball-2.10.2.tar.gz'
url = 'https://files.pythonhosted.org/packages/98/63/b111538b5db47b8081d8ca82280fadaa145fbd31aa249f49675a01abb8eb/gfootball-2.10.2.tar.gz'
expected = '1b0fdcfa78b7fadc3730585ee7f0f412ba825c27e422b2b85ea0cf7ba57800b6'
if not archive.exists():
    temporary = archive.with_suffix('.download')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Football archive differs from the audited checksum')
if not source.exists():
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != source.name:
                raise RuntimeError(f'Unexpected archive path: {member.name}')
        bundle.extractall(root, filter='data')
script = source / 'gfootball/build_game_engine.sh'
original = script.read_text()
lines = original.splitlines()
for index, line in enumerate(lines):
    if line.startswith('PARALLELISM='):
        lines[index] = 'PARALLELISM="${SLURM_CPUS_PER_TASK:-2}"'
    if '&& cmake . &&' in line:
        lines[index] = line.replace('&& cmake . &&', '&& cmake -DPython_ROOT_DIR="$BTA_FOOTBALL_PREFIX" -DBOOST_ROOT="$BTA_FOOTBALL_PREFIX" . &&')
patched = '\n'.join(lines) + '\n'
if '-DPython_ROOT_DIR=' not in patched or 'PARALLELISM="${SLURM_CPUS_PER_TASK:-2}"' not in patched:
    raise RuntimeError('Unexpected upstream build script')
script.write_text(patched)
(root / 'SOURCE.json').write_text(json.dumps({
    'version': '2.10.2', 'url': url, 'sha256': expected,
    'upstream_tag_commit': 'ba9952130898bde2879b01ff2b49d3d48d26ce81',
    'build_patch': 'bounded parallelism and explicit private Python/Boost discovery only',
    'patched_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}, indent=2) + '\n')
PY

cd "$build_dir/gfootball-2.10.2"
"$BTA_FOOTBALL_PREFIX/bin/python" -m pip wheel --no-build-isolation --no-deps . --wheel-dir "$build_dir/wheels"
"$BTA_FOOTBALL_PREFIX/bin/python" -m pip install --no-deps --force-reinstall "$build_dir"/wheels/gfootball-2.10.2-*.whl
"$uv_bin" pip install --python "$BTA_FOOTBALL_PREFIX/bin/python" --no-deps -e "$project_dir"
"$BTA_FOOTBALL_PREFIX/bin/python" -m pip check
"$BTA_FOOTBALL_PREFIX/bin/python" "$project_dir/scripts/check_environments.py" \
  --profile football --strict --imports > "$build_dir/environment-report.json"
"$BTA_FOOTBALL_PREFIX/bin/python" -m pip freeze > "$build_dir/pip-freeze.txt"
echo "Installed Football; provenance: $build_dir/environment-report.json"
echo "Native replay check: $BTA_FOOTBALL_PREFIX/bin/python $project_dir/scripts/smoke_football.py --output $build_dir/native-smoke.json"
