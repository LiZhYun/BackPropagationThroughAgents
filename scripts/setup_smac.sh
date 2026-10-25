#!/usr/bin/env bash
# Rebuild only this project's CPU SMAC prefix; never download the SC2 engine.
set -euo pipefail
case "${1:-}" in
  --help|-h) printf '%s\n' 'Usage: bash scripts/setup_smac.sh [--check-only]' 'Default: create/update .venv-smac from pinned dependencies.' '--check-only: read versions/imports/dependencies; install nothing.' 'SC2PATH optionally selects an existing read-only StarCraft II 4.10 installation.'; exit 0 ;;
  --check-only) mode=check ;;
  '') mode=install ;;
  *) printf '%s\n' 'Unknown option; use --help.' >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then printf '%s\n' 'Use --help for supported arguments.' >&2; exit 2; fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
sim_env_name="${BTA_SMAC_ENV_NAME:-.venv-smac}"
if [[ ! "$sim_env_name" =~ ^\.venv-smac(-[A-Za-z0-9_]+)?$ ]]; then
  printf '%s\n' 'BTA_SMAC_ENV_NAME must be .venv-smac or .venv-smac-SUFFIX within this project.' >&2
  exit 2
fi
sim_prefix="$project_dir/$sim_env_name"
lock_dir="$project_dir/configs/environment_locks"
uv_bin="${BTA_UV:-$(command -v uv)}"
export UV_CACHE_DIR="$project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_dir/.python"
export UV_NO_CONFIG=1 UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
unset UV_INDEX UV_EXTRA_INDEX_URL UV_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
if [[ -L "$sim_prefix" ]]; then printf '%s\n' 'Refusing a symlinked environment prefix.' >&2; exit 1; fi
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) printf '%s\n' 'This verified profile supports Linux x86_64 only.' >&2; exit 1 ;;
esac
uv_version="$("$uv_bin" --version)"
case "$uv_version" in
  'uv 0.11.7'*) ;;
  *) printf '%s\n' "Expected uv 0.11.7, found $uv_version; set BTA_UV to that binary." >&2; exit 1 ;;
esac
if [[ "$mode" == install && ! -x "$sim_prefix/bin/python" ]]; then
  if [[ -e "$sim_prefix" ]]; then printf '%s\n' 'Existing prefix has no Python; inspect it before rerunning.' >&2; exit 1; fi
  "$uv_bin" python install --no-bin --no-registry --install-dir "$UV_PYTHON_INSTALL_DIR" 3.11.15
  "$uv_bin" venv --no-project --managed-python --python 3.11.15 "$sim_prefix"
fi
if [[ ! -x "$sim_prefix/bin/python" ]]; then printf '%s\n' 'Missing .venv-smac; run setup without --check-only first.' >&2; exit 1; fi
"$sim_prefix/bin/python" - "$sim_prefix" <<'PY'
from pathlib import Path
import sys
if Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Interpreter does not belong to the private SMAC prefix.')
if sys.version_info[:3] != (3, 11, 15):
    raise SystemExit('Expected Python 3.11.15; existing prefix was not changed.')
PY
if [[ "$mode" == install ]]; then
  # Explicit indexes and --no-deps prevent a resolver from replacing legacy pins.
  # The complete runtime list below is followed by a dependency consistency check.
  "$uv_bin" pip install --python "$sim_prefix/bin/python" --no-deps \
    --default-index https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
  "$uv_bin" pip install --python "$sim_prefix/bin/python" --no-deps \
    --default-index https://pypi.org/simple \
    --build-constraints "$lock_dir/smac-build-constraints.txt" \
    --requirements "$lock_dir/smac-cpu-requirements.txt"
  "$uv_bin" pip install --python "$sim_prefix/bin/python" --no-deps \
    --default-index https://pypi.org/simple \
    --build-constraints "$lock_dir/smac-build-constraints.txt" --editable "$project_dir"
fi
"$uv_bin" pip check --python "$sim_prefix/bin/python" >&2
probe_args=(--profile smac --strict --imports)
if [[ -n "${SC2PATH:-}" ]]; then probe_args+=(--sc2-path "$SC2PATH"); fi
if [[ "$mode" == install ]]; then
  record_dir="$project_dir/.native/${sim_env_name#.venv-}"
  mkdir -p "$record_dir"
  "$uv_bin" pip freeze --python "$sim_prefix/bin/python" > "$record_dir/pip-freeze.txt"
  printf '%s\n' "$uv_version" > "$record_dir/uv-version.txt"
  sha256sum "$lock_dir/smac-cpu.json" "$lock_dir/smac-cpu-requirements.txt" \
    "$lock_dir/smac-build-constraints.txt" "$project_dir/scripts/setup_smac.sh" > "$record_dir/recipe-sha256.txt"
  "$sim_prefix/bin/python" "$project_dir/scripts/check_environments.py" "${probe_args[@]}" > "$record_dir/environment-report.json"
  cat "$record_dir/environment-report.json"
else
  "$sim_prefix/bin/python" "$project_dir/scripts/check_environments.py" "${probe_args[@]}"
fi
