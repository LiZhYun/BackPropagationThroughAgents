# Runtime versions and isolation

Use one private environment for each simulator family. Optional simulators are
not required for the core matrix tests. All profiles below currently use the
PyTorch 2.6.0 CPU wheel. A GPU profile has not been validated; changing PyTorch,
CUDA, NumPy, or a simulator creates a new profile requiring separate checks.

| Profile | Interpreter and key packages | Native simulator |
| --- | --- | --- |
| Core `.venv` | Python 3.11.15; PyTorch 2.6.0+cpu; observed NumPy 2.4.6 (reported, not pinned by checker) | None |
| SMAC `.venv-smac` | Python 3.11.15; NumPy 1.23.5; protobuf 3.20.3; PySC2 4.0.0; SMACv2 1.0.0 at commit `577ab5a2cff2391f8df582da5731ea9cd6adf3c6` | StarCraft II 4.10, Base75689 |
| MuJoCo `.venv-mujoco` | Python 3.10.22; NumPy 1.23.5; Cython 0.29.36; Gym 0.10.8; mujoco-py 2.1.2.14; MAMujoco 1.1.0 at commit `b212ddd74b258e7cea006ff1d642b5ffada4b99d` | MuJoCo 2.1.0; Mesa 25.0.5 OSMesa |

The SMAC compatibility pins retain the old NumPy aliases and protobuf behavior
used by the pinned PySC2 stack. The MuJoCo profile retains the original Gym and
mujoco-py API and physics engine. Gymnasium/MuJoCo 3 is a separate migration,
not the canonical benchmark profile. See [MuJoCo setup](mujoco_setup.md) and
[adapter semantics](mujoco_adapter.md) for its native build and action mapping.

## Rebuild or check SMAC

Run from the repository root with `uv 0.11.7` available, or set `BTA_UV` to that
binary. Git is needed to acquire the pinned SMACv2 source. The installer accepts
Linux x86_64 only and writes packages to `.venv-smac`, downloads/cache to `.cache/uv`,
and a new managed Python to `.python`. It rejects a symlinked environment prefix
and an existing prefix with a different Python version. It never uses `sudo`,
changes another environment, or downloads StarCraft II.

```bash
# Optional: inspect an existing, separately supplied engine installation.
export SC2PATH=/path/to/StarCraftII
bash scripts/setup_smac.sh
# Read-only verification of an already installed prefix:
bash scripts/setup_smac.sh --check-only
# Validate a fresh installation without changing the existing prefix:
BTA_SMAC_ENV_NAME=.venv-smac-verify bash scripts/setup_smac.sh
```

Re-running installation applies the same package versions and VCS commit, then
checks dependency consistency and imports. The complete 47-package inventory is
[smac-cpu.json](../configs/environment_locks/smac-cpu.json); the installer reads
[smac-cpu-requirements.txt](../configs/environment_locks/smac-cpu-requirements.txt).
Torch is fetched separately from the official CPU wheel index; the other
registry packages use PyPI. SMACv2 comes from its exact upstream Git commit, and
this project is installed editable without resolving additional dependencies.
Runtime versions and VCS content are pinned; registry wheel contents and every
build dependency are not hash-locked.

[smac-build-constraints.txt](../configs/environment_locks/smac-build-constraints.txt)
chooses setuptools 80.9.0 and wheel 0.45.1 for future isolated builds. These are
recipe choices, not a claim about the original installed wheels' build tools.
Installation records a package freeze, uv version, recipe hashes, and JSON
version report under `.native/smac/`. The existing environment contains no pip
module; the script uses `uv pip check` and `uv pip freeze` directly.

On this host, the inspected SMAC prefix uses a read-only uv-managed Python at
`/home/liz23/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu`; its packages
are local to this repository. Fresh setup uses `.python` for the base interpreter
as well. The verified host is Linux x86_64 with glibc 2.34. Existing native SC2
files were supplied under `/scratch/work/liz23/3rdparty/StarCraftII` and were not
modified. The manifest records SHA256 hashes for its Base75689 executable and
three `10gen_*` maps. Those identify the inspected files; they do not establish
the acquisition history or integrity of the complete distribution.

The upstream [SMACv2 installation instructions](https://github.com/oxwhirl/smacv2/tree/577ab5a2cff2391f8df582da5731ea9cd6adf3c6#installation)
explain separate engine/map acquisition. Our checker requires the Base75689
executable and `10gen_protoss`, `10gen_terran`, and `10gen_zerg` maps in either
`Maps/` or `Maps/SMAC_Maps/` when SC2PATH is supplied. Missing files fail the
check. If SC2PATH is absent, engine status is `not_requested`: successful Python
checks alone do not establish that an SC2 game can run.

## Inspect without training

The checker uses only Python's standard library until `--imports` is selected.
Run it with the interpreter being inspected; it never switches environments.
It emits one JSON object with versions, package installer/VCS metadata,
interpreter paths, native engine presence when requested, and mismatches.

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/check_environments.py --profile core --imports
PYTHONNOUSERSITE=1 .venv-smac/bin/python scripts/check_environments.py --profile smac --strict --imports
PYTHONNOUSERSITE=1 .venv-mujoco/bin/python scripts/check_environments.py --profile mujoco --strict
```

`--strict` fails on version/source mismatches. Import errors and missing
explicitly requested engine files fail even without `--strict`. SMAC imports
load the capability wrapper but start no game. MuJoCo inspection does **not**
import `mujoco_py`, whose import may compile native bindings; its `--imports`
option checks NumPy, PyTorch, and Gym only. Set `MUJOCO_PY_MUJOCO_PATH` to include
MuJoCo library presence and the downloaded archive's `SOURCE.json` provenance.
The MuJoCo checker accepts the Python 3.10 series; the exact current interpreter
and native dependency builds are recorded in `.native/mujoco-conda-explicit.txt`,
`.native/mujoco-pip-freeze-with-training.txt`, and the engine's `SOURCE.json`.
The official 2.1 archive SHA256 is
`a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012`.

The read-only SMAC check passed on the existing prefix: all 47 package pins,
exact VCS commit, NumPy/Torch/wrapper imports, and dependency consistency;
Base75689 and all three maps were present. The setup script's shell syntax and
help were checked. The core imports/version report passed, the runtime
requirements agree with the JSON manifest, and a missing-engine test returned
structured errors with exit status 1. A fresh rebuild subsequently passed in
`.venv-smac-verify`, using a new project-private Python3.11.15 installation,
all47 pinned packages, dependency checks, and imports. Its report and recipe
hashes are in `.native/smac-verify/`; the existing training prefix was unchanged.
Import checks and version reports do not test episode behavior or
constitute reproduced benchmark results; simulator smoke tests and training
validation remain separate.
