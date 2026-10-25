# Legacy MuJoCo setup

The canonical simulator track uses **MuJoCo 2.1.0, Gym 0.10.8 and mujoco-py 2.1.2.14**. It keeps the paper's `*-v2` task definitions. The accepted sources do not provide an exact historical package lock, so this is an explicitly pinned reconstruction of the upstream legacy stack, not a claim that every package version matches the original runs.

Run from the project root:

```bash
bash scripts/setup_mujoco.sh
```

The script installs only inside this subproject. `.venv-mujoco/` is a **conda prefix**, despite its name. The installation uses the cluster's `/appl/manual_installations/software/micromamba/2.3.0/bin/micromamba`; set `BTA_MICROMAMBA` to another micromamba 2.3 executable when needed. No shell initialization, system package installation, shared environment change, or training job is performed.

| Component | Pin |
|---|---|
| Python | 3.10; exact solved patch in the exported lock |
| NumPy | 1.23.5 |
| Cython | 0.29.36 |
| SciPy | 1.10.1 |
| Gym | 0.10.8 |
| mujoco-py | 2.1.2.14 |
| MuJoCo engine | 2.1.0 |
| MA-MuJoCo | `b212ddd74b258e7cea006ff1d642b5ffada4b99d` |
| Mesa | 25.0.5, including OSMesa |

The script also installs private GL headers, GLEW, GLFW and `patchelf`. Mesa is deliberately pinned: newer releases removed OSMesa, which the legacy CPU extension requires. Python dependencies and native packages are exported after installation to `.native/mujoco-pip-freeze.txt` and `.native/mujoco-conda-explicit.txt`.

The official engine archive lives under `.simulators/mujoco210/`. Its URL and SHA-256 are recorded in `.simulators/mujoco210/SOURCE.json`. Reruns reject a checksum change against that recorded artifact. `MUJOCO_PY_MUJOCO_PATH` selects this engine; the script forces the CPU extension and uses one thread. The first import compiles the extension inside the private Python prefix.

For subsequent simulator processes, use these environment settings:

```bash
project_dir="$PWD"
export MUJOCO_PY_MUJOCO_PATH="$project_dir/.simulators/mujoco210"
export MUJOCO_PY_FORCE_CPU=1
export PYTHONNOUSERSITE=1
export SUPPRESS_GR_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$project_dir/.venv-mujoco/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$project_dir/.venv-mujoco/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$project_dir/.venv-mujoco/include:${CPATH:-}"
export LIBRARY_PATH="$project_dir/.venv-mujoco/lib:${LIBRARY_PATH:-}"
```

The setup script has a three-step, zero-action HalfCheetah test. Its purpose is to check import, observation construction, stepping and timeout reporting. It is not a learning result. PyTorch and the reproduction package are installed separately when the training adapter is ready.

`SUPPRESS_GR_PROMPT=1` suppresses the upstream package's interactive maintenance notice. The source is intentionally retained for comparisons with legacy results. The smoke-test record is `.native/mujoco-smoke.json`.

Verified on 2026-10-25: the CPU extension compiled, `pip check` passed, and HalfCheetah-v2 `2x3`, `obsk=0` completed three finite zero-action steps with a timeout on step 3. Its raw local observation size is 6, critic state size is 17 and local action dimension is 3. This establishes simulator availability only. No training or benchmark score is claimed. The downloaded engine archive SHA-256 is `a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012`.

## Host inspection

On `login5.triton.aalto.fi`, the conventional `~/.mujoco` directory and `/scratch/work/liz23/3rdparty/mujoco210` were absent. The inherited loader path pointed to a nonexistent old cluster MuJoCo installation. Existing user-level Python packages had no usable Linux extension: only macOS/Python 3.7 shared objects and an unfinished Python 3.9 build. The system did not expose OSMesa or the required GL development headers. The private prefix handles these dependencies without changing the user's packages.

Micromamba normally registers created environments in `~/.conda/environments.txt`. To preserve that file, the script initializes private `conda-meta/history` files and uses `micromamba install` into those prefixes. It hashes the user registry before and after the transaction and verifies it is unchanged. Both package and environment caches are explicitly redirected because this host inherits a shared `CONDA_PKGS_DIRS` setting.

## Simulator semantics for the adapter

The pinned upstream class is `multiagent_mujoco.mujoco_multi.MujocoMulti(env_args=...)`. `get_env_info()` provides `n_agents`, maximum local action dimension `n_actions`, `obs_shape`, `state_shape`, `episode_limit` and local `action_spaces`. `reset()` returns local observations; query `get_state()` separately. `step(local_actions)` returns `(scalar_team_reward, done, info)`. `info['episode_limit']` identifies timeout when `done` is true. Observations are padded local joint features for finite `agent_obsk`; `None` gives the base environment's full observation to every agent. The critic state is the base Gym `_get_obs()` vector. Do not confuse rollout length with the environment's episode limit.

The legacy class normalizes action inputs from `[-1,1]` to the underlying control bounds and maps each local action through the partition node's `act_ids`. Use that anatomical mapping. The historical BPTA repository instead concatenates actions, which can assign actions to the wrong joints when partition order differs from actuator order. For example, diagonal Ant partitions have actuator indices `[2,3,6,7]` and `[4,5,0,1]`. This repair must be documented when comparing results.

The upstream package does not contain all paper-specific partitions. The [audited adapter extension](mujoco_adapter.md) now supplies all 12 required task/partition pairs, including `3x2`, Ant `8x1`, Walker `6x1` and Humanoid `17x1`. The historical wrapper also appended agent identity and normalized each observation/state vector. Those preprocessing choices must be explicit rather than accidentally inherited.

Upstream `MujocoMulti.seed()` is a no-op and `close()` is unimplemented. Seed and close the underlying Gym environment in the adapter. Avoid passing `seed` inside `env_args` without checking the superclass: it can shadow the `seed` method with an integer. New adapters must also preserve true terminal versus timeout distinctions and report actual engine and task settings.

## Primary sources

- [Legacy MA-MuJoCo package requirements](https://github.com/schroederdewitt/multiagent_mujoco/blob/b212ddd74b258e7cea006ff1d642b5ffada4b99d/setup.py) and [environment API](https://github.com/schroederdewitt/multiagent_mujoco/blob/b212ddd74b258e7cea006ff1d642b5ffada4b99d/multiagent_mujoco/mujoco_multi.py).
- [mujoco-py engine installation and native requirements](https://github.com/openai/mujoco-py/blob/master/README.md).
- [NumPy 1.23.5 supports Python 3.8–3.11](https://numpy.org/doc/stable/release/1.23.5-notes.html); Python 3.10 is selected for this legacy build.
- [Mesa's removal of OSMesa](https://github.com/conda-forge/mesalib-feedstock/pull/123).
- [Historical BPTA environment wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py) and [partition definitions](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/obsk.py).

Gymnasium-Robotics is the maintained successor, but its current MA-MuJoCo implementation constructs `*-v5` environments and uses the official `mujoco` bindings and PettingZoo-style dictionaries. Results from that stack must be labeled as a modern migration and must not silently replace the legacy paper comparison.
