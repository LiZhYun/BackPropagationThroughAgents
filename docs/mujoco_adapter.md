# Legacy MA-MuJoCo adapter

`bta_repro.mujoco_env.MujocoEnv` uses the private stack in [mujoco_setup.md](mujoco_setup.md). Construction verifies Gym 0.10.8, mujoco-py 2.1.2.14, MuJoCo engine 2.1.0 and MA-MuJoCo commit `b212ddd74b258e7cea006ff1d642b5ffada4b99d`. Importing the adapter requires only NumPy; construction loads the optional simulator.

```python
from bta_repro.mujoco_env import MujocoEnv

env = MujocoEnv({
    "scenario": "HalfCheetah-v2", "agent_conf": "2x3",
    "agent_obsk": 0, "episode_limit": 1000, "seed": 0,
    "action_units": "normalized",
})
observation, info = env.reset()
# raw_actions: floating array [env.num_agents, env.action_dim]
next_observation, reward, terminated, truncated, info = env.step(raw_actions)
env.close()
```

The runner reads `num_agents`, `obs_dim`, `state_dim`, `action_dim` and `is_discrete=False`. Observations contain float32 `obs[agents, local_features]`, float32 `state[global_features]` and boolean `active_agents[agents]`. There is no automatic reset. Final observations remain available for timeout bootstrapping; call `reset()` before stepping after either episode-ending flag. All agents remain active together until the episode ends. Rewards are the original scalar team reward.

`config` contains resolved settings; `versions` records measured package/engine versions and installed upstream commit. Unknown config keys fail. Optional `k_categories` and `global_categories` strings pass explicit observation categories to upstream; `None` retains its defaults. `agent_obsk=None` gives every actor the full base observation; nonnegative integers select graph neighborhoods. BPTA specifies fully observed Ant, while AgentMixer specifies Ant with `agent_obsk=0`; keep these distinct.

Observations and critic state follow pinned upstream MA-MuJoCo. No agent-ID vector or per-vector normalization is added. Both were present in the historical author wrapper; adding them requires explicit preprocessing configuration and a reported comparison. The default physical horizon is 1000 simulator decisions, separate from PPO rollout length. A paper's rollout/episode-length hyperparameter must not silently change the physical horizon.

## Actions and audited partitions

PPO likelihoods use unchanged raw Gaussian samples. The default `action_units="normalized"` follows both upstream and author `NormalizedActions` wrappers: `step()` clips a private copy to `[-1,1]`, maps it affinely into each physical actuator's range, then scatters it into global actuator order. It never mutates samples retained by PPO or applies tanh. `info['clipped_action_components']` counts clipped policy coordinates.

Read-only `action_low`/`action_high` describe the policy input coordinates (`-1`/`1` by default); `actuator_low`/`actuator_high` always describe the actual physical bounds in local agent order. `actuator_indices` maps those local entries to global actuators. Explicit `action_units="physical"` skips affine scaling and clips raw samples directly to physical bounds; it is a different exploration parameterization and must be reported as such. For example, with zero-mean standard-normal samples, normalized mode clips about 31.7% of components, while physical mode on Humanoid's `±0.4` actuators clips about 68.9%.

The historical author wrapper concatenated local action vectors, which can control different joints from those observed by an agent. Explicit upstream `act_ids` give these supported mappings (zero-based global indices):

| Task | Partition | Local-to-global actuator indices |
|---|---|---|
| HalfCheetah-v2 | 2x3 | `[2,1,0]`, `[5,4,3]` |
| HalfCheetah-v2 | 3x2 | `[2,1]`, `[0,5]`, `[4,3]` |
| HalfCheetah-v2 | 6x1 | `[2]`, `[1]`, `[0]`, `[5]`, `[4]`, `[3]` |
| Ant-v2 | 2x4 | `[2,3,4,5]`, `[6,7,0,1]` |
| Ant-v2 | 2x4d | `[2,3,6,7]`, `[4,5,0,1]` |
| Ant-v2 | 4x2 | `[2,3]`, `[4,5]`, `[6,7]`, `[0,1]` |
| Ant-v2 | 8x1 | `[2]`, `[3]`, `[4]`, `[5]`, `[6]`, `[7]`, `[0]`, `[1]` |
| Walker2d-v2 | 2x3 | `[2,1,0]`, `[5,4,3]` |
| Walker2d-v2 | 3x2 | `[2,1]`, `[0,5]`, `[4,3]` |
| Walker2d-v2 | 6x1 | `[2]`, `[1]`, `[0]`, `[5]`, `[4]`, `[3]` |
| Humanoid-v2 | 17x1 | One actuator per agent: `14,15,2,0,1,11,12,13,16,7,9,8,3,5,4,6,10` |
| HumanoidStandup-v2 | 17x1 | Same agent order as Humanoid-v2 |

Construction checks exact mapping, unique complete coverage, equal local dimensions, and finite ordered bounds. It also compares each node's position/velocity indices against the actual model's actuator transmission and joint addresses. A permutation alone is insufficient: the observed joint must be the commanded joint.

Scaling occurs exactly once before calling the base Gym task. Physical bounds come from the model's double-precision `actuator_ctrlrange`, cross-checked against Gym's action space. Gym rounds its Box limits to float32; using those rounded values would send `0.4000000059604645` to Humanoid's exact `0.4` boundary. The adapter uses the exact model boundary. Existing Ant/HalfCheetah/Walker `±1` ranges have identical normalized and physical units.

## Episode endings and seeds

Gym 0.10.8 `TimeLimit` overwrites `done=True` at the horizon without exposing whether the base task also terminated. Upstream then labels every horizon-ending transition a timeout, including a fall on that exact step. The adapter calls the same underlying Gym v2 physics task directly and manages its horizon. Physical termination takes precedence; a still-running task at its horizon returns `terminated=False, truncated=True`. Thus failures do not receive timeout value bootstrapping. `info['episode_limit']` agrees with `truncated`; `elapsed_steps` counts transitions.

`reset()` advances the environment's random stream; `reset(seed=n)` seeds the actual base Gym task and restarts it. Upstream `MujocoMulti.seed()` is a no-op. NumPy and Python global draws are also isolated during construction, reset, step and close, preserving caller streams and independent interleaved MuJoCo environments in the synchronous runner. Constructor-only draws are discarded so the first reset and explicit reseeding agree. Use separate processes for parallel simulation: unrelated simultaneous threads manipulating global RNGs are not supported. `close()` uses base Gym because upstream `MujocoMulti.close()` is unimplemented.

Backend exceptions and nonfinite transitions require reset before another step. Invalid action input fails before advancing physics. No synthetic transition replaces simulator failure.

## Author-specific partition extension

The table covers every MuJoCo task in `configs/paper_targets.json`: 21 paper entries and 12 distinct task/partition pairs. Every pair has equal local action dimensions, so there are no dummy action coordinates or padded dimensions in policy log probabilities. Other partitions still fail explicitly.

`mujoco_partitions.py` adds the six author-specific configurations missing from pinned upstream. It loads an existing upstream grouping of the same robot, copies the complete per-instance node/edge/global graph, and regroups its nodes using explicit names from author commit `13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0`. Nodes retain their observation callbacks, body associations and graph edges. The extension rebuilds neighborhood dictionaries with upstream `get_joints_at_kdist`, clears stale observation padding before measuring the new size, and rebuilds local action/observation spaces. It neither patches module globals nor copies the whole environment implementation.

Each extended node's actuator is resolved from the model's named physical joint, and its observation position/velocity coordinates are cross-checked. The resulting mapping must match the audited table. Any repaired node IDs appear in `versions['partition_actuator_corrections']`; `partition_extension` and `partition_author_commit` identify the extension and source. The pinned Gym/MuJoCo models required no node-ID repairs, including Humanoid's abdomen despite the upstream warning comment. Correct scattering still repairs the historical author wrapper's concatenation error.

The author Humanoid order is left shoulder 1/2, abdomen x/y/z, right shoulder 1/2, right elbow, left elbow, left hip x/y/z, right hip x/y/z, right knee, left knee. This order is retained for both Humanoid variants. Upstream's temporary `9|8` grouping only loads the original graph; it is replaced with `17x1` before dimensions or observations reach the runner. Unequal-action `9|8` policies are not supported and are not required by either paper's target manifest.

The `3x2` HalfCheetah groups follow the author source exactly: back foot/shin, back thigh/front foot, front shin/thigh. Walker uses the analogous foot/leg, thigh/opposite foot, opposite leg/thigh groups. Their unusual middle groups must not be silently replaced with a different anatomical partition.

## Validation

`tests/test_mujoco_env.py` checks normalized scaling and physical clipping with asymmetric/exact model bounds, unchanged raw samples, global-RNG preservation, deterministic interleaving/reseeding, malformed/unsupported partitions, physical-joint mismatch, terminal/timeout precedence and failed-transition lifecycle. `tests/test_mujoco_partitions.py` checks exact manifest coverage, graph isolation, observation-cache rebuilding at `obsk=None/0/1`, named-joint actuator repair, and rejection of inconsistent anatomy.

All 67 adapter/partition tests passed in the private legacy Python environment on 2026-10-25. An additional real-engine check exercised asymmetric nonzero samples at `obsk=1` for every extended partition and at `obsk=None` for Ant `8x1`. Physical controls exactly matched the expected clipped, affine-scaled, anatomically scattered values; raw samples remained unchanged. Ant with `obsk=None` returned the full 111-element state to every actor.

With the exports from [mujoco_setup.md](mujoco_setup.md), run:

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/smoke_mujoco_adapter.py
```

The script constructs all 12 supported configurations, takes two zero-action transitions each and replays one after reseeding. It checks exact state/reward replay, finite values, physical mappings, timeout reporting and unchanged caller RNG. JSON output contains configurations, dimensions and actual versions. This checks integration only; it claims no learning performance.

Verified on 2026-10-25 with Python 3.10.22 and NumPy 1.23.5: all 12 configurations passed the real CPU simulator check, including exact seeded replay and timeout at step 2. With `obsk=0`, HalfCheetah/Walker local dimensions are 6/4/2 for `2x3`/`3x2`/`6x1`; Ant local dimensions are 50/50/28/20 for `2x4`/`2x4d`/`4x2`/`8x1`. Both Humanoid variants have local dimension 3. State dimensions are 17 for HalfCheetah/Walker, 111 for Ant, and 376 for Humanoid variants. Humanoid uses physical control bounds `[-0.4, 0.4]`.

Primary references: [pinned partitions](https://github.com/schroederdewitt/multiagent_mujoco/blob/b212ddd74b258e7cea006ff1d642b5ffada4b99d/multiagent_mujoco/obsk.py), [pinned wrapper](https://github.com/schroederdewitt/multiagent_mujoco/blob/b212ddd74b258e7cea006ff1d642b5ffada4b99d/multiagent_mujoco/mujoco_multi.py), [Gym 0.10.8 TimeLimit](https://github.com/openai/gym/blob/0.10.8/gym/wrappers/time_limit.py), and [author partitions](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/obsk.py).
