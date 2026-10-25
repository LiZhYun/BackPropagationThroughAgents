# SMACv2 adapter

[`src/bta_repro/smac_env.py`](../src/bta_repro/smac_env.py) provides an optional simulator adapter. Importing it requires NumPy only; constructing a real environment imports SMACv2. Matrix tests and algorithm tests do not require StarCraft II.

The backend API is pinned to official SMACv2 commit `577ab5a2cff2391f8df582da5731ea9cd6adf3c6`, using `StarCraftCapabilityEnvWrapper` from `smacv2.env.starcraft2.wrapper`. Create its private pinned environment using the [verified setup recipe](environment_versions.md):

```bash
bash scripts/setup_smac.sh
```

StarCraft II is a separate prerequisite. Set `SC2PATH` to an existing installation containing `Versions` and `Maps`; the local shared installation is `/scratch/work/liz23/3rdparty/StarCraftII`, with version 4.10.0 / Base75689. The adapter defaults to `game_version="4.10"`. It does not install software, copy maps, save replays, or modify that installation. A `sc2_path` configuration value can select a path for backend calls while restoring the caller's environment afterwards. Startup failures remain errors; no synthetic episodes replace them.

## Scenarios and provenance

```python
from bta_repro.smac_env import SmacEnv, smac_config

env = SmacEnv(smac_config("protoss_5v5", seed=42))
try:
    observation, info = env.reset()
    # observation contains obs, state, available_actions, and active_agents.
finally:
    env.close()
```

The factory accepts Protoss, Terran, or Zerg combined with `5v5`, `10v10`, or `10v11`. The BPTA figures include all nine; AgentMixer's active table includes the six symmetric scenarios. Counts use a `10gen_{race}` map with capability-defined teams. Team types, weights, exception types, and randomized start positions follow the pinned official [Protoss](https://github.com/oxwhirl/smacv2/blob/577ab5a2cff2391f8df582da5731ea9cd6adf3c6/smacv2/examples/configs/sc2_gen_protoss.yaml), [Terran](https://github.com/oxwhirl/smacv2/blob/577ab5a2cff2391f8df582da5731ea9cd6adf3c6/smacv2/examples/configs/sc2_gen_terran.yaml), and [Zerg](https://github.com/oxwhirl/smacv2/blob/577ab5a2cff2391f8df582da5731ea9cd6adf3c6/smacv2/examples/configs/sc2_gen_zerg.yaml) examples. These details are a documented configuration choice: the accepted papers do not fully specify their capability settings. Store the complete resolved configuration with every run.

## Episode and action contract

`reset(seed=None)` returns `(observation, info)`. `step(actions)` returns `(observation, reward, terminated, truncated, info)`. Observations are copied arrays: `obs` float32 `[agents, obs_dim]`, `state` float32 `[state_dim]`, `available_actions` bool `[agents, actions]`, and `active_agents` bool `[agents]`. Actions must be legal integer indices. An agent is active if any non-noop action is available; dead agents retain the backend's noop action.

The adapter never auto-resets. A terminal or truncated step returns the final observation for value bootstrapping, and subsequent steps require an explicit reset. Losses/wins set `terminated`; `info["episode_limit"]` sets `truncated`. The backend exposes the time-limit flag only with `continuing_episode=True`, so the adapter requires this setting as a documented instrumentation change. Unexpected simulator restart-counter changes invalidate the transition and raise `SmacSimulatorError`, preventing engine failures from becoming training losses. These behaviors follow the inspected [backend implementation](https://github.com/oxwhirl/smacv2/blob/577ab5a2cff2391f8df582da5731ea9cd6adf3c6/smacv2/env/starcraft2/starcraft2.py).

StarCraft II ends a game after2^19 native ticks, according to Blizzard's
[`ResponseStep` protocol](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/sc2api.proto).
Ordinary SMAC episode resets do not reset this clock. Two200k development runs
failed after129,536 recorded training transitions with two environments, near
this limit; the old failure artifacts are retained.

The adapter now checks the native clock at `reset()`. Once it reaches
`restart_game_loop=480000`, it calls the pinned backend's `full_restart()` before
starting the next episode. The threshold leaves a whole episode and1024 ticks
of setup allowance before expiry; incompatible settings are rejected. This
changes native process lifecycle, not reward or termination rules. Capability
generators keep advancing without reseeding; the native process reuses its
constructor seed as defined by upstream. Reset/step info records the clock and
planned restart count. An additional unexpected restart remains an error.

To exercise the lifecycle quickly with a smaller diagnostic threshold:

```bash
PYTHONNOUSERSITE=1 .venv-smac/bin/python scripts/smoke_smac.py \
  --steps 512 --restart-game-loop 128 --require-restarts 1 \
  --output results/smac-restart-smoke.json
```

The512-step check is runtime validation, not benchmark evidence. Long-run
training must also pass beyond the previous failure point.

On2026-10-25 this native check passed512 transitions and12 planned restarts;
each new process reset the observed clock to4 ticks before the next episode.
Job21198088 completed with exit0. Evidence is retained in
`results/smac_restart_validation_20261025/report.json`, alongside source hashes
and native logs. The refreshed200k learning runs use the normal480000 threshold.

## Seeding

The constructor seed reaches the SC2 launch. Upstream capability distributions additionally create unseeded NumPy generators, including nested position generators; the adapter replaces them with independent child streams derived from that seed. It preserves per-environment Python and legacy NumPy random states around backend calls and restores the caller's states. This addresses the separate randomness sources visible in [distributions.py](https://github.com/oxwhirl/smacv2/blob/577ab5a2cff2391f8df582da5731ea9cd6adf3c6/smacv2/env/starcraft2/distributions.py).

The first reset may supply the matching constructor seed. Later resets use `seed=None` and advance the stream; any request to reseed in place raises an error. Construct a new environment for independent evaluation seeds or replay from the beginning. Upstream's `seed()` is a getter, not a reliable reseeding API. Real-engine reproducibility still needs empirical verification for the selected version and platform.

Only team/start-position capabilities and `prob_obs_enemy=1` are supported by this adapter. Additional stochastic capability paths require an RNG audit before enabling them. Adapter calls are serialized against one another while global random state is installed; use one environment per process for parallel rollout, since unrelated threads are outside this lock.

## Validation status

The fake-backend suite in `tests/test_smac_env.py` verifies shapes, masks, episode semantics, final observations, legal actions, simulator restarts, nested-generator seeding, independent streams, unsupported reseeding, configuration factories, and cleanup. It does not launch StarCraft II. A real-engine smoke run and multi-seed performance evaluation are separate requirements; this adapter provides no benchmark performance claim.
