# Predator–Prey protocol audit

The accepted AgentMixer archive specifies capture reward 10 and single-agent
penalties 0 and −1, but its Predator–Prey appendix is commented out
(`references/agentmixer/aaai25.tex:1009–1015`). The active paper points to those
additional experiments. The supplied `Prey_main.pdf` has a 1M-step axis.
The released code resolves the mechanics below; it does **not** prove which
historical run produced the figure.

The implementation is pinned to author repository commit
`13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0`. Sources inspected through the GitHub
connector were:

- [Environment](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/predator_prey/predator_prey.py): constructor lines 42–95; reset/observation 114–168; movement 179–235; capture/prey movement/reward 269–308; seed 349–352.
- [Wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/predator_prey/predatorprey_wrapper.py): all actions available; shared observation concatenates local observations; the extra predator-visibility channel is commented out.
- [Training entry point](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/train/train_predatorprey.py): construction 21/40; count defaults 56–60; seeding 26/45/134–137.
- [Temporal launcher](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/predator_prey/lumi/train_predator_prey.sh): array/seed IDs 0–4, penalty argument, 1M transitions, rollout length 200, 50 training environments, one minibatch, 100 evaluation environments, evaluation every 10k transitions, modifier enabled.
- [Temporal runner](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/predator_runner.py): evaluation action selection 224–254 and reduction 290–323.
- [Vector wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/env_wrappers.py#L1345-L1359): automatically resets a completed environment before returning its observation.

The environment Git blob is `18ad171d94148f86e653b10fbd58c81fb910971c`, SHA256
`7f93327c4923f48c62961c1bb91620562124cf13fa67f1a71e64a1dbc1d45f47`.
The wrapper blob is `fad0e85e9568fc86bdc281a6e0606dfac0d6f78a` and temporal runner
blob is `252571af6412ac581b218ad0e57dfcd75fe5a2a3`. These are source identities,
not evidence that the historical experiments used that commit or dependency set.

## Exact released mechanics

| Item | Protocol |
| --- | --- |
| Geometry | Bounded 5×5 empty grid; no wraparound, walls inside the grid, or obstacles |
| Entities | Two controlled predators and one prey; each occupies a distinct cell |
| Initialization | Place predators by independent uniform row/column rejection draws, in index order; place prey in an empty cell with no cardinally adjacent predator |
| Actions | 0 down, 1 left, 2 up, 3 right, 4 stay; all five remain available even at boundaries or beside occupied cells |
| Predator movement | Process agent 0 then agent 1; only enter currently empty in-bounds cells. A later agent can follow into a vacated cell; swaps are blocked |
| Observation | 28 floats: row/5, column/4, row-major 5×5 binary prey-visibility window centered on the predator, elapsed_steps/128 |
| Hidden information | No other-predator channel, agent ID, prey identity, or explicit boundary channel; off-grid/unseen entries are zero |
| Critic state | Concatenation of the two local vectors, length 56. This is joint observation, not an omniscient prey-position vector |
| Step reward | Shared −0.01, then single-neighbor penalty 0 or −1, or cooperative capture reward +10 |
| Capture | After both predators move, count cardinal neighbors of prey; exactly one causes a penalty without capture; two capture and remove prey immediately |
| Prey proposals | Probabilities `(0.175, 0.175, 0.175, 0.175, 0.3)` in the same action order; up to five proposals while alive |
| Prey rejection | Reject a proposal if its destination has a cardinal predator neighbor. Once accepted, move only if the destination is vacant and in bounds; otherwise stay without another draw. Five rejected proposals also mean stay |
| End | Capture ends immediately; otherwise the episode ends after 128 steps |

Several misleading legacy comments are not the executable protocol: the
constructor's description mentions reward 1, uniform prey movement, agent IDs,
and a 7×7/four-predator variant. The actual launcher uses the defaults above,
reward 10, and biased movement probabilities. Its `episode_length=200` controls
rollout collection and evaluation windows; it is never passed as `max_steps` to
the environment. A single capture adds 10 to **each** predator's identical
reward, so the new scalar team reward retains one copy rather than summing to 20.
For an episode of length L with C captures and S single-neighbor attempts,
return is `10*C + penalty*S - 0.01*L`. With one prey, C is at most one and raw
episode return cannot exceed 9.99. Do not compare this bound directly with an
unverified plotted-unit transformation.

The asymmetry row/5 versus column/4 and the danger-before-vacancy rejection
order are preserved deliberately. For example, a prey proposal into a predator's
own cell can pass the neighbor test and then fail vacancy; a wall-directed
proposal can likewise end the sampling loop. These affect transition
probabilities and are covered by tests.

## Seeding and adapter differences

The legacy `seed()` initializes `self.np_random`, but reset uses Python's global
`random.randint` and prey movement uses global `np.random.choice`. Neither uses
that seeded member. The entry point seeds global NumPy once but does not seed
Python random. Thus the advertised training seeds `seed + rank*1000` and
evaluation seeds `seed*50000 + rank*10000` do not establish independent,
reproducible environment streams. Exact original seed trajectories cannot be
recovered from this source alone.

[PreyEnv](../src/bta_repro/prey_env.py) uses a private `random.Random(seed)` for
placement and a private `np.random.RandomState(seed)` for prey proposals.
`reset()` advances those streams; `reset(seed=...)` recreates both. Environment
instances cannot perturb one another or the caller's RNG. RandomState is
intentional: when both legacy global generators are manually seeded to the same
value, the new mechanics reproduce their draws exactly. This repairs isolation
without changing the stochastic transition or initialization distributions.

The adapter exposes `reset() -> (observation, info)` and
`step(actions) -> (observation, reward, terminated, truncated, info)`. It never
autoresets, retains the final observation, and rejects step calls until reset
after an episode ends. Capture is termination; reaching the time limit without
capture is truncation. Capture takes precedence if both happen on step 128.
The legacy environment exposes only a combined done flag, and its subprocess
wrapper loses the terminal observation by autoresetting. Our separate flags
follow this repository's timeout-bootstrap convention; they do not reproduce
the old runner's treatment of every done as terminal.

Observations/state are new float32 arrays, masks are fresh boolean arrays, and
incoming actions are never mutated. All predators remain active; an ended
episode is represented by the episode flags, not agent death. Rendering and
adjacency helpers unused by this benchmark are omitted. No Gym, PettingZoo,
Pillow, engine install, or vendored legacy environment is required. This is an
independent implementation of the released task, not a replacement environment.

Only the paper-target 5×5/two-predator/one-prey partial-observation protocol is
supported. Other geometry/count/full-observation arguments are rejected.
`episode_limit` may explicitly shorten the 128-step limit for diagnostics;
that is a configuration change and must not be reported as the canonical task.

## Selected comparison metric and historical qualifications

The [figure extraction](other_curve_targets.md) estimates AgentMixer endpoints
at 1M as 2.21 [2.19, 2.23] for penalty 0 and 0.36 [0.35, 0.36] for penalty −1.
These ranges are digitization allowances, not uncertainty across seeds. The
commented text calls the quantity average test-episode return; the released
runner computes a different quantity:

1. Reset each evaluation environment, then collect exactly 200 decisions per
   environment, automatically resetting after captures or timeouts.
2. Select deterministic individual-policy actions. With the action modifier
   enabled for training, evaluation uses zero cross-agent execution masks.
3. Sum the rewards over those 200 decisions for each agent/environment, average
   across environments, then divide by 200 and multiply by observation upper
   bound 1. The resulting metric is **mean reward per decision**.

The launcher's 100 evaluation environments therefore yield 20,000 decisions per
evaluation call. Although it passes `eval_episodes=100`, this runner's evaluation
loop does not use that argument to stop after 100 completed episodes.
On 2026-10-25 the author [confirmed evaluation return](author_clarifications.md)
for both Prey figures. On 2026-10-26 the author selected the released evaluation
protocol for reproduction. The **primary comparison metric is therefore mean
reward per decision over 100×200-decision windows**, including unfinished
segments. This is an author-selected convention; no plotting script/run ID
links the historical values to this exact logger key.
Do not relabel or rescale the original plotted endpoints as proven raw episodic
returns. Cross-seed aggregation and the band statistic remain unresolved. The
launcher seeds 0–4 are an example released configuration, not established
historical seed IDs for the figure; its broken environment seeding adds another
qualification.

New experiments should retain raw episodic return, episode length, capture
rate, and mean reward per decision as separate named metrics. An exact legacy
logger comparison additionally needs a fixed 200-decision, autoreset evaluation
window per environment, rather than an average of per-episode reward rates.
The author selection supplies the reduction for the reproduction comparison.
Before claiming superiority, match the selected evaluation protocol and budget,
report both mean and median across independent training seeds while the original
aggregation is unknown, and account for the digitization reading range. No training result
or reproduction claim is supplied by this implementation audit.

## Integration and validation

The adapter has `is_discrete=True`, `num_agents=2`, `num_actions=5`, `obs_dim=28`,
`state_dim=56`, `config`, `versions`, and `close()`. Example construction:

```python
from bta_repro.prey_env import PreyEnv

env = PreyEnv({"penalty": -1, "seed": 42, "episode_limit": 128})
observation, info = env.reset()
# observation keys: obs, state, available_actions, active_agents
observation, reward, terminated, truncated, info = env.step([4, 4])
env.close()
```

The shared trainer accepts `environment="prey"` and constructs `PreyEnv` with
the existing independent training seeds. Optional observation normalization is
an explicit protocol choice. Capture success is `info['capture_success']` and
is logged separately from SMAC/Football win rate. Projected predecessor actions
are available for AR-PPO/BPPO through `action_embed_dim`. Experimental two-agent
joint AgentMixer is selected with `agentmixer_correlation="joint"`; its joint
likelihood is scored once per transition while evaluation uses local actors.

Set `eval_window_steps=200` and `eval_windows=100` to add the released logger's
fixed-window reduction. Each independently seeded window starts from reset,
autoresets after captures/timeouts, and includes rewards from any unfinished
episode at the final step. Results retain each window's reward sum, completed
captures, and unfinished segment. The aggregate is total reward divided by
all20,000 evaluation decisions. These decisions are counted separately from
training, and the episode evaluator's metrics remain unchanged. Frozen training
normalization and preserved RNG states keep evaluation from affecting learning.

The100k development pilot used episode evaluation only. The300k coordination
controls add20 windows of200 steps for faster diagnosis; this is fewer evaluation
windows than the released100, explicitly recorded in configuration. Matching
the logger's reduction does not resolve the historical figure-to-logger link.

Validation on 2026-10-25:

- `tests/test_prey_env.py`: 64 tests passed, covering initialization, exact
  feature layout, action numbering, sequential collisions, capture/penalties,
  rejection limits, blocked proposals, terminal precedence, array ownership,
  independent seed streams, invalid input, lifecycle behavior, and a real rollout
  collector check of timeout-observation bootstrapping before reset.
- A differential check against the verified legacy environment blob matched
  10,240 transitions: both penalties × 20 seeds × 256 random joint decisions.
  It included 90 completed episodes and 44 captures. Float32 observations,
  concatenated state, positions, identical per-agent/shared rewards, prey-alive
  flags, and combined done flags matched exactly, including resets.
- The differential harness extracted the original class unchanged, replacing
  only Gym space metadata objects and the unused image drawing method. Both
  legacy global RNGs were manually seeded to correspond to the repaired
  private streams. It exercised environment behavior, not policy training.
  The temporary original source and its MIT license were kept outside the
  package; no legacy class or rendering code is vendored in the adapter/tests.

The author repository's [MIT license](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/LICENSE)
credits Zhiyuan Li (2024), Git blob `7be2abfcbb4f82a6b97165411ae7093a810a8a1b`.
The new adapter is independently written from the audited mechanics, with
source attribution retained here and in its module docstring. It uses the
repository's existing NumPy dependency; no legacy Gym/Pillow version or native
engine is needed to execute it. Historical package versions and exact figure
run provenance remain unverified.
