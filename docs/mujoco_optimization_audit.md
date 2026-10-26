# MuJoCo optimization protocol audit

Audit date: 2026-10-25. This document compares the accepted manuscript sources, active commands in the pinned public release, and the completed 100k-step HalfCheetah `2x3`, observation-distance `0` pilot. It identifies protocol differences; it does not establish which difference caused the pilot returns or claim that the release commands generated the published curves.

The largest verified differences are fresh batch size (4,000 versus 800 transitions), value-target normalization and clipped Huber loss versus raw MSE, and initial Gaussian standard deviation (about 0.366 versus 1). The release also uses recurrent networks, LayerNorm, agent-specific centralized critics, and a different observation-normalization convention. The pilot's modifier learning rate already matches the active Lumi command.

## Evidence and scope

- Canonical BPPO source: [accepted BPTA TeX](../references/bpta/Formatting-Instructions-LaTeX-2024.tex), common parameters at lines 471–499 and MuJoCo table at 584–607. These are active manuscript content. The Ant observation override is at 613–626.
- Canonical AgentMixer source: [accepted AgentMixer TeX](../references/agentmixer/aaai25.tex). The continuous IGC construction at approximately 404–419 is active. Its appendix parameter tables, including modifier learning rate `5e-5` at 1117–1121 and MuJoCo parameters at 1275–1310, are **commented source**, not active accepted text. They remain useful provenance but have lower evidential status.
- Public implementation: `LiZhYun/BackPropagationThroughAgents`, immutable commit `13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0`. Links below resolve to this commit. Files were read through the GitHub connector after checking existing local captures; references were not modified. Active shell commands establish what those scripts configure, not proof of historical execution or linkage to a particular figure.
- Completed pilot: [frozen source](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/), [MAPPO configuration](../results/mujoco_hc2x3_pilot_20261024T233938Z/mappo_seed0/config.json), and [AgentMixer configuration](../results/mujoco_hc2x3_pilot_20261024T233938Z/agentmixer_seed0/config.json). This audit uses that snapshot rather than later edits in `src/`. The subsequent longer control is outside this audit.

## Which release commands apply

The Lumi [modifier launcher](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/lumi/train_mujoco.sh#L13-L42) and [no-modifier launcher](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/lumi/train_mujoco_ar.sh#L13-L31) both select algorithm `temporal`, `agent_obsk=0`, and Slurm seeds 0–4. Their respective [modifier wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/lumi/train_mujoco_all.sh#L4-L6) and [no-modifier wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/lumi/train_mujoco_all_ar.sh#L4-L6) actively submit HalfCheetah `2x3`, `3x2`, and `6x1`.

The `_ar.sh` filename alone does not identify plain ARPPO: the selected temporal runner dispatches no-modifier training to `train_seq_agent_m` under default `train_sim_seq=0`; with `use_action_attention`, it dispatches to `joint_train`. This is the BPPO-style versus AgentMixer-style release path audited here. See [runner dispatch](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/mujoco_runner.py#L100) and [configuration](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/config.py#L275-L280).

The Triton [launcher](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/triton/train_mujoco.sh) omits `--agent_obsk`, inheriting `None`. Its [wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/temporal/mujoco/triton/train_mujoco_all.sh#L4-L10) comments out HalfCheetah submissions and activates Ant submissions. It should not be substituted for the Lumi HalfCheetah observation protocol.

## Batch and optimizer settings

| Setting | Active BPTA table / commented AgentMixer table | Effective Lumi release paths | Completed 100k pilot |
|---|---|---|---|
| Fresh training batch | 40 environments × 100 rollout steps | 4,000 transitions | 8 × 100 = 800 transitions |
| PPO epochs; minibatches | 5; 1 | 5; 1 | 5; one full batch per epoch |
| Total interaction budget | 10 million | 10 million | 100,000 |
| Environment episode limit | Not established by the table's “episode length 100” | 1,000 in environment construction; 100 is rollout length | 1,000; rollout length 100 |
| Actor / critic learning rate | `3e-4` / `3e-4` | `3e-4` / `3e-4` | `3e-4` / `3e-4` |
| Modifier learning rate | AgentMixer commented table: `5e-5` | `3e-5` with modifier | `3e-5` |
| Learning-rate decay | Unspecified in these tables | Disabled by default; launchers do not enable it | Constant |
| Adam epsilon; weight decay | `1e-5`; 0 | `1e-5`; 0 | `1e-5`; 0 |
| Gradient norm cap | 10 | 10 | 10 |
| Policy clipping; entropy coefficient | 0.2; 0 | 0.2; 0 | 0.2; 0 |
| Discount; GAE lambda | 0.99; 0.95 | 0.99; 0.95 | 0.99; 0.95 |
| Value-loss coefficient | Unspecified | 1, applied to the configured Huber loss | 0.5, applied to raw mean squared error |

Launcher overrides are at the links above. Remaining effective settings come from the actual `argparse` declarations in [config.py](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/config.py#L339-L420), not its sometimes stale help prose. Actor and critic each use their own [Adam optimizer](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/bta/algorithm/temporalPolicy.py#L23-L34); the [modifier optimizer](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/base_runner.py#L192-L195) is separate. The no-modifier launcher lists `attention_lr=3e-4`, but no modifier optimizer is constructed on that path. `use_linear_lr_decay=False`; the actor/critic decay routine is called only if that flag is enabled. The active modifier command does not introduce another LR schedule.

With five epochs and one minibatch, at 100k interactions the release configuration performs 25 rollout updates, whereas the pilot performs 125. Thus the pilot takes five times as many optimizer steps per interaction for each updated component, on batches one fifth as large. Both reuse each collected transition five times; these are different batch/update schedules, not different nominal epoch counts. The comparison does not assume that an AgentMixer joint step and a BPPO per-agent reverse step are computationally identical.

## Rewards, returns, and critics

The release copies the scalar environment reward to each agent without reward standardization. `ValueNorm` normalizes **return targets**, not rewards: the trainer updates its return normalizer, compares predictions to normalized returns, and uses denormalized value predictions when computing GAE. Defaults are ValueNorm enabled, PopArt disabled, clipped value loss enabled with width 0.2, and Huber loss enabled with delta 10. The clipped/unclipped losses are combined by their maximum. See [value loss](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/bta/t_policy.py#L98-L143), [buffer returns](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/utils/separated_buffer.py#L176-L258), and [environment reward handling](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py#L146-L170).

The actual [ValueNorm implementation](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/utils/valuenorm.py#L10-L70) uses debiased exponential first and second moments, beta `0.99999`, epsilon `1e-5`, and variance floor `1e-2`. Default `per_element_update=False` means one moment update per call. `cal_value_loss` updates these moments again on each training minibatch, including repeated epochs. ValueNorm does not perform PopArt's output-layer rescaling.

The release creates **one distinct actor, critic, trainer, value normalizer, and buffer per agent**, for both audited paths. Each critic receives that agent's `share_obs` entry. With default `use_centralized_V=True`, this is the full environment state with the corresponding one-hot agent ID appended, then normalized as described below. It is not one parameter-shared critic, nor a critic of concatenated local observations. `use_obs_instead_of_state=False` by default; neither Lumi command enables it. The MuJoCo runner only substitutes local observations when `use_centralized_V` is disabled. See [per-agent construction](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/base_runner.py#L158-L190), [warmup](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/mujoco_runner.py#L147-L156), and [state construction](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py#L206-L216).

The pilot uses one centralized critic of the global state, raw returns, and `0.5 * mean((value-return)^2)`, with no target normalization, value clipping, or PopArt. Raw rewards are retained in both implementations. A scalar coefficient comparison alone therefore does not describe the effective difference in critic loss. Frozen implementation: [runner.py](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/runner.py), value update around lines 343–351.

The release defaults `use_proper_time_limits=False`, and these launchers leave it unchanged; its GAE path uses done masks at time limits. The redesigned pilot distinguishes termination from truncation and bootstraps a time-limit transition from its terminal next state. This is an intentional semantic correction and should not be silently removed to match historical defaults.

## Observation normalization and network structure

The author wrapper appends an agent ID to each local observation and to each agent's copy of global state, then computes `(vector - vector.mean()) / vector.std()` **within that single vector**. The ID entries participate in the normalization. There are no running moments, clipping, or epsilon in these two expressions. See [observations](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py#L173-L204) and [states](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py#L206-L215).

The pilot instead maintains training-only running moments **per observation feature and agent**, plus per global-state feature, and clips normalized values to ±10. Evaluation freezes the training normalizer. The recorded pilot dimensions are local observation 6 and global state 17, without appended IDs. Consequently, “observations normalized” does not establish protocol equivalence. See its [normalization implementation](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/normalization.py) and saved observation-normalization metadata.

The manuscript table specifies two hidden layers of width 64 and ReLU. The release sets `layer_N=2`, but [MLPLayer](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/mlp.py#L10-L31) implements one input affine/ReLU/LayerNorm block **plus two additional hidden blocks**. Input LayerNorm is separately enabled by `use_feature_normalization=True`. Hidden and output weights use orthogonal initialization; the action-head gain defaults to 0.01.

Further, `use_recurrent_policy=True` by default and is not disabled by either Lumi command. Only MAT/MAT-Dec are forced non-recurrent by the entry point. Thus the [actor and critic](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/bta/algorithm/r_actor_critic.py) and [modifier](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/action_attention.py#L28-L56) include GRUs. `data_chunk_length=1` truncates the training chunks; it does not remove recurrent state carried during rollout/evaluation. The frozen pilot uses two width-64 tanh hidden layers and feedforward policies/critic. Accepted source does not establish that these release architecture details were used for the plotted experiments.

## Gaussian scales and action handling

The release's [local DiagGaussian](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/distributions.py#L112-L134) uses a state-independent parameter named `log_std`, but transforms it as

```text
local_std = sigmoid(parameter / std_x) * std_y
parameter_initial = std_x = 1
std_y = 0.5
initial local_std = sigmoid(1) * 0.5 ≈ 0.365529
```

This is not an exponentiated log standard deviation. Its mathematical scale range is `(0, 0.5)`. The Box [ACTLayer constructor](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/act.py#L23-L30) does not pass `args` into `DiagGaussian`, so the local head uses those built-in defaults even though launchers also list `std_x_coef` and `std_y_coef`.

For AgentMixer, the modifier computes `joint_std = 0.75 * sigmoid(0.3 * head(features))` under the Lumi overrides. It retains the individual means and **replaces** the local scale; it does not add a log-scale residual to local scales. The head has orthogonal gain 0.01 and zero bias, so scale is near 0.375 initially, not identically 0.375 for every input. The live collection path always samples this joint Normal; the threshold-based mixture alternatives are commented. See [modifier forward](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/action_attention.py#L107-L152) and [sampling](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/mujoco_runner.py#L210-L244). The active accepted IGC construction preserves means and allows modified scales, but does not specify these numerical bounds or initializers.

The pilot MAPPO and autoregressive policies exponentiate state-independent log-scale parameters initialized to zero, giving initial standard deviation 1, bounded by log-scale clipping `[-5, 2]`. AgentMixer instead has an observation-dependent local log-scale head initialized to zero and a zero-initialized modifier log-scale residual; joint standard deviation also starts at 1 with the same clipping. The `agentmixer_identity` control retains this AgentMixer local family while disabling the modifier, so it is the direct control for the modifier. MAPPO additionally differs in local scale parameterization. Frozen implementations: [core.py](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/core.py), [bppo_continuous.py](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/bppo_continuous.py), and [agentmixer_continuous.py](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/agentmixer_continuous.py).

Both use unsquashed Gaussian samples. The release's `NormalizedActions.action` affinely maps nominal normalized coordinates `[-1,1]` to physical action bounds, **without explicit clipping in that wrapper**. The pilot stores/scores raw Gaussian actions and clips only when applying them to the environment. This audit does not infer additional clipping inside the historical physics backend. See [author wrapper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/envs/ma_mujoco/multiagent_mujoco/mujoco_multi.py#L13-L22) and [current adapter rationale](mujoco_adapter.md). The corrected joint-action index mapping must also be retained as a documented repair.

The release `FixedNormal.log_probs` returns per-coordinate log densities rather than an event sum. This audit does not certify every downstream surrogate reduction in the release. The redesigned algorithms intentionally sum event log probabilities before likelihood ratios; those algorithmic repairs are documented separately and should not be replaced merely to match this source. See [distribution helper](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/distributions.py#L50-L59), [continuous BPPO](bppo_continuous.md), and [continuous AgentMixer](agentmixer_continuous.md).

## Evaluation convention

| Setting | Active Lumi release protocol | Completed pilot |
|---|---|---|
| Training run seeds | Slurm array 0–4 | One development seed, 0 |
| Training environment seed | `seed + rank * 1000` | Separate reproducible `SeedSequence`-derived environment streams recorded in run metadata |
| Evaluation environment seed | `seed * 50000 + rank * 10000`, seeded at construction | 1,000,000, used to create a fresh evaluation environment on each call |
| Evaluation environments / episodes | 1 by default / 5 complete episodes | 1 / 5 complete episodes |
| Action selection | Deterministic means | Deterministic means |
| AgentMixer execution | Individual actors, modifier omitted | Individual actors, modifier omitted |
| BPPO execution | Ordered autoregressive actors | Ordered autoregressive actors |
| Evaluation schedule | Every 25 rollout updates, including update index 0 | Every 20,000 steps and final evaluation |
| Within-call aggregation | Mean of complete episode return arrays; agent rewards are duplicated team rewards | Mean and uncertainty statistics over complete scalar team episode returns |

Release evidence: [environment seeds and 1,000-step limit](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/scripts/train/train_mujoco.py#L16-L57), [deterministic execution](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/mujoco_runner.py#L250-L277), and [episode evaluation](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/runner/temporal/mujoco_runner.py#L328-L368). Pilot evidence: frozen [simulator_train.py](../results/mujoco_hc2x3_pilot_20261024T233938Z/source/bta_repro/simulator_train.py), around lines 240–297.

The release resets the persistent evaluation environments at each checkpoint but does not reseed them there; the pilot recreates the same seeded evaluation stream on each call. In the release seed-0 run, train rank 0 and evaluation rank 0 both start from seed 0. The pilot deliberately uses independent train/evaluation streams; final held-out evaluation must remain separate from configuration selection.

Derived from the runner loop and launch settings, release evaluations occur at 4,000, 104,000, …, 9,904,000 interactions, not at exactly 0 and 10 million. These coordinates agree with the extracted figure sampling grid, but this agreement alone does not prove that those launchers generated the figures. The BPTA text says mean and standard deviation over five seeds; the exact historical seed IDs are not established by the accepted text. AgentMixer's MuJoCo cross-seed mean/median convention remains unresolved; its SMAC median caption does not settle MuJoCo aggregation. See [curve target provenance](curve_targets.md).

## Implications for the next controlled pilot

1. Keep the frozen longer control as the reference. A separate `40 × 100`, five-epoch, one-minibatch condition tests the batch schedule explicitly. Report interaction budget and rollout-update count together; a 100k development run covers only 1% of the release's 10M budget.
2. Add an explicit, named Gaussian-scale control with initial standard deviation approximately 0.366 and documented bounds. Apply matched local-family controls to AgentMixer with and without the modifier. If testing the release's bounded replacement scale, label it separately from the present log-scale residual construction; changing initialization alone does not reproduce that family.
3. Test the critic protocol as a separate declared change: return-target ValueNorm plus clipped Huber loss, recording raw returns and value predictions in original reward units. Keep PopArt disabled unless it is an explicit additional condition. Compare at a matched batch schedule and environment budget so the critic change has an interpretable control.

ReLU/LayerNorm, recurrence, agent IDs, per-vector normalization, and per-agent critics remain additional departures to track, rather than silently bundle into a claimed exact reproduction. Preserve intentional repairs to action mapping, event likelihoods, independent evaluation, and terminal/truncated bootstrapping. None of the source evidence demonstrates that any single protocol change will recover the paper's performance. The completed single-seed pilot establishes executable training and measurable behavior, not reproduced or exceeded MuJoCo results.

## Implemented initial-exploration control

The three continuous policy constructors now accept keyword-only `initial_std=1.0`: `ContinuousIndependentPolicy`, `ContinuousAutoregressivePolicy`, and `ContinuousAgentMixerPolicy`. The value must be finite and lie in `[exp(-5), exp(2)]`, the existing representable scale range. Independent and autoregressive actors initialize their state-independent log-scale parameters to `log(initial_std)`; AgentMixer initializes each local log-scale head's bias to that value while keeping its weights zero. Mean initialization, log-scale bounds, subsequent parameterization, and the modifier's zero residual are unchanged. The default remains bitwise identical under the same initialization seed, with no checkpoint shape changes.

For example, `initial_std=0.36552928931500245` matches the audited release's initial local Gaussian standard deviation. This is an **initial-noise-only control**: it does not reproduce the legacy upper bound of 0.5, the modifier's bounded replacement scale, or its initial joint scale near 0.375. AgentMixer's initial joint distribution still equals its local distribution. Compare this setting with the default at otherwise matched training settings; any performance conclusion requires actual controlled runs.

An additional 2026-10-26 control exposes `continuous_modifier_scale="bounded"` for AgentMixer. It applies the released `0.75 * sigmoid(0.3 * head)` replacement rule with an explicit `exp(-5)` numerical floor. Network initialization remains the same zero head as the residual control, giving exactly 0.375 initially; it does not copy the release's small orthogonal head. Local scale features remain observation-dependent and undetached. See [continuous AgentMixer](agentmixer_continuous.md) for the precise scope and identity-control restriction. This is an isolated output-parameterization experiment, not a claim to reproduce all released architecture choices.
