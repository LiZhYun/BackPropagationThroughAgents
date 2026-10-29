# Continuous AgentMixer

The accepted source, the accepted AgentMixer LaTeX source (locally supplied; excluded from public checkout), defines continuous Individual-Global-Consistency by retaining the local Gaussian means while allowing the centralized policy modifier to determine standard deviations. [`src/bta_repro/agentmixer_continuous.py`](../src/bta_repro/agentmixer_continuous.py) implements that construction directly.

Each agent has an independently parameterized actor whose local observation produces a mean vector. By default, the observation also produces diagonal standard deviations; an optional state-independent scale control is described below. The centralized modifier embeds each agent's mean and standard deviations, concatenates global state features, and applies the same alternating agent/channel mixing blocks as the discrete implementation. Its output is a residual to the local log standard deviations:

```text
local_std = exp(clamp(local_log_std, -5, 2))
joint_mean = local_mean
joint_std = exp(clamp(log(local_std) + modifier(local_mean, local_std, state), -5, 2))
```

The bounds are numerical and exploration choices, not constants specified by the theorem. They give strictly positive standard deviations between approximately `0.0067` and `7.389`. The modifier's final layer initializes to zero, so the initial centralized and local distributions agree. The modifier controls exploration without any path from the state into the means. Means are unbounded. Since a nonsingular Gaussian's mode equals its mean, local and centralized **raw Gaussian** modes agree exactly.

The local actors and modifier receive PPO gradients through the centralized distribution. The modifier receives mean and standard deviation tensors without detaching them. Its hidden layers begin learning after its zero-initialized output layer changes. `local_actions(obs)` returns only local means; `sample(..., decentralized=True)` samples the local policies without invoking the modifier or using a state.

## Bounded replacement scale control

The default is `continuous_modifier_scale="residual"`. The simulator configuration can instead select `"bounded"`, which uses the output map audited in the [released implementation](mujoco_optimization_audit.md):

```text
score = modifier(local_mean, local_std, state)
joint_std = max(exp(-5), 0.75 * sigmoid(0.3 * score))
joint_mean = local_mean
```

This replaces the local standard deviations rather than adding their logarithms. Local scales still enter the modifier's features without detachment, and local means remain exactly unchanged. The upper scale bound is 0.75. The positive floor `exp(-5)` is a documented numerical safeguard; it is not present in the released sigmoid expression. At the floor, the clamp has zero gradient below the boundary.

Both options use identical network weights and RNG consumption at initialization. A zero modifier head therefore gives a centralized standard deviation of exactly 0.375 for the bounded option, independently of `initial_std`. This deliberately holds initialization fixed to isolate the output map; the release used a small orthogonal head and started near, rather than exactly at, 0.375. This control does not reproduce the release's state-independent bounded local scales, recurrent architecture, or normalization. The parameter name `log_std_head` remains in saved weights for compatibility, but represents an unconstrained score under the bounded option.

The bounded option requires continuous AgentMixer with the modifier enabled. A frozen zero bounded head would produce constant 0.375 exploration, not an identity modifier, so that combination is rejected. The existing identity control uses the residual map. The selected option is stored in run and checkpoint configurations and reconstructed during saved-policy evaluation.

`configs/mujoco_hc2x3_modifier_scale1m.json` compares residual, bounded and identity variants at the same 1M-step budget in development seeds 0–2. Performance conclusions require those runs; the scale bounds alone do not establish better exploration or learning.

## Local uncertainty control

The simulator option `continuous_local_scale="observation"` retains the original observation-dependent local scale head. Selecting `"state_independent"` gives each agent its own trainable vector, with one log standard deviation per action coordinate. The means still depend only on that agent's observations. Both choices use `exp(clamp(log_std, -5, 2))`, the same initial scale and the same modifier features, with gradients through those features. A state-independent **local** scale can still produce a state-dependent **centralized** scale through the modifier.

This isolates a difference identified in the [release audit](mujoco_optimization_audit.md): the released local Gaussian uses state-independent uncertainty. Its bounded `0.5 * sigmoid(parameter)` map is a separate difference and is not introduced here. Recurrent networks, activation functions and normalization also remain separate questions.

Both choices initialize identical mean and modifier weights, consume identical RNG, and produce identical initial distributions within each modifier variant. The constructor initializes the original scale head before replacing it with its bias vector for the state-independent option; the unused head is then removed. The default retains its existing checkpoint keys. The selected option is recorded in configurations and restored by the saved-policy evaluator. It is supported only for continuous AgentMixer, including the identity control.

The [registered study](../experiments/hc2x3_local_scale_study_20261029.json) compares residual, bounded and identity variants under both local-scale choices, retaining all three development seeds per condition. It keeps entropy at 0.01: the completed matched entropy-zero study lowered all three final means and curve areas. Six native 12k-step integration runs and checkpoint replays precede the 18-run 1M learning comparison. Native checks are queued; learning is unsubmitted. Neither budget establishes paper reproduction, and any selected setting requires evidence matched to the claimed budget and unused confirmation seeds for performance claims; full-budget execution is optional.

The [compatibility audit](../experiments/continuous_local_scale_default_parity_20261029.json) verifies exact default weights, scores, samples, gradients, RNG and one Adam step against the previous committed implementation in 24 cases. Native default-prefix equivalence remains a gate to check when the queued runs finish.

## Optional local history

Set `recurrent_actor=true` for continuous AgentMixer to give each local Gaussian actor a private GRU. The identity control and both local-scale choices support it. The default remains feedforward; continuous MAPPO, AR-PPO and BPPO still reject recurrence.

```text
features = two_layer_tanh_encoder(local_observation)
memory = GRU(features, stop_gradient(incoming_memory))
local_mean = mean_head(memory)
local_std = exp(clamp(local_scale_head(memory), -5, 2))
joint_mean = local_mean
joint_std = existing_modifier_scale_map(local_mean, local_std, state)
```

With `continuous_local_scale="state_independent"`, the learned per-agent vector replaces `local_scale_head(memory)`. The modifier receives the resulting means and scales, so it can use local history through those features. Global state enters only the modifier, and no other agent's observation or memory enters a local actor. The modifier and shared critic remain feedforward. Local and centralized raw Gaussian modes remain exactly equal for the same incoming histories, including after PPO updates.

The shared runner stores incoming memory with each raw observation, recomputes the encoder and GRU during PPO, and holds the incoming memory fixed. Gradients span one decision. Memory continues across rollout boundaries and resets separately for each terminated or truncated environment. Evaluation starts from empty memory for each episode; checkpoints reconstruct the architecture from the existing configuration flag. [Memory contract](recurrent_actors.md).

This tests the documented local-history difference from the release. It does not add the released recurrent modifier/critics, ReLU/LayerNorm architecture, or bounded local scale map. GRU construction adds random draws, so feedforward and recurrent models do not promise identical common initial tensors. All existing modes are unchanged: [29 compatibility cases](../experiments/continuous_recurrent_default_parity_20261029.json) match bitwise through two PPO epochs, including gradients, Adam states, samples, memory advancement and RNG.

The [frozen study](../experiments/hc2x3_recurrent_study_20261029.json) contains twelve native checks: residual, bounded and identity modifiers × two local-scale choices × feedforward/GRU. Each trains for 44k decisions, eleven 40×100 rollouts, crossing the 1,000-step horizon in all 40 training environments before a further rollout. A read-only memory observer, exact final checkpoint replay, six feedforward prefix comparisons and resource review are required before learning. These checks are now queued as **21284086**, followed by corrected complete-family auditor **21289017**. The [execution record](../experiments/continuous_recurrent_gate_execution_20261029.json) freezes all eight helper files, 25 source modules and twelve configurations. The original study records its earlier unsubmitted state at freeze time.

Eighteen recurrent 1M development configurations are separately frozen, to be compared with all eighteen earlier feedforward local-scale configurations. Both families retain seeds 0–2 and every modifier/scale combination. The earlier study keeps priority; no learning job has been added. One million steps is only 10% of the paper budget. Any performance claim still needs matched evidence and unused final training seeds. Full-budget development is optional under the current release scope.

The observer recomputes each memory transition with the collection-time weights and checks incoming memory, carry across rollouts, and individual termination/truncation resets. It also checks that observation consumes no RNG and changes no policy or critic weights. In both the [core runtime](../experiments/continuous_recurrent_gate_preflight_20261029.json) and [private MuJoCo runtime](../experiments/continuous_recurrent_gate_private_preflight_20261029.json), all twelve observed/unobserved policy conditions match bitwise through three synthetic PPO updates; 47 deliberately corrupted cases are rejected. Each runtime also passes a synthetic 44k-step test with all forty horizon resets and exact final checkpoint replay. These synthetic checks do not establish native MuJoCo behavior.

The native collector requires every declared condition, all 480 training resets, 264,000 GRU memory transitions, 120 original evaluation episodes, 24 replay episodes, and six exact feedforward prefixes against the prior local-scale gates. Missing or invalid conditions fail the audit. The original six local-scale gates still require their own complete audit; measured native resources and a free allocation are required before any learning release. [Dependencies and allocation](worker_schedule.md#independent-allocations-and-prey-learning-29-october-2026).

The current [verification](../experiments/continuous_recurrent_verification_20261029.json) includes 41 new checks for continuous memory, finite-difference gradients, mode equality, raw Normal likelihoods, independent resets, optimizer ownership, and exact saved-policy replay on synthetic environments. That source suite passed 1,044 tests with one CUDA-only skip; the current scale-control verification below passes 1,093 tests. This is correctness evidence; native MuJoCo behavior and performance improvement remain unverified.

## Raw policy and actuator commands

`sample` returns raw Normal actions `[batch, agents, action_dim]`, together with log probabilities summed over action coordinates `[batch, agents]`. `evaluate_actions` accepts those same raw actions and also returns coordinate-summed differential entropies. It does not sum across agents.

An environment adapter can clip the raw action to the actuator's legal interval. The rollout must keep the original raw action for PPO scoring. Clipping is a deterministic environment-facing map: the return is observed after applying that map, while the sampled random variable optimized by the score-function estimator remains the raw action. Scoring the clipped action as though it were the sampled Normal action gives the wrong likelihood ratio. No tanh transformation or change-of-variables correction is applied here.

Clipping can introduce probability mass at actuator boundaries. The equality of Gaussian modes is therefore a statement about the raw distributions, not a guarantee about the modes of the resulting clipped actuator distributions. At deterministic evaluation, applying the same clipping map to the identical local and centralized means still produces identical actuator commands.

## Scope and verification

Conditioned on observations and state, the implementation uses independent Normal noise across agents and action coordinates. State-dependent standard deviations alone do not create conditionally correlated samples. Mode consistency does not establish convergence or performance superiority.

The implementation supports homogeneous continuous action dimensions, with feedforward or private recurrent local actors. Heterogeneous actuator sets need explicit padding/masking or another adapter convention. The environment adapter owns actuator bounds and simulator details.

`tests/test_agentmixer_continuous.py` verifies exact mean equality, state-independent local execution, local information isolation, bounded positive standard deviations, coordinate-summed likelihoods and entropies, raw action handling, deterministic means, actor and modifier gradients, and checkpoint restoration. Those checks establish implementation properties, not reproduction of benchmark returns.


## Automatic local-scale native collection

The six local-scale native gates now have collector **21288137**, which runs after the complete existing HC branch. It checks all six conditions, all 24 original evaluation episodes, twelve exact replay episodes, eighteen optimizer rollouts and three default first-three-update prefixes. Its exact reporting source is frozen, with no change to training configurations or native jobs. [Collector specification](../experiments/remaining_native_collectors_plan_20261029.json); [verified schedule](../experiments/remaining_native_collector_hc_schedule_20261029.json).

This read-only task also audits the fifteen remaining Football gates and reuses the completed HC allocation. The eighteen-run feedforward study retains that reservation after the collector; measured native memory/runtime and a separate learning release remain required. The twelve memory gates and their existing collector keep their original data dependencies. Neither learning family is released automatically.


## State-independent sigmoid local scale

Use `continuous_local_scale="state_independent_sigmoid"` with continuous AgentMixer, including its identity control and private GRU actors. Each agent learns one unconstrained score per action coordinate:

```text
local_std = max(exp(-5), 0.5 * sigmoid(scale_score))
joint_mean = local_mean
```

This isolates the local scale map documented in the pinned [released Gaussian](https://github.com/LiZhYun/BackPropagationThroughAgents/blob/13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0/bta/algorithms/utils/distributions.py). The positive floor is our numerical safeguard against underflow; the release has no floor. Local scales remain independent of observations and state. The selected centralized modifier map is unchanged and retains the exact local means.

Set `initial_std` explicitly in `[exp(-5),0.5)`. The inverse sigmoid initializes the score to the requested scale. `initial_std=0.36552928931500245` gives score1, matching the release's initial `0.5*sigmoid(1)`. Mean and modifier tensors, including GRU tensors, and constructor RNG match the exponential state-independent control. Initial scales agree numerically; inverse-map rounding does not promise bitwise identical stochastic samples between the two parameterizations. Existing choices retain their checkpoint keys and behavior. [35-case bitwise compatibility](../experiments/continuous_sigmoid_default_parity_20261029.json).

The [registered comparison](../experiments/hc2x3_sigmoid_study_20261029.json) freezes six 44k native checks and eighteen 1M development runs. Three modifier/identity choices are crossed with FF/GRU, retaining seeds 0–2 for learning and all prior FF/GRU outcomes. Every matched learning pair changes only the local-scale option. All 24 frozen parsers pass. The six native checks are now queued as 21289365; eighteen learning runs remain unsubmitted. Native memory/reset witnesses, complete raw evaluation audits, exact final replay, measured resources and a verified allocation are required before learning. Earlier studies retain priority; no improvement is promised. [Correctness evidence](../experiments/continuous_sigmoid_verification_20261029.json).

## Memory collector schedule correction

The original twelve 44k memory configurations evaluate at 0, 12k, 24k, 36k and 44k. Their first auditor incorrectly expected only two endpoints. The corrected collector checks the frozen grid and stages, requiring 120 original episodes and 24 replay episodes. Both saved complete synthetic 44k runs pass the corrected grid; twelve missing/stage corruptions fail. The actual twelve native runs remain pending. [Correction](../experiments/continuous_memory_audit_grid_fix_plan_20261029.json); [preflight](../experiments/continuous_memory_audit_grid_fix_preflight_20261029.json).

Only the owned,never-started read-only collector 21284087 was cancelled and replaced by 21289017, with the same afterany dependency and 1 CPU/2 GiB/5 min allocation. All training sources, configurations, seeds, helpers and jobs remain unchanged. Original audit code and receipts are preserved. [Replacement verification](../experiments/continuous_memory_audit_grid_fix_schedule_20261029.json).


## Observed sigmoid native gates

Jobs **21289365 → 21289366** run and audit all six frozen 44k conditions. They reuse the Prey allocation only after its complete learning collector 21288041 and the corrected original memory auditor 21289017. The latter remains a scientific prerequisite: every original twelve conditions must pass with unchanged hashed artifacts before a new sigmoid gate can start. Earlier feedforward/recurrent learning reservations remain intact. [Execution](../experiments/continuous_sigmoid_gate_execution_20261029.json); [actual capacity](../experiments/continuous_sigmoid_gate_capacity_20261029.json); [submission](../experiments/continuous_sigmoid_gate_schedule_20261029.json).

Each native run includes eleven 40×100 rollouts, all forty training horizon resets, the complete five-evaluation grid and exact final replay. The family requires264k training decisions, 132k GRU transitions, 240 resets, 60 original evaluation episodes and 12 replay episodes. At most two 1 CPU/4 GiB/30 min workers are active; the complete-family auditor uses 1 CPU/2 GiB/5 min afterward. New logs are explicitly routed to their own manifest directory.

Both core and private MuJoCo runtimes pass six observed/unobserved bitwise comparisons through three synthetic PPO updates, reject 26 corruptions, complete a synthetic 44k horizon with 40 resets and exact final replay, and check 9,090 existing native evaluation episodes. The full configured grid is checked explicitly. [Release preflight](../experiments/continuous_sigmoid_gate_release_preflight_20261029.json). These witnesses do not execute the new native study or establish learning gain. All eighteen sigmoid learning runs remain unsubmitted pending complete native/resource/allocation review. No cross-map bitwise update-prefix claim is made: the two local scale maps intentionally have different gradients.

The exact installed wheel for this source now passes 1,093 tests with one CUDA-only skip outside the checkout. [Wheel verification](clean_checkout.md).


## Release-first scope, 29 October 2026

All remaining continuous research sweeps and collectors are stopped. The54 prepared learning runs remain unsubmitted. The six local-scale gates produced outputs, but their exact historical-prefix audit failed; no complete-family pass or learning gain is claimed. Older scheduling descriptions above are preserved history. Correctness and explicit method documentation govern code release; additional tuning is optional. [Current scope](../STATUS.md).
