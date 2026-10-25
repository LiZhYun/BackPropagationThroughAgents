# Joint modifier development diagnosis

This is a read-only diagnosis of the experimental two-agent joint modifier, not a reproduced result or a convergence claim. It examines the final checkpoints from:

- `results/matrix_joint_20261024T231609Z`: actor and modifier LR `5e-4`, entropy `0.01`, scales hard-clamped to `[exp(-8), exp(8)]`.
- `results/matrix_joint_slow_entropy01_20261024T233313Z`: modifier LR `5e-5`, entropy decaying from `0.1` to `0.01`, otherwise the same scale range.
- `results/matrix_joint_slow_bounded_20261024T233315Z`: the slower modifier and entropy schedule, with smooth log-scale bound `log(10)`.

Each contains three development seeds for Climbing and Penalty, trained for 100,000 one-step joint decisions. None reaches optimal greedy execution. No optimizer updates or new training were performed for this diagnosis.

## Exact gradient calculation

The implementation uses

\[
g_i(a)=\max_b z_i(b)-z_i(a),\qquad
E(a,b)=s(a,b)[g_0(a)+g_1(b)],\qquad
p(a,b)\propto e^{-E(a,b)}.
\]

All nine actions were enumerated from each checkpoint. Gradients were calculated in float64 for

\[
F=\mathbb E_p[R]/\sigma+0.01H(p),
\]

holding the current reward standard deviation `sigma` fixed. At the behavior policy this has the expected normalized-advantage PPO ascent gradient, before clipping becomes active. It does not reproduce finite-batch noise or subsequent PPO epochs.

The local-logit gradient was separated into the direct gap path, with modifier outputs held fixed, and the additional path through the modifier's local-policy inputs. The masked maximum remains differentiated in both cases.

For slow bounded Climbing seed 0:

| Quantity | Agent 0, action A | Agent 1, action A |
|---|---:|---:|
| Full ascent gradient | -0.0337866 | -0.0504686 |
| Direct gap gradient | -0.0337382 | -0.0504331 |
| Modifier-input contribution | -0.0000484 | -0.0000355 |

Here `p(AA)=0.47150`, `p(AC)=0.00334`, `p(CA)=0.00401`, and `p(CC)=0.52083`. The AA scale is about `0.1001`, while AC and CA scales are about `9.97`. Raising a local A logit increases the harmful mismatches roughly 100 times more sensitively than it increases AA. Their negative contribution outweighs AA's positive contribution. Detaching modifier inputs would leave the adverse direct gradient essentially intact. This is not evidence of a missing maximum derivative or a dominant modifier-feedback cancellation.

## Fixed-mode optima explain the plateaus

Exact IGC requires the local greedy tuple to remain a global probability mode. With that tuple held fixed, the centralized expected return can be optimized without improving decentralized execution:

| Game and fixed local mode | Expected-return supremum | Limiting distribution | Observed slow unbounded seed 0 |
|---|---:|---|---:|
| Climbing, CC | 8 | Half AA, half CC | 7.99861 |
| Penalty, BB | 22/3 | Equal AC, CA, BB | 7.33235 |

For Climbing, `p(AA) <= p(CC)` pairs AA's excess above 8 with CC's equal deficit; every other reward is below 8. For Penalty, `p(AC), p(CA) <= p(BB)` similarly bounds the two rewards of 10 against BB's reward of 2. These are suprema when the fixed tuple must remain the unique mode: the other listed probabilities approach it from below.

Mode changes introduce genuine barriers. At a local tie, this parameterization makes every Cartesian product of tied local modes a joint mode. For example, Climbing's first-agent A/C tie with the second agent fixed at C forces both AC and CC to be modes; even an unrestricted distribution satisfying that constraint has expected-return supremum 5.8. Penalty's A/B tie with the second agent fixed at B forces AB and BB to be modes, with supremum 5.5. Thus continuous ascent need not escape a well-optimized distribution with the wrong local mode. These constraints concern this exact mode-set interpretation; they do not transfer a theorem from the paper.

## Saturation and optimizer motion

The fast unbounded runs show a second issue: useful scales reach the hard lower clamp and unwanted outcomes are nearly suppressed. For Climbing seed 0, the modifier parameter gradient is exactly zero, while the local-actor gradient norm is `7.49e-4`. The local A-logit ascent gradients remain positive, about `1.38e-4` for each agent. Penalty's actor gradient norm is `8.14e-4`; both A and C are pushed upward relative to B, while the modifier gradient is effectively zero.

Using the saved Adam moments and the exact current gradient, the estimated next increase in Climbing's `z(A)-z(C)` is:

| Adam epsilon | Agent 0 | Agent 1 |
|---|---:|---:|
| `1e-5` | 0.000526 | 0.000502 |
| `1e-8` | 0.000573 | 0.000564 |

The remaining gaps are about 0.530 and 0.519. Lower epsilon improves this estimated functional motion only 9–12%; a fivefold actor learning rate changes it much more. These are first-order next-step estimates, not forecasts of convergence. Penalty shows a similar 12–13% epsilon effect and retains the symmetry between its two optimal off-diagonal outcomes.

The float32 positive-energy guard also matters near ties: its floor is about `2.095e-6`, so at scale `exp(-8)` it gives zero gap derivative when the pair-gap sum is below about `0.00625`. The diagnosed checkpoints are outside this region, so it does not explain their current gradients. It adds a small near-tie region that finite updates must cross. Float64 energy arithmetic is a possible later numerical control; changing the backward derivative through a floor would require a separately named surrogate.

## Controlled next comparisons

The signs and step estimates support three reward-independent comparisons at the same 200,000-step budget, retaining the fast unbounded modifier LR `5e-4`:

1. Unchanged actor LR `5e-4`, controlling for the longer budget.
2. Actor LR `2.5e-3` throughout.
3. Actor LR `5e-4` for the first half, then `2.5e-3`.

These test whether larger finite local updates can leave the narrow plateau after the modifier suppresses mismatches. They preserve the forward IGC construction and require no payoff-specific initialization or policy. Lower Adam epsilon is a secondary control; modifier-input detachment is low priority given the measured decomposition. An identity-only modifier warmup is not clearly favorable because it initially restores the independent-policy optimization landscape. Any initialization-gain experiment should use a fixed, reward-independent gain and all preregistered development seeds.

The bounded runs have a broader adverse-gradient barrier, so these unbounded optimizer controls should not be generalized to them. Success remains an empirical question, and full discrete AgentMixer benchmark runs should wait for stable development results and independent final evaluation.

### Configuration and schedule convention

All three planned configurations use `algorithm="agentmixer"`, `agentmixer_correlation="joint"`, `steps=200000`, `batch_size=1024`, `modifier_learning_rate=0.0005`, `critic_learning_rate=0.0005`, `entropy_coef=0.01`, `entropy_final=null`, and `modifier_log_scale_bound=null`. Their actor settings are:

| Control | `learning_rate` | `late_actor_lr_multiplier` | `actor_lr_change_fraction` |
|---|---:|---:|---:|
| Longer budget | 0.0005 | 1 | 0.5 |
| Fixed fivefold actor rate | 0.0025 | 1 | 0.5 |
| Late fivefold actor rate | 0.0005 | 5 | 0.5 |

The schedule is evaluated from completed environment steps before collecting each rollout. It preserves rollout sizes and optimizer moments. With 1,024-step batches, the late control's nominal threshold is 100,000 steps and the first affected rollout starts at **100,352** steps. Its update ends at 101,376 steps. No rollout is split to hit the threshold. All PPO epochs for a rollout use the same active actor rate.

`metrics.jsonl` records `actor_lr` and `actor_lr_schedule_steps` (the rollout's starting step). The first changed rollout is always logged, even between regular evaluation intervals, with `actor_lr_changed=true`. Only actor optimizer groups change: the modifier and critic retain their configured rates. BPPO changes each actor's separate optimizer; factorized and joint AgentMixer change only their local-actor group. The default multiplier is 1, preserving the previous constant-rate behavior. Multipliers must be finite and positive, and change fractions must be finite in `[0, 1)`; a fraction of zero selects the multiplied rate from the first rollout. Adam epsilon remains `1e-5`.
