# Method fidelity and implementation choices

This repository reconstructs the core algorithms in the accepted [BPTA paper](https://ojs.aaai.org/index.php/AAAI/article/view/29277) and [AgentMixer paper](https://ojs.aaai.org/index.php/AAAI/article/view/34048). It is a readable redesign with documented corrections, rather than a promise of identical historical trajectories. Local author-provided source archives informed the [source audit](source_audit.md); they are not bundled with the public code.

## Preserved mechanisms

| Method | Preserved mechanism | Implementation and checks |
| --- | --- | --- |
| BPPO | Each actor conditions on its observation and recorded predecessor actions; actors update in reverse order, receiving feedback through successor likelihood dependencies. | [Discrete policy/objective](../src/bta_repro/bppo.py), [continuous policy/objective](../src/bta_repro/bppo_continuous.py); [gradient and update checks](../tests/test_bppo.py) |
| AR-PPO | The same autoregressive execution with the own PPO objective and no peer-feedback term. | Same policy classes; matched no-peer paths in the BPPO tests |
| AgentMixer | A centralized modifier receives state and local policy features during training; deployment uses local policies. Local and centralized policy modes agree under the corrected parameterization. | [Discrete modifier](../src/bta_repro/agentmixer.py), [Gaussian modifier](../src/bta_repro/agentmixer_continuous.py); [discrete checks](../tests/test_agentmixer.py), [continuous checks](../tests/test_agentmixer_continuous.py) |
| Shared training | Explicit clipped PPO, a global critic, GAE, legal-action/active-agent masks, and separate termination and time-limit handling. | [Core](../src/bta_repro/core.py), [runner](../src/bta_repro/runner.py); [core checks](../tests/test_core.py), [runner checks](../tests/test_runner.py) |

BPPO requires predecessor-action communication at execution. AgentMixer local execution requires neither the global state nor the modifier. These are different information contracts, and evaluation does not interchange them.

## Explicit corrections

**BPPO objective.** The printed own-learning term omits an advantage multiplier. We include it so the last-agent/no-peer limit is PPO. We reduce the own objective and peer dot product to scalars separately, select clipping from the scalar own surrogate, and document differentiation through both numerator and behavior-denominator action dependence. Discrete straight-through Gumbel gradients are biased estimators. See [the exact objective and detach boundaries](bppo.md); no unbiasedness or convergence guarantee is inferred from the implementation.

**Discrete AgentMixer.** Arbitrary learned additive action offsets can change the local maximizing action. The default implementation instead scales each action’s nonnegative gap from the local maximum by a positive factor, preserving the full set of available-action modes. Conditional action samples are factorized across agents; shared state features do not establish a correlated joint distribution. See [the construction and numerical guards](agentmixer.md).

**Continuous AgentMixer.** Local Gaussian means are unchanged by the modifier; only standard deviations change. Mode equality concerns raw Gaussian distributions. The runner scores raw samples even when the environment clips actuator commands. See [continuous semantics](agentmixer_continuous.md).

## Defaults and optional extensions

The short matrix examples use raw predecessor one-hot inputs for BPPO and the corrected factorized discrete AgentMixer. The temporal example uses the same factorized modifier with local execution. Neither enables an experimental auxiliary objective.

Action projections, a two-agent correlated modifier, private recurrent actors, alternative Gaussian scale maps and local distillation remain explicit configuration options. They are research controls, with their own assumptions and tests, rather than evidence of the papers’ performance. In particular, recurrent updates use one-decision gradient chunks; the released architecture is not claimed to be reproduced layer for layer. [Configuration types](../configs/README.md), [recurrent memory](recurrent_actors.md), [joint modifier](agentmixer_joint.md), [distillation](prey_distillation.md).

## What validation establishes

The core suite checks numerical and behavioral properties, and saved-policy evaluation checks source/version provenance and replay. Native integrations and learning performance are separate evidence. A historical MuJoCo exact-prefix comparison failed in the optional local-scale study; that failure is retained and exact historical trajectory equivalence is unproven. It does not justify silently relaxing the check or claiming a learning gain.

Release readiness concerns clear code, documented mechanisms/corrections, runnable examples and installation/correctness checks. Full-budget sweeps and all-task performance reproduction are not release prerequisites. Any future performance claim must report its actual budget, simulator protocol, every declared seed and uncertainty. [Current scope](../STATUS.md).
