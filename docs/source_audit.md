# Accepted-source audit

This audit reads the supplied accepted-paper sources and figure assets. It records **no reproduced results**. The machine-readable targets are in `configs/paper_targets.json`; numeric results are populated only where the source explicitly prints them. A curve endpoint is not an exact numeric result.

The archive names and SHA-256 hashes are preserved in `references/sources.json`. The target manifest also hashes the canonical `.tex` files. Reference files were read without modification.

## Source status and changes to the initial assessment

- BPTA: `references/bpta/Formatting-Instructions-LaTeX-2024.tex` includes an active appendix with experiment settings. Its Climbing matrix is `[[11,-30,0],[-30,7,0],[0,6,5]]`, not another conventional Climbing variant. Penalty is `[[-100,0,10],[0,2,0],[10,0,-100]]` (lines 278–309).
- AgentMixer: `references/agentmixer/aaai25.tex` contains an active main paper, but the appendix material after the bibliography is commented out. Its archived hyperparameters and additional figures remain useful evidence, but compiling the supplied source unchanged does not include them. The manifest distinguishes this material from active text.
- Earlier claims about legacy buffer/trainer defects are code-audit questions; the paper sources do not verify those claims. This audit does not treat them as established code findings.
- BPTA's SMAC prose lists 20v20, 10v11 and 20v23 (line 330). Both supplied result figures instead show 5v5, 10v10 and 10v11 for Protoss, Zerg and Terran. The confirmed suite contains those nine plotted tasks; the six prose-only 20-unit tasks remain unresolved.
- BPTA explicitly uses mean and standard deviation over five seeds (line 276). AgentMixer's active SMAC table explicitly uses **median and standard deviation**, at 10 million steps (lines 545–561). Do not substitute a mean for the median. Five seeds and averaging the last five evaluations appear in AgentMixer's commented statistical-test paragraph (line 1063); the latter is not explicitly the SMAC table's aggregation rule.
- Ant observation settings differ: BPTA explicitly overrides `agent_obsk` to `None` (lines 613–626), whereas AgentMixer's archived settings use `0` (lines 1274–1312) and its active text emphasizes local observation. Combining these into one identical Ant benchmark would change the comparison.

## Method contracts

**BPTA/BPPO.** Agents act in a fixed auto-regressive order. Predecessor actions reach successors through direct and skip connections. The learning signal contains the agent's own update and feedback differentiated through successor action dependencies. BPPO updates agents in reverse execution order, with successor likelihood factors, reparameterized actions, and a global value function trained with GAE (BPTA lines 195–264). A forward-conditioned PPO baseline without peer feedback is ARMAPPO; a rename to `ar_ppo` must preserve that distinction.

The printed BPPO objective, Eq. 6 (`eq:bppo`, lines 225–237), remains unresolved: the own-learning term is printed without an explicit advantage multiplier, while the peer term includes it. Exact detach boundaries and whether the old-policy denominator is differentiated with respect to predecessor actions also require an explicit implementation decision. Any repaired objective must be documented as a repair and checked analytically or with gradient tests; it cannot be silently described as a literal transcription.

**AgentMixer.** The policy modifier combines state and individual policy features using residual agent-mixing and channel-mixing MLPs with LayerNorm. Individual-Global-Consistency preserves joint and local policy modes while permitting centralized exploration. Training uses PPO; deployment uses individual policies with local information (AgentMixer lines 308–317, 378–435, 493–497). In continuous actions, the joint Gaussian mean is the concatenation of individual means, with modifier-produced scale.

The discrete construction, Eq. 12 (`eq:cate_joint`, lines 417–435), remains unresolved. The text alternates between probabilities and logits. Arbitrary additive learned noise can change the maximizer, and a temperature approaching zero selects `argmax(log(alpha) + epsilon)`, which need not equal `argmax(alpha)`. A sigmoid output is not by itself a random uniform draw. A redesigned implementation must specify its actual joint distribution, randomness, likelihood and mode guarantee. The source's convergence claim is not an unconditional guarantee for an arbitrary replacement parameterization.

## Confirmed benchmark inventory

| Paper | Domain | Tasks supported by supplied figures or tables | Budget per run |
|---|---|---|---|
| BPTA | Matrix | Climbing; Penalty | 1,000,000 steps |
| BPTA | SMACv2 | Protoss, Zerg, Terran × 5v5, 10v10, 10v11 | 10,000,000 steps |
| BPTA | MA-MuJoCo | Ant-v2: 2x4, 4x2, 8x1; HalfCheetah-v2: 2x3, 3x2, 6x1; Walker2d-v2: 2x3, 3x2, 6x1 | 10,000,000 steps |
| BPTA | GRF | 3-vs-1 with keeper; counterattack hard; corner; full-game 5-vs-5 | 20,000,000 steps |
| AgentMixer | Matrix | Climbing | 200,000 steps |
| AgentMixer | SMACv2 | Protoss, Terran, Zerg × 5v5, 10v10 | 10,000,000 steps |
| AgentMixer | MA-MuJoCo main | Ant-v2: 2x4, 4x2, 8x1, 2x4d; HalfCheetah-v2: 6x1; Humanoid-v2: 17x1 | 10,000,000 steps |
| AgentMixer | MA-MuJoCo appendix | HalfCheetah-v2: 2x3, 3x2; Walker2d-v2: 2x3, 3x2, 6x1; HumanoidStandup-v2: 17x1 | 10,000,000 steps |
| AgentMixer | Predator-Prey appendix | Single-capture penalties 0 and -1 | 1,000,000 steps, from plot axis |

These are 24 BPTA and 21 AgentMixer tasks. Figure labels were extracted with the installed `pypdf` package. Curves were not digitized. For GRF, prose-normalized scenario names still require confirmation against the actual engine identifiers and reward settings. Predator-Prey dynamics, agent counts and full training configuration are not specified in the supplied text.

## Exact performance targets and metric distinctions

AgentMixer's active `tab:expsmac` supplies these exact results:

| SMACv2 map | AgentMixer median win rate | SD | MAPPO median | ARMAPPO median |
|---|---:|---:|---:|---:|
| Protoss 5v5 | 0.68 | 0.04 | 0.51 | 0.50 |
| Terran 5v5 | 0.71 | 0.10 | 0.48 | 0.41 |
| Zerg 5v5 | 0.66 | 0.03 | 0.48 | 0.44 |
| Protoss 10v10 | 0.47 | 0.02 | 0.40 | 0.36 |
| Terran 10v10 | 0.50 | 0.11 | 0.29 | 0.20 |
| Zerg 10v10 | 0.52 | 0.06 | 0.38 | 0.36 |

The manifest transcribes all baseline scores and SDs from this table. Other primary results remain `null` with a figure locator. Qualitative claims of convergence or superiority do not justify invented decimal targets.

BPTA provides both training and evaluation win-rate curves for SMACv2 and the three GRF academy scenarios. The GRF full game reports episode return and explicitly remains negative; it must not be relabeled as win rate. MA-MuJoCo main figures say episode return without fully defining the evaluation convention. AgentMixer's Ant ablation separates training and evaluation; its Predator-Prey text says average test-episode return, without fully defining aggregation across seeds.

Matrix reward normalization needs resolution. BPTA's plot scale is consistent with 200 repeated decisions: theoretical episode maxima are 2,200 for Climbing and 2,000 for Penalty. These are derived ceilings, not measured results. AgentMixer's Climbing plot is on a scale near per-step payoff 11 even though its archived table also says episode length 200. Do not compare these plotted returns directly. Once optimal return is attained, the acceptance criterion must concern convergence speed or reliability; exceeding the mathematical maximum is impossible.

## Settings and baselines

Both sources report independent per-agent parameters for the PPO baselines and proposed algorithms, with AgentMixer excepting MAT and MAT-Dec. Shared defaults are Adam, epsilon `1e-5`, weight decay `0`, gradient norm `10`, chunk length `1`, ReLU, PPO clip `0.2`, discount `0.99` and GAE lambda `0.95`. Exact domains and source line locators are encoded in the manifest.

| Domain | Actor/critic LR | Entropy | Hidden layers × width | PPO epochs / minibatches | Rollout threads / length |
|---|---:|---:|---|---|---|
| Matrix | 5e-4 | 0.01 | 1 × 64 | 15 / 1 | 50 / 200 |
| SMACv2 | 5e-4 | 0.01 | 1 × 64 | 5 / 1 | 50 / 200 |
| MA-MuJoCo | 3e-4 | 0 | 2 × 64 | 5 / 1 | 40 / 100 |
| BPTA GRF | 5e-4 | 0.01 | 1 × 64 | 15 / 2 | 50 / 200 |

The MAPPO and ARMAPPO baseline configurations inherit these same per-paper domain settings. ARMAPPO's action projection is a separately labeled BPTA ablation, with dimension unspecified. The AgentMixer modifier's archived dimensions are 32 for agent mixing, 256 for channel mixing and LR `5e-5`; block count appears only in a doubly commented draft table and remains unconfirmed.

BPTA also compares HAPPO and, on HalfCheetah-v2 6x1, GCS. AgentMixer includes HAPPO, MAVEN, MAT, MAT-Dec and the AIL ablation. MAT and MAT-Dec have richer evaluation information than the local AgentMixer actors; preserve that distinction in results. AgentMixer's local-information claim for ARMAPPO does not fully explain how predecessor-action communication is accounted for.

The manifest retains the additional update-order, execution-order, projection, AIL, full-information, SAC, modifier-architecture and runtime studies. Runtime comparisons require matching hardware allocation; a shared server model alone does not establish comparability.

## Conditions for a completed reproduction

The original random seed identifiers, precise package versions, evaluation episode counts, deterministic versus sampled evaluation rule and checkpoint-selection convention are not supplied. Five training seeds are explicit for BPTA and supported by AgentMixer's commented significance analysis. Record all choices made to fill the gaps.

No experiment result is claimed by this audit. Completion requires per-task actual results with configuration, seed-level data, simulator versions, environment-step counts, uncertainty and independent final evaluation seeds. Report both mean and median so each paper's statistic can be recovered. Preserve original budgets for direct comparisons and label any added training or architecture changes. Unresolved plot targets need original data or a documented digitization protocol before claiming numerical improvement over them.
