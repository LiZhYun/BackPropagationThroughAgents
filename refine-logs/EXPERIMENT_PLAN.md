# Experiment plan

Date: 2026-10-25. Status: active; no full benchmark reproduction claimed.

## Claim map

| Claim | Minimum evidence |
|---|---|
| C1: the redesign preserves the two papers' core mechanisms | Exact toy references, causal gradient checks, AgentMixer mode consistency, execution-information isolation, and documented equation-to-code mapping |
| C2: the shared implementation exceeds published performance under comparable conditions | Matched task/observation/reward/step protocols, independent final seeds, all-run metrics and uncertainty, plus ablations isolating peer feedback and policy modification |

Anti-claims: improvement only comes from changed simulator physics, more environment steps, selected successful seeds, or an algorithm silently replacing the papers' ideas. No LLM or other unrelated component is part of these methods.

## Required experiment blocks

1. **Numerical correctness.** Exact matrix payoffs/gradients, scalar PPO reductions, likelihood identity, causal masks, legal actions, termination/truncation, and checkpoint restoration. Failure stops algorithm benchmarking. Unit and integration tests provide evidence; they do not prove convergence.
2. **Development matrix games.** Independent MAPPO, AR-PPO, BPPO, and AgentMixer on climbing and penalty. Start with training seeds 0, 1, 2 and 100k joint steps per run. Track exact expected reward, greedy reward, optimum probability, first threshold attainment, and all seed outcomes. These are one-step development games, matching pinned released environment termination. Legacy training logs multiply step reward by200; the exact historical figure-to-logger mapping remains unverified.
3. **Shared simulator benchmarks.** Multi-Agent MuJoCo and SMACv2, with distinct observation protocols where the papers differ. Use `configs/paper_targets.json`; never replace missing numeric thresholds with invented values. Run baseline smoke checks, method pilots, then paper-matched budgets. Compare final performance and learning-curve area; preserve the paper's mean/median statistic and additionally publish seed-level scores.
4. **Paper-specific coverage and ablations.** BPTA Football, AgentMixer Predator-Prey, no-peer BPPO/AR-PPO, and identity-modifier AgentMixer/MAPPO comparisons. Keep architecture and budget differences explicit. Required for full coverage, with lower priority than shared suites.
5. **Independent confirmation and release.** Freeze configurations and code snapshots before at least five final training seeds disjoint from development seeds. Record evaluation episode seeds, environment versions, raw returns, all failures, checkpoints, and exact commands. Final seed IDs will be fixed before launch. Add confidence intervals and distinguish point-score improvements from statistically established improvements.

## Execution order and resource limits

| Stage | Gate | Initial resource use |
|---|---|---|
| M0 correctness | All relevant tests and all-method smoke runs pass | Single-threaded CPU checks |
| M1 matrix development | Stable training and interpretable across-seed results | CPU array, at most two concurrent tasks |
| M2 simulator pilot | Action/observation semantics and metric parity verified | One smoke job per new simulator |
| M3 full benchmarks | Development configuration frozen; protocol resolved | Start with at most two training jobs; size resources from pilot timings |
| M4 release | All required tasks and release artifacts validated | Independent final seeds and documentation checks |

After the100k HalfCheetah pilot (136–186seconds perrun), allow at most four concurrent CPU workers across this project: two for the longer MuJoCo check and two for short matrix/SMAC diagnostics. Each requests at most2CPUs/8GB, so the ceiling is8CPUs/32GB. Record job dependencies to enforce this limit. This supersedes the initial two-worker ceiling above.

Do not invent GPU-hour estimates before measuring simulator throughput. Known budgets are in the target manifest. Full reproduction spans dozens of tasks and hundreds of seed runs; measured pilot costs determine scheduling. No paid external compute is assumed. Preserve the concurrent unrelated reproduction.

## Risks and interpretation

- BPPO's printed scalar/vector expression and advantage placement require documented corrections; numerical tests must match the chosen surrogate, not an unsupported unbiasedness assertion.
- AgentMixer's original discrete offsets do not enforce the stated mode constraint. Positive gap scaling is a labeled corrected parameterization; the paper's theorem is not automatically inherited.
- Plot-only thresholds need extraction with uncertainty or author-provided raw data. Environment migration is a separate comparison track unless equivalence is established.
- Matrix return ceilings are 11 and 10. A strict return improvement above a ceiling is impossible; report reliability and sample efficiency instead.
- Simulator initialization and scheduler failures are infrastructure outcomes, not negative algorithm results.

## Completion checklist

- [x] Isolated code and source provenance.
- [x] Initial numerical correctness suite.
- [x] All-method end-to-end checks on matrix, SMACv2 and MA-MuJoCo.
- [ ] Development matrix sweep and diagnosis.
- [ ] Continuous/recurrent method coverage where required.
- [ ] All benchmark protocols and numeric/plot targets resolved.
- [ ] Full baseline and method evidence.
- [ ] Independent final seeds and ablations.
- [ ] Reproducible community release with verified performance claims.
