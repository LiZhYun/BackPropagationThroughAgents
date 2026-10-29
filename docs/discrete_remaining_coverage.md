# Remaining discrete simulator coverage

The 29 October 2026 coverage review found three Football scenarios and eight SMAC maps with short native integrations, but no integrations at the recurrent rollout layouts now used for development. The [registered native study](../experiments/remaining_discrete_native_study_20261029.json) closes that gap before learning pilots. **All new checks are queued; no new performance result is available.**

Each scenario retains five controls: MAPPO, AR-PPO, BPPO, AgentMixer and its identity-modifier control. Every actor has private GRU memory. Each condition uses development seed 0, performs one complete 10,000-transition rollout and optimizer update, evaluates four complete episodes before and after training, then replays all four final episodes from its checkpoint. Exact replay includes raw episode records, not just their mean.

| Scenario | Agents | Paper targets covered by the check | Paper training budget |
| --- | ---: | --- | ---: |
| Football `academy_counterattack_hard` | 4 | BPTA | 20M |
| Football `academy_corner` | 10 | BPTA | 20M |
| Football `5_vs_5` | 4 controlled players | BPTA | 20M |
| SMAC `protoss_5v5`, `zerg_5v5` | 5 | BPTA and AgentMixer for each map | 10M |
| SMAC `protoss_10v10`, `terran_10v10`, `zerg_10v10` | 10 | BPTA and AgentMixer for each map | 10M |
| SMAC `protoss_10v11`, `terran_10v11`, `zerg_10v11` | 10 | BPTA for each map | 10M |

These are 15 Football and 40 SMAC conditions associated with 16 canonical target entries. Football 3v1 and SMAC Terran 5v5 already have complete layout gates and ongoing full-budget development studies, so they are not duplicated here. A 10k integration is 0.05% of the Football budget or 0.1% of the SMAC budget.

## Source and protocol

Football uses the exact 25-module snapshot from its completed 1M study and active 20M follow-up: 50 environments × 200 decisions, 15 epochs and two minibatches, with value coefficient 1. SMAC uses the exact 23-module snapshot from its completed layout study and active 10M follow-up: 25 environments × 400 decisions, five epochs and one minibatch. The latter matches the audited released modifier layout; the non-modifier launcher and manuscript table use 50 × 200. We retain this difference explicitly.

Within each method, the new configuration changes only the scenario, short gate budget, evaluation count/cadence and checkpoint cadence. Initialization rules, optimizer settings, observation handling and method implementation remain fixed. Both source sets predate the optional continuous local-scale control; that control is irrelevant to these discrete studies. All 55 configurations pass their own frozen parser, and the registration records every source and configuration hash.

The [Football](../experiments/football_remaining_policy_shapes_20261029.json) and
[SMAC](../experiments/smac_remaining_policy_shapes_20261029.json) policy preflights
also pass all 55 conditions on synthetic tensors under those frozen sources.
They check packed GRU memory, masked sampling/scoring and finite actor gradients;
AgentMixer conditions additionally check exact local/global greedy-action equality.
The mode-check flag applies only to AgentMixer. These checks do not execute a
simulator and do not replace the pending native gates.

The prior native integrations supply expected observation/state/action dimensions for each scenario. The older SMAC integration did not record backend versions; its shapes are used as shape evidence only. Runtime versions must agree with the recent audited recurrent source cohort. Resolved SMAC capabilities and restart settings must also match the declared task.

Football keeps `simple115v2`, `scoring,checkpoints`, summed player rewards and complete native matches. A full 5v5 match takes 3,001 API steps. Its final-score win rate and evaluation return remain separate. The historical 5v5 figure's exact evaluation reduction is unresolved; the author confirmed evaluation return, which does not establish equivalence to the released incomplete-match evaluator. See the [Football protocol](football.md).

SMAC retains the pinned capability distributions and planned engine restart at 480,000 ticks. The papers do not specify every capability setting, so these are documented reconstruction choices. Training and evaluation win rates remain distinct; AgentMixer's published SMAC aggregation is a cross-seed median. These seed-0 gates cannot support that comparison. See the [SMAC protocol](smac.md).

## Scheduling and release requirements

| Job | New conditions | Dependency | Maximum allocation |
| --- | ---: | --- | --- |
| 21281001 | 15 Football | Entire local-scale gate array 21280689 | One worker, 1 CPU, 8 GiB, 45 minutes per condition |
| 21281002 | 40 SMAC | Entire current SMAC 10M array 21272799 | Two workers, each 2 CPUs, 32 GiB, 60 minutes per condition |

The Football worker reuses the two 4-GiB slots from the confirmation/replay/local-scale chain. The frozen but unsubmitted 18-run local-scale learning study must now wait for the HC 3×2 final auditor **21284586**, which follows **21281001**, or use another verified free capacity chain. [Current allocation graph](worker_schedule.md#independent-allocations-and-prey-learning-29-october-2026). The SMAC checks reuse the current SMAC allocation. Existing jobs remain unchanged, and the current project ceiling is fourteen workers, sixteen CPUs and 120 GiB after the independent-allocation revision. [Submission records](../experiments/remaining_discrete_native_schedules_20261029.json).

The prior Football native workers peaked at 3.48 GiB; the prior 25-engine SMAC gate peaked at 21.24 GiB. Larger agent counts and observation shapes may need more memory. The new allocations are bounded diagnostic limits, not measured requirements for those new shapes. Automatic requeue is disabled.

Before releasing any learning pilot, require complete source/configuration and raw episode audits, finite losses, exact rollout and optimizer counts, correct independent environment seeds, resolved native configurations and exact final checkpoint replays. Review each task's Slurm memory and timing records separately. All five controls for a scenario must pass together; retain failed or unfinished conditions. Short native returns do not select a winning configuration. Full-budget development and unused final training seeds remain later requirements.

The [latest coverage snapshot](../experiments/benchmark_coverage_20261029_native_collectors.provenance.json) indexes 907 runs across 75 manifests and all 45 targets, with no basic provenance errors. Ten targets have completed exact-budget method artifacts, including unsuccessful outcomes; 35 still lack them. Queued checks do not increase the number of reproduced targets.


## Automatic whole-family collection

Two read-only collectors now cover all remaining native families. **21288137** waits for the entire HC branch through final auditor **21284586**, as well as explicit completion of local-scale array **21280689** and Football array **21281001**. It audits all six local-scale and all fifteen Football conditions separately, retaining failures from either family. **21288138** waits for all forty SMAC conditions in **21281002**. [Frozen collector plan](../experiments/remaining_native_collectors_plan_20261029.json); [verified submissions](../experiments/remaining_native_collectors_schedules_20261029.json).

Each task requests one CPU/2 GiB for ten minutes and replaces its completed allocation. The ceiling remains fourteen workers/sixteen CPUs/120 GiB. The previous eighteen-run local-scale learning study keeps its HC reservation and now follows collector **21288137**, or separately verified free capacity. No existing job or scientific dependency changes, and no learning is submitted automatically.

The collector freezes the exact 25-module reporting package. Copies of the previously prepared auditors change only the reporting import path; their native checks are retained. All 61 configuration hashes and training source snapshots pass preflight; seven deliberately changed inputs and all three unfinished families are rejected without producing completion records. [Preflight](../experiments/remaining_native_collectors_preflight_20261029.json). Positive whole-family execution remains pending on the actual native artifacts. Short native scores, completed artifact counts and a passing collector cannot substitute for full-budget learning, resource review or unused final seeds.
