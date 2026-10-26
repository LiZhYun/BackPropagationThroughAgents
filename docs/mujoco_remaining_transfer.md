# Remaining MuJoCo learning coverage

Job **21221614** runs37 native checks for the remaining task/method gaps.
The paired500k learning plans are frozen but **not submitted**. This work
covers nine task/observation groups without learning pilots, plus the missing
MAPPO/AgentMixer/identity controls for HalfCheetah3×2 and6×1.

The [study record](../experiments/mujoco_remaining_transfer_study_20261026.json)
derives the conditions from canonical paper targets. It retains the exact
20-module source used by the Ant/Walker pilots and the selected HalfCheetah
training settings. The [preflight](../experiments/mujoco_remaining_transfer_preflight_20261026.json)
validates all74 expanded configurations using that frozen trainer; all outputs
were unstarted at verification. It checks source/configuration hashes and
resolved defaults, not just the current package's configuration parser.

| Task | Observation radius | Methods checked |
| --- | --- | --- |
| Ant-v2 4x2 | Global (`null`) | mappo, arppo, bppo |
| Ant-v2 8x1 | Global (`null`) | mappo, arppo, bppo |
| HalfCheetah-v2 3x2 | Local (`0`) | mappo, agentmixer, identity |
| HalfCheetah-v2 6x1 | Local (`0`) | mappo, agentmixer, identity |
| Walker2d-v2 3x2 | Local (`0`) | mappo, arppo, bppo, agentmixer, identity |
| Walker2d-v2 6x1 | Local (`0`) | mappo, arppo, bppo, agentmixer, identity |
| Ant-v2 4x2 | Local (`0`) | mappo, agentmixer, identity |
| Ant-v2 8x1 | Local (`0`) | mappo, agentmixer, identity |
| Ant-v2 2x4d | Local (`0`) | mappo, agentmixer, identity |
| Humanoid-v2 17x1 | Local (`0`) | mappo, agentmixer, identity |
| HumanoidStandup-v2 17x1 | Local (`0`) | mappo, agentmixer, identity |

The two Ant observation protocols remain distinct. Identity denotes an
AgentMixer policy with its modifier disabled. Existing HalfCheetah BPPO/AR
pilots and Ant2×4/Walker2×3 studies remain separate. This extension does not
select new settings from their unfinished comparisons.

Every native check uses three40×100 rollouts, five PPO epochs, two complete
evaluation episodes at initialization and12k, and the canonical1,000-step
horizon. Training seed0 and evaluation seed1,000,000 remain separate. The
old64-step integrations used a16-step horizon; they do not replace this check.

The37 pilots retain the same configurations except for500k training steps,
ten evaluation episodes every100k, and checkpoint frequency. Training uses
entropy coefficient0, initial standard deviation0.365529, actor/critic rate
0.0003, modifier rate0.00003, observation normalization and normalized clipped
Huber value loss. The residual modifier remains the default. These pilots
measure transfer and runtime below10M; they are not independent confirmation.

All37 gates must finish validly before learning release. Audit every source,
configuration, seed, rollout, optimizer/ValueNorm count and raw episode
reduction. Measure scheduler memory and runtime, then size each learning task
from those measurements. The prepared pilot's120-minute header is provisional;
it is not an approved runtime estimate. Retain all declared outcomes and
failures, without early-peak substitution or selective reporting. Full10M
budgets, at least five disjoint final training seeds and the other paper
suites remain required.

Two gate workers run19 and18 conditions serially, each capped at two CPUs,
8 GiB and two hours, without automatic requeue. This is a bounded diagnostic
allocation, not a measured prediction for the37 pilots. Worker0 follows
Ant/Walker21212297_0 and precedes pending HC-partition21220995_0; worker1
follows Prey-bounds21212197_0 and precedes matrix21220499_0. All32 serial
edges were checked before release in the
[schedule receipt](../experiments/mujoco_remaining_gate_schedule_20261026.json).
The eight-worker,16-CPU/64-GiB project ceiling remains unchanged. Active jobs
and the unrelated reproduction were untouched.
