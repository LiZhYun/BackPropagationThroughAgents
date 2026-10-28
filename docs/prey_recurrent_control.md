# Predator–Prey recurrent control

The released temporal launcher inherits **recurrent actors and 15 PPO epochs**.
It uses 50 training environments, 200 decisions per rollout, one minibatch,
value-loss coefficient 1, and evaluation every 10,000 training decisions.
The completed controls with the released rollout size used feed-forward
actors, five epochs, coefficient 0.5 and evaluation every 250,000 decisions.
Those experiments therefore did not test this released optimization setting.

The [source audit](../experiments/prey_released_training_protocol_20261026.json)
records exact Git blobs and SHA-256 hashes for the pinned launcher and entry
point. Selected parser defaults were read from Python's syntax tree and
combined with the launcher arguments without executing upstream training code.
`use_recurrent_policy` defaults to true; its disabling flag is absent from
the launcher. The entry point disables recurrence only for MAT/MAT-dec, while
this launcher selects `temporal`. The accepted archive's Prey and
hyperparameter appendices are commented out. This is released-code evidence,
with no claim that the historical figure used exactly this version.

## Frozen comparison

The [study](../experiments/prey_recurrent_protocol_study_20261026.json) contains
eight 1M-step development runs, all training seed 0:

| Factor | Settings |
| --- | --- |
| Penalty | 0, −1 |
| Actor | Feed-forward, private GRU |
| Modifier | Factorized AgentMixer, frozen identity control |

Within a penalty and modifier choice, each pair differs only in recurrence.
Within a penalty and actor choice, each pair differs only in whether the
modifier learns. All use the released rollout, epoch, minibatch, value-loss
coefficient and evaluation cadence above. Actor/critic learning rates remain
5e-4, modifier rate 5e-5, entropy 0.01, hidden width 64 and gradient norm limit 10.
The smooth modifier-range experiment remains separate.

The primary endpoint is the author's selected **100 independent windows × 200
decisions**, using local greedy actors. Completed episodes reset within each
window; unfinished rewards remain included. The aggregate is total team reward
divided by 20,000 decisions. Initial evaluation and evaluations at every 10k
through 1M produce 101 scheduled points, requiring 2.02M window-evaluation
decisions per run, accounted separately from training. Twenty complete episodes
per point provide additional diagnostics. Final endpoints are retained without
selecting earlier peaks or pooling the two penalties.

## Runtime and correctness gate

Job **21219646** first runs one recurrent AgentMixer penalty −1 configuration
at 10k decisions: a complete 50×200 rollout, 15 actor and 15 critic steps, and
the full initial/final evaluation protocol. A copied evaluator then replays
the final checkpoint. Source hashes, finite losses, raw windows, episode
accounting and exact replay must pass. Measured training/evaluation time and
Slurm peak memory determine the later worker allocation.

This single worker requests 2 CPUs/8GB for at most 30 minutes. It follows
AgentMixer confirmation `21218175_0` and precedes native GRU check `21213860_0`.
The [schedule proof](../experiments/prey_recurrent_protocol_schedule_20261026.json)
records the four-worker ceiling used then. The gate has now passed, and the eight-run
learning study is queued as job **21219719** behind the existing Prey controls.

These are reconstructed tanh actors and a shared feed-forward critic. The
historical ReLU/LayerNorm encoders, separate recurrent critics and attention
modifier are not copied. The positive gap-scale modifier preserves the core
mode-consistency idea; native seeds and timeout handling retain documented
corrections. The experiment tests recurrence and modifier effects within this
explicit reconstruction. Independent training seeds and evidence beyond the
paper target remain necessary after development.

## Completed gate and learning release

The [complete gate audit](../experiments/prey_recurrent_protocol_check_20261026.json)
verifies 15 policy and 15 critic steps, 200 raw evaluation windows, 40 complete
episodes, independent window seed derivation and exact checkpoint replay.
The training run uses 10,000 training and 45,120 evaluation decisions; replay
adds 22,560 separately recorded evaluation decisions. The whole job takes 112s
and peaks at 1.33GiB resident memory. It is a correctness/runtime check, not a
learning result.

Measured phases are 28.84s for initial evaluation/setup, 21.37s for collection
and updating, and 24.36s for final checkpoint/evaluation. Scaling 100 rollout
blocks and 101 evaluations at the slower observed evaluation cost estimates
84.17 minutes per 1M run. Four serial runs per worker, a 50% margin and rounding
up to a whole hour produce a **nine-hour worker cap**. Learning changes episode
lengths and reset counts, so this is a planning estimate rather than a runtime
guarantee. Timeouts remain visible and do not trigger hidden retries.

Job 21219719 has two 2CPU/8GB workers, each waiting for its matching existing
Prey worker `21212197_i`. The [release proof](../experiments/prey_recurrent_release_schedule_20261026.json)
records the passed gate, exact frozen manifest, resource override and verified
holds/dependencies/releases. The original upper-bound submission script is
unchanged; `sbatch --time=09:00:00` sets the measured allocation. The frozen
study's initial `prepared_not_submitted` status remains a pre-launch record;
this release record and the job ledger describe the subsequent submission.


## First completed 1M run

The zero-penalty recurrent AgentMixer run finishes at exactly1M training
steps. Its [raw audit](../experiments/prey_recurrent_release1m_progress_20261026T220300Z.json)
checks all101 evaluation points:10,100 windows (2.02M decisions),2,020 complete
diagnostic episodes and100 full50×200 updates. Source/configuration hashes,
seed derivation,15 actor/critic steps per rollout and ValueNorm counts agree.
All eight declared conditions remain visible; seven are unfinished in this
snapshot.

| Quantity | Final result |
| --- | ---: |
| Primary100×200-window reward per decision | 2.0515 |
| Normalized area of the101-point primary curve | 1.59706 |
| Diagnostic episode mean return | 9.9590 |
| Diagnostic captures | 20/20 |
| Captures across the100 windows | 4,123 |
| Windows ending inside an unfinished episode | 76 |

Every zero-penalty episode/window satisfies the independent reward check
`reward = 10 × captures − 0.01 × decisions`. This counts the shared team reward
once and includes unfinished rewards. The run uses2,043,517 evaluation
interactions separately from its1M training interactions and takes71m33s.

A [fresh checkpoint replay](../experiments/prey_recurrent_first_complete_verification_20261026.json)
using the exact frozen source and runtime reproduces all100 final windows and
all20 episode records exactly. Replay adds20,082 evaluation decisions; these
are not independent training seeds.

The primary score remains below the figure reading range[2.19,2.23]. The
released reduction follows the author's instruction, while the historical
plot-to-logger association remains qualified. Complete episode captures do
not by themselves satisfy this reward-rate target. The matched FF and identity
controls, penalty−1 conditions and independent final seeds remain required;
this one run cannot establish a recurrence or modifier advantage.

The current [worker schedule](worker_schedule.md#remaining-mujoco-native-gates-2026-10-26)
keeps this study in the B chains under the eight-worker project ceiling.


## Complete eight-condition comparison — 2026-10-28

All eight 1M runs pass the [complete audit](../experiments/prey_recurrent_release1m_complete_20261028.json): 16,160 raw complete episodes, 80,800 released 100×200 windows and 800 updates. The primary metric includes unfinished-window rewards and counts team reward once, as selected by the author.

| Penalty | AgentMixer FF | AgentMixer GRU | Identity FF | Identity GRU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.9520 | 2.0515 | 2.2160 | 2.2465 |
| −1 | 0.07125 | 0.04055 | −0.00875 | 0.26760 |

These are final window reward/decision values, not episode returns. Learned-modifier zero-penalty endpoints remain below the figure reading [2.19, 2.23]. Identity GRU exceeding that interval does not establish AgentMixer reproduction. One development seed cannot establish recurrence or modifier superiority. The earlier exact checkpoint replay covers only AgentMixer GRU at penalty 0; the complete audit checks all logged reductions, not all native trajectories.
