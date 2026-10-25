# Experiment tracker

| ID | Stage | Purpose | System | Status | Evidence |
|---|---|---|---|---|---|
| R001 | M0 | Numerical and integration correctness | Shared core + both methods | PASS | 551 tests passed,1CUDA-onlyskip, 2026-10-25 |
| R002 | M0 | Pipeline smoke | MAPPO climbing, 2,048 steps | PASS | results/smoke/mappo_20261025 |
| R003 | M0 | Pipeline smoke | AR-PPO, BPPO, AgentMixer | PASS | results/smoke/{algorithm}_20261025; 2,048 steps each |
| R004 | M1 | Matrix development | Four algorithms × two games × seeds 0–2 | FAIL learning gate | Job21192344; all24 completed, zero final greedy optima |
| R005 | M2 | Simulator availability | Multi-Agent MuJoCo, SMACv2 | PASS smoke only | .native/mujoco-smoke.json; results/smoke/smac_21192452.json |
| R006 | M3 | Full shared benchmarks | Both papers | TODO | configs/paper_targets.json |
| R007 | M3 | Paper-specific suites | Football, Predator-Prey | TODO | configs/paper_targets.json |
| R008 | M4 | Independent confirmation | Frozen final configurations | TODO | At least five disjoint final training seeds |
| R009 | M0 | Continuous and temporal correctness | Gaussian policies, GAE, masks, raw actions | PASS targeted checks | Continuous BPPO14, AgentMixer9, independent3; runner/BPPO54 tests |
| R010 | M1 | Stable normalization rerun | Same24 matrix development controls | FAIL learning gate | Job21192446; all24 remain suboptimal |
| R011 | M1 | Exploration and projection controls | Shared ARPPO/BPPO projection; bounded AgentMixer scales | PASS AR/BPPO development | Jobs21192609/21192611: both methods optimal in3/3seeds onbothgames; matched baseline also succeeds |
| R012 | M2 | Real all-method integration | Four methods,SMAC/MuJoCo,128steps | PASS smoke | Job21192629; all8runs complete |
| R013 | M1 | Experimental correlated modifier | Exact joint gap scaling,matched factorized control | FAIL initial settings | Jobs21192619/21192625/21192700/21192705; retain negative results;21192712 also0/6optima |
| R014 | M2 | MuJoCo pilot | HalfCheetah2x3,five methods/controls,100ksteps,seed0 | COMPLETED early pilot | Job21192732; final mean return MAPPO−86.84,AR−97.78,BPPO−289.44,AMidentity−2.01,AM39.32;136–186seconds/run; no10Mcomparison |
| R015 | M0 | MuJoCo partition protocol | All12 task/partitionpairs | PASS simulator checks | 67adapter/partitiontests; exactanatomy,normalizedcontrols,seededreplay |
| R016 | Targets | Published curves | MuJoCo21panels | EXTRACTED with uncertainty | configs/curve_estimates_mujoco.json; source-statistic and appendix ambiguities retained |
| R017 | Targets | Remaining published curves and matrix protocol |18primary+14secondarycurves | EXTRACTED with uncertainty | configs/curve_estimates_other.json; one-step termination verified in pinned code; historical logger mapping unresolved |
| R018 | M2 | Longer MuJoCo learning check | SamefiveHC2x3controls,1Msteps,seed0 | COMPLETED | Job21192905; finalreturns MAPPO1260.62,AR2087.35,BPPO2007.54,AMidentity1980.50,AM1695.45; no10Mcomparison |
| R019 | M1 | Actor learning-rate controls | JointAgentMixer,200k,3seeds×2games | MIXED | Jobs21192929/21192931/21192932; fixedfastactorLR solvesClimbing3/3 by61–72k; failsPenalty3/3; lateLR solvesClimbing2/3 |
| R020 | M1 | Matched modifier/exploration controls | Factorizedfastactor andjointzeroentropy | FAIL robustness | Jobs21193031/21193032; factorized0/6; noentropyClimbing0/3,Penalty1/3 |
| R021 | M2 | MuJoCo protocol controls |7variants,500ksteps,seed0 | COMPLETED | Job21195730; returns: normonly2698.45,noiseonly1871.94,MAPPOprotocol2193.79,AR2191.15,BPPO2399.97,AMidentity1527.66,AM2128.55; development seed only |
| R022 | M2 | Predator–Prey pilot |5methods×2penalties,100ksteps,seed0 | COMPLETED | Job21195734; zero penalty capture90–100%; minus1 capture0–5%; episode metrics only, no exact paper-scale comparison |
| R023 | Setup | FreshSMAC installation | Separateprivateverifyenvironment | PASS | .native/smac-verify/environment-report.json;47pins,privatePython,imports; existingprefixunchanged |

| R024 | Setup | Native Football engine | Four scenarios, complete matches and seed replay | PASS | Build21195904; replay21196022;5v5 takes3001 API steps. Source2.10.2 checksum pinned |
| R025 | M0 | Football training integration | Four algorithms × four scenarios ×128steps | PASS | All16 runs in job21196049 completed; complete-match evaluations, no performance claim |
| R026 | Setup | Fresh portable Football installation |130 native packages,23 Python versions | PASS |21196130 completes package/import checks and allfour native replays; traces match originalbuild exactly. Earlier live-script interruption21196106 retained; private libopengl fixes login-node imports |
| R027 | M0 | Prey fixed-window metric and joint runner | Reward denominator, partial episodes, local execution, float64 likelihoods | PASS | Fixed-window tests plus shared-runner integration; joint event scored once, partial rewards retained |
| R028 | M2 | Prey coordination controls | Six variants × two penalties,300k,seed0 | COMPLETED |21196038; zero-penalty projectedAR/BPPO rates2.1975/2.185; factorizedAM1.9875,jointAM1.76. Penalty−1 captures0–5% across allsix controls |
| R029 | M2 | SMACv2 learning pilot | Five methods,terran5v5,20k,seed0 | COMPLETED |21196113; allfive methods completed with0/8 final wins; roughlyfour minutes/run |
| R030 | M2 | Longer normalized-critic MuJoCo follow-up | Five matched HC2x3 controls,3M,seed0 | RUNNING |21196118; dependencies21196038 and21196130;2workers,4h cap based on500k timings |
| R031 | M1 | Joint-distribution precision control | Two actor rates,both games,three seeds,200k | COMPLETED mixed |21196198 fast:Climbing2/3,Penalty2/3;21196199 slow:0/3both. Networks remainfloat32. Float32 fastcontrol3/3Climbing,0/3Penalty; no robust convergence |
| R032 | Targets | Target inventory linkage |45 source targets | PASS coverage | Six original numeric values unchanged;39 digitized selectors resolve uniquely; stale matrix episode interpretation qualified. scripts/check_paper_targets.py |
| R033 | M2 | Longer SMAC learning check | Five controls,terran5v5,200k,seed0 | FAILED runtime |21196276; MAPPO/AR-PPO stop at129,536 recorded steps on native restart; three methods not started; retry R038 |
| R034 | M0 | Complete method/observation coverage |15 MuJoCo variants+nine SMAC scenarios×four methods | PASS integration |21196515; all96 runs×64steps complete,diagnostic MuJoCo horizon16; not performance evidence |
| R035 | M2 | Prey rollout-batch controls | Eight variants×two penalties,1M,seed0 | RUNNING |21196495; all eight penalty0 runs complete, penalty−1 continues;100 fixed200-step evaluation windows |
| R036 | M2 | Football learning pilot | Five methods,3v1,100k,seed0 | QUEUED |21196497; full-match score/evaluation, no early-cap zeros |

| R037 | M0 | Native SMAC process renewal | Planned restart between episodes before game-clock expiry | UNIT PASS; native QUEUED |34 SMAC tests;21198088 requires a native restart with diagnostic128-tick threshold; source frozen |
| R038 | M2 | SMAC200k retry | Five methods,seed0,native clock guard | QUEUED |21198089; after Prey and successful R037; prior failed artifacts retained |
| R039 | M2 | Longer MuJoCo protocol comparison | Five matched methods,3M,seed0 | QUEUED |21198090;40×100 batch,initialstd0.365529,normalized clipped Huber critic; selected from existing500k development controls |
| R040 | Targets | Author metric clarification | Prey and Football5v5 figures | RECORDED |2026-10-25: “eval return”; docs/author_clarifications.md; exact reduction remains unresolved |
| R041 | M2 | Continuous BPPO noise coupling | Fresh versus behavior noise,two settings,seeds0–2,500k | QUEUED; numerical PASS |21198169; analytic two-epoch scale gradient, default identity and full trainer integration pass; no learning evidence yet |
