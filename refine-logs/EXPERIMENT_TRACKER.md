# Experiment tracker

| ID | Stage | Purpose | System | Status | Evidence |
|---|---|---|---|---|---|
| R001 | M0 | Numerical and integration correctness | Shared core + both methods | PASS | 607 tests passed, 1 CUDA-only skip, 2026-10-26 |
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
| R030 | M2 | Longer normalized-critic MuJoCo follow-up | Five matched HC2x3 controls, 3M, seed 0 | COMPLETED | 21196118; final returns MAPPO 3045.06, AR 3599.84, BPPO 599.40, AM 2607.24, identity 1359.66 |
| R031 | M1 | Joint-distribution precision control | Two actor rates,both games,three seeds,200k | COMPLETED mixed |21196198 fast:Climbing2/3,Penalty2/3;21196199 slow:0/3both. Networks remainfloat32. Float32 fastcontrol3/3Climbing,0/3Penalty; no robust convergence |
| R032 | Targets | Target inventory linkage |45 source targets | PASS coverage | Six original numeric values unchanged;39 digitized selectors resolve uniquely; stale matrix episode interpretation qualified. scripts/check_paper_targets.py |
| R033 | M2 | Longer SMAC learning check | Five controls,terran5v5,200k,seed0 | FAILED runtime |21196276; MAPPO/AR-PPO stop at129,536 recorded steps on native restart; three methods not started; retry R038 |
| R034 | M0 | Complete method/observation coverage |15 MuJoCo variants+nine SMAC scenarios×four methods | PASS integration |21196515; all96 runs×64steps complete,diagnostic MuJoCo horizon16; not performance evidence |
| R035 | M2 | Prey rollout-batch controls | Eight variants × two penalties, 1M, seed 0 | COMPLETED; hard task unsolved | 21196495; penalty −1 fixed-window reward remains negative in all eight variants; entropy control captures 9/20 but return remains negative |
| R036 | M2 | Football learning pilot | Five methods, 3v1, 100k, seed 0 | COMPLETED; zero wins | 21196497; all five final evaluations 0/20 wins despite positive shaped return |

| R037 | M0 | Native SMAC process renewal | Planned restart between episodes before game-clock expiry | PASS native |34 SMAC tests;21198088 passes512 transitions and12 planned restarts with diagnostic128-tick threshold; scheduler exit0 |
| R038 | M2 | SMAC 200k retry | Five methods, seed 0, native clock guard | COMPLETED | 21198089; all five pass previous failure point, each logs two planned native restarts; final wins MAPPO 1/32, AR 4/32, BPPO 0/32, AM 2/32, identity 0/32 |
| R039 | M2 | Longer MuJoCo protocol comparison | Five matched methods, 3M, seed 0 | COMPLETED | 21198090; MAPPO 4041.12, AR 4739.43, BPPO 5228.01, AM 2411.14, identity 2975.45; BPPO +10.31% versus AR in one development seed |
| R040 | Targets | Author metric clarification | Prey and Football5v5 figures | RECORDED |2026-10-25: “eval return”; docs/author_clarifications.md; exact reduction remains unresolved |
| R041 | M2 | Continuous BPPO noise coupling | Two estimators × two settings × three seeds, 500k | COMPLETED; mixed | 21198169; fresh/behavior mean±SD: small batch 1844.05±1053.90 / 2853.01±343.67; combined protocol 2547.40±546.97 / 1959.81±805.73 |
| R042 | M0 | Saved-policy evaluation | All methods,both action spaces,normalization/source provenance | PASS |560 full-suite tests; native Prey100k AgentMixer replay exactly equals original20-episode evaluation using frozen source; scripts/evaluate_simulator.py |

| R043 | M2 | First paper-budget HalfCheetah comparison | BPPO and AR-PPO, 10M, seeds 0–2 | SUBMITTED | 21208906; configs/mujoco_hc2x3_bppo_ar10m.json; 32 evaluation episodes, two workers, 12h cap; development selection, not final independent seeds |
| R044 | M2 | Continuous AgentMixer scale map | Residual, bounded replacement, identity; 1M, seeds 0–2 | 6/9 complete; bounded not better | 21209002; residual mean±SD2722.35±291.72, bounded2546.83±60.04; full paired seed results retained; identity pending in snapshot; numerical invariants pass |
| R045 | Analysis | Completed development results | All 48 runs from six suites | RECORDED | docs/development_results_20261026.md; raw CSV with per-run summary/manifest hashes; no final superiority claim |
| R046 | M2 | Longer SMAC learning | Five methods, terran5v5, 1M, seed 0 | QUEUED | 21209234 after 21209002 and 21210161; same runtime and optimizer settings as completed 200k; two workers, 12h cap |
| R047 | M2 | Longer Football learning | Five methods, 3v1, 1M, seed 0 | QUEUED | 21209248 after 21208906; same settings as zero-win 100k pilot; two workers, 12h cap |
| R048 | Diagnosis | Prey execution behavior | All16 final policies, original20 evaluation episodes | PASS replay; learning gap found | Frozen source reproduces every per-episode return within1e-12. Small-batch AM recent training captures83–100%, greedy evaluation0/20; repeated blocked moves. Fresh100-episode distribution diagnostic21210161 completed; see R051; docs/prey_behavior_diagnosis.md |
| R049 | Reporting | Full-budget learning curves and seed uncertainty | All48 completed native runs, plus live incomplete controls | PASS | 24 targeted checks; full suite607 passed/1skip. Missing seeds, changed protocols, source mismatches and absent endpoints block aggregates; PNG/PDF and data exported. docs/learning_curve_reports.md |
| R050 | M2 | Continuous exploration control | Three AgentMixer variants, entropy0.01, 1M, seeds0–2 | QUEUED | 21211006 after21209234. Compare all nine against zero-entropy controls; same core and tasks. First residual/bounded seed0 scale≈0.043 motivates hypothesis, not an established cause |
| R051 | Diagnosis | Frozen Prey execution distributions | Six policies × three execution modes ×100 fresh shared episode seeds | COMPLETED; deployment gap | 21210161 completed0:0 in6m18s. Small-batch central captures85/100 and98/100 versus greedy0/100 and local-sampling3/100 or0/100. Modes checked equal at every visited state; no centralized score used as benchmark success |
