# Experiment tracker

| ID | Stage | Purpose | System | Status | Evidence |
|---|---|---|---|---|---|
| R001 | M0 | Numerical and integration correctness | Shared core + both methods | PASS | 277 tests passed, 2026-10-25 |
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
| R018 | M2 | Longer MuJoCo learning check | SamefiveHC2x3controls,1Msteps,seed0 | PREPARED | configs/mujoco_hc2x3_1m.json; no final seed claim |
