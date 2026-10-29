# BPTA and AgentMixer reproduction

This subdirectory is the only writable project scope for this reproduction. Another Codex session is reproducing an unrelated paper in the parent workspace. Preserve all unrelated files, environments, jobs, and changes. Never stop jobs by user name. Use unique bpta-agentmixer job names and record submitted job IDs.

The user authorizes a readable redesign preserving the core ideas of BPTA/BPPO and AgentMixer. On 2026-10-29 the user clarified that the priority is an open, clear, faithful repo ASAP: full-budget runs and exhaustive experiments are not required. All research sweeps are stopped. Do not launch or resume research sweeps without a new user request; run only correctness, packaging and small runnable-example checks needed for release. Performance beyond published results is not a release gate and must never be claimed without evidence. Once the release checks pass, push to the user-specified GitHub repository. Accepted AAAI sources in references/ are canonical. Do not edit reference sources. Keep original archives in the parent untouched.

Do not report smoke tests or toy results as full reproduction. Document corrected parameterizations and unresolved differences from the paper. New results must identify configuration, seeds, simulator versions, steps, and uncertainty. Use independent final evaluation seeds after selecting configurations.

Implementation uses a small Python package src/bta_repro. Keep algorithm-specific losses explicit. Default logging is local; no account or telemetry required. Optional simulators must not be required for matrix games. Every source or code change belongs to this subproject.
