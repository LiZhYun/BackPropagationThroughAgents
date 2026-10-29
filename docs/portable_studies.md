# Exporting a completed simulator study

Export the whole frozen study after every declared run completes. The exporter rejects incomplete or inconsistent runs and requires a nonempty final checkpoint. It verifies the frozen source, configurations, full training budgets, evaluation schedules and summary/log agreement. It does not certify learning performance or independently replay the native engine.

```bash
python scripts/export_simulator_study.py results/STUDY/manifest.json \
  --output results/exported-study
python results/exported-study/bundle.py verify
python results/exported-study/bundle.py list
```

Copy or archive the resulting directory to another machine. Its operational manifest contains relative paths; original configs, metadata, logs, summaries, checkpoints and source bytes are preserved. The original manifest remains under `provenance/`. Native engines and environments are excluded. Use each run's recorded dependency and environment versions to configure the destination machine.

From any working directory, use the Python interpreter with the required dependencies:

```bash
python /path/to/exported-study/bundle.py train 0 --output /new/path/training
python /path/to/exported-study/bundle.py evaluate 0 --output /new/path/replay.json
```

The index selects a run shown by `list`. Both commands verify the bundle first, use its exact source through `PYTHONPATH`, and require a new output path outside the bundle. The evaluation command preserves the saved protocol and preprocessing. Check its `source_matches_training`, `versions_match_training`, final-checkpoint flag and raw episode records before claiming replay equivalence.

`verify` and `list` use only the standard library. Hashes detect byte changes; they do not authenticate results or replace scientific audits. Original workspace paths may remain in the preserved provenance, but are not used to launch training or evaluation. Do not treat missing native assets as a reason to substitute a different simulator silently.

The [verified export](../experiments/portable_study_verification_20261029.json) contains all 12 completed HalfCheetah partition development runs, 19 source modules and 85 inventoried files. After moving the bundle, all per-run curves and group statistics match exactly. Eleven focused tests also remove their synthetic original inputs and successfully execute the relocated frozen training module. Native replay of one real checkpoint is queued as job 21279857; export validation alone does not claim that replay has passed.
