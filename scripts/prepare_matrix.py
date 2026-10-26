"""Freeze matrix configurations and verified source; never submit or train."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil

from bta_repro.experiment_source import source_snapshot
from bta_repro.train import Config


MATRIX_MODULES = ("__init__.py", "envs.py", "core.py", "bppo.py", "agentmixer.py",
                  "agentmixer_joint.py", "train.py", "checkpoint.py")


def prepare(root, *, label="matrix_dev", overrides=None, algorithms=("mappo", "arppo", "bppo", "agentmixer"),
            games=("climbing", "penalty"), seeds=(0, 1, 2), steps=None, workers=2,
            wall_time_minutes=30, source_manifest=None, stamp=None):
    root = Path(root).resolve()
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("matrix sweeps support one or two workers")
    if type(wall_time_minutes) is not int or not 1 <= wall_time_minutes <= 1440:
        raise ValueError("wall_time_minutes must be an integer between 1 and 1440")
    if not label.replace("_", "").isalnum():
        raise ValueError("label must contain only letters, digits, and underscores")
    for name, values in (("algorithms", algorithms), ("games", games), ("seeds", seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("training seeds must be uint32 integers")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("configuration overrides must be a JSON object")
    base = {"steps": 100000, "batch_size": 1024, "epochs": 4, "threads": 1, **(overrides or {})}
    if steps is not None:
        base["steps"] = steps
    entries = []
    for algorithm in algorithms:
        for game in games:
            for seed in seeds:
                config = {**base, "algorithm": algorithm, "game": game, "seed": seed}
                Config(**config).validate()
                entries.append(config)
    files, hashes, provenance = source_snapshot(
        root, source_manifest, required_modules=("__init__.py", "train.py"),
        module_names=MATRIX_MODULES if source_manifest is None else None)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "results" / f"{label}_{stamp}"
    snapshot = directory / "source/bta_repro"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (snapshot / name).write_bytes(data)
    runs = []
    for config in entries:
        algorithm, game, seed = (config[k] for k in ("algorithm", "game", "seed"))
        name = f"{algorithm}_{game}_seed{seed}"
        config_path = directory / (name + ".json")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        runs.append({"algorithm": algorithm, "game": game, "seed": seed,
                     "config_path": str(config_path), "output_path": str(directory / name)})
    manifest = {"scope": "Prepared matrix experiment; validation context requires a separate frozen study plan.",
                "source_sha256": hashes, "source_provenance": provenance, "runs": runs}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(root / "scripts/run_matrix_array.py", directory / "run_array.py")
    workers = min(workers, len(runs))
    quote = lambda path: shlex.quote(str(path))
    (directory / "submit.sbatch").write_text(f'''#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-matrix-dev
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time={wall_time_minutes // 60:02d}:{wall_time_minutes % 60:02d}:00
#SBATCH --array=0-{workers-1}%2
#SBATCH --output={directory}/slurm_%A_%a.out
#SBATCH --error={directory}/slurm_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BTA_MATRIX_WORKERS={workers}
export PYTHONPATH={quote(directory / 'source')}
cd {quote(root)}
exec {quote(root / '.venv/bin/python')} {quote(directory / 'run_array.py')} {quote(directory / 'manifest.json')}
''')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="matrix_dev")
    parser.add_argument("--overrides", type=Path, help="flat JSON settings applied to every run")
    parser.add_argument("--algorithms", nargs="+", default=["mappo", "arppo", "bppo", "agentmixer"])
    parser.add_argument("--games", nargs="+", default=["climbing", "penalty"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, help="override the JSON budget; default is 100000 when omitted from both")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wall-time-minutes", type=int, default=30)
    parser.add_argument("--source-manifest", type=Path, help="reuse a previous experiment's exact verified source")
    args = vars(parser.parse_args())
    args["overrides"] = json.loads(args["overrides"].read_text()) if args["overrides"] else None
    print(prepare(Path(__file__).resolve().parents[1], **args))


if __name__ == "__main__":
    main()
