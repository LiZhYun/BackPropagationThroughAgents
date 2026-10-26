"""Check an installed wheel's contents, console commands and checkpoint replay.

Run with the wheel environment's Python, from outside the source checkout.
The short training runs check integration only, not learning performance.
"""

import argparse
import hashlib
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def module_hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory outside the source checkout")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if Path.cwd().resolve().is_relative_to(checkout) or output.is_relative_to(checkout):
        parser.error("run from outside the checkout and choose an output outside it")
    if output.exists():
        parser.error("output already exists; choose a new path")

    import bta_repro

    installed = Path(bta_repro.__file__).resolve().parent
    if installed == checkout / "src/bta_repro":
        raise RuntimeError("the package is imported from the source checkout")
    dist = distribution("bta-repro")
    if any(p.name.endswith(".pth") for p in dist.files or []):
        raise RuntimeError("install the wheel, not an editable source checkout")
    hashes = module_hashes(installed)
    assert hashes == module_hashes(checkout / "src/bta_repro"), "installed modules differ"
    assert dist.metadata["License-Expression"] == "MIT"
    assert dist.metadata.get_payload().strip() == (checkout / "README.md").read_text().strip()
    licenses = [p for p in dist.files or [] if str(p).endswith("/licenses/LICENSE")]
    assert len(licenses) == 1
    assert dist.locate_file(licenses[0]).read_bytes() == (checkout / "LICENSE").read_bytes()

    output.mkdir(parents=True)
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    commands = []

    def run(name, command):
        with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
            result = subprocess.run([str(p) for p in command], cwd=output,
                                    env=environment, stdout=stdout, stderr=stderr)
        commands.append({"name": name, "command": [str(p) for p in command],
                         "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.stderr')}")

    def config(name, settings):
        path = output / f"{name}.json"
        path.write_text(json.dumps(settings, indent=2) + "\n")
        return path

    # sys.executable's directory contains the Linux console entry points.
    console = Path(sys.executable).parent
    for method, example in [("bppo", "matrix_projected_entropy01.json"),
                            ("agentmixer", "matrix_joint_lr_fast.json")]:
        settings = json.loads((checkout / "configs" / example).read_text())
        settings.update(steps=64, batch_size=32, hidden_dim=8, epochs=1, eval_every=1, threads=1)
        path = config(method, settings)
        run(method, [console / "bta-train", "--config", path, "--algorithm", method,
                     "--game", "climbing", "--seed", "0", "--output", output / method])
        summary = json.loads((output / method / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["steps"] == 64
        assert math.isfinite(summary["execution_policy_expected_return"])

    path = config("prey", {
        "environment": "prey", "env_config": {"penalty": -1, "episode_limit": 4},
        "algorithm": "agentmixer", "recurrent_actor": True, "steps": 16,
        "num_envs": 2, "rollout_steps": 4, "hidden_dim": 8, "epochs": 1,
        "eval_episodes": 2, "eval_windows": 2, "eval_window_steps": 8,
        "normalize_value_targets": True, "threads": 1,
    })
    run("prey", [console / "bta-train-simulator", "--config", path, "--output", output / "prey"])
    run("replay", [sys.executable, checkout / "scripts/evaluate_simulator.py", output / "prey",
                   "--output", output / "replay.json"])
    summary = json.loads((output / "prey/summary.json").read_text())
    replay = json.loads((output / "replay.json").read_text())
    assert summary["status"] == "completed" and summary["steps"] == 16
    assert replay["source_matches_training"] and replay["checkpoint_is_final"]
    assert replay["evaluation"] == summary["final_evaluation"], "checkpoint replay differs"
    record = {"scope": "installed-package integration only; not learning performance",
              "installed_package": str(installed), "module_sha256": hashes,
              "commands": commands, "exact_checkpoint_replay": True}
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Installed package verified: {len(hashes)} modules, two matrix commands and recurrent Prey replay.")


if __name__ == "__main__":
    main()
