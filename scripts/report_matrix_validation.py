"""Report a frozen matrix validation or budget-extension study without training."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bta_repro.matrix_validation import report_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.csv.resolve() or args.output.exists() or args.csv.exists():
        parser.error("outputs must be distinct new files")
    report = report_study(args.plan, ROOT)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    columns = ["algorithm", "game", "seed", "status", "eligible", "threshold_success",
               "execution_policy_expected_return", "greedy_return", "optimal_action_probability",
               "training_policy_expected_return", "training_optimal_action_probability",
               "normalized_expected_return_auc", "first_observed_sustained_95pct_expected_return_step", "errors"]
    with args.csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run in report["runs"]:
            writer.writerow({**run, **run.get("final", {}), "errors": "; ".join(run["errors"])})
    print(json.dumps({key: report[key] for key in ("study_id", "errors", "groups", "paired_comparisons", "validation_decision", "paper_reproduction")}, indent=2))


if __name__ == "__main__":
    main()
