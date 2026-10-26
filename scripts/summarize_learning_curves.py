"""Export full-budget simulator learning curves and training-seed statistics."""

import argparse
import json
from pathlib import Path

from bta_repro.learning_curves import METRICS, summarize_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--metric", choices=METRICS, default="return_mean")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()
    report = summarize_curves(args.manifest, args.metric)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for group in report["groups"]:
        aggregate = group["aggregate"]
        record = {key: group[key] for key in ("variant", "declared_seeds", "eligible_seeds", "aggregation_errors")}
        if aggregate:
            record.update(final=aggregate["final"], normalized_auc=aggregate["normalized_auc"])
        print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
