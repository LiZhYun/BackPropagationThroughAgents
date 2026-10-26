"""Audit a frozen episodic-return confirmation and retain every declared seed."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bta_repro.simulator_confirmation import report_confirmation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.csv.resolve() or args.output.exists() or args.csv.exists():
        parser.error('outputs must be distinct new files')
    report = report_confirmation(args.plan, ROOT)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    fields = ['algorithm', 'variant', 'seed', 'status', 'eligible', 'final_return', 'normalized_auc', 'errors']
    with args.csv.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for run in report['runs']:
            writer.writerow({**run, 'errors': '; '.join(run['errors'])})
    print(json.dumps({k: report[k] for k in ('study_id', 'errors', 'groups', 'paired_comparison',
                                            'figure_comparison', 'validation_decision', 'paper_reproduction')}, indent=2))


if __name__ == '__main__':
    main()
