#!/usr/bin/env python3
"""Show all paper tasks and their indexed experiment artifacts (standard library only)."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bta_repro.benchmark_coverage import summarize_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new JSON evidence snapshot')
    parser.add_argument('--csv', type=Path, help='new CSV coverage table')
    args = parser.parse_args()
    destinations = [p.resolve() for p in (args.output, args.csv) if p is not None]
    if len(destinations) != len(set(destinations)) or any(p.exists() for p in destinations):
        parser.error('outputs must be distinct new paths')
    report = summarize_coverage(ROOT)
    fields = ['benchmark', 'paper_budget', 'declared_method_runs',
              'completed_consistent_method_runs', 'max_completed_method_steps',
              'exact_paper_budget_method_runs']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['benchmarks'])
    print('\t'.join(fields))
    for row in report['benchmarks']:
        print('\t'.join(str(row[field]) for field in fields))
    print(f"Manifest errors: {len(report['manifest_errors'])}; "
          f"run errors: {sum(bool(r['errors']) for r in report['runs'])}; "
          f"unmatched runs: {len(report['unmatched_run_indices'])}.")
    print(report['interpretation'])


if __name__ == '__main__':
    main()
