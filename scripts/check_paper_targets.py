#!/usr/bin/env python3
"""Check target inventory coverage; this does not check achieved performance."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'configs/paper_targets.json').read_text())
    files, errors, numeric, digitized = {}, [], 0, 0
    identifiers = set()
    for benchmark in manifest['benchmarks']:
        identifier = benchmark['id']
        if identifier in identifiers:
            errors.append(f'duplicate benchmark: {identifier}')
        identifiers.add(identifier)
        values = [m['published_value'] for m in benchmark['metrics'] if m['published_value'] is not None]
        reference = benchmark.get('digitized_target')
        if values:
            numeric += 1
            if reference:
                errors.append(f'{identifier}: numeric and digitized targets require an explicit precedence policy')
            continue
        if not reference:
            errors.append(f'{identifier}: no numeric or digitized target')
            continue
        filename = reference['file']
        if filename not in files:
            files[filename] = json.loads((root / filename).read_text())['targets']
        selector = reference['selector']
        matches = [row for row in files[filename] if row.get('primary_benchmark_target', True)
                   and all(row.get(key) == value for key, value in selector.items())]
        if len(matches) != 1:
            errors.append(f'{identifier}: expected one primary curve, found {len(matches)}')
            continue
        target = matches[0]
        if (target['paper'], target['task']) != (benchmark['paper'], benchmark['task']):
            errors.append(f'{identifier}: curve belongs to a different benchmark')
        lower, upper = target['digitization_reading_range']
        if not lower <= target['reported_endpoint_rounded'] <= upper:
            errors.append(f'{identifier}: endpoint is outside its reading range')
        if target['digitization_uncertainty']['is_statistical_confidence_interval']:
            errors.append(f'{identifier}: geometric reading uncertainty was relabeled as statistical')
        digitized += 1
    print(json.dumps({'purpose': 'target coverage only, not reproduced performance',
                      'benchmarks': len(identifiers), 'numeric_source_targets': numeric,
                      'digitized_curve_targets': digitized, 'ok': not errors, 'errors': errors}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
