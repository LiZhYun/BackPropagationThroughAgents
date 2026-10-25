"""Summarize every declared run, retaining incomplete/failed runs visibly."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest', type=Path)
parser.add_argument('--full', action='store_true', help='Also print every seed record')
args = parser.parse_args()
plan = json.loads(args.manifest.read_text())
groups = defaultdict(list)
missing = []
for entry in plan['runs']:
    path = Path(entry['output_path']) / 'summary.json'
    if not path.exists():
        missing.append({'algorithm': entry['algorithm'], 'game': entry['game'], 'seed': entry['seed']})
        continue
    summary = json.loads(path.read_text())
    summary['seed'] = entry['seed']
    groups[(entry['algorithm'], entry['game'])].append(summary)
report = {'scope': plan['scope'], 'missing_runs': missing, 'groups': []}
for (algorithm, game), runs in sorted(groups.items()):
    scores = [run['greedy_return'] for run in runs]
    expected = [run['execution_policy_expected_return'] for run in runs]
    report['groups'].append({
        'algorithm': algorithm, 'game': game, 'completed_seeds': [r['seed'] for r in runs],
        'greedy_mean': statistics.mean(scores),
        'greedy_sd': statistics.stdev(scores) if len(scores) > 1 else None,
        'greedy_median': statistics.median(scores),
        'optimal_greedy_seeds': sum(r['greedy_return'] == r['optimum_per_step'] for r in runs),
        'execution_expected_mean': statistics.mean(expected), 'runs': runs,
    })
text = json.dumps(report, indent=2) + '\n'
(args.manifest.parent / 'aggregate.json').write_text(text)
if args.full:
    print(text)
else:
    print(json.dumps({**report, 'groups': [{k:v for k,v in group.items() if k != 'runs'}
                                         for group in report['groups']]}, indent=2))
