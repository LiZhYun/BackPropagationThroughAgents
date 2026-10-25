"""Summarize every declared simulator run, including missing and failed runs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    records = []
    groups = {}
    for entry in manifest['runs']:
        output = Path(entry['output_path'])
        summary_path = output / 'summary.json'
        record = {key: entry[key] for key in ('variant', 'algorithm', 'seed')}
        record['output_path'] = str(output)
        if not summary_path.exists():
            record['status'] = 'unfinished' if output.exists() else 'not_started'
        else:
            result = json.loads(summary_path.read_text())
            record['status'] = result.get('status', 'unknown')
            record['steps'] = result.get('steps')
            record['elapsed_seconds'] = result.get('elapsed_seconds')
            evaluation = result.get('final_evaluation', {})
            record['evaluation_return_mean'] = evaluation.get('return_mean')
            record['evaluation_win_rate'] = evaluation.get('win_rate')
            record['evaluation_capture_rate'] = evaluation.get('capture_rate')
            record['evaluation_reward_per_environment_step'] = evaluation.get('reward_per_environment_step')
            record['evaluation_fixed_window_reward_per_step'] = evaluation.get('fixed_window', {}).get('reward_per_environment_step')
            record['evaluation_episode_count'] = evaluation.get('count')
            record['evaluation_seed'] = evaluation.get('seed')
            record['error'] = result.get('error')
        records.append(record)
        groups.setdefault(entry['variant'], []).append(record)
    aggregates = []
    for variant, runs in groups.items():
        complete = [run for run in runs if run['status'] == 'completed']
        group = {'variant': variant, 'declared_seeds': [run['seed'] for run in runs],
                 'completed_seeds': [run['seed'] for run in complete],
                 'all_declared_runs_completed': len(complete) == len(runs)}
        for metric in ('evaluation_return_mean', 'evaluation_win_rate', 'evaluation_capture_rate',
                       'evaluation_reward_per_environment_step', 'evaluation_fixed_window_reward_per_step'):
            values = [run[metric] for run in complete if run.get(metric) is not None]
            group[metric] = ({'mean': statistics.mean(values),
                              'median': statistics.median(values),
                              'sample_sd_across_training_seeds': statistics.stdev(values) if len(values) > 1 else None,
                              'seed_count': len(values)} if values else None)
        aggregates.append(group)
    return {'scope': manifest['scope'],
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'interpretation': 'Development results; variation across evaluation episodes is not training-seed uncertainty. Missing or failed seeds remain listed.',
            'runs': records, 'groups': aggregates}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    arguments = parser.parse_args()
    report = summarize(arguments.manifest)
    destination = arguments.manifest.parent / 'aggregate.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'scope': report['scope'], 'groups': report['groups'],
                      'unfinished_runs': [run for run in report['runs'] if run['status'] != 'completed']},
                     indent=2, allow_nan=False))
