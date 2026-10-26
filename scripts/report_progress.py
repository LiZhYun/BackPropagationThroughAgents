#!/usr/bin/env python3
"""Report recorded run artifacts without querying or changing scheduler jobs."""
import argparse
import json
from pathlib import Path


def last_metric(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        lines = stream.read().splitlines()
    # A writer may be partway through its final JSON line. Read the preceding
    # complete record instead; do not call an unfinished file a failed run.
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


def report(root, latest):
    root = Path(root).resolve()
    jobs = {}
    for line in (root / 'experiments/jobs.jsonl').read_text().splitlines():
        entry = json.loads(line)
        jobs.setdefault(str(entry['job_id']), {}).update(entry)
    selected = [(job, entry) for job, entry in jobs.items() if entry.get('manifest')]
    if latest:
        selected = selected[-latest:]
    records = []
    for job, entry in selected:
        manifest_path = (root / entry['manifest']).resolve()
        try:
            if not manifest_path.is_relative_to(root):
                raise ValueError('recorded manifest is outside this repository')
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict) or not isinstance(manifest.get('runs'), list):
                raise ValueError('manifest has no run list')
        except (OSError, ValueError) as error:
            records.append({'job_id': job, 'manifest': entry['manifest'],
                            'artifact_status': 'manifest_missing' if isinstance(error, FileNotFoundError) else 'manifest_unreadable',
                            'error': str(error), 'counts': None, 'declared_runs': None,
                            'last_recorded_training_steps': None, 'runs': []})
            continue
        runs = []
        for run in manifest['runs']:
            directory = Path(run['output_path'])
            if not directory.is_absolute():
                directory = root / directory
            directory = directory.resolve()
            summary = directory / 'summary.json'
            item = {'run': directory.name, 'status': 'not_started', 'last_recorded_training_steps': 0}
            if not directory.is_relative_to(root):
                item.update(status='unavailable', error='recorded output directory is outside this repository')
                runs.append(item)
                continue
            if summary.exists():
                data = json.loads(summary.read_text())
                item.update(status=data.get('status', 'unknown'),
                            last_recorded_training_steps=data.get('steps', 0))
                if data.get('error'):
                    item['error'] = data['error']
            elif directory.exists():
                data = last_metric(directory / 'metrics.jsonl')
                item.update(status='incomplete', last_recorded_training_steps=data.get('steps', 0))
            runs.append(item)
        counts = {state: sum(run['status'] == state for run in runs)
                  for state in ('completed', 'failed', 'incomplete', 'not_started', 'unknown', 'unavailable')}
        records.append({'job_id': job, 'manifest': entry['manifest'], 'scope': manifest.get('scope', 'unspecified'),
                        'artifact_status': 'manifest_present',
                        'counts': counts, 'declared_runs': len(runs),
                        'last_recorded_training_steps': sum(run['last_recorded_training_steps'] for run in runs),
                        'runs': runs})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest', type=int, default=10, help='most recent manifests;0 means all')
    parser.add_argument('--json', action='store_true', help='include individual run statuses and scope')
    args = parser.parse_args()
    if args.latest < 0:
        parser.error('--latest must be nonnegative')
    records = report(Path(__file__).resolve().parents[1], args.latest)
    if args.json:
        print(json.dumps({'note': 'Artifact status only: incomplete does not imply a currently running scheduler job.',
                          'experiments': records}, indent=2))
    else:
        print('job\tcomplete/declared\tfailed\tincomplete\tnot_started\tunavailable\tlast_recorded_steps\texperiment')
        for entry in records:
            if entry['counts'] is None:
                print(f"{entry['job_id']}\tunknown\tunknown\tunknown\tunknown\tunknown\tunknown\t"
                      f"{Path(entry['manifest']).parent.name} ({entry['artifact_status']})")
                continue
            c = entry['counts']
            print(f"{entry['job_id']}\t{c['completed']}/{entry['declared_runs']}\t{c['failed']}\t"
                  f"{c['incomplete']}\t{c['not_started']}\t{c['unavailable']}\t{entry['last_recorded_training_steps']}\t"
                  f"{Path(entry['manifest']).parent.name}")
        print('Incomplete describes artifacts, not scheduler state. Missing manifests are unknown, not failed training runs.')


if __name__ == '__main__':
    main()
