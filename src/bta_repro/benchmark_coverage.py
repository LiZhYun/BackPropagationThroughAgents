"""Map recorded artifacts to paper tasks; never certify performance or release.

This standard-library report checks a small, explicit set of task fields and
artifact provenance. It does not audit every simulator setting, learning
curve, metric reduction, independent seed registration, or method assumption.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path


def _read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path is outside project: {name}")
    return path


def _settings(configurations, name, seen=()):
    if name in seen:
        raise ValueError("cycle in paper configuration inheritance")
    entry = configurations[name]
    parent = (_settings(configurations, entry['inherits'], (*seen, name))
              if entry.get('inherits') else {})
    return {**parent, **entry.get('values', {})}


def task_fields(benchmark, configurations):
    """Basic task fields only; source ambiguities remain a separate audit."""
    suite, task = benchmark['suite'], benchmark['task']
    settings = _settings(configurations, benchmark['configuration'])
    if suite == 'matrix':
        return {'game': task}, {}
    if suite == 'mamujoco':
        scenario, partition = task.rsplit('_', 1)
        return {'environment': 'mujoco', 'scenario': scenario, 'agent_conf': partition}, {
            'agent_obsk': settings['agent_obsk'], 'episode_limit': 1000,
            'action_units': 'normalized'}
    if suite == 'smacv2':
        return {'environment': 'smac', 'scenario': task.removeprefix('10gen_')}, {'game_version': '4.10'}
    if suite == 'grf':
        return {'environment': 'football', 'scenario': task}, {}
    if suite == 'predator_prey':
        penalty = int(task.removeprefix('predator_prey_penalty_'))
        return {'environment': 'prey', 'penalty': penalty}, {'episode_limit': 128}
    raise ValueError(f"unsupported benchmark suite: {suite}")


def _run(root, manifest_path, manifest, entry, source_errors):
    result = {'manifest': str(manifest_path.relative_to(root)),
              'variant': entry.get('variant', Path(entry['output_path']).name),
              'algorithm': entry['algorithm'], 'seed': entry.get('seed'),
              'status': 'not_started', 'artifact_consistent': False,
              'errors': list(source_errors)}
    try:
        output = _path(root, entry['output_path'])
        result['output_path'] = str(output.relative_to(root))
        declared, digest = _read(_path(root, entry['config_path']))
        result.update(config=declared, declared_config_sha256=digest,
                      declared_steps=declared['steps'], seed=declared['seed'],
                      seed_source='manifest_and_declared_config' if 'seed' in entry else 'declared_config')
        if (isinstance(declared['steps'], bool) or not isinstance(declared['steps'], int)
                or declared['steps'] <= 0):
            raise ValueError('declared steps must be a positive integer')
        if declared['algorithm'] != entry['algorithm'] or ('seed' in entry and declared['seed'] != entry['seed']):
            raise ValueError('manifest algorithm/seed differs from declared config')
        if any(key in entry and declared.get(key) != entry[key] for key in ('game', 'environment')):
            raise ValueError('manifest task differs from declared config')
        result['fields'] = {**declared.get('env_config', {}),
                            **{k: declared[k] for k in ('environment', 'game') if k in declared}}
        if not output.exists():
            return result
        result['status'] = 'incomplete'
        summary = None
        if (output / 'summary.json').exists():
            summary, digest = _read(output / 'summary.json')
            result.update(status=summary.get('status', 'unknown'), summary_sha256=digest,
                          recorded_steps=summary.get('steps'))
            if summary.get('error'):
                result['training_error'] = summary['error']
        metadata, digest = _read(output / 'config.json')
        result['metadata_sha256'] = digest
        config = metadata['config']
        if any(config.get(k) != v for k, v in declared.items()):
            raise ValueError('recorded config differs from declared config')
        if metadata['source_sha256'] != manifest['source_sha256']:
            raise ValueError('recorded source hashes differ from manifest')
        result.update(config=config, method_variant=metadata.get('method_variant'),
                      environment_versions=metadata.get('environment_versions'))
        resolved = metadata.get('resolved_training_env_configs', [])
        result['resolved_env_configs'] = resolved
        if resolved:
            first = {k: v for k, v in resolved[0].items() if k != 'seed'}
            if any({k: v for k, v in env.items() if k != 'seed'} != first for env in resolved):
                raise ValueError('resolved training environments differ beyond seed')
            result['fields'] = {**first, **{k: config[k] for k in ('environment', 'game') if k in config}}
        if summary is not None and result['status'] == 'completed':
            if (isinstance(summary.get('steps'), bool)
                    or not isinstance(summary.get('steps'), int)
                    or summary.get('steps') != declared['steps']):
                raise ValueError('completed summary does not reach declared budget')
            result['artifact_consistent'] = not result['errors']
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['errors'].append(str(error))
    return result


def summarize_coverage(root):
    root = Path(root).resolve()
    targets, targets_hash = _read(root / 'configs/paper_targets.json')
    ledger = root / 'experiments/jobs.jsonl'
    ledger_data = ledger.read_bytes()
    jobs = {}
    for line in ledger_data.splitlines():
        entry = json.loads(line)
        jobs.setdefault(str(entry['job_id']), {}).update(entry)
    paths = dict.fromkeys(entry['manifest'] for entry in jobs.values() if entry.get('manifest'))
    records, errors, manifest_hashes = [], [], {}
    seen_outputs, seen_manifests = {}, set()
    for name in paths:
        try:
            path = _path(root, name)
            if path in seen_manifests:
                continue
            seen_manifests.add(path)
            manifest, digest = _read(path)
            manifest_hashes[str(path.relative_to(root))] = digest
            source_errors = []
            if not manifest['source_sha256']:
                source_errors.append('manifest has no frozen source hashes')
            for filename, expected in manifest['source_sha256'].items():
                try:
                    source = _path(root, path.parent / 'source/bta_repro' / filename)
                    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                        source_errors.append(f'frozen source hash mismatch: {filename}')
                except (OSError, ValueError) as error:
                    source_errors.append(f'{filename}: {error}')
            for entry in manifest['runs']:
                row = _run(root, path, manifest, entry, source_errors)
                key = row.get('output_path')
                if key is not None and key in seen_outputs:
                    previous = seen_outputs[key]
                    message = f'duplicate output directory in indexed manifests: {key}'
                    previous['errors'].append(message)
                    previous['artifact_consistent'] = False
                    row['errors'].append(message)
                    row['artifact_consistent'] = False
                elif key is not None:
                    seen_outputs[key] = row
                records.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'manifest': name, 'error': str(error)})
    rows = []
    identifiers = [b['id'] for b in targets['benchmarks']]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('duplicate paper benchmark identifier')
    matched = set()
    for benchmark in targets['benchmarks']:
        task, checks = task_fields(benchmark, targets['configurations'])
        budget = _settings(targets['configurations'], benchmark['configuration'])['environment_steps']
        related = []
        for index, record in enumerate(records):
            fields = record.get('fields', {})
            if any(k not in fields or fields[k] != v for k, v in task.items()):
                continue
            matched.add(index)
            config = record.get('config', {})
            algorithm = 'bppo' if benchmark['paper'] == 'bpta' else 'agentmixer'
            method = (record['algorithm'] == algorithm and
                      (algorithm != 'agentmixer' or config.get('modifier_enabled', True)))
            differences = [f'{k}: expected {v!r}, found {fields.get(k, "<missing>")!r}'
                           for k, v in checks.items() if k not in fields or fields[k] != v]
            related.append({'run_index': index, 'role': 'method' if method else 'control',
                            'basic_task_differences': differences})
        method_runs = [records[r['run_index']] for r in related if r['role'] == 'method']
        consistent = [records[r['run_index']] for r in related
                      if r['role'] == 'method' and not r['basic_task_differences']
                      and records[r['run_index']]['artifact_consistent']]
        rows.append({'benchmark': benchmark['id'], 'paper_budget': budget,
                     'checked_task_fields': {**task, **checks},
                     'declared_method_runs': len(method_runs),
                     'method_artifact_status_counts': dict(Counter(r['status'] for r in method_runs)),
                     'completed_consistent_method_runs': len(consistent),
                     'max_completed_method_steps': max((r['recorded_steps'] for r in consistent), default=0),
                     'exact_paper_budget_method_runs': sum(r['recorded_steps'] == budget for r in consistent),
                     'related_runs': related,
                     'performance_and_independent_confirmation': 'not_assessed'})
    return {'scope': __doc__, 'target_sha256': targets_hash,
            'ledger_sha256': hashlib.sha256(ledger_data).hexdigest(),
            'reporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifest_sha256': manifest_hashes, 'manifest_errors': errors,
            'unmatched_run_indices': sorted(set(range(len(records))) - matched),
            'benchmarks': rows, 'runs': records,
            'interpretation': 'Counts describe indexed artifacts only. Basic task checks are not full protocol '
                              'equivalence. No scores are compared, seeds pooled, or release claims certified. '
                              'Incomplete is an artifact state, not a scheduler query.'}
