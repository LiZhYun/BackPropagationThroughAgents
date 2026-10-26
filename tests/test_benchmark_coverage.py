import hashlib
import json
from pathlib import Path

import pytest

from bta_repro.benchmark_coverage import summarize_coverage, task_fields


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def project(tmp_path):
    targets = {'configurations': {
        'base': {'values': {'environment_steps': 1000, 'agent_obsk': 0}},
        'ant_global': {'inherits': 'base', 'values': {'agent_obsk': None}}},
        'benchmarks': [
            {'id': 'bpta:Ant-v2_2x4', 'paper': 'bpta', 'suite': 'mamujoco',
             'task': 'Ant-v2_2x4', 'configuration': 'ant_global'},
            {'id': 'agentmixer:Ant-v2_2x4', 'paper': 'agentmixer', 'suite': 'mamujoco',
             'task': 'Ant-v2_2x4', 'configuration': 'base'},
            {'id': 'bpta:Walker2d-v2_2x3', 'paper': 'bpta', 'suite': 'mamujoco',
             'task': 'Walker2d-v2_2x3', 'configuration': 'base'}]}
    write(tmp_path / 'configs/paper_targets.json', targets)
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments/jobs.jsonl').write_text('')
    return tmp_path


def add_run(root, name='first', *, algorithm='bppo', obsk=None, horizon=1000,
            steps=1000, status='completed', modifier=True, seed=0):
    directory = root / 'results' / name
    source = directory / 'source/bta_repro/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('# frozen source\n')
    hashes = {'model.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    env = {'scenario': 'Ant-v2', 'agent_conf': '2x4', 'agent_obsk': obsk,
           'episode_limit': horizon, 'action_units': 'normalized'}
    config = {'algorithm': algorithm, 'seed': seed, 'steps': steps,
              'environment': 'mujoco', 'env_config': env, 'modifier_enabled': modifier}
    declared = directory / 'run.json'
    output = directory / 'run'
    write(declared, config)
    entry = {'algorithm': algorithm, 'seed': seed, 'variant': name,
             'config_path': str(declared), 'output_path': str(output)}
    manifest = {'source_sha256': hashes, 'runs': [entry]}
    write(directory / 'manifest.json', manifest)
    if status != 'not_started':
        write(output / 'config.json', {'config': config, 'source_sha256': hashes,
                                       'resolved_training_env_configs': [{**env, 'seed': 42}]})
    if status not in ('not_started', 'incomplete'):
        write(output / 'summary.json', {'status': status, 'steps': steps,
                                        'error': 'engine failed' if status == 'failed' else None})
    with (root / 'experiments/jobs.jsonl').open('a') as stream:
        stream.write(json.dumps({'job_id': name, 'manifest': str(directory / 'manifest.json')}) + '\n')
    return directory


def test_observation_protocol_identity_control_and_missing_targets(project):
    add_run(project, 'global_bppo')
    add_run(project, 'wrong_local_bppo', obsk=0)
    add_run(project, 'local_am', algorithm='agentmixer', obsk=0)
    add_run(project, 'identity', algorithm='agentmixer', obsk=0, modifier=False)
    report = summarize_coverage(project)
    bpta, am, missing = report['benchmarks']
    assert bpta['declared_method_runs'] == 2
    assert bpta['completed_consistent_method_runs'] == 1
    assert am['declared_method_runs'] == am['completed_consistent_method_runs'] == 1
    assert am['exact_paper_budget_method_runs'] == 1
    assert missing['completed_consistent_method_runs'] == missing['declared_method_runs'] == 0
    assert any(r['basic_task_differences'] for r in bpta['related_runs'])
    assert all(r['performance_and_independent_confirmation'] == 'not_assessed' for r in report['benchmarks'])


def test_integration_horizon_and_short_budget_remain_distinct(project):
    add_run(project, 'short_horizon', horizon=16)
    add_run(project, 'short_budget', steps=128)
    add_run(project, 'unfinished', status='incomplete')
    add_run(project, 'pending', status='not_started')
    add_run(project, 'failure', status='failed')
    report = summarize_coverage(project)
    row = report['benchmarks'][0]
    assert len(report['runs']) == 5
    assert row['completed_consistent_method_runs'] == 1
    assert row['max_completed_method_steps'] == 128
    assert row['exact_paper_budget_method_runs'] == 0
    assert row['method_artifact_status_counts'] == {'completed': 2, 'incomplete': 1, 'not_started': 1, 'failed': 1}
    assert report['runs'][-1]['training_error'] == 'engine failed'


@pytest.mark.parametrize('problem', ['changed_source', 'missing_source', 'config_mismatch',
                                    'short_summary', 'float_steps', 'source_metadata_mismatch'])
def test_inconsistent_artifacts_cannot_count_as_completed(project, problem):
    directory = add_run(project)
    metadata_path = directory / 'run/config.json'
    metadata = json.loads(metadata_path.read_text())
    if problem == 'changed_source':
        (directory / 'source/bta_repro/model.py').write_text('# changed\n')
    elif problem == 'missing_source':
        (directory / 'source/bta_repro/model.py').unlink()
    elif problem == 'config_mismatch':
        metadata['config']['steps'] = 2000
    elif problem == 'source_metadata_mismatch':
        metadata['source_sha256']['model.py'] = 'bad hash'
    else:
        write(directory / 'run/summary.json', {'status': 'completed',
                                               'steps': 500 if problem == 'short_summary' else 1000.0})
    write(metadata_path, metadata)
    report = summarize_coverage(project)
    assert len(report['runs']) == 1 and report['runs'][0]['errors']
    assert report['benchmarks'][0]['completed_consistent_method_runs'] == 0


def test_ledger_updates_do_not_duplicate_runs_but_conflicting_manifests_fail(project):
    directory = add_run(project)
    ledger = project / 'experiments/jobs.jsonl'
    with ledger.open('a') as stream:
        stream.write(json.dumps({'job_id': 'first', 'event': 'completed'}) + '\n')
    assert len(summarize_coverage(project)['runs']) == 1
    duplicate = directory / 'duplicate.json'
    duplicate.write_bytes((directory / 'manifest.json').read_bytes())
    with ledger.open('a') as stream:
        stream.write(json.dumps({'job_id': 'other', 'manifest': str(duplicate)}) + '\n')
    report = summarize_coverage(project)
    assert len(report['runs']) == 2
    assert all(row['errors'] and not row['artifact_consistent'] for row in report['runs'])


def test_missing_manifest_and_external_paths_are_explicit_errors(project):
    ledger = project / 'experiments/jobs.jsonl'
    ledger.write_text(json.dumps({'job_id': 'missing', 'manifest': 'results/missing.json'}) + '\n'
                      + json.dumps({'job_id': 'external', 'manifest': str(project.parent / 'other.json')}) + '\n')
    report = summarize_coverage(project)
    assert len(report['manifest_errors']) == 2
    assert report['runs'] == []
    assert len(report['benchmarks']) == 3


def test_older_smoke_manifest_uses_explicit_config_seed_without_guessing(project):
    directory = add_run(project, seed=37)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['runs'][0]['seed']
    del manifest['runs'][0]['variant']
    write(path, manifest)
    row = summarize_coverage(project)['runs'][0]
    assert row['seed'] == 37 and row['seed_source'] == 'declared_config'
    assert row['artifact_consistent']


def test_absolute_and_relative_aliases_of_one_manifest_are_not_duplicate_evidence(project):
    directory = add_run(project)
    with (project / 'experiments/jobs.jsonl').open('a') as stream:
        stream.write(json.dumps({'job_id': 'alias', 'manifest': 'results/first/manifest.json'}) + '\n')
    report = summarize_coverage(project)
    assert len(report['runs']) == 1 and report['runs'][0]['artifact_consistent']


@pytest.mark.parametrize('suite,task,expected', [
    ('matrix', 'climbing', {'game': 'climbing'}),
    ('smacv2', '10gen_terran_5v5', {'environment': 'smac', 'scenario': 'terran_5v5'}),
    ('grf', '5_vs_5', {'environment': 'football', 'scenario': '5_vs_5'}),
    ('predator_prey', 'predator_prey_penalty_-1', {'environment': 'prey', 'penalty': -1}),
])
def test_task_identity_mapping(suite, task, expected):
    identity, _ = task_fields({'suite': suite, 'task': task, 'configuration': 'base'},
                             {'base': {'values': {}}})
    assert identity == expected
