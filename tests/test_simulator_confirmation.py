"""Synthetic studies exercise integrity failures, raw reductions and paired seeds."""

import json
import math
from pathlib import Path
import random
import shutil

import pytest

from bta_repro.simulator_confirmation import digest, expected_grid, report_confirmation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def evaluation(value):
    episodes = [{'episode': i, 'return': value + delta, 'length': 4,
                 'terminated': False, 'truncated': True, 'info': {}}
                for i, delta in enumerate((-1., 0., 1.))]
    return {'count': 3, 'seed': 2000000, 'execution': 'autoregressive', 'deterministic': True,
            'return_mean': value, 'return_std': 1., 'length_mean': 4.,
            'terminated': 0, 'truncated': 3, 'environment_steps': 12,
            'reward_per_environment_step': value / 4, 'reward_per_step_scope': 'completed_episodes_only',
            'episodes': episodes}


def register(root, plan):
    write(root / 'study.json', plan)
    write(root / 'experiments/jobs.jsonl', {'plan': 'study.json', 'plan_sha256': digest(root / 'study.json'),
                                          'recorded_utc': '2026-01-02T01:00:00+00:00'})


def study(root, bppo=(12., 13., 14., 15., 16.)):
    directory = root / 'runs'
    source = directory / 'source/bta_repro'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('')
    (source / 'simulator_train.py').write_text('# frozen fixture, not executable training\n')
    hashes = {p.name: digest(p) for p in source.glob('*.py')}
    for name in ('run_simulator_worker.py', 'run_simulator_array.py', 'submit.sbatch'):
        (directory / name).write_text('fixture\n')
    versions = {'python': 'test', 'torch': 'test'}
    env_versions = {'engine': 'pinned_fixture'}
    inventory = {'recorded_utc': '2026-01-01T00:00:00+00:00',
                 'runs': [{'seed': s, 'eval_seed': 1000000} for s in (0, 1, 2)]}
    write(root / 'inventory.json', inventory)
    write(root / 'selected/manifest.json', {'source_sha256': hashes})
    write(root / 'selected/curves.json', {'manifest_sha256': digest(root / 'selected/manifest.json'),
                                        'runs': [{'algorithm': a, 'eligible': True, 'environment_versions': env_versions}
                                                 for a in ('bppo', 'arppo')]})
    seeds = list(range(1000, 1005))
    plan = {'study_id': 'fixture', 'scope': 'synthetic confirmation, not learning evidence',
            'frozen_utc': '2026-01-02T00:00:00+00:00', 'manifest': 'runs/manifest.json',
            'source_sha256': hashes, 'versions': versions, 'training_seeds': seeds,
            'prior_recorded_training_seeds': [0, 1, 2], 'prior_inventory': 'inventory.json',
            'prior_inventory_sha256': digest(root / 'inventory.json'), 'prior_inventory_run_count': 3,
            'launchers': {name: digest(directory / name) for name in
                          ('run_simulator_worker.py', 'run_simulator_array.py', 'submit.sbatch')},
            'selection_evidence': {'manifest': 'selected/manifest.json',
                                   'manifest_sha256': digest(root / 'selected/manifest.json'),
                                   'validated_curves': 'selected/curves.json',
                                   'curves_sha256': digest(root / 'selected/curves.json')},
            'evaluation': {'metric': 'final_evaluation.return_mean', 'seed': 2000000,
                           'budget': 10, 'episodes': 3, 'episode_limit': 4},
            'training_environment_seeds': {str(s): [10000 + 2*i, 10001 + 2*i] for i, s in enumerate(seeds)},
            'paper_reference': {'reading_range': [9., 10.]}, 'runs': []}
    for algorithm in ('bppo', 'arppo'):
        for i, seed in enumerate(seeds):
            value = bppo[i] if algorithm == 'bppo' else 8. + i
            config = {'algorithm': algorithm, 'seed': seed, 'environment': 'mujoco',
                      'env_config': {'episode_limit': 4}, 'steps': 10, 'num_envs': 2,
                      'rollout_steps': 2, 'eval_every_steps': 3, 'eval_episodes': 3, 'eval_seed': 2000000}
            name = f'{algorithm}_seed{seed}'
            cp = directory / f'{name}.json'
            write(cp, config)
            entry = {'algorithm': algorithm, 'variant': algorithm, 'seed': seed,
                     'config_path': str(cp.relative_to(root)), 'config_sha256': digest(cp),
                     'resolved_config': config, 'output_path': f'runs/{name}'}
            plan['runs'].append(entry)
            output = root / entry['output_path']
            write(output / 'config.json', {'config': config, 'source_sha256': hashes, 'versions': versions,
                                           'environment_versions': env_versions,
                                           'created_utc': '2026-01-03T00:00:00+00:00',
                                           'evaluation_environment_seed': 2000000,
                                           'training_environment_seeds': plan['training_environment_seeds'][str(seed)]})
            rows = [{'event': 'evaluation', 'stage': 'initial', 'steps': 0, 'evaluation': evaluation(0.)}]
            for update, step in enumerate((4, 8, 10), 1):
                rows.append({'event': 'update', 'steps': step, 'update': update,
                             'rollout_shape': [2 if step < 10 else 1, 2]})
                rows.append({'event': 'evaluation', 'stage': 'final' if step == 10 else 'periodic',
                             'steps': step, 'evaluation': evaluation(value * step / 10)})
            (output / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            write(output / 'summary.json', {'status': 'completed', 'steps': 10, 'train_environment_steps': 10,
                                            'evaluation_environment_steps': 48, 'updates': 3,
                                            'initial_evaluation': rows[0]['evaluation'], 'final_evaluation': rows[-1]['evaluation']})
            (output / 'checkpoint.pt').write_bytes(b'presence fixture; this standard-library report never loads a model')
    write(root / plan['manifest'], {'source_sha256': hashes, 'runs': [
        {k: r[k] for k in ('algorithm', 'variant', 'seed', 'config_path', 'output_path')} for r in plan['runs']]})
    plan['manifest_sha256'] = digest(root / plan['manifest'])
    register(root, plan)
    return plan


def report(root):
    return report_confirmation(root / 'study.json', root)


def edit(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write(path, value)


def test_complete_study_recomputes_episodes_and_pairs_by_seed(tmp_path):
    plan = study(tmp_path)
    # Reverse one method's listing: pairing must not depend on row order.
    plan['runs'][5:] = reversed(plan['runs'][5:])
    register(tmp_path, plan)
    state = random.getstate()
    result = report(tmp_path)
    assert random.getstate() == state
    assert result['errors'] == [] and all(r['eligible'] for r in result['runs'])
    aggregate = result['groups'][0]['aggregate']
    assert aggregate['final']['mean'] == 14.
    assert aggregate['final']['sample_sd_across_training_seeds'] == pytest.approx(math.sqrt(2.5))
    assert aggregate['normalized_auc']['mean'] == 7.
    paired = result['paired_comparison']
    assert [p['seed'] for p in paired['pairs']] == plan['training_seeds']
    assert paired['final_return']['mean'] == 4.
    assert paired['final_return']['bootstrap_95pct_mean'] == [4., 4.]
    assert paired['normalized_auc']['mean'] == pytest.approx(2.)
    assert result['validation_decision'] == 'above_reading_bound_with_bootstrap_support'
    assert result['paper_reproduction'] == 'not_established'


@pytest.mark.parametrize('values,decision', [
    ((9., 9., 9., 9., 9.), 'mean_not_above_reading_bound'),
    ((0., 0., 0., 0., 100.), 'point_mean_above_reading_bound'),
])
def test_point_improvement_and_interval_support_are_separate(tmp_path, values, decision):
    study(tmp_path, values)
    result = report(tmp_path)
    assert result['validation_decision'] == decision
    assert result['figure_comparison']['lower_bootstrap_mean_endpoint_above_bound'] is False
    assert result['paper_reproduction'] == 'not_established'


@pytest.mark.parametrize('state', ['not_started', 'unfinished', 'failed'])
def test_missing_or_failed_seed_blocks_group_and_comparison(tmp_path, state):
    plan = study(tmp_path)
    output = tmp_path / plan['runs'][0]['output_path']
    if state == 'not_started':
        shutil.rmtree(output)
    elif state == 'unfinished':
        (output / 'summary.json').unlink()
    else:
        edit(output / 'summary.json', lambda x: x.update(status='failed'))
    result = report(tmp_path)
    assert len(result['runs']) == 10 and result['runs'][0]['status'] == state
    assert result['groups'][0]['aggregate'] is None
    assert result['groups'][1]['aggregate'] is not None
    assert result['paired_comparison'] is None and result['figure_comparison'] is None
    assert result['validation_decision'] == 'incomplete_or_invalid'


@pytest.mark.parametrize('artifact', ['source', 'extra_source', 'manifest', 'launcher', 'plan', 'inventory', 'selection', 'config'])
def test_changed_frozen_input_blocks_all_claims_but_retains_run_table(tmp_path, artifact):
    plan = study(tmp_path)
    paths = {'source': 'runs/source/bta_repro/simulator_train.py', 'extra_source': 'runs/source/bta_repro/extra.py',
             'manifest': plan['manifest'], 'launcher': 'runs/run_simulator_worker.py', 'plan': 'study.json',
             'inventory': 'inventory.json', 'selection': 'selected/curves.json', 'config': plan['runs'][0]['config_path']}
    path = tmp_path / paths[artifact]
    with path.open('a') as stream:
        stream.write(' ')
    result = report(tmp_path)
    assert result['errors'] and len(result['runs']) == 10
    assert not any(r['eligible'] for r in result['runs'])
    assert result['validation_decision'] == 'not_assessed'


@pytest.mark.parametrize('defect', ['software', 'engine', 'source', 'config', 'early', 'training_seed', 'evaluation_seed'])
def test_recorded_metadata_must_match_preregistered_reconstruction(tmp_path, defect):
    plan = study(tmp_path)
    path = tmp_path / plan['runs'][0]['output_path'] / 'config.json'
    changes = {
        'software': lambda x: x['versions'].update(torch='changed'),
        'engine': lambda x: x['environment_versions'].update(engine='changed'),
        'source': lambda x: x['source_sha256'].update(extra='changed'),
        'config': lambda x: x['config'].update(rollout_steps=1),
        'early': lambda x: x.update(created_utc='2025-01-01T00:00:00+00:00'),
        'training_seed': lambda x: x.update(training_environment_seeds=[1, 2]),
        'evaluation_seed': lambda x: x.update(evaluation_environment_seed=1),
    }
    edit(path, changes[defect])
    result = report(tmp_path)
    assert result['runs'][0]['errors'] and not result['runs'][0]['eligible']
    assert result['groups'][0]['aggregate'] is None and result['paired_comparison'] is None


@pytest.mark.parametrize('defect', ['mean', 'std', 'length', 'count', 'index', 'unfinished', 'seed', 'execution', 'grid', 'stage'])
def test_raw_evaluations_and_complete_schedule_are_verified(tmp_path, defect):
    plan = study(tmp_path)
    output = tmp_path / plan['runs'][0]['output_path']
    path = output / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    periodic = next(r for r in rows if r.get('stage') == 'periodic')
    e = periodic['evaluation']
    if defect == 'mean': e['episodes'][0]['return'] += 1
    elif defect == 'std': e['return_std'] = 0
    elif defect == 'length': e['environment_steps'] = 1
    elif defect == 'count': e['episodes'].pop()
    elif defect == 'index': e['episodes'][0]['episode'] = 1
    elif defect == 'unfinished': e['episodes'][0]['truncated'] = False
    elif defect == 'seed':
        for row in rows:
            if row['event'] == 'evaluation': row['evaluation']['seed'] = 0
        edit(output / 'summary.json', lambda x: x['final_evaluation'].update(seed=0))
    elif defect == 'execution':
        for row in rows:
            if row['event'] == 'evaluation': row['evaluation']['execution'] = 'centralized'
        edit(output / 'summary.json', lambda x: x['final_evaluation'].update(execution='centralized'))
    elif defect == 'grid': periodic['steps'] = 3
    else: periodic['stage'] = 'initial'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = report(tmp_path)
    assert result['runs'][0]['errors'] and not result['runs'][0]['eligible']
    assert result['groups'][0]['aggregate'] is None


@pytest.mark.parametrize('defect', ['summary_episodes', 'training_count', 'evaluation_count', 'checkpoint'])
def test_final_artifacts_and_interaction_accounting_agree(tmp_path, defect):
    plan = study(tmp_path)
    output = tmp_path / plan['runs'][0]['output_path']
    if defect == 'checkpoint': (output / 'checkpoint.pt').unlink()
    else:
        changes = {'summary_episodes': lambda x: x['final_evaluation']['episodes'][0].update(length=1),
                   'training_count': lambda x: x.update(train_environment_steps=9),
                   'evaluation_count': lambda x: x.update(evaluation_environment_steps=47)}
        edit(output / 'summary.json', changes[defect])
    result = report(tmp_path)
    assert result['runs'][0]['errors'] and not result['runs'][0]['eligible']
    assert result['figure_comparison'] is None


def test_environment_seed_reuse_is_rejected_even_if_plan_hash_is_registered(tmp_path):
    plan = study(tmp_path)
    plan['training_seeds'][0] = 0
    register(tmp_path, plan)
    result = report(tmp_path)
    assert 'reuses' in result['errors'][0]


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'shape', 'summary'])
def test_training_budget_is_checked_against_every_rollout(tmp_path, defect):
    plan = study(tmp_path)
    output = tmp_path / plan['runs'][0]['output_path']
    path = output / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first = next(i for i, row in enumerate(rows) if row['event'] == 'update')
    if defect == 'missing': rows.pop(first)
    elif defect == 'duplicate': rows.insert(first, dict(rows[first]))
    elif defect == 'order': rows[first]['update'] = 2
    elif defect == 'shape': rows[first]['rollout_shape'] = [1, 4]
    else: edit(output / 'summary.json', lambda x: x.update(updates=2))
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = report(tmp_path)
    assert result['runs'][0]['errors'] and not result['runs'][0]['eligible']
    assert result['figure_comparison'] is None


def test_missing_planned_seed_cannot_be_silently_removed(tmp_path):
    plan = study(tmp_path)
    plan['runs'].pop()
    register(tmp_path, plan)
    assert 'planned run set' in report(tmp_path)['errors'][0]


def test_method_pair_cannot_mix_different_selected_engines(tmp_path):
    plan = study(tmp_path)
    path = tmp_path / 'selected/curves.json'
    edit(path, lambda x: x['runs'][1]['environment_versions'].update(engine='different'))
    plan['selection_evidence']['curves_sha256'] = digest(path)
    register(tmp_path, plan)
    result = report(tmp_path)
    assert 'different native environment versions' in result['errors'][0]
    assert result['paired_comparison'] is None


def test_unfinished_partial_line_is_retained_without_becoming_final(tmp_path):
    plan = study(tmp_path)
    output = tmp_path / plan['runs'][0]['output_path']
    (output / 'summary.json').unlink()
    with (output / 'metrics.jsonl').open('a') as stream:
        stream.write('{"event":')
    result = report(tmp_path)
    assert result['runs'][0]['ignored_unfinished_tail']
    assert not result['runs'][0]['errors'] and not result['runs'][0]['eligible']


def test_metric_symlink_cannot_read_another_checkout(tmp_path):
    root = tmp_path / 'checkout'
    plan = study(root)
    path = root / plan['runs'][0]['output_path'] / 'metrics.jsonl'
    external = tmp_path / 'outside.jsonl'
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    result = report(root)
    assert 'leaves the reproduction' in result['runs'][0]['errors'][0]


def test_evaluation_grid_rounds_to_real_rollouts_and_handles_disabled_cadence():
    assert expected_grid({'steps': 10, 'num_envs': 2, 'rollout_steps': 2, 'eval_every_steps': 3}) == [0, 4, 8, 10]
    assert expected_grid({'steps': 10, 'num_envs': 2, 'rollout_steps': 2, 'eval_every_steps': 1}) == [0, 4, 8, 10]
    assert expected_grid({'steps': 10, 'num_envs': 2, 'rollout_steps': 2, 'eval_every_steps': 0}) == [0, 10]
