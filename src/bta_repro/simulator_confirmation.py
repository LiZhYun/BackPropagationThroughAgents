"""Audit frozen BPPO/AR-PPO episodic-return confirmation studies without engines.

This checks the declared reconstruction, not equivalence to historical paper
logs. Every planned run remains visible, including missing and invalid runs.
"""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from .learning_curves import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, read_curve_run, seed_statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, value):
    path = (root / value).resolve()
    require(path.is_relative_to(root), "artifact path leaves the reproduction repository")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid integer {name}")
    return value


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamps must include a timezone")
    return result


def expected_grid(config):
    budget = integer(config['steps'], 'budget', 1)
    batch = integer(config['num_envs'], 'num_envs', 1) * integer(config['rollout_steps'], 'rollout_steps', 1)
    cadence = integer(config['eval_every_steps'], 'evaluation cadence')
    grid, target = [0], cadence
    while cadence and target < budget:
        step = ((target + batch - 1) // batch) * batch
        if step >= budget:
            break
        grid.append(step)
        target = (step // cadence + 1) * cadence
    return grid + [budget]


def check_episode_evaluation(evaluation, config):
    """Recompute the return reduction and interaction count from complete episodes."""
    count = integer(config['eval_episodes'], 'evaluation episode count', 1)
    require(type(evaluation['count']) is int and evaluation['count'] == count, 'evaluation episode count changed')
    require(type(evaluation['seed']) is int and evaluation['seed'] == config['eval_seed'], 'evaluation seed changed')
    require(evaluation['deterministic'] is True and evaluation['execution'] == 'autoregressive',
            'execution must use deterministic autoregressive actors')
    episodes = evaluation['episodes']
    require(len(episodes) == count, 'raw evaluation episode count disagrees')
    returns, lengths = [], []
    for index, episode in enumerate(episodes):
        require(integer(episode['episode'], 'episode index') == index, 'episode indices are missing or duplicated')
        returns.append(number(episode['return'], 'episode return'))
        length = integer(episode['length'], 'episode length', 1)
        require(length <= config['env_config']['episode_limit'], 'evaluation episode exceeds configured horizon')
        require(type(episode['terminated']) is bool and type(episode['truncated']) is bool
                and (episode['terminated'] or episode['truncated']), 'unfinished evaluation episode')
        lengths.append(length)
    steps = sum(lengths)
    require(integer(evaluation['environment_steps'], 'evaluation interactions') == steps,
            'evaluation interaction count disagrees with episode lengths')
    for key in ('terminated', 'truncated'):
        require(integer(evaluation[key], key) == sum(e[key] for e in episodes), f'{key} count disagrees')
    expected = {'return_mean': statistics.mean(returns),
                'return_std': statistics.stdev(returns) if count > 1 else 0.,
                'length_mean': statistics.mean(lengths),
                'reward_per_environment_step': math.fsum(returns) / steps}
    for key, value in expected.items():
        require(math.isclose(number(evaluation[key], key), value, rel_tol=1e-9, abs_tol=1e-8),
                f'{key} disagrees with raw evaluation episodes')
    require(evaluation['reward_per_step_scope'] == 'completed_episodes_only', 'episode reward scope changed')
    return steps


def check_training_interactions(rows, config, summary):
    """Verify the budget against every recorded synchronous rollout update."""
    budget = integer(config['steps'], 'budget', 1)
    environments = integer(config['num_envs'], 'training environments', 1)
    rollout = integer(config['rollout_steps'], 'rollout steps', 1)
    require(budget % environments == 0, 'budget is not divisible by training environments')
    batch = environments * rollout
    updates = [row for row in rows if row.get('event') == 'update']
    require(len(updates) == (budget + batch - 1) // batch, 'missing or duplicated rollout update')
    for index, row in enumerate(updates):
        expected_steps = min((index + 1) * batch, budget)
        expected_length = min(rollout, (budget - index * batch) // environments)
        require(integer(row['steps'], 'update steps') == expected_steps
                and integer(row['update'], 'update number', 1) == index + 1, 'rollout update order or budget changed')
        require(canonical(row['rollout_shape']) == canonical([expected_length, environments]), 'rollout interaction shape changed')
    require(integer(summary['updates'], 'summary updates') == len(updates), 'summary update count disagrees')
    require(integer(summary['steps'], 'completed budget') == budget
            and integer(summary['train_environment_steps'], 'training interactions') == budget,
            'training interaction accounting mismatch')


def check_plan(root, path, plan):
    """Check the pre-submission record, complete run set and immutable inputs."""
    require(BOOTSTRAP_SAMPLES == 10000 and BOOTSTRAP_SEED == 1729, 'bootstrap implementation differs from frozen study rule')
    events = [json.loads(line) for line in (root / 'experiments/jobs.jsonl').read_text().splitlines()]
    registered = [e for e in events if e.get('plan') == str(path.relative_to(root))]
    require(bool(registered), 'study has no recorded submission')
    require(all(e['plan_sha256'] == digest(path) for e in registered), 'plan differs from submission hash')
    frozen = timestamp(plan['frozen_utc'])
    require(all(timestamp(e['recorded_utc']) >= frozen for e in registered), 'submission predates plan freeze')
    manifest_path = checked_path(root, plan['manifest'])
    require(digest(manifest_path) == plan['manifest_sha256'], 'frozen manifest changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['source_sha256'] == plan['source_sha256'], 'manifest source hashes changed')
    source = checked_path(root, manifest_path.parent / 'source/bta_repro')
    require({p.name for p in source.glob('*.py')} == set(plan['source_sha256']), 'frozen source file set changed')
    for name, expected in plan['source_sha256'].items():
        require(Path(name).name == name and name.endswith('.py'), 'invalid source filename')
        require(digest(checked_path(root, source / name)) == expected, f'frozen source changed: {name}')
    for name, expected in plan['launchers'].items():
        require(Path(name).name == name, 'invalid launcher filename')
        require(digest(checked_path(root, manifest_path.parent / name)) == expected, f'launcher changed: {name}')
    inventory_path = checked_path(root, plan['prior_inventory'])
    require(digest(inventory_path) == plan['prior_inventory_sha256'], 'prior seed inventory changed')
    inventory = json.loads(inventory_path.read_text())
    require(len(inventory['runs']) == plan['prior_inventory_run_count'], 'prior inventory run count changed')
    require(timestamp(inventory['recorded_utc']) <= frozen, 'seed inventory postdates freeze')
    seeds = plan['training_seeds']
    require(len(seeds) >= 5 and len(set(seeds)) == len(seeds), 'confirmation requires at least five unique seeds')
    for seed in seeds:
        require(integer(seed, 'training seed') < 2**32, 'training seed exceeds supported range')
    for field in ('budget', 'episodes', 'episode_limit'):
        integer(plan['evaluation'][field], field, 1)
    require(integer(plan['evaluation']['seed'], 'evaluation seed') < 2**32, 'evaluation seed exceeds supported range')
    prior = {r['seed'] for r in inventory['runs'] if r['seed'] is not None}
    require(not set(seeds) & prior, 'confirmation reuses a previously prepared training seed')
    require(sorted(prior) == plan['prior_recorded_training_seeds'], 'prior seed summary disagrees with inventory')
    require(plan['evaluation']['metric'] == 'final_evaluation.return_mean', 'unsupported confirmation metric')
    require(plan['evaluation']['seed'] not in {r.get('eval_seed') for r in inventory['runs']},
            'confirmation reuses a previously recorded evaluation stream')

    selection = plan['selection_evidence']
    selected_path = checked_path(root, selection['manifest'])
    require(digest(selected_path) == selection['manifest_sha256'], 'selected development manifest changed')
    selected = json.loads(selected_path.read_text())
    require(selected['source_sha256'] == plan['source_sha256'], 'source differs from selected development source')
    curves_path = checked_path(root, selection['validated_curves'])
    require(digest(curves_path) == selection['curves_sha256'], 'selected development evidence changed')
    selected_curves = json.loads(curves_path.read_text())
    require(selected_curves['manifest_sha256'] == selection['manifest_sha256'], 'selection report manifest mismatch')
    environment_versions = {}
    for algorithm in ('bppo', 'arppo'):
        references = [r for r in selected_curves['runs'] if r['algorithm'] == algorithm and r['eligible']]
        require(bool(references), f'no valid development environment reference for {algorithm}')
        require(len({canonical(r['environment_versions']) for r in references}) == 1,
                'selected environment versions disagree')
        environment_versions[algorithm] = references[0]['environment_versions']
    require(canonical(environment_versions['bppo']) == canonical(environment_versions['arppo']),
            'matched methods use different native environment versions')

    key = lambda r: (r['algorithm'], r['seed'])
    entries = plan['runs']
    expected_keys = {(a, s) for a in ('bppo', 'arppo') for s in seeds}
    require(len(entries) == len(expected_keys) and {key(r) for r in entries} == expected_keys,
            'planned run set is missing, duplicated or unsupported')
    counterparts = {key(r): r for r in manifest['runs']}
    require(len(manifest['runs']) == len(expected_keys) and set(counterparts) == expected_keys,
            'manifest run set differs from frozen study')
    outputs, configurations = set(), set()
    for entry in entries:
        counterpart = counterparts[key(entry)]
        require(entry['variant'] == counterpart['variant'], 'manifest variant mismatch')
        for field in ('config_path', 'output_path'):
            require(checked_path(root, entry[field]) == checked_path(root, counterpart[field]), 'manifest path mismatch')
        output = checked_path(root, entry['output_path'])
        require(output not in outputs, 'duplicate output directory')
        outputs.add(output)
        cp = checked_path(root, entry['config_path'])
        require(digest(cp) == entry['config_sha256'], 'frozen run configuration changed')
        config = entry['resolved_config']
        require(canonical(json.loads(cp.read_text())) == canonical(config), 'full resolved configuration mismatch')
        require(config['algorithm'] == entry['algorithm'] and type(config['seed']) is int
                and config['seed'] == entry['seed'], 'run identity disagrees with configuration')
        require(config['steps'] == plan['evaluation']['budget'] and config['eval_seed'] == plan['evaluation']['seed']
                and config['eval_episodes'] == plan['evaluation']['episodes'], 'declared evaluation protocol mismatch')
        require(config['env_config']['episode_limit'] == plan['evaluation']['episode_limit'], 'episode horizon mismatch')
        configurations.add(canonical({k: v for k, v in config.items() if k not in ('algorithm', 'seed')}))
    require(len(configurations) == 1, 'methods or training seeds differ in matched training settings')
    streams = [plan['training_environment_seeds'][str(s)] for s in seeds]
    flat = [n for stream in streams for n in stream]
    require(all(len(stream) == entries[0]['resolved_config']['num_envs'] for stream in streams),
            'planned training environment stream count changed')
    require(all(integer(n, 'training environment seed') < 2**32 for n in flat), 'environment seed exceeds supported range')
    require(len(flat) == len(set(flat)) and plan['evaluation']['seed'] not in flat, 'environment seed streams collide')
    return environment_versions, min(timestamp(e['recorded_utc']) for e in registered)


def read_confirmation_run(root, entry, plan, environment_versions, submitted):
    record = {k: entry[k] for k in ('algorithm', 'variant', 'seed', 'output_path')}
    record.update(status='not_started', eligible=False, errors=[])
    try:
        output = checked_path(root, entry['output_path'])
        if not output.exists():
            return record
        record['status'] = 'unfinished'
        for name in ('config.json', 'summary.json', 'metrics.jsonl', 'checkpoint.pt'):
            checked_path(root, output / name)
        if not (output / 'config.json').exists():
            return record
        metadata = json.loads((output / 'config.json').read_text())
        config = entry['resolved_config']
        require(canonical(metadata['config']) == canonical(config), 'recorded configuration differs from frozen study')
        require(metadata['source_sha256'] == plan['source_sha256'], 'recorded source differs from frozen study')
        require(metadata['versions'] == plan['versions'], 'recorded software versions differ from selected versions')
        require(metadata['environment_versions'] == environment_versions[entry['algorithm']], 'native environment versions changed')
        require(timestamp(metadata['created_utc']) >= submitted, 'run predates frozen study submission')
        require(metadata['evaluation_environment_seed'] == config['eval_seed'], 'recorded evaluation stream changed')
        require(metadata['training_environment_seeds'] == plan['training_environment_seeds'][str(entry['seed'])]
                and len(metadata['training_environment_seeds']) == config['num_envs'], 'recorded training streams changed')
        absolute = {**entry, 'output_path': str(output), 'config_path': str(checked_path(root, entry['config_path']))}
        record.update(read_curve_run(absolute, plan['source_sha256'], 'return_mean'))
        record['output_path'] = entry['output_path']
        record['metadata_sha256'] = digest(output / 'config.json')
        if not record['eligible']:
            return record
        require([p['steps'] for p in record['points']] == expected_grid(config), 'predeclared evaluation grid changed')
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        evaluations = [r for r in rows if r.get('event') == 'evaluation']
        require([r['stage'] for r in evaluations] == ['initial'] + ['periodic'] * (len(evaluations) - 2) + ['final'],
                'evaluation stages disagree with schedule')
        steps = sum(check_episode_evaluation(r['evaluation'], config) for r in evaluations)
        summary = json.loads((output / 'summary.json').read_text())
        require(canonical(summary['initial_evaluation']) == canonical(evaluations[0]['evaluation'])
                and canonical(summary['final_evaluation']) == canonical(evaluations[-1]['evaluation']),
                'summary and curve episode records disagree')
        check_training_interactions(rows, config, summary)
        require(integer(summary['evaluation_environment_steps'], 'all evaluation interactions') == steps,
                'total evaluation interaction accounting mismatch')
        checkpoint = output / 'checkpoint.pt'
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, 'completed run lacks a checkpoint')
        record['checkpoint_sha256'] = digest(checkpoint)
        record['final_return'] = summary['final_evaluation']['return_mean']
        record['normalized_auc'] = record['full_budget_auc']['normalized_by_training_steps']
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        record['eligible'] = False
        record['errors'].append(str(error))
    return record


def report_confirmation(plan_path, root):
    root = Path(root).resolve()
    path = checked_path(root, plan_path)
    plan = json.loads(path.read_text())
    report = {'study_id': plan['study_id'], 'scope': plan['scope'], 'plan_sha256': digest(path),
              'reporter_sha256': digest(Path(__file__)), 'curve_reader_sha256': digest(Path(__file__).with_name('learning_curves.py')),
              'errors': [], 'runs': [{**{k: r[k] for k in ('algorithm', 'variant', 'seed', 'output_path')},
                                     'status': 'unverified', 'eligible': False, 'errors': []} for r in plan['runs']],
              'groups': [], 'paired_comparison': None, 'figure_comparison': None,
              'validation_decision': 'not_assessed', 'paper_reproduction': 'not_established',
              'uncertainty': {'unit': 'whole training seed; paired differences retain the same seed pairs',
                              'method': 'two-sided 95% percentile bootstrap with linear percentile interpolation',
                              'resamples': BOOTSTRAP_SAMPLES, 'rng_seed': BOOTSTRAP_SEED},
              'interpretation': 'Intervals describe independent training-seed scores under the frozen reconstruction. '
                                'Historical evaluation, smoothing and simulator equivalence remain unverified. '
                                'Figure comparison and paired peer-feedback evidence are separate conclusions.'}
    try:
        versions, submitted = check_plan(root, path, plan)
        report['runs'] = [read_confirmation_run(root, entry, plan, versions, submitted) for entry in plan['runs']]
        for algorithm in ('bppo', 'arppo'):
            runs = [r for r in report['runs'] if r['algorithm'] == algorithm]
            complete = all(r['eligible'] for r in runs)
            group = {'algorithm': algorithm, 'declared_seeds': plan['training_seeds'],
                     'eligible_seeds': [r['seed'] for r in runs if r['eligible']], 'complete': complete, 'aggregate': None}
            if complete:
                group['aggregate'] = {metric: seed_statistics([r[key] for r in runs], confidence_intervals=True)
                                      for metric, key in [('final', 'final_return'), ('normalized_auc', 'normalized_auc')]}
            report['groups'].append(group)
        if not all(g['complete'] for g in report['groups']):
            report['validation_decision'] = 'incomplete_or_invalid'
            return report
        by_key = {(r['algorithm'], r['seed']): r for r in report['runs']}
        pairs = [{'seed': seed, **{key: by_key['bppo', seed][key] - by_key['arppo', seed][key]
                                 for key in ('final_return', 'normalized_auc')}} for seed in plan['training_seeds']]
        report['paired_comparison'] = {'direction': 'BPPO minus AR-PPO', 'pairs': pairs,
                                      **{key: seed_statistics([p[key] for p in pairs], confidence_intervals=True)
                                         for key in ('final_return', 'normalized_auc')}}
        final = report['groups'][0]['aggregate']['final']
        bounds = plan['paper_reference']['reading_range']
        require(len(bounds) == 2 and number(bounds[0], 'lower reading bound') <= number(bounds[1], 'upper reading bound'),
                'invalid paper reading range')
        point = final['mean'] > bounds[1]
        supported = final['bootstrap_95pct_mean'][0] > bounds[1]
        report['figure_comparison'] = {'reading_range': bounds, 'seed_mean': final['mean'],
                                      'mean_above_upper_reading_bound': point,
                                      'lower_bootstrap_mean_endpoint_above_bound': supported,
                                      'mean_margin_over_upper_bound': final['mean'] - bounds[1]}
        report['validation_decision'] = ('above_reading_bound_with_bootstrap_support' if supported else
                                         'point_mean_above_reading_bound' if point else 'mean_not_above_reading_bound')
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        report['errors'].append(str(error))
        report['validation_decision'] = 'not_assessed'
    return report
