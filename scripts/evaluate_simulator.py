"""Evaluate a saved simulator policy with its recorded preprocessing and protocol."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bta_repro import simulator_train as training
from bta_repro.normalization import ObservationNormalizer


def evaluate_saved(run, *, eval_seed=None, episodes=None, device=None, allow_source_change=False):
    run = Path(run)
    metadata = json.loads((run / 'config.json').read_text())
    checkpoint_path = run / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint version')
    if checkpoint['config'] != metadata['config']:
        raise ValueError('Checkpoint and run metadata configurations differ')
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(Path(training.__file__).parent.glob('*.py'))}
    source_matches = source_hashes == metadata['source_sha256']
    if not source_matches and not allow_source_change:
        raise ValueError('Source differs from training; use its frozen source via PYTHONPATH, '
                         'or explicitly pass --allow-source-change for a labeled compatibility check')
    arguments = dict(checkpoint['config'])
    for name, value in [('eval_seed', eval_seed), ('eval_episodes', episodes), ('device', device)]:
        if value is not None:
            arguments[name] = value
    config = training.Config(**arguments)
    config.validate()
    torch.set_num_threads(config.threads)
    spec = metadata['environment_spec']
    dimensions = SimpleNamespace(**spec, num_actions=spec['action_dim'])
    # Model construction must not consume the caller's training RNG stream.
    with training.evaluation_rng(config.eval_seed):
        policy = training.make_policy(config, dimensions).to(config.device)
        policy.load_state_dict(checkpoint['policy'], strict=True)
    normalizer = None
    if config.normalize_observations:
        state = checkpoint.get('extra_state', {}).get('observation_normalizer')
        if state is None:
            raise ValueError('Normalized policy checkpoint is missing observation moments')
        normalizer = ObservationNormalizer(spec['num_agents'], spec['obs_dim'], spec['state_dim'],
                                            clip=config.observation_clip)
        normalizer.load_state_dict(state)
    result = training.evaluate(policy, config, spec, normalizer)
    versions = training._versions()
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        'scope': 'Saved-policy evaluation; new evaluation episodes are not independent training seeds.',
        'run': str(run.resolve()), 'checkpoint_step': checkpoint['step'],
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_run_status': summary.get('status', 'incomplete'),
        'checkpoint_is_final': summary.get('status') == 'completed' and summary.get('steps') == checkpoint['step'],
        'source_matches_training': source_matches, 'source_sha256': source_hashes,
        'versions': versions, 'versions_match_training': versions == metadata['versions'],
        'config': asdict(config), 'evaluation': result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='directory containing config.json and checkpoint.pt')
    parser.add_argument('--output', type=Path, required=True, help='new evaluation JSON file')
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--device')
    parser.add_argument('--allow-source-change', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    result = evaluate_saved(args.run, eval_seed=args.eval_seed, episodes=args.episodes,
                            device=args.device, allow_source_change=args.allow_source_change)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(training._json(result, indent=2) + '\n')
    evaluation = result['evaluation']
    print(json.dumps({'output': str(args.output), 'checkpoint_step': result['checkpoint_step'],
                      'source_matches_training': result['source_matches_training'],
                      'return_mean': evaluation['return_mean'], 'win_rate': evaluation['win_rate'],
                      'capture_rate': evaluation['capture_rate']}))


if __name__ == '__main__':
    main()
