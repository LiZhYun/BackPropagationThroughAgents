#!/usr/bin/env python3
"""Check real Football matches and exact native-seed replay, without learning."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from bta_repro.football_env import FootballEnv, SCENARIOS


def trace(env, *, reset_seed=None):
    observation, reset_info = env.reset(seed=reset_seed)
    digest = hashlib.sha256(observation['obs'].tobytes())
    actions = np.random.default_rng(7291)
    total = 0.0
    for step in range(1, env.native_horizon + 2):
        observation, reward, terminated, truncated, info = env.step(
            actions.integers(0, env.num_actions, size=env.num_agents))
        digest.update(observation['obs'].tobytes())
        digest.update(np.asarray([reward, terminated, truncated], dtype=np.float64).tobytes())
        total += reward
        if terminated or truncated:
            assert terminated and not truncated, 'native match must terminate'
            # Engine game time starts at -1. The 3000-tick 5v5 match takes
            # 3001 Python step calls; preserve the native ending unchanged.
            assert step <= env.native_horizon + 1
            return {'sha256': digest.hexdigest(), 'steps': step, 'return': total,
                    'score': info['score'], 'won': info['match_won'],
                    'engine_seed': reset_info['engine_seed']}
    raise AssertionError('native match exceeded its declared horizon')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    report = {'purpose': 'native simulator validation, not a learning result',
              'versions': {name: version(name) for name in ('gfootball', 'numpy', 'gym', 'torch')},
              'scenarios': {}}
    for scenario in SCENARIOS:
        env = FootballEnv({'scenario': scenario, 'seed': 37})
        try:
            first = trace(env)
            replay = trace(env, reset_seed=37)
            assert first == replay, f'{scenario}: reseeded replay differs'
        finally:
            env.close()
        env = FootballEnv({'scenario': scenario, 'seed': 37})
        try:
            fresh = trace(env)
            assert first == fresh, f'{scenario}: fresh-instance replay differs'
        finally:
            env.close()
        report['scenarios'][scenario] = {**first, 'reseed_replay': True, 'fresh_instance_replay': True}
        print(json.dumps({'scenario': scenario, **report['scenarios'][scenario]}), flush=True)
    report.update(status='passed', elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
