"""Small real-engine check; no training or performance claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from bta_repro.smac_env import SmacEnv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scenario', default='terran_5v5')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--restart-game-loop', type=int, default=480000)
parser.add_argument('--require-restarts', type=int, default=0)
args = parser.parse_args()
if args.steps < 1 or args.require_restarts < 0:
    parser.error('steps must be positive and require-restarts must be nonnegative')
rng = np.random.default_rng(1234)
env = SmacEnv({'scenario': args.scenario, 'seed': 1234,
               'restart_game_loop': args.restart_game_loop})
records = []
restarts = []
try:
    observation, info = env.reset()
    for step in range(args.steps):
        assert np.isfinite(observation['obs']).all()
        assert np.isfinite(observation['state']).all()
        action = np.array([rng.choice(np.flatnonzero(mask)) for mask in observation['available_actions']])
        observation, reward, terminated, truncated, info = env.step(action)
        records.append({'step':step+1, 'reward':float(reward), 'terminated':terminated,
                        'truncated':truncated, 'native_game_loop':info['native_game_loop'],
                        'planned_restarts':info['planned_restarts']})
        if terminated or truncated:
            observation, info = env.reset()
            if info['planned_restart']:
                restarts.append({'after_step':step+1, **info})
    if env.planned_restarts < args.require_restarts:
        raise AssertionError(f'Observed {env.planned_restarts} restarts; required {args.require_restarts}')
    report = {'status':'passed', 'scope':f'{args.steps}-step simulator smoke only', 'scenario':args.scenario,
              'restart_game_loop':args.restart_game_loop, 'planned_restarts':env.planned_restarts,
              'restart_events':restarts,
              'num_agents':env.num_agents, 'obs_dim':env.obs_dim, 'state_dim':env.state_dim,
              'num_actions':env.num_actions, 'transitions':records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('transitions', 'restart_events')}))
finally:
    env.close()
