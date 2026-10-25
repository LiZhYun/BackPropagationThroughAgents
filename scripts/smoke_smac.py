"""Small real-engine check; no training or performance claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from bta_repro.smac_env import SmacEnv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scenario', default='terran_5v5')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
rng = np.random.default_rng(1234)
env = SmacEnv({'scenario': args.scenario, 'seed': 1234})
records = []
try:
    observation, info = env.reset()
    for step in range(32):
        assert np.isfinite(observation['obs']).all()
        assert np.isfinite(observation['state']).all()
        action = np.array([rng.choice(np.flatnonzero(mask)) for mask in observation['available_actions']])
        observation, reward, terminated, truncated, info = env.step(action)
        records.append({'step':step+1, 'reward':float(reward), 'terminated':terminated, 'truncated':truncated})
        if terminated or truncated:
            observation, info = env.reset()
    report = {'status':'passed', 'scope':'32-step simulator smoke only', 'scenario':args.scenario,
              'num_agents':env.num_agents, 'obs_dim':env.obs_dim, 'state_dim':env.state_dim,
              'num_actions':env.num_actions, 'transitions':records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='transitions'}))
finally:
    env.close()
