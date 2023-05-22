#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from bta.config import get_config

from bta.envs.overcooked.Overcooked_Env import Overcooked
from bta.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from bta.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, featurize_type=("ppo",)*all_args.num_agents)
                else:
                    env = Overcooked_new(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, featurize_type=("ppo",)*all_args.num_agents)
                else:
                    env = Overcooked_new(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--layout_name", type=str, default='random1', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true') 
    parser.add_argument("--random_index", default=False, action='store_true') 
    parser.add_argument("--w0", type=str, default="1,1,1,1", help="Weight vector of dense reward 0 in overcooked env.")
    parser.add_argument("--w1", type=str, default="1,1,1,1", help="Weight vector of dense reward 1 in overcooked env.") 
    parser.add_argument("--predict_other_shaped_info", default=False, action='store_true', help="Predict other agent's shaped info within a short horizon, default False")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int, help="Horizon for shaped info target, default 50")
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int, help="Event count for shaped info target, default 10")
    parser.add_argument("--use_task_v_out", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        # # sweep
        # sweep_config = {
        #     'method': 'bayes',
        #     'metric': {
        #     'name': 'eval_ep_sparse_r',
        #     'goal': 'maximize'   
        #     }
        # }
        # # 参数范围
        # parameters_dict = {
        #     'threshold': {
        #         # a flat distribution between 0 and 1.0
        #         'distribution': 'uniform',
        #         'min': 0,
        #         'max': 1.0
        #     }
        # }

        # sweep_config['parameters'] = parameters_dict
        # sweep_id = wandb.sweep(sweep_config, project=all_args.env_name + '_' + all_args.layout_name + '_sweep')
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.layout_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         tags=all_args.wandb_tags)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents
    all_args.device = device
    if "gcs" in all_args.algorithm_name:
        all_args.n_xdims = 144
        all_args.nhead = 1
        all_args.gat_nhead = 2
        all_args.decoder_hidden_dim = 64
        all_args.node_num = num_agents
        all_args.act_graph = True

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments

    if "gcs" in all_args.algorithm_name:
        from bta.runner.gcs.overcooked_runner import OvercookedRunner as Runner
    elif "ar" in all_args.algorithm_name:
        from bta.runner.ar.overcooked_runner import OvercookedRunner as Runner
    elif "happo" in all_args.algorithm_name:
        from bta.runner.happo.overcooked_runner import OvercookedRunner as Runner
    elif "temporal" in all_args.algorithm_name:
        from bta.runner.temporal.overcooked_runner import OvercookedRunner as Runner
    else: # mappo
        from bta.runner.mappo.overcooked_runner import OvercookedRunner as Runner

    # # sweep
    # def train(wconfig=None):
    #     with wandb.init(config=wconfig,project=all_args.env_name + '_' + all_args.layout_name + '_sweep',entity=all_args.wandb_name,name=str(all_args.algorithm_name) + "_" +
    #                         str(all_args.experiment_name) +
    #                         "_seed" + str(all_args.seed),group=all_args.scenario_name,dir=str(run_dir),):
    #         config['all_args'].threshold = wandb.config.threshold
    #         runner = Runner(config)
    #         runner.run()

    # wandb.agent(sweep_id, train, count=30)
    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])