#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

import torch
from easydict import EasyDict

from bta.config import get_config

from bta.envs.gobigger.gobigger_env import GoBiggerEnv
from bta.envs.env_wrappers import GoBiggerSubprocVecEnv, DummyVecEnv

def make_train_env(all_args, env_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GoBigger":
                env = GoBiggerEnv(env_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return GoBiggerSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args, env_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GoBigger":
                env = GoBiggerEnv(env_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return GoBiggerSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="t4p3", 
                        help="which scenario to run on.")
    parser.add_argument("--team_num", type=int,
                        default=2, 
                        help="team numbers")
    parser.add_argument("--player_num_per_team", type=int, default=2,
                        help="number of players per team.")
    parser.add_argument("--max_ball_num", type=int, default=64,
                        help="max_ball_num.")
    parser.add_argument("--max_food_num", type=int, default=256,
                        help="max_food_num.")
    parser.add_argument("--max_spore_num", type=int, default=32,
                        help="max_spore_num.")
    parser.add_argument("--direction_num", type=int, default=12,
                        help="direction_num.")
    parser.add_argument("--step_mul", type=int, default=8,
                        help="step_mul.")
    parser.add_argument("--second_per_frame", type=float, default=0.05,
                        help="second_per_frame.")
    parser.add_argument("--frame_limit", type=int, default=10*60*20,
                        help="frame_limit: 10*60*20.")
    parser.add_argument("--map_width", type=int, default=64,
                        help="map width.")
    parser.add_argument("--map_height", type=int, default=64,
                        help="map height.")
    parser.add_argument("--match_time", type=int, default=1200,
                        help="match time.")
    parser.add_argument("--spatial", action="store_false", 
                        default=True, 
                        help="by default true.")
    parser.add_argument("--train", action="store_false", 
                        default=True, 
                        help="by default True.")
    parser.add_argument("--speed", action="store_false", 
                        default=True, 
                        help="by default True.")
    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
                        
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
                   0] + "/results") / all_args.env_name / ("t"+str(all_args.team_num)) / ("p"+str(all_args.player_num_per_team)) / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        # # sweep
        # sweep_config = {
        #     'method': 'bayes',
        #     'metric': {
        #     'name': 'average_episode_rewards',
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
        # sweep_id = wandb.sweep(sweep_config, project=all_args.env_name + '_' + all_args.scenario_name + '_sweep')
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
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

    gif_dir = str(run_dir / 'gifs')
    if all_args.use_render:
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    env_config = EasyDict(
        player_num_per_team=all_args.player_num_per_team,
        team_num=all_args.team_num,
        max_ball_num=all_args.max_ball_num,
        max_food_num=all_args.max_food_num,
        max_spore_num=all_args.max_spore_num,
        direction_num=all_args.direction_num,
        step_mul=all_args.step_mul,
        second_per_frame=all_args.second_per_frame,
        frame_limit=all_args.frame_limit,
        match_time=all_args.match_time,
        map_height=all_args.map_height,
        map_width=all_args.map_width,
        spatial=all_args.spatial,
        train=all_args.train,
        speed=all_args.speed,
        use_render=all_args.use_render,
        device=torch.device("cpu"),
        obs_settings=dict(
        obs_type='all', # ['partial', 'all']
        ),
        playback_settings=dict(
        playback_type='by_frame' if all_args.use_render else 'none',
        by_frame=dict(
            save_frame=True,
            save_dir=gif_dir if all_args.use_render else None,
            save_name_prefix='test',
        ),)
    )
    envs = make_train_env(all_args, env_config)
    eval_envs = make_eval_env(all_args, env_config) if all_args.use_eval else None
    num_agents = all_args.player_num_per_team
    all_args.num_agents = num_agents
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
        from bta.runner.gcs.gobigger_runner import GoBiggerRunner as Runner
    elif "ar" in all_args.algorithm_name:
        from bta.runner.ar.gobigger_runner import GoBiggerRunner as Runner
    elif "happo" in all_args.algorithm_name:
        from bta.runner.happo.gobigger_runner import GoBiggerRunner as Runner
    elif "temporal" in all_args.algorithm_name:
        from bta.runner.temporal.gobigger_runner import GoBiggerRunner as Runner
    else: # mappo
        from bta.runner.mappo.gobigger_runner import GoBiggerRunner as Runner

    # # sweep
    # def train(wconfig=None):
    #     with wandb.init(config=wconfig,project=all_args.env_name + '_' + all_args.scenario_name + '_sweep',entity=all_args.wandb_name,name=str(all_args.algorithm_name) + "_" +
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
    # torch.multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])