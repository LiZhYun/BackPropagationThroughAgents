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

from bta.envs.predator_prey.predatorprey_wrapper import PredatorPreyWrapper
from bta.envs.env_wrappers import ShareSubprocVecEnv_Mujoco, ShareDummyVecEnv
from datetime import datetime

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "predator_prey":
                env = PredatorPreyWrapper(num_agents=all_args.num_agents, n_preys=all_args.num_preys, penalty=all_args.penalty)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv_Mujoco([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "predator_prey":
                env = PredatorPreyWrapper(num_agents=all_args.num_agents, n_preys=all_args.num_preys, penalty=all_args.penalty)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv_Mujoco([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=2,
                        help="number of controlled players.")
    parser.add_argument("--num_preys", type=int, default=1,
                        help="number of preys.")
    parser.add_argument("--penalty", type=float, default=-0.5,
                        help="by default True. If False, sample action according to probability")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    
    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = False
    
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
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                        #  group=all_args.penalty,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         tags=["iclr24"],
                         )
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

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        # all_args.penalty, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
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
        from bta.runner.gcs.predator_runner import PredatorRunner as Runner
    elif "ar" in all_args.algorithm_name:
        from bta.runner.ar.predator_runner import PredatorRunner as Runner
    elif "ha" in all_args.algorithm_name:
        from bta.runner.happo.predator_runner import PredatorRunner as Runner
    elif "temporal" in all_args.algorithm_name:
        from bta.runner.temporal.predator_runner import PredatorRunner as Runner
    elif "mat" in all_args.algorithm_name:
        from bta.runner.mat.predator_runner import PredatorRunner as Runner
    elif "maven" in all_args.algorithm_name:
        from bta.runner.maven.predator_runner import PredatorRunner as Runner
    elif "macpf" in all_args.algorithm_name:
        from bta.runner.macpf.predator_runner import PredatorRunner as Runner
    elif "mappo" in all_args.algorithm_name:
        from bta.runner.mappo.predator_runner import PredatorRunner as Runner
    else:
        raise NotImplementedError

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
