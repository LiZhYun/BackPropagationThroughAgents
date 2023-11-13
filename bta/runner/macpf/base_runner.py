    
import time
import os
import numpy as np
from itertools import chain
import wandb
import torch
from tensorboardX import SummaryWriter
from bta.utils.shared_buffer_macpf import SharedReplayBuffer
from bta.utils.util import update_linear_schedule, flatten, generate_mask_from_order

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # if self.use_render:
        self.run_dir = config["run_dir"]
        self.gif_dir = str(self.run_dir / 'gifs')
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        # else:
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


        if self.algorithm_name == "macpf":
            from bta.algorithms.macpf.macpf_trainer import MacpfTrainer as TrainAlgo
            from bta.algorithms.macpf.algorithm.macpf_policy import MacpfPolicy as Policy
        else:
            raise NotImplementedError
        
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            self.action_dim = self.envs.action_space[0].n
            self.action_shape = 1
        elif self.envs.action_space[0].__class__.__name__ == "Box":
            self.action_dim = self.envs.action_space[0].shape[0]
            self.action_shape = self.action_dim
        else:
            self.mix_actions = True
            self.continous_dim = self.envs.action_space[0][0].shape[0]
            self.discrete_dim = self.envs.action_space[0][1].n
            self.action_dim = self.continous_dim + self.discrete_dim
            self.action_shape = self.continous_dim + 1

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def train(self, step):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, step)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + ".pt")
        qmixer = self.trainer.policy.mixer
        torch.save(qmixer.state_dict(), str(self.save_dir) + "/mixer_agent" + ".pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + '.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        qmixer_state_dict = torch.load(str(self.model_dir) + '/mixer_agent' + '.pt')
        self.policy.mixer.load_state_dict(qmixer_state_dict)
        
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
