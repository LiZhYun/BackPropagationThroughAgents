import time
from typing import Any, Dict
import wandb
import os
import socket
import numpy as np
import igraph as ig
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pickle

from bta.utils.separated_buffer_ar import SeparatedReplayBuffer
from bta.utils.util import update_linear_schedule, get_gard_norm, flatten, generate_mask_from_order
from bta.algorithms.utils.util import check

import psutil
import socket

def _t2n(x):
    return x.detach().cpu().numpy()

def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    if "temporal" in algorithm_name:
        from bta.algorithms.bta.t_policy import T_POLICY as TrainAlgo
        from bta.algorithms.bta.algorithm.temporalPolicy import TemporalPolicy as Policy
        from bta.algorithms.bta.algorithm.temporal_model import Mixer_per_node as Graph
        return TrainAlgo, Policy, Graph
    elif "mappo" in algorithm_name:
        from bta.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from bta.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        return TrainAlgo, Policy
    elif "ar" in algorithm_name:
        from bta.algorithms.ar_mappo.ar_mappo import AR_MAPPO as TrainAlgo
        from bta.algorithms.ar_mappo.algorithm.arMAPPOPolicy import AutoRegressivePolicy as Policy
        return TrainAlgo, Policy
    # elif algorithm_name == "traj":
    #     from bta.algorithms.population.traj import Traj_Trainer as TrainAlgo
    #     from bta.algorithms.population.policy_pool import PolicyPool as Policy
    else:
        raise NotImplementedError
    

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
        self.use_single_network = self.all_args.use_single_network
        self.recurrent_N = self.all_args.recurrent_N
        self.mix_actions = False
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
        # self.action_dim = self.envs.action_space[0].n if self.envs.action_space[0].__class__.__name__ == "Discrete" else self.envs.action_space[0].shape[0]
        self.all_args.action_dim = self.action_dim
        self.all_args.action_shape = self.action_shape
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

        TrainAlgo, Policy = make_trainer_policy_cls(self.algorithm_name, use_single_network=self.use_single_network)

        # # dump policy config to allow loading population in yaml form
        # share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        # self.policy_config = (self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0])
        # policy_config_path = os.path.join(self.run_dir, 'policy_config.pkl')
        # pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        # print(f"Pickle dump policy config at {policy_config_path}")

        self.policy = []
        for agent_id in range(self.num_agents):
            #print(len(self.envs.share_observation_space))
            #print(len(self.envs.observation_space))
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], agent_id, self.envs.action_space[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            #print("Base runner", agent_id, share_observation_space)
            bu = SeparatedReplayBuffer(self.all_args,
                                    self.envs.observation_space[agent_id],
                                    share_observation_space,
                                    self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            # self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1],
                                                                )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, steps=None):
        postfix = f"_{steps}.pt" if steps else ".pt"
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + postfix)
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + postfix)
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + postfix)

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                if not self.use_render:
                    policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log(self, infos: Dict[str, Any], step):
        if self.use_wandb:
            wandb.log(infos, step=step)
        else:
            [self.writter.log(k, v, step) for k, v in infos.items()]

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def log_system(self):
        # RRAM
        mem = psutil.virtual_memory()
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        if used_mem/total_mem > 0.95:
            slack = slackweb.Slack(url=webhook_url)
            host_name = socket.gethostname()
            slack.notify(text="Host {}: occupied memory is *{:.2f}*%!".format(host_name, used_mem/total_mem*100))