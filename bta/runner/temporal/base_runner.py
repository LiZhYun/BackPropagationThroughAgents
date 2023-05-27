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
from tensorboardX import SummaryWriter
import pickle
import math
from bta.utils.separated_buffer import SeparatedReplayBuffer
from bta.utils.util import update_linear_schedule, is_acyclic, pruning, get_gard_norm, flatten, generate_mask_from_order
from bta.algorithms.utils.util import check

import psutil
import socket

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
        self.use_single_network = self.all_args.use_single_network
        self.use_graph = self.all_args.use_graph
        self.recurrent_N = self.all_args.recurrent_N
        self.temperature = self.all_args.temperature
        self.agent_order = None
        self.inner_clip_param = self.all_args.inner_clip_param
        self.mix_actions = False
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            self.action_dim = self.envs.action_space[0].n
            self.action_shape = 1
        elif self.envs.action_space[0].__class__.__name__ == "Box":
            self.action_dim = self.envs.action_space[0].shape[0]
            self.action_shape = self.envs.action_space[0].shape[0]
        else:
            self.mix_actions = True
            self.continous_dim = self.envs.action_space[0][0].shape[0]
            self.discrete_dim = self.envs.action_space[0][1].n
            self.action_dim = self.continous_dim + self.discrete_dim
            self.action_shape = self.continous_dim + 1
        # self.action_dim = self.envs.action_space[0].n if self.envs.action_space[0].__class__.__name__ == "Discrete" else self.envs.action_space[0].shape[0]
        
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
            self.run_dir = self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from bta.algorithms.bta.t_policy import T_POLICY as TrainAlgo
        from bta.algorithms.bta.algorithm.temporalPolicy import TemporalPolicy as Policy
        from bta.algorithms.bta.algorithm.temporal_model import Mixer_per_node as Graph

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
                        agent_id,
                        device = self.device)
            self.policy.append(po)

        if self.use_graph:
            self.all_args.node_feat_size = self.policy[0].actor.abs_size + self.num_agents
            self.all_args.edge_feat_size = 1 + self.num_agents*2 + 2

            edge_predictor_configs = {
                'dim_in_time': self.all_args.hidden_size,
                'dim_in_node': self.all_args.node_feat_size,
                'hidden_channels' : self.all_args.hidden_size, 
            }

            mixer_configs = {
                'per_graph_size'  : self.all_args.max_edges, 
                'time_channels'   : self.all_args.time_channels, 
                'input_channels'  : self.all_args.edge_feat_size, 
                'hidden_channels' : self.all_args.hidden_size, 
                'out_channels'    : self.all_args.hidden_size,
                'num_layers'      : self.all_args.num_layers,
                'use_single_layer' : False
            }
            self.graph_policy = Graph(mixer_configs, edge_predictor_configs, device = self.device)
            self.graph_optimizer = torch.optim.Adam(self.graph_policy.parameters(), lr=self.all_args.graph_lr, eps=self.all_args.opti_eps, weight_decay=self.all_args.weight_decay)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], agent_id, device = self.device)
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
        # random update order
        if self.use_graph:
            self.graph_policy.train()
            for agent_id in range(self.num_agents):
                all_loss = torch.zeros(1).to(self.device)
                graph_loss = self.trainer[agent_id].train_graph(self.buffer[agent_id])
                all_loss += graph_loss
            all_loss /= self.num_agents
            self.graph_optimizer.zero_grad()
            all_loss.backward()
            if self.all_args.use_max_grad_norm:
                _ = nn.utils.clip_grad_norm_(self.graph_policy.parameters(), self.all_args.max_grad_norm)
            self.graph_optimizer.step()

        action_dim=self.buffer[0].one_hot_actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        old_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        new_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        action_grad = np.zeros((self.num_agents, self.num_agents, self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        ordered_vertices = self.agent_order[0]

        for idx, agent_id in enumerate(reversed(ordered_vertices)):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)

            # other agents' gradient to agent_id
            action_grad_per_agent = np.zeros((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
            for updated_agent in reversed(ordered_vertices)[0:idx]:
                numerator = np.concatenate([new_actions_probs[agent_id+1:updated_agent], new_actions_probs[updated_agent+1:]],0)
                denominator = np.concatenate([old_actions_probs[agent_id+1:updated_agent], old_actions_probs[updated_agent+1:]],0)
                numerator = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32) if numerator is None else np.prod(numerator, 0)
                denominator = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32) if denominator is None else np.prod(denominator, 0)
                action_grad_per_agent += action_grad[updated_agent][agent_id] * (numerator / denominator)
            self.buffer[agent_id].update_action_grad(action_grad_per_agent)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            tmp_agent_order = ordered_vertices.clone()
            agent_order = torch.stack([tmp_agent_order for _ in range(self.episode_length*self.n_rollout_threads)]).to(self.device)
            # agent_order = torch.stack([torch.randperm(self.num_agents) for _ in range(self.episode_length*self.n_rollout_threads)]).to(self.device)
            execution_masks_batch = generate_mask_from_order(
                agent_order, ego_exclusive=False).to(
                    self.device).float()[:, agent_id]  # [bs, n_agents, n_agents]
                
            one_hot_actions = torch.from_numpy(self.buffer[agent_id].one_hot_actions.reshape(-1, *self.buffer[agent_id].one_hot_actions.shape[2:])).to(self.device)
            one_hot_actions.requires_grad = True

            if self.env_name == "GoBigger":
                obs_batch = flatten(self.buffer[agent_id].obs[:-1])
            else:
                obs_batch = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:])
            
            if self.env_name == "GoBigger":
                old_actions_logprobs = []
                batch_size = self.n_rollout_threads * self.episode_length
                rand = list(range(batch_size))
                mini_batch_size = batch_size // self.all_args.num_mini_batch
                sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.all_args.num_mini_batch)]
                for indices in sampler:
                    _, old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch[indices],
                                                                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices],
                                                                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:])[indices],
                                                                self.buffer[agent_id].one_hot_actions.reshape(-1, *self.buffer[agent_id].one_hot_actions.shape[2:])[indices],
                                                                # self.buffer[agent_id].execution_masks.reshape(-1, *self.buffer[agent_id].execution_masks.shape[2:]),
                                                                execution_masks_batch[indices],
                                                                available_actions[indices],
                                                                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])[indices],
                                                                tau=self.temperature)
                    old_actions_logprobs.append(old_actions_logprob)
                old_actions_logprob = torch.cat(old_actions_logprobs, dim=0)
            else:
                _, old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch,
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            self.buffer[agent_id].one_hot_actions.reshape(-1, *self.buffer[agent_id].one_hot_actions.shape[2:]),
                                                            # self.buffer[agent_id].execution_masks.reshape(-1, *self.buffer[agent_id].execution_masks.shape[2:]),
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            old_actions_probs[agent_id] = _t2n(torch.exp(old_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,1)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], tmp_agent_order, tau=self.temperature)

            if self.env_name == "GoBigger":
                new_actions_logprobs = []
                batch_size = self.n_rollout_threads * self.episode_length
                rand = list(range(batch_size))
                mini_batch_size = batch_size // self.all_args.num_mini_batch
                sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.all_args.num_mini_batch)]
                for indices in sampler:
                    _, new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch[indices],
                                                                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices],
                                                                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:])[indices],
                                                                one_hot_actions[indices],
                                                                # self.buffer[agent_id].execution_masks.reshape(-1, *self.buffer[agent_id].execution_masks.shape[2:]),
                                                                execution_masks_batch[indices],
                                                                available_actions[indices],
                                                                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])[indices],
                                                                tau=self.temperature)
                    new_actions_logprobs.append(new_actions_logprob)
                new_actions_logprob = torch.cat(new_actions_logprobs, dim=0)
            else:
                _, new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch,
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            one_hot_actions,
                                                            # self.buffer[agent_id].execution_masks.reshape(-1, *self.buffer[agent_id].execution_masks.shape[2:]),
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            new_actions_probs[agent_id] = _t2n(torch.exp(new_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,1)

            self.trainer[agent_id].policy.actor_optimizer.zero_grad()
            torch.sum(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob.detach()), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1), dim=-1).mean().backward()
            for i in range(self.num_agents):
                action_grad[agent_id][i] = _t2n(one_hot_actions.grad[:,i]).reshape(self.episode_length,self.n_rollout_threads,action_dim)
            factor = factor*_t2n(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            if self.use_graph:
                train_info['graphic_loss'] = all_loss.item()
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
        if self.use_graph:
            graph_policy = self.graph_policy                 
            torch.save(graph_policy.state_dict(), str(self.save_dir) + "/graph_agent" + postfix)

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
        if self.use_graph:
            graph_state_dict = torch.load(str(self.model_dir) + '/graph_agent.pt')
            self.graph_policy.load_state_dict(graph_state_dict)

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