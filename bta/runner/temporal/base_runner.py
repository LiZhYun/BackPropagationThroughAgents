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
from torch.distributions import kl_divergence
from tensorboardX import SummaryWriter
from collections import defaultdict
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
        self.recurrent_N = self.all_args.recurrent_N
        self.temperature = self.all_args.temperature
        self.agent_order = None
        self.clip_param = self.all_args.clip_param
        self.ppo_epoch = self.all_args.ppo_epoch
        self.num_mini_batch = self.all_args.num_mini_batch
        self.data_chunk_length = self.all_args.data_chunk_length
        self.policy_value_loss_coef = self.all_args.policy_value_loss_coef
        self.value_loss_coef = self.all_args.value_loss_coef
        self.entropy_coef = self.all_args.entropy_coef
        self.max_grad_norm = self.all_args.max_grad_norm  
        self.huber_delta = self.all_args.huber_delta
        self._use_recurrent_policy = self.all_args.use_recurrent_policy
        self._use_naive_recurrent = self.all_args.use_naive_recurrent_policy
        self._use_max_grad_norm = self.all_args.use_max_grad_norm
        self._use_clipped_value_loss = self.all_args.use_clipped_value_loss
        self._use_huber_loss = self.all_args.use_huber_loss
        self._use_popart = self.all_args.use_popart
        self._use_valuenorm = self.all_args.use_valuenorm
        self._use_value_active_masks = self.all_args.use_value_active_masks
        self._use_policy_active_masks = self.all_args.use_policy_active_masks
        self._use_policy_vhead = self.all_args.use_policy_vhead
        self.gamma = self.all_args.gamma
        self.gae_lambda = self.all_args.gae_lambda
        self._use_gae = self.all_args.use_gae
        self._use_popart = self.all_args.use_popart
        self._use_valuenorm = self.all_args.use_valuenorm
        self._use_proper_time_limits = self.all_args.use_proper_time_limits
        self.automatic_kl_tuning = self.all_args.automatic_kl_tuning
        if self.automatic_kl_tuning:
            self.log_kl_coef = torch.tensor(np.log(self.all_args.kl_coef), requires_grad=True, device=self.device)
            self.kl_coef = self.log_kl_coef.exp()
            self.kl_lr = self.all_args.kl_lr
            self.opti_eps = self.all_args.opti_eps
            self.weight_decay = self.all_args.weight_decay
            self.kl_coef_optim = torch.optim.Adam([self.log_kl_coef], lr=self.kl_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        else:
            self.kl_coef = self.all_args.kl_coef
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.inner_clip_param = self.all_args.inner_clip_param
        self.dual_clip_coeff = torch.tensor(1.0 + 0.005).to(self.device)
        self.skip_connect = self.all_args.skip_connect
        self.use_action_attention = self.all_args.use_action_attention
        self.mix_actions = False
        self.discrete = False
        self.continuous = False
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            self.discrete = True
            self.action_dim = self.envs.action_space[0].n
            self.action_shape = 1
        elif self.envs.action_space[0].__class__.__name__ == "Box":
            self.continuous = True
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
        self.obs_emb_size = self.policy[0].actor.abs_size
        
        if self.use_action_attention:
            from bta.algorithms.utils.action_attention import Action_Attention
            self.action_attention = Action_Attention(self.all_args, self.envs.action_space[0], device = self.device)
            self.attention_optimizer = torch.optim.Adam(self.action_attention.parameters(), lr=self.all_args.attention_lr, eps=self.all_args.opti_eps, weight_decay=self.all_args.weight_decay)

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
                                    self.envs.action_space[agent_id],
                                    agent_id)
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

    @torch.no_grad()
    def joint_compute(self):
        self.rewards = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        for agent_id in range(self.num_agents):
            # self.trainer[agent_id].prep_rollout()
            self.rewards[agent_id] = self.buffer[agent_id].rewards.copy()
            # next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
            #                                                     self.buffer[agent_id].rnn_states_critic[-1],
            #                                                     self.buffer[agent_id].masks[-1],
            #                                                     )
            # next_value = _t2n(next_value)
            # self.buffer[agent_id].value_preds[-1] = next_value
        self.compute_returns(self.rewards.mean(0), [self.trainer[i].value_normalizer for i in range(self.num_agents)])
    
    def compute_returns(self, rewards, value_normalizer=None):
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.advg = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        if self._use_gae:
            gae = 0
            for step in reversed(range(rewards.shape[0])):
                if self._use_popart or self._use_valuenorm:
                    norm_next_value = 0
                    norm_value = 0
                    for i in range(self.num_agents):
                        norm_next_value += value_normalizer[i].denormalize(self.buffer[i].value_preds[step + 1])
                        norm_value += value_normalizer[i].denormalize(self.buffer[i].value_preds[step])
                    norm_next_value /= self.num_agents
                    norm_value /= self.num_agents
                    delta = rewards[step] + self.gamma * norm_next_value * self.buffer[0].masks[step + 1] - norm_value
                    gae = delta + self.gamma * self.gae_lambda * self.buffer[0].masks[step + 1] * gae
                    self.advg[step] = gae
                    self.returns[step] = gae + norm_value
        # for agent_id in range(self.num_agents):
        #     self.buffer[agent_id].returns = self.returns.copy()

    def train_single(self):
        train_infos = []

        action_dim=self.buffer[0].one_hot_actions.shape[-1]
        factor = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        old_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, self.action_shape), dtype=np.float32)
        new_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, self.action_shape), dtype=np.float32)
        action_grad = np.zeros((self.num_agents, self.num_agents, self.episode_length, self.n_rollout_threads, self.action_shape), dtype=np.float32)
        ordered_vertices = [i for i in range(self.num_agents)]

        for idx, agent_id in enumerate(reversed(ordered_vertices)):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(np.prod(np.concatenate([factor[:agent_id], factor[agent_id+1:]],0), 0))

            # other agents' gradient to agent_id
            action_grad_per_agent = np.zeros((self.episode_length, self.n_rollout_threads, self.action_shape), dtype=np.float32)
            updated_agents_order = list(reversed(ordered_vertices))[0:idx] if idx < self.num_agents else list(reversed(ordered_vertices))[idx-self.num_agents+1:idx]
            for updated_agent in updated_agents_order:
                multiplier = np.concatenate([factor[:agent_id], factor[agent_id+1:]],0)
                multiplier = np.concatenate([multiplier[:updated_agent], multiplier[updated_agent+1:]],0)
                multiplier = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32) if multiplier is None else np.prod(multiplier, 0)
                action_grad_per_agent += action_grad[updated_agent][agent_id] * multiplier
            self.buffer[agent_id].update_action_grad(action_grad_per_agent)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            if self.skip_connect:
                execution_masks_batch = torch.stack([torch.ones(self.episode_length*self.n_rollout_threads)] * agent_id +
                                                [torch.zeros(self.episode_length*self.n_rollout_threads)] *
                                                (self.num_agents - agent_id), -1).to(self.device)
            else:
                if idx != self.num_agents - 1:
                    execution_masks_batch = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(list(reversed(ordered_vertices))[idx+1]), 1.0)\
                        .unsqueeze(0).repeat(self.episode_length*self.n_rollout_threads, 1).to(self.device)
                else:
                    execution_masks_batch = torch.stack([torch.zeros(self.episode_length*self.n_rollout_threads)] * self.num_agents, -1).to(self.device)

            one_hot_actions = torch.from_numpy(self.buffer[agent_id].one_hot_actions[:,:,0:self.num_agents].reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])).to(self.device)
            old_one_hot_actions = self.buffer[agent_id].one_hot_actions[:,:,0:self.num_agents].reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])
            
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
                    _, old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch[indices],
                                                                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices],
                                                                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:])[indices],
                                                                old_one_hot_actions[indices],
                                                                execution_masks_batch[indices],
                                                                available_actions[indices],
                                                                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])[indices],
                                                                tau=self.temperature)
                    old_actions_logprobs.append(old_actions_logprob)
                old_actions_logprob = torch.cat(old_actions_logprobs, dim=0)
            else:
                _, old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch,
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            old_one_hot_actions,
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            old_actions_probs[agent_id] = _t2n(torch.exp(old_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,self.action_shape)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], idx, list(reversed(ordered_vertices)), tau=self.temperature)

            if self.env_name == "GoBigger":
                new_actions_logprobs = []
                batch_size = self.n_rollout_threads * self.episode_length
                rand = list(range(batch_size))
                mini_batch_size = batch_size // self.all_args.num_mini_batch
                sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.all_args.num_mini_batch)]
                for indices in sampler:
                    _, new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch[indices],
                                                                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices],
                                                                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:])[indices],
                                                                one_hot_actions[indices],
                                                                execution_masks_batch[indices],
                                                                available_actions[indices],
                                                                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])[indices],
                                                                tau=self.temperature)
                    new_actions_logprobs.append(new_actions_logprob)
                new_actions_logprob = torch.cat(new_actions_logprobs, dim=0)
            else:
                _, new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(obs_batch,
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            one_hot_actions,
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            new_actions_probs[agent_id] = _t2n(torch.exp(new_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,self.action_shape)

            self.trainer[agent_id].policy.actor_optimizer.zero_grad()
            if self.inner_clip_param == 0.:
                torch.sum(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob.detach()),dim=-1), dim=-1).mean().backward()
            else:
                torch.sum(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob.detach()), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1), dim=-1).mean().backward()
            for i in range(self.num_agents):
                action_grad[agent_id][i] = _t2n(one_hot_actions.grad[:,i]).reshape(self.episode_length,self.n_rollout_threads,self.action_shape)
            if self.inner_clip_param == 0.:
                factor[agent_id] = _t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            else:
                factor[agent_id] = _t2n(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos
    
    def train(self):
        advs = []
        train_infos = []
        for agent_idx in range(self.num_agents):
            advs.append(self.trainer[agent_idx].train_adv(self.buffer[agent_idx]))
            train_info = defaultdict(float)
            train_info['value_loss'] = 0
            train_info['policy_loss'] = 0
            train_info['dist_entropy'] = 0
            train_info['actor_grad_norm'] = 0
            train_info['critic_grad_norm'] = 0
            train_info['ratio'] = 0
            train_info['factor'] = 0
            train_info['action_grad'] = 0
            train_infos.append(train_info)

            self.trainer[agent_idx].prep_training()
            
        batch_size = self.n_rollout_threads * self.episode_length
        mini_batch_size = batch_size // self.num_mini_batch

        for epoch in range(self.ppo_epoch):
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.num_mini_batch)]
            if self._use_recurrent_policy:
                data_generators = [self.buffer[agent_idx].recurrent_generator(advs[agent_idx], self.num_mini_batch, self.data_chunk_length, sampler=sampler) for agent_idx in range(self.num_agents)]
            elif self._use_naive_recurrent:
                data_generators = [self.buffer[agent_idx].naive_recurrent_generator(advs[agent_idx], self.num_mini_batch, sampler=sampler) for agent_idx in range(self.num_agents)]
            else:
                data_generators = [self.buffer[agent_idx].feed_forward_generator(advs[agent_idx], self.num_mini_batch, sampler=sampler) for agent_idx in range(self.num_agents)]
            
            for batch_idx in range(self.num_mini_batch):
                action_dim=self.buffer[0].one_hot_actions.shape[-1]
                factor = np.ones((self.num_agents, mini_batch_size, self.action_shape), dtype=np.float32)
                action_grad = np.zeros((self.num_agents, self.num_agents, mini_batch_size, self.action_shape), dtype=np.float32)
                ordered_vertices = [i for i in range(self.num_agents)]

                for idx, agent_idx in enumerate(reversed(ordered_vertices)):
                    # other agents' gradient to agent_id
                    action_grad_per_agent = np.zeros((mini_batch_size, self.action_shape), dtype=np.float32)
                    updated_agents_order = list(reversed(ordered_vertices))[0:idx] if idx < self.num_agents else list(reversed(ordered_vertices))[idx-self.num_agents+1:idx]
                    for updated_agent in updated_agents_order:
                        multiplier = np.concatenate([factor[:agent_idx], factor[agent_idx+1:]],0)
                        multiplier = np.concatenate([multiplier[:updated_agent], multiplier[updated_agent+1:]],0)
                        multiplier = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32) if multiplier is None else np.prod(multiplier, 0)
                        action_grad_per_agent += action_grad[updated_agent][agent_idx] * multiplier

                    share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
                    value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch, _,_,_,_ = next(data_generators[agent_idx])

                    factor_batch = check(factor).to(**self.tpdv)
                    action_grad_batch = check(action_grad_per_agent).to(**self.tpdv)
                    old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
                    adv_targ = check(adv_targ).to(**self.tpdv)
                    value_preds_batch = check(value_preds_batch).to(**self.tpdv)
                    return_batch = check(return_batch).to(**self.tpdv)
                    active_masks_batch = check(active_masks_batch).to(**self.tpdv)

                    if self.skip_connect:
                        execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * agent_idx +
                                                        [torch.zeros(actions_batch.shape[0])] *
                                                        (self.num_agents - agent_idx), -1).to(self.device)
                    else:
                        if idx != self.num_agents - 1:
                            execution_masks_batch = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(list(reversed(ordered_vertices))[idx+1]), 1.0)\
                                .unsqueeze(0).repeat(actions_batch.shape[0], 1).to(self.device)
                        else:
                            execution_masks_batch = torch.stack([torch.zeros(actions_batch.shape[0])] * self.num_agents, -1).to(self.device)                    

                    # old action logprob
                    _, old_actions_logprob, _, _, _, _ =self.trainer[agent_idx].policy.actor.evaluate_actions(obs_batch,
                                                                rnn_states_batch,
                                                                actions_batch,
                                                                masks_batch,
                                                                one_hot_actions_batch,
                                                                execution_masks_batch,
                                                                available_actions_batch,
                                                                active_masks_batch,
                                                                tau=self.temperature)
                    
                    # train
                    values, train_actions, action_log_probs, _, dist_entropy, _, _ = self.trainer[agent_idx].policy.evaluate_actions(share_obs_batch,
                                                                                        obs_batch, 
                                                                                        rnn_states_batch, 
                                                                                        rnn_states_critic_batch, 
                                                                                        actions_batch, 
                                                                                        masks_batch, 
                                                                                        one_hot_actions_batch,
                                                                                        execution_masks_batch,
                                                                                        available_actions_batch,
                                                                                        active_masks_batch,
                                                                                        tau=self.temperature
                                                                                        )
                    actions = torch.from_numpy(actions_batch).to(self.device)
                    # if self.continuous:
                    #     train_actions = torch.exp(action_log_probs) / ((-torch.exp(action_log_probs) * (actions - train_actions.mean) / (train_actions.stddev ** 2)).detach() + torch.finfo(torch.float32).eps)
                    # elif self.discrete:
                    #     train_actions = torch.exp(action_log_probs) / ((torch.exp(action_log_probs)*(1-torch.exp(action_log_probs))).detach() + torch.finfo(torch.float32).eps)

                    # is_inf_or_nan = torch.logical_or(torch.isinf(train_actions), torch.isnan(train_actions))
                    # train_actions = torch.where(is_inf_or_nan, torch.zeros_like(train_actions), train_actions)

                    # actor update
                    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
                    factor_batch = torch.prod(factor_batch,dim=0)
                    surr1 = (imp_weights * factor_batch + (imp_weights.detach()) * action_grad_batch * train_actions) * adv_targ
                    surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * factor_batch \
                            + (torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param)) * action_grad_batch * train_actions) * adv_targ
                    
                    if self._use_policy_active_masks:
                        policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                        dim=-1,
                                                        keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                    else:
                        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

                    policy_loss = policy_action_loss

                    self.trainer[agent_idx].policy.actor_optimizer.zero_grad()

                    (policy_loss - dist_entropy * self.entropy_coef).backward()
                    
                    if self._use_max_grad_norm:
                        actor_grad_norm = nn.utils.clip_grad_norm_(self.trainer[agent_idx].policy.actor.parameters(), self.max_grad_norm)
                    else:
                        actor_grad_norm = get_gard_norm(self.trainer[agent_idx].policy.actor.parameters())

                    self.trainer[agent_idx].policy.actor_optimizer.step()

                    # critic update
                    value_loss = self.trainer[agent_idx].cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
                    value_loss = value_loss * self.value_loss_coef
                    self.trainer[agent_idx].policy.critic_optimizer.zero_grad()

                    (value_loss * self.value_loss_coef).backward()

                    if self._use_max_grad_norm:
                        critic_grad_norm = nn.utils.clip_grad_norm_(self.trainer[agent_idx].policy.critic.parameters(), self.max_grad_norm)
                    else:
                        critic_grad_norm = get_gard_norm(self.trainer[agent_idx].policy.critic.parameters())

                    self.trainer[agent_idx].policy.critic_optimizer.step()

                    # new action logprob
                    one_hot_actions = torch.from_numpy(one_hot_actions_batch).to(self.device)
                    one_hot_actions.requires_grad = True
                    _, new_actions_logprob, _, _, _, _ =self.trainer[agent_idx].policy.actor.evaluate_actions(obs_batch,
                                                                rnn_states_batch,
                                                                actions_batch,
                                                                masks_batch,
                                                                one_hot_actions,
                                                                execution_masks_batch,
                                                                available_actions_batch,
                                                                active_masks_batch,
                                                                tau=self.temperature)
                    
                    self.trainer[agent_idx].policy.actor_optimizer.zero_grad()
                    if self.inner_clip_param == 0.:
                        torch.sum(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob.detach()),dim=-1), dim=-1).mean().backward()
                    else:
                        torch.sum(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob.detach()), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1), dim=-1).mean().backward()
                    for i in range(self.num_agents):
                        if self.discrete:
                            action_grad[agent_idx][i] = _t2n(one_hot_actions.grad[:,i].gather(1, actions.long()))
                        else:
                            action_grad[agent_idx][i] = _t2n(one_hot_actions.grad[:,i])
                    if self.inner_clip_param == 0.:
                        factor[agent_idx] = _t2n(torch.exp(new_actions_logprob-old_actions_logprob))
                    else:
                        factor[agent_idx] = _t2n(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param))
                    
                    train_infos[agent_idx]['value_loss'] += value_loss.item()
                    train_infos[agent_idx]['policy_loss'] += policy_loss.item()
                    train_infos[agent_idx]['dist_entropy'] += dist_entropy.item()
                    train_infos[agent_idx]['ratio'] += imp_weights.mean().item()
                    train_infos[agent_idx]['action_grad'] += action_grad_batch.mean().item()
                    train_infos[agent_idx]['factor'] += factor_batch.mean().item()
                    
                    if int(torch.__version__[2]) < 5:
                        train_infos[agent_idx]['actor_grad_norm'] += actor_grad_norm
                        train_infos[agent_idx]['critic_grad_norm'] += critic_grad_norm
                    else:
                        train_infos[agent_idx]['actor_grad_norm'] += actor_grad_norm.item()
                        train_infos[agent_idx]['critic_grad_norm'] += critic_grad_norm.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for agent_idx in range(self.num_agents):
            for k in train_infos[agent_idx].keys():
                train_infos[agent_idx][k] /= num_updates    
            self.buffer[agent_idx].after_update()
        return train_infos
    
    def joint_train(self):
        train_infos = []
        advantages_all = self.advg[:-1]
        advantages_copy = advantages_all.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages_all = (advantages_all - mean_advantages) / (std_advantages + 1e-5)
        advs = []
        for agent_idx in range(self.num_agents):
            advs.append(self.trainer[agent_idx].train_adv(self.buffer[agent_idx]))
            train_info = defaultdict(float)
            train_info['value_loss'] = 0
            train_info['policy_loss'] = 0
            train_info['joint_policy_loss'] = 0
            train_info['dist_entropy'] = 0
            train_info['joint_dist_entropy'] = 0
            train_info['actor_grad_norm'] = 0
            train_info['critic_grad_norm'] = 0
            train_info['attention_grad_norm'] = 0
            train_info['ratio'] = 0
            train_info['joint_ratio'] = 0
            train_infos.append(train_info)

            self.trainer[agent_idx].prep_training()
            
        batch_size = self.n_rollout_threads * self.episode_length
        mini_batch_size = batch_size // self.num_mini_batch

        new_actions_logprobs = np.ones((self.num_agents, batch_size, self.action_shape), dtype=np.float32)
        new_actions = np.ones((self.num_agents, batch_size, self.action_shape))
        new_advg = np.ones((self.num_agents, batch_size, 1), dtype=np.float32)
        new_returns = np.ones((self.num_agents, batch_size, 1), dtype=np.float32)

        for epoch in range(self.ppo_epoch):
            advantages_all = advantages_all.reshape(-1, 1)
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.num_mini_batch)]
            if self._use_recurrent_policy:
                data_generators = [self.buffer[agent_idx].recurrent_generator(advs[agent_idx], self.num_mini_batch, self.data_chunk_length, sampler=sampler) for agent_idx in range(self.num_agents)]
            elif self._use_naive_recurrent:
                data_generators = [self.buffer[agent_idx].naive_recurrent_generator(advs[agent_idx], self.num_mini_batch, sampler=sampler) for agent_idx in range(self.num_agents)]
            else:
                data_generators = [self.buffer[agent_idx].feed_forward_generator(advs[agent_idx], self.num_mini_batch, sampler=sampler) for agent_idx in range(self.num_agents)]
            
            for batch_idx in range(self.num_mini_batch):
                adv_targ_all = check(advantages_all[sampler[batch_idx]]).to(**self.tpdv)
                available_actions_all = torch.ones(mini_batch_size, self.num_agents, self.action_dim).to(self.device)
                active_masks_all = torch.zeros(mini_batch_size, self.num_agents, 1).to(self.device)
                logits_all = torch.zeros(mini_batch_size, self.num_agents, self.action_dim).to(self.device)
                obs_feats_all = torch.zeros(mini_batch_size, self.num_agents, self.obs_emb_size).to(self.device)
                individual_loss = torch.zeros(self.num_agents).to(self.device)
                for agent_idx in range(self.num_agents):
                    share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
                    value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch, factor_batch, action_grad, joint_actions_batch, joint_action_log_probs_batch = next(data_generators[agent_idx])
                    adv_targ = check(adv_targ).to(**self.tpdv)
                    old_joint_action_log_probs = check(joint_action_log_probs_batch).to(**self.tpdv)
                    old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
                    active_masks_batch = check(active_masks_batch).to(**self.tpdv)
                    active_masks_all[:, agent_idx] = check(active_masks_batch).to(**self.tpdv)
                    if available_actions_batch is not None:
                        available_actions_all[:, agent_idx] = check(available_actions_batch).to(**self.tpdv)

                    ego_exclusive_action = one_hot_actions_batch[:,0:self.num_agents]
                    execution_mask = torch.stack([torch.zeros(mini_batch_size)] * self.num_agents, -1).to(self.device)

                    actions_batch = check(new_actions[agent_idx, sampler[batch_idx]]).to(self.device) if epoch != 0
                    old_action_log_probs_batch = check(new_actions_logprobs[agent_idx, sampler[batch_idx]]).to(self.device) if epoch != 0
                    adv_targ = check(new_advg[agent_idx, sampler[batch_idx]]).to(self.device) if epoch != 0
                    return_batch = check(new_returns[agent_idx, sampler[batch_idx]]).to(self.device) if epoch != 0

                    values, individual_dist, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat = self.trainer[agent_idx].policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            ego_exclusive_action,
                                                                            execution_mask,
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            tau=self.temperature,
                                                                            kl=True,
                                                                            joint_actions=joint_actions_batch[:,agent_idx]
                                                                            )
                
                    logits_all[:, agent_idx] = logits.clone()
                    obs_feats_all[:, agent_idx] = obs_feat.clone()

                    # actor update
                    if epoch != 0:
                        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                        # off_factor = torch.exp(old_action_log_probs_batch - old_joint_action_log_probs[:, agent_idx])

                        # dual clip
                        # cliped_ratio = torch.min(ratio, off_factor * (1.0 + self.clip_param))

                        surr1 = ratio * adv_targ
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

                        # clip decay
                        self.clip_param = self.clip_param * 0.96
                    else:
                        ratio = torch.exp(action_log_probs_kl - old_joint_action_log_probs[:, agent_idx])
                        surr1 = action_log_probs_kl
                        surr2 = action_log_probs_kl
                    
                    if self._use_policy_active_masks:
                        policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                        dim=-1,
                                                        keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                    else:
                        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

                    policy_loss = policy_action_loss

                    individual_loss[agent_idx] = (policy_loss)

                    #critic update
                    value_loss = self.trainer[agent_idx].cal_value_loss(values, check(value_preds_batch).to(**self.tpdv), 
                                    check(return_batch).to(**self.tpdv), check(active_masks_batch).to(**self.tpdv))
                    value_loss = value_loss * self.value_loss_coef
                    self.trainer[agent_idx].policy.critic_optimizer.zero_grad()

                    (value_loss * self.value_loss_coef).backward()

                    if self._use_max_grad_norm:
                        critic_grad_norm = nn.utils.clip_grad_norm_(self.trainer[agent_idx].policy.critic.parameters(), self.max_grad_norm)
                    else:
                        critic_grad_norm = get_gard_norm(self.trainer[agent_idx].policy.critic.parameters())

                    self.trainer[agent_idx].policy.critic_optimizer.step()

                    train_infos[agent_idx]['value_loss'] += value_loss.item()
                    train_infos[agent_idx]['policy_loss'] += policy_loss.item()
                    # train_infos[agent_idx]['actor_grad_norm'] += actor_grad_norm
                    train_infos[agent_idx]['critic_grad_norm'] += critic_grad_norm
                    train_infos[agent_idx]['ratio'] += ratio.mean().item()
                    train_infos[agent_idx]['dist_entropy'] += dist_entropy.item()

                joint_action_log_probs, joint_dist_entropy, joint_logits, joint_dist = self.action_attention.evaluate_actions(logits_all, obs_feats_all, joint_actions_batch, available_actions=available_actions_all, tau=self.temperature)

                # actor update
                ratio = torch.prod(torch.prod(torch.exp(joint_action_log_probs - old_joint_action_log_probs),dim=-1,keepdim=True),dim=-2)

                surr1 = ratio * adv_targ_all
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_all
            
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
              
                policy_loss = policy_action_loss

                for agent_idx in range(self.num_agents):
                    self.trainer[agent_idx].policy.actor_optimizer.zero_grad()
                self.attention_optimizer.zero_grad()

                (policy_loss - joint_dist_entropy * self.entropy_coef + individual_loss.sum()).backward()
                
                if self._use_max_grad_norm:
                    attention_grad_norm = nn.utils.clip_grad_norm_(self.action_attention.parameters(), self.max_grad_norm)
                    for agent_idx in range(self.num_agents): 
                        actor_grad_norm = nn.utils.clip_grad_norm_(self.trainer[agent_idx].policy.actor.parameters(), self.max_grad_norm)
                        train_infos[agent_idx]['actor_grad_norm'] += actor_grad_norm
                        train_infos[agent_idx]['attention_grad_norm'] += attention_grad_norm
                else:
                    attention_grad_norm = get_gard_norm(self.action_attention.parameters())
                    for agent_idx in range(self.num_agents): 
                        actor_grad_norm = get_gard_norm(self.trainer[agent_idx].policy.actor.parameters())
                        train_infos[agent_idx]['actor_grad_norm'] += actor_grad_norm
                        train_infos[agent_idx]['attention_grad_norm'] += attention_grad_norm

                for agent_idx in range(self.num_agents):
                    self.trainer[agent_idx].policy.actor_optimizer.step()
                self.attention_optimizer.step()
                
                for agent_idx in range(self.num_agents):
                    train_infos[agent_idx]['joint_policy_loss'] += policy_loss.item()
                    train_infos[agent_idx]['joint_dist_entropy'] += joint_dist_entropy.item()
                    train_infos[agent_idx]['joint_ratio'] += ratio.mean().item()
            
            if epoch == 0:
                for agent_id in range(self.num_agents):
                    if self.env_name == "GoBigger":
                        obs_batch = flatten(self.buffer[agent_id].obs[:-1])
                    else:
                        obs_batch = self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:])
                    
                    execution_masks_batch = torch.stack([torch.zeros(self.episode_length*self.n_rollout_threads)] * self.num_agents, -1).to(self.device)

                    one_hot_actions = self.buffer[agent_id].one_hot_actions[:,:,0:self.num_agents].reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])

                    available_actions = None if self.buffer[agent_id].available_actions is None \
                        else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

                    new_action, new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor(obs_batch,
                                                                    self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                                    one_hot_actions,
                                                                    execution_masks_batch,
                                                                    available_actions,
                                                                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                                    tau=self.temperature)
                    
                    new_actions_logprobs[agent_id] = _t2n(new_actions_logprob)
                    new_actions[agent_id] = _t2n(torch.argmax(new_action, -1, keepdim=True).to(torch.int)) if self.discrete = True else _t2n(new_action)

                    advg, returns = self.buffer[agent_id].compute_returns(self.trainer[agent_id].value_normalizer)
                    new_advg[agent_id] = advg
                    new_returns[agent_id] = returns
                
        num_updates = self.ppo_epoch * self.num_mini_batch

        for agent_idx in range(self.num_agents):
            for k in train_infos[agent_idx].keys():
                train_infos[agent_idx][k] /= num_updates    
            self.buffer[agent_idx].after_update()
        return train_infos

    def save(self, steps=None):
        postfix = f"_{steps}.pt" if steps else ".pt"
        if self.use_action_attention:
            torch.save(self.action_attention.state_dict(), str(self.save_dir) + "/joint_agent" + postfix)
            torch.save(self.attention_optimizer.state_dict(), str(self.save_dir) + "/joint_opti" + postfix)
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + postfix)
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(self.trainer[agent_id].policy.actor_optimizer.state_dict(), str(self.save_dir) + "/actor_opti" + str(agent_id) + postfix)
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + postfix)
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + postfix)
                torch.save(self.trainer[agent_id].policy.critic_optimizer.state_dict(), str(self.save_dir) + "/critic_opti" + str(agent_id) + postfix)

    def restore(self):
        if self.use_action_attention:
            joint_agent_state_dict = torch.load(str(self.model_dir) + '/joint_agent.pt')
            self.action_attention.load_state_dict(joint_agent_state_dict)
            joint_opti_state_dict = torch.load(str(self.model_dir) + '/joint_opti.pt')
            self.attention_optimizer.load_state_dict(joint_opti_state_dict)
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                actor_opti_state_dict = torch.load(str(self.model_dir) + '/actor_opti' + str(agent_id) + '.pt')
                self.policy[agent_id].actor_optimizer.load_state_dict(actor_opti_state_dict)
                if not self.use_render:
                    policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
                    critic_opti_state_dict = torch.load(str(self.model_dir) + '/critic_opti' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic_optimizer.load_state_dict(critic_opti_state_dict)

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