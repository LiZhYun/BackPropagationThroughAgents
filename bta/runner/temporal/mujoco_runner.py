import copy
import os
import time
import numpy as np
import itertools
from itertools import chain
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import imageio
import warnings
import functools
from bta.utils.util import update_linear_schedule, is_acyclic, pruning, generate_mask_from_order
from bta.runner.temporal.base_runner import Runner
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict
from icecream import ic
from scipy.stats import rankdata
import igraph as ig
import wandb
from bta.algorithms.utils.util import check
from bta.algorithms.utils.distributions import FixedCategorical, FixedNormal
import math


def _t2n(x):
    if type(x) == float:
        return x
    else:
        return x.detach().cpu().numpy()

class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            if self.decay_id == 0:
                # self.threshold = max(self.initial_threshold - (self.initial_threshold * ((episode*self.decay_factor) / float(episodes))), 0.)
                # self.temperature = max(self.all_args.temperature - (self.all_args.temperature * (episode*self.decay_factor / float(episodes))), 0.05)
                self.temperature = min(0.1 + ((self.all_args.temperature - 0.1) * (episode*self.decay_factor / float(episodes))), self.all_args.temperature)
            elif self.decay_id == 1:
                # self.threshold = 0. + (self.initial_threshold - 0.) * \
                    # (1 + math.cos(math.pi * (episode*self.decay_factor) / (episodes-1))) / 2 if episode*self.decay_factor <= episodes else 0.
                # self.temperature = 0.05 + (self.all_args.temperature - 0.05) * \
                #     (1 + math.cos(math.pi * (episode*self.decay_factor) / (episodes-1))) / 2
                self.temperature = 0.1 + (self.all_args.temperature - 0.1) * \
                    (1 + math.cos(math.pi * (episode*self.decay_factor) / (episodes-1) + math.pi)) / 2 if episode*self.decay_factor <= episodes else self.all_args.temperature
            elif self.decay_id == 2:
                # self.threshold = 0.1 + self.all_args.threshold * math.pow(1.01,math.floor((episode)/10))
                self.temperature = self.all_args.temperature * math.pow(0.99,math.floor((episode)/10))
            else:
                pass
            self.agent_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            
            for step in range(self.episode_length):
                # Sample actions
                values, actions, hard_actions, action_log_probs, rnn_states, \
                    rnn_states_critic, joint_actions, joint_action_log_probs, rnn_states_joint, thresholds, bias, logits = self.collect(step)

                env_actions = joint_actions if self.use_action_attention else hard_actions
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(env_actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
                    rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, rnn_states_joint, thresholds, bias, logits

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            if self.use_action_attention:
                self.joint_compute()
            train_infos = self.joint_train(episode) if self.use_action_attention else [self.train_seq_agent_m, self.train_seq_agent_a, self.train_sim_a][self.train_sim_seq]()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                total_mean = 0.0
                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards_per_agent"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    total_mean += train_infos[a]["average_episode_rewards_per_agent"]
                    # print("average episode rewards agent {} is {}".format(a, train_infos[a]["average_episode_rewards_per_agent"]))
                total_mean /= self.num_agents
                print("average episode rewards for team is {}".format(total_mean))
                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = total_mean
                    # train_infos[a]["threshold"] = _t2n(self.threshold_dist().mean) if self.decay_id == 3 else self.threshold
                # print("threshold is {}".format(train_infos[0]["threshold"]))
                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    if self.use_wandb:
                        wandb.log({"train_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                                total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    def collect(self, step):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_dim))
        logits = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_dim).to(**self.tpdv)
        obs_feats = torch.zeros(self.n_rollout_threads, self.num_agents, self.obs_emb_size).to(**self.tpdv)
        hard_actions = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_shape).to(**self.tpdv)
        action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape))
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        if not self.discrete:
            stds = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_dim).to(**self.tpdv)
  
        ordered_vertices = [i for i in range(self.num_agents)]
        for idx, agent_idx in enumerate(ordered_vertices):
            self.trainer[agent_idx].prep_rollout()
            # ego_exclusive_action = actions.copy()
            ego_exclusive_action = actions[:,0:self.num_agents]
            # tmp_execution_mask = execution_masks[:, agent_idx]
            if self.use_action_attention:
                tmp_execution_mask = torch.stack([torch.zeros(self.n_rollout_threads)] * self.num_agents, -1).to(self.device)
            else:
                tmp_execution_mask = torch.stack([torch.ones(self.n_rollout_threads)] * agent_idx +
                                                [torch.zeros(self.n_rollout_threads)] *
                                                (self.num_agents - agent_idx), -1).to(self.device)
                
            value, action, action_log_prob, rnn_state, rnn_state_critic, logit, obs_feat \
                = self.trainer[agent_idx].policy.get_actions(self.buffer[agent_idx].share_obs[step],
                                                            self.buffer[agent_idx].obs[step],
                                                            self.buffer[agent_idx].rnn_states[step],
                                                            self.buffer[agent_idx].rnn_states_critic[step],
                                                            self.buffer[agent_idx].masks[step],
                                                            ego_exclusive_action,
                                                            tmp_execution_mask,
                                                            # deterministic=True,
                                                            # tau=self.temperature
                                                            )
            hard_actions[:, agent_idx] = action
            actions[:, agent_idx] = _t2n(action)
            logits[:, agent_idx] = logits if self.discrete else logit.mean
            if not self.discrete:
                stds[:, agent_idx] = logit.stddev
            obs_feats[:, agent_idx] = obs_feat
            action_log_probs[:, agent_idx] = _t2n(action_log_prob)
            values[:, agent_idx] = _t2n(value)
            rnn_states[:, agent_idx] = _t2n(rnn_state)
            rnn_states_critic[:, agent_idx] = _t2n(rnn_state_critic)

        joint_actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape))
        joint_action_log_probs, rnn_states_joint = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape)), np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        if self.use_action_attention:
            share_obs = np.concatenate(np.stack([self.buffer[i].share_obs[step] for i in range(self.num_agents)], 1))
            rnn_states_joint = np.concatenate(np.stack([self.buffer[i].rnn_states_joint[step] for i in range(self.num_agents)], 1))
            masks = np.concatenate(np.stack([self.buffer[i].masks[step] for i in range(self.num_agents)], 1))
            bias_, action_std, rnn_states_joint = self.action_attention(logits.reshape(-1, self.action_dim), obs_feats.reshape(-1, self.obs_emb_size), share_obs, rnn_states_joint, masks, hard_actions)
            rnn_states_joint = _t2n(rnn_states_joint)
            # if self.decay_id == 3:
            #     self.threshold = self.threshold_dist().sample([self.n_rollout_threads*self.num_agents]).view(self.n_rollout_threads, self.num_agents, 1)
            #     self.threshold = torch.clamp(self.threshold, 0, 1)
            if self.discrete:
                mixed_ = (logits + action_std) / self.temperature  # ~Gumbel(logits,tau)
                mixed_ = mixed_ - mixed_.logsumexp(dim=-1, keepdim=True)
                # mixed_ = bias_
                ind_dist = FixedCategorical(logits=logits)
                mix_dist = FixedCategorical(logits=mixed_)
            else:
                ind_dist = FixedNormal(logits, stds)
                mix_dist = FixedNormal(logits, action_std)
            if self.threshold >= torch.rand(1):
                mix_actions = mix_dist.sample()
            else:
                mix_actions = hard_actions.clone()
            if (self.threshold > 0.) and (self.threshold < 1.):
                mix_action_log_probs = (mix_dist.log_probs(mix_actions) + torch.tensor(self.threshold, device=self.device).log()) if not self.discrete else (mix_dist.log_probs_joint(mix_actions) + torch.tensor(self.threshold, device=self.device).log())
                ind_action_log_probs = (ind_dist.log_probs(mix_actions) + torch.tensor(1-self.threshold, device=self.device).log()) if not self.discrete else (ind_dist.log_probs_joint(mix_actions) + torch.tensor(1-self.threshold, device=self.device).log())
                log_probs = torch.stack([ind_action_log_probs, mix_action_log_probs],dim=-1)
                action_log_probs = _t2n(torch.logsumexp(log_probs,-1))
            elif self.threshold == 0.:
                ind_action_log_probs = ind_dist.log_probs(mix_actions) if not self.discrete else ind_dist.log_probs_joint(mix_actions)
                action_log_probs = _t2n(ind_action_log_probs)
            elif self.threshold == 1.:
                mix_action_log_probs = mix_dist.log_probs(mix_actions) if not self.discrete else mix_dist.log_probs_joint(mix_actions)
                action_log_probs = _t2n(mix_action_log_probs)  
            # mix_actions = mix_dist.sample()
            # mix_actions = hard_actions.clone()
            # mix_action_log_probs = mix_dist.log_probs(mix_actions) if not self.discrete else mix_dist.log_probs_joint(mix_actions)
            # ind_action_log_probs = ind_dist.log_probs(mix_actions) if not self.discrete else ind_dist.log_probs_joint(mix_actions)
            joint_actions = _t2n(mix_actions)
            # action_log_probs = _t2n(ind_action_log_probs)  
            # joint_action_log_probs = _t2n(mix_action_log_probs)  

        return values, actions, _t2n(hard_actions), action_log_probs, rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, rnn_states_joint, _t2n(self.threshold), _t2n(bias_), _t2n(logits)

    def collect_eval(self, step, eval_obs, eval_rnn_states, eval_masks):
        actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim))
        hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_shape))

        ordered_vertices = [i for i in range(self.num_agents)]
        for idx, agent_idx in enumerate(ordered_vertices):
            self.trainer[agent_idx].prep_rollout()
            # ego_exclusive_action = actions.copy()
            # tmp_execution_mask = execution_masks[:, agent_idx]
            ego_exclusive_action = actions[:,0:self.num_agents]
            # tmp_execution_mask = execution_masks[:, agent_idx]
            if self.use_action_attention:
                tmp_execution_mask = torch.stack([torch.zeros(self.n_eval_rollout_threads)] * self.num_agents, -1).to(self.device)
            else:
                tmp_execution_mask = torch.stack([torch.ones(self.n_eval_rollout_threads)] * agent_idx +
                                                [torch.zeros(self.n_eval_rollout_threads)] *
                                                (self.num_agents - agent_idx), -1).to(self.device)
            action, rnn_state = self.trainer[agent_idx].policy.act(eval_obs[:, agent_idx],
                                                        eval_rnn_states[:, agent_idx],
                                                        eval_masks[:, agent_idx],
                                                        ego_exclusive_action,
                                                        tmp_execution_mask,
                                                        deterministic=True)
            hard_actions[:, agent_idx] = _t2n(action)
            actions[:, agent_idx] = _t2n(action)
            eval_rnn_states[:, agent_idx] = _t2n(rnn_state)

        return actions, hard_actions, eval_rnn_states

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
            rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, rnn_states_joint, thresholds, bias, logits = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id],
                                        obs[:, agent_id],
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, :],
                                        hard_actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id],
                                        joint_actions=joint_actions[:, agent_id],
                                        joint_action_log_probs=joint_action_log_probs[:, agent_id],
                                        rnn_states_joint=rnn_states_joint[:, agent_id],
                                        thresholds=thresholds,
                                        bias=bias[:, agent_id],
                                        logits=logits[:, agent_id]
                                        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_rnn_states_collector = []

            _, hard_actions, eval_rnn_states = self.collect_eval(0, eval_obs, eval_rnn_states, eval_masks)

            eval_actions = hard_actions

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_max_episode_rewards': [np.max(eval_episode_rewards)]}
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                break