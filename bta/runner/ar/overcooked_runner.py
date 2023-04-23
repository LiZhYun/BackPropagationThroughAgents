import time
import wandb
import copy
import os
import numpy as np
import itertools
from itertools import chain
import torch
import torch.nn.functional as F
import imageio
import warnings
import functools
from bta.utils.util import update_linear_schedule, is_acyclic, pruning
from bta.runner.ar.base_runner import Runner
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict
from icecream import ic
from scipy.stats import rankdata
import igraph as ig
from bta.algorithms.utils.util import check


def _t2n(x):
    return x.detach().cpu().numpy()

class OvercookedRunner(Runner):
    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += (self.n_rollout_threads)
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic, available_actions
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.layout_name,
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
                    print("average episode rewards agent {} is {}".format(a, train_infos[a]["average_episode_rewards_per_agent"]))
                total_mean /= self.num_agents
                print("average episode rewards for team is {}".format(total_mean))
                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = total_mean
                
                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos['ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                            env_infos['ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                            env_infos['ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                            env_infos['ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                            env_infos[f'ep_sparse_r_by_agent{a}'].append(info['episode']['ep_sparse_r_by_agent'][a])
                            env_infos[f'ep_shaped_r_by_agent{a}'].append(info['episode']['ep_shaped_r_by_agent'][a])

                        env_infos['ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        env_infos['ep_shaped_r'].append(info['episode']['ep_shaped_r'])
                        
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        if not self.use_centralized_V:  
            share_obs = obs      

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        # with torch.autograd.set_detect_anomaly(True):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_shape).to(dtype=torch.int ,device=self.device)
        action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        
        for agent_idx in range(self.num_agents):
            self.trainer[agent_idx].prep_rollout()
            # construct ego-exclusive one-hot actions based on current available actions
            ego_exclusive_action = torch.cat(
                [actions[:, :agent_idx], actions[:, agent_idx + 1:]],
                -2).squeeze(-1)
            execution_mask = torch.stack([torch.ones(self.n_rollout_threads)] * agent_idx +
                                            [torch.zeros(self.n_rollout_threads)] *
                                            (self.num_agents - 1 - agent_idx), -1).to(self.device)
            
            onehot_action = F.one_hot(ego_exclusive_action.long(), self.action_dim).float()

            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_idx].policy.get_actions(self.buffer[agent_idx].share_obs[step],
                                                            self.buffer[agent_idx].obs[step],
                                                            self.buffer[agent_idx].rnn_states[step],
                                                            self.buffer[agent_idx].rnn_states_critic[step],
                                                            self.buffer[agent_idx].masks[step],
                                                            onehot_action,
                                                            execution_mask)
            actions[:, agent_idx] = action.clone()
            action_log_probs[:, agent_idx] = _t2n(action_log_prob)
            values[:, agent_idx] = _t2n(value)
            rnn_states[:, agent_idx] = _t2n(rnn_state)
            rnn_states_critic[:, agent_idx] = _t2n(rnn_state_critic)
        actions = _t2n(actions)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic, available_actions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer[agent_id].insert(share_obs[:, agent_id],
                                        obs[:, agent_id],
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, :],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id],
                                        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        eval_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int32)
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                ego_exclusive_action = torch.cat(
                [check(eval_actions[:, :agent_id]), check(eval_actions[:, agent_id + 1:])],
                -2).squeeze(-1)
                execution_mask = torch.stack([torch.ones(self.n_eval_rollout_threads)] * agent_id +
                                                [torch.zeros(self.n_eval_rollout_threads)] *
                                                (self.num_agents - 1 - agent_id), -1).to(self.device)
                
                onehot_action = F.one_hot(ego_exclusive_action.long(), self.action_dim).float()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                onehot_action,
                                                                                execution_mask,
                                                                                deterministic=True)

                eval_actions[:, agent_id] = _t2n(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        for eval_info in eval_infos:
            for a in range(self.num_agents):
                # eval_env_infos[f'eval_ep_dish_r_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][0])
                # eval_env_infos[f'eval_ep_soup_r_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][1])
                # eval_env_infos[f'eval_ep_place_r_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][2])
                eval_env_infos[f'eval_ep_sparse_r_by_agent{a}'].append(eval_info['episode']['ep_sparse_r_by_agent'][a])
                eval_env_infos[f'eval_ep_shaped_r_by_agent{a}'].append(eval_info['episode']['ep_shaped_r_by_agent'][a])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])

        eval_env_infos['eval_average_episode_rewards'] = np.sum(eval_episode_rewards, axis=0)
        print("eval average sparse rewards: " + str(np.mean(eval_env_infos['eval_ep_sparse_r'])))
        
        self.log_env(eval_env_infos, total_num_steps)
        
    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)     

        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            trajectory = []

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                actions = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(np.array(obs)[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(obs)[:, agent_id],
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    actions.append(action[0])
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # Obser reward and next obs
                print("action:",actions)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step([actions])
                obs = np.stack(obs) 
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            for info in infos:
                for a in range(self.num_agents):
                    ic(info['episode']['ep_sparse_r_by_agent'][a])
                    ic(info['episode']['ep_shaped_r_by_agent'][a])
                ic(info['episode']['ep_sparse_r'])
                ic(info['episode']['ep_shaped_r'])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            #print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

    # def save(self, step):
    #     for agent_id in range(self.num_agents):
    #         if self.use_single_network:
    #             policy_model = self.trainer[agent_id].policy.model
    #             torch.save(policy_model.state_dict(), str(self.save_dir) + f"/model_agent{agent_id}_periodic_{step}.pt")
    #         else:
    #             policy_actor = self.trainer[agent_id].policy.actor
    #             torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_agent{agent_id}_periodic_{step}.pt")
    #             policy_critic = self.trainer[agent_id].policy.critic
    #             torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_agent{agent_id}_periodic_{step}.pt")