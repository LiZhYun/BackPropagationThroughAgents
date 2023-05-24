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
from bta.utils.util import update_linear_schedule, is_acyclic, pruning, default_collate_with_dim
from bta.runner.ar.base_runner import Runner
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict
from icecream import ic
from scipy.stats import rankdata
from bta.algorithms.utils.util import check
from gobigger.agents import BotAgent
import math 


def _t2n(x):
    return x.detach().cpu().numpy()

class GoBiggerRunner(Runner):
    def __init__(self, config):
        super(GoBiggerRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        self.setup_action()
       
    def run(self):
        obs_raw = self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        bot_agents = []
        for env_idx in range(self.n_rollout_threads):
            bot_agent = []
            for player in range(self.all_args.player_num_per_team, self.all_args.team_num * self.all_args.player_num_per_team):
                bot_agent.append(BotAgent(player))
            bot_agents.append(bot_agent)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                env_actions = []
                trans_actions = self.transform_action(actions)
                for env_idx in range(self.n_rollout_threads):
                    env_action = {i: [trans_actions[env_idx][i][0], trans_actions[env_idx][i][1], trans_actions[env_idx][i][2]] for i in range(self.num_agents)} 
                    env_action.update({bot.name: bot.step(obs_raw[env_idx][1][bot.name]) for bot in bot_agents[env_idx]})
                    env_actions.append(env_action)
                
                # Obser reward and next obs
                share_obs, obs, obs_raw, rewards, dones, infos = self.envs.step(env_actions)
                share_obs = np.stack([default_collate_with_dim(share_obs[env_idx], device=self.device) for env_idx in range(self.n_rollout_threads)])
                total_num_steps += (self.n_rollout_threads)
                data = obs_raw, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic
                
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
            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
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
                if len(self.env_infos["win_rate"]) > 0:
                    print("Win rate: ", sum(self.env_infos["win_rate"]) / len(self.env_infos["win_rate"]))
                    
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        share_obs, obs, obs_raw = self.envs.reset()
        
        share_obs = np.stack([default_collate_with_dim(share_obs[env_idx], device=self.device) for env_idx in range(self.n_rollout_threads)])     

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = share_obs.copy()
        
        return obs_raw

    @torch.no_grad()
    def collect(self, step):
        # with torch.autograd.set_detect_anomaly(True):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_shape).to(self.device)
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
            if self.mix_actions:
                discrete_ = F.one_hot(ego_exclusive_action[:, :, 2].long(), self.discrete_dim).float()
                # discrete_ = torch.eye(3).to(self.device)[ego_exclusive_action[:, :, 2].unsqueeze(0).to(torch.int)]
                onehot_action = torch.cat([ego_exclusive_action[:, :, :2], discrete_], -1)
            else:
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
        obs_raw, obs, share_obs, rewards, dones_env, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data
        
        if np.any(dones_env):
            for done, info, ob_raw in zip(dones_env, infos, obs_raw):
                if done:
                    sorted_leaderboard_ = sorted(ob_raw[0]["leaderboard"].items(), key=lambda item: item[1])
                    if sorted_leaderboard_[-1][0] == 0:
                        self.env_infos["win_rate"].append(1)
                    else:
                        self.env_infos["win_rate"].append(0)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer[agent_id].insert(share_obs,
                                        share_obs,
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, :],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        np.expand_dims(rewards[:, agent_id], -1),
                                        masks[:, agent_id],
                                        )
    def setup_action(self):
        theta = math.pi * 2 / self.all_args.direction_num
        self.x_y_action_List = [[0.3 * math.cos(theta * i), 0.3 * math.sin(theta * i), 0] for i in
                                range(self.all_args.direction_num)] + \
                               [[math.cos(theta * i), math.sin(theta * i), 0] for i in
                                range(self.all_args.direction_num)] + \
                               [[0, 0, 0], [0, 0, 1], [0, 0, 2]]

    def transform_action(self, agent_outputs):
        env_num = agent_outputs.shape[0]
        actions = {}
        for env_id in range(env_num):
            actions[env_id] = {}
            for game_player_id in range(self.all_args.player_num_per_team):
                action_idx = agent_outputs[env_id][game_player_id]
                actions[env_id][game_player_id] = self.x_y_action_List[int(action_idx)]
        return actions
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_env_infos = defaultdict(list)
        eval_share_obs, eval_obs, eval_obs_raw = self.eval_envs.reset()
        eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        bot_agents = []
        for env_idx in range(self.n_eval_rollout_threads):
            bot_agent = []
            for player in range(self.all_args.player_num_per_team, self.all_args.team_num * self.all_args.player_num_per_team):
                bot_agent.append(BotAgent(player))
            bot_agents.append(bot_agent)

        eval_average_episode_rewards = []

        # loop until enough episodes
        for eval_step in range(self.episode_length):
            # get actions
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
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(eval_share_obs,
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                onehot_action,
                                                                                execution_mask,
                                                                                deterministic=True)
                eval_actions[:, agent_id] = _t2n(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            env_actions = []
            trans_actions = self.transform_action(eval_actions)
            for env_idx in range(self.n_eval_rollout_threads):
                env_action = {i: [trans_actions[env_idx][i][0], trans_actions[env_idx][i][1], trans_actions[env_idx][i][2]] for i in range(self.num_agents)} 
                env_action.update({bot.name: bot.step(eval_obs_raw[env_idx][1][bot.name]) for bot in bot_agents[env_idx]})
                env_actions.append(env_action)

            # step
            eval_share_obs, eval_obs, eval_obs_raw, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(env_actions)
            eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    
            eval_average_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        for ob_raw in eval_obs_raw:
            sorted_leaderboard_ = sorted(ob_raw[0]["leaderboard"].items(), key=lambda item: item[1])
        if sorted_leaderboard_[-1][0] == 0:
            eval_env_infos['win_rate'].append(1)
        else:
            eval_env_infos['win_rate'].append(0)

        eval_env_infos['eval_average_episode_rewards'] = np.mean(np.sum(eval_average_episode_rewards, axis=0))
        print("eval average episode rewards: " + str(eval_env_infos['eval_average_episode_rewards']))
        eval_env_infos['win_rate'] = np.mean(eval_env_infos['win_rate'])
        print("eval win rate: " + str(eval_env_infos['win_rate']))
        
        if self.use_wandb:
            wandb.log({"eval_win_rate": eval_env_infos['win_rate']}, step=total_num_steps)
            wandb.log({"eval_average_episode_rewards": eval_env_infos['eval_average_episode_rewards']}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_env_infos['win_rate']}, total_num_steps)
            self.writter.add_scalars("eval_average_episode_rewards", {"eval_average_episode_rewards": eval_env_infos['eval_average_episode_rewards']}, total_num_steps)
            self.writter.add_scalars("eval_step", {"expected_step": eval_step}, total_num_steps)
        
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