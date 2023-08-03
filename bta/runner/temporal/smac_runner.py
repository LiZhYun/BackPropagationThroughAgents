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
from bta.algorithms.utils.util import check
from bta.algorithms.utils.distributions import FixedCategorical, FixedNormal
from functools import reduce

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            self.temperature = max(self.all_args.temperature - (self.all_args.temperature * (episode / float(episodes))), 1.0)
            self.agent_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            # self.agent_order = torch.randperm(self.num_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            for step in range(self.episode_length):
                # Sample actions
                values, actions, hard_actions, action_log_probs, rnn_states, \
                    rnn_states_critic, joint_actions, joint_action_log_probs = self.collect(step)
                    
                # Obser reward and next obs
                env_actions = joint_actions if joint_actions is not None else hard_actions
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)
                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, hard_actions, action_log_probs, \
                    rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            if self.use_action_attention:
                self.joint_compute()
            train_infos = self.joint_train() if self.use_action_attention else [self.train_seq_agent_m, self.train_seq_agent_a, self.train_sim_a][self.train_sim_seq]()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                if self.env_name == "StarCraft2" or self.env_name == "SMACv2" or self.env_name == "SMAC" or self.env_name == "StarCraft2v2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                
                for i in range(self.num_agents):
                    train_infos[i]['dead_ratio'] = 1 - self.buffer[i].active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer[i].active_masks.shape)) 
                    
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:  
            share_obs = obs      

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    def collect(self, step):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_dim))
        logits = torch.zeros(self.n_rollout_threads, self.num_agents, self.action_dim).to(self.device)
        obs_feats = torch.zeros(self.n_rollout_threads, self.num_agents, self.obs_emb_size).to(self.device)
        hard_actions = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
        action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
  
        ordered_vertices = [i for i in range(self.num_agents)]
        for idx, agent_idx in enumerate(ordered_vertices):
            self.trainer[agent_idx].prep_rollout()
            # ego_exclusive_action = actions.copy()
            ego_exclusive_action = actions[:,0:self.num_agents]
            # tmp_execution_mask = execution_masks[:, agent_idx]
            if self.use_action_attention:
                tmp_execution_mask = torch.stack([torch.zeros(self.n_rollout_threads)] * self.num_agents, -1).to(self.device)
            else:
                if self.skip_connect:
                    tmp_execution_mask = torch.stack([torch.ones(self.n_rollout_threads)] * agent_idx +
                                                    [torch.zeros(self.n_rollout_threads)] *
                                                    (self.num_agents - agent_idx), -1).to(self.device)
                else:
                    if idx != 0:
                        tmp_execution_mask = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(ordered_vertices[idx-1]), 1.0)\
                            .unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
                    else:
                        tmp_execution_mask = torch.stack([torch.zeros(self.n_rollout_threads)] * self.num_agents, -1).to(self.device)

            value, action, action_log_prob, rnn_state, rnn_state_critic, logit, obs_feat \
                = self.trainer[agent_idx].policy.get_actions(self.buffer[agent_idx].share_obs[step],
                                                            self.buffer[agent_idx].obs[step],
                                                            self.buffer[agent_idx].rnn_states[step],
                                                            self.buffer[agent_idx].rnn_states_critic[step],
                                                            self.buffer[agent_idx].masks[step],
                                                            ego_exclusive_action,
                                                            tmp_execution_mask,
                                                            available_actions=self.buffer[agent_idx].available_actions[step],
                                                            tau=self.temperature)
            hard_actions[:, agent_idx] = _t2n(torch.argmax(action, -1, keepdim=True).to(torch.int))
            actions[:, idx] = _t2n(action)
            logits[:, idx] = logit.clone()
            obs_feats[:, idx] = obs_feat.clone()
            action_log_probs[:, agent_idx] = _t2n(action_log_prob)
            values[:, agent_idx] = _t2n(value)
            rnn_states[:, agent_idx] = _t2n(rnn_state)
            rnn_states_critic[:, agent_idx] = _t2n(rnn_state_critic)

        joint_actions, joint_action_log_probs = None, None
        if self.use_action_attention:
            available_actions_all = np.stack([self.buffer[agent_idx].available_actions[step] for agent_idx in range(self.num_agents)],1)
            available_actions_all = check(available_actions_all).to(**self.tpdv)
            bias_, action_std = self.action_attention(logits, obs_feats)
            if self.discrete:
                mixed_ = logits+bias_
                mixed_[available_actions_all == 0] = -1e10
                joint_dist = FixedCategorical(logits=mixed_)
            else:
                action_mean = logits+bias_
                joint_dist = FixedNormal(action_mean, action_std)
            joint_actions = joint_dist.sample()
            joint_action_log_probs = joint_dist.log_probs_joint(joint_actions)
            joint_actions = _t2n(joint_actions)
            joint_action_log_probs = _t2n(joint_action_log_probs)
            # joint_values = _t2n(joint_values)
            for agent_idx in range(self.num_agents):
                ego_exclusive_action = actions[:,0:self.num_agents]
                tmp_execution_mask = torch.stack([torch.zeros(self.n_rollout_threads)] * self.num_agents, -1).to(self.device)
                _, action_log_prob, _, _, _, _ = self.trainer[agent_idx].policy.actor.evaluate_actions(
                    self.buffer[agent_idx].obs[step],
                    self.buffer[agent_idx].rnn_states[step],
                    joint_actions[:,agent_idx],
                    self.buffer[agent_idx].masks[step],
                    ego_exclusive_action,
                    tmp_execution_mask,
                    available_actions=self.buffer[agent_idx].available_actions[step],
                    tau=self.temperature
                )
                action_log_probs[:, agent_idx] = _t2n(action_log_prob)

        return values, actions, hard_actions, action_log_probs, rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs

    def collect_eval(self, step, eval_obs, eval_rnn_states, eval_masks, available_actions):
        actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim))
        hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int32)

        ordered_vertices = [i for i in range(self.num_agents)]
        for idx, agent_idx in enumerate(ordered_vertices):
            self.trainer[agent_idx].prep_rollout()
            # ego_exclusive_action = actions.copy()
            # tmp_execution_mask = execution_masks[:, agent_idx]
            ego_exclusive_action = actions[:,0:self.num_agents]
            # tmp_execution_mask = execution_masks[:, agent_idx]
            if self.skip_connect:
                tmp_execution_mask = torch.stack([torch.ones(self.n_eval_rollout_threads)] * agent_idx +
                                                [torch.zeros(self.n_eval_rollout_threads)] *
                                                (self.num_agents - agent_idx), -1).to(self.device)
            else:
                if idx != 0:
                    tmp_execution_mask = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(ordered_vertices[idx-1]), 1.0)\
                        .unsqueeze(0).repeat(self.n_eval_rollout_threads, 1).to(self.device)
                else:
                    tmp_execution_mask = torch.stack([torch.zeros(self.n_eval_rollout_threads)] * self.num_agents, -1).to(self.device)

            action, rnn_state \
                = self.trainer[agent_idx].policy.act(eval_obs[:, agent_idx],
                                                            eval_rnn_states[:, agent_idx],
                                                            eval_masks[:, agent_idx],
                                                            ego_exclusive_action,
                                                            tmp_execution_mask,
                                                            available_actions=available_actions[:, agent_idx],
                                                            deterministic=True)
            hard_actions[:, agent_idx] = _t2n(action.to(torch.int))
            actions[:, idx] = _t2n(F.one_hot(action.long(), self.action_dim).squeeze(1))
            eval_rnn_states[:, agent_idx] = _t2n(rnn_state)

        return actions, hard_actions, eval_rnn_states

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, hard_actions, action_log_probs, \
            rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs = data
        
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs

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
                                        bad_masks=bad_masks[:, agent_id],
                                        active_masks=active_masks[:, agent_id],
                                        available_actions=available_actions[:, agent_id],
                                        joint_actions=joint_actions,
                                        joint_action_log_probs=joint_action_log_probs,
                                        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            
            _, hard_actions, eval_rnn_states = self.collect_eval(0, eval_obs, eval_rnn_states, eval_masks, eval_available_actions)

            eval_actions = hard_actions
            # step
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
        
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
                    # self.trainer[agent_id].prep_rollout()
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