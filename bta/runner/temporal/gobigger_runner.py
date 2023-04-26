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
from bta.utils.util import update_linear_schedule, is_acyclic, pruning, default_collate_with_dim, generate_mask_from_order
from bta.utils.separated_buffer import SeparatedReplayBufferEval
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

            self.temperature = max(self.all_args.temperature - (self.all_args.temperature * (episode / float(episodes))), 1.0)
            self.agent_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            for step in range(self.episode_length):
                # Sample actions
                values, actions, hard_actions, action_log_probs, rnn_states, \
                    rnn_states_critic, execution_masks, neighbors_agents, edges_agents, edges_agents_ts, adjs = self.collect(step)
                
                env_actions = []
                trans_actions = self.transform_action(hard_actions)
                for env_idx in range(self.n_rollout_threads):
                    env_action = {i: [trans_actions[env_idx][i][0], trans_actions[env_idx][i][1], trans_actions[env_idx][i][2]] for i in range(self.num_agents)} 
                    env_action.update({bot.name: bot.step(obs_raw[env_idx][1][bot.name]) for bot in bot_agents[env_idx]})
                    env_actions.append(env_action)
                
                # Obser reward and next obs
                share_obs, obs, obs_raw, rewards, dones, infos = self.envs.step(env_actions)
                share_obs = np.stack([default_collate_with_dim(share_obs[env_idx], device=self.device) for env_idx in range(self.n_rollout_threads)])
                total_num_steps += (self.n_rollout_threads)
                data = obs_raw, obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
                    rnn_states, rnn_states_critic, execution_masks, neighbors_agents, edges_agents, edges_agents_ts, adjs
                
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
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        share_obs, obs, obs_raw = self.envs.reset()
        
        share_obs = np.stack([default_collate_with_dim(share_obs[env_idx], device=self.device) for env_idx in range(self.n_rollout_threads)])    

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = share_obs.copy()
        
        return obs_raw

    def collect(self, step):
        # with torch.autograd.set_detect_anomaly(True):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_dim))
        if self.mix_actions:
            hard_actions = np.zeros((self.n_rollout_threads, self.num_agents, self.continous_dim + 1))
        else:
            hard_actions = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        new_dist_entropies = np.zeros((self.n_rollout_threads, self.num_agents))
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))

        # for agent_id in range(self.num_agents):
        #     self.trainer[agent_id].prep_rollout()
        if self.use_graph:
            edge_feats = torch.stack([check(self.buffer[agent_idx].temporal_neighbors_edge_feat) for agent_idx in range(self.num_agents)], 1)
            edts = torch.stack([check(self.buffer[agent_idx].temporal_neighbors_edge_timestamps) for agent_idx in range(self.num_agents)], 1)
            non_act_edge_mask = edts >= 0
            assert not torch.isinf(edts).any(), "InF!!"

            temporal_inputs = [
                edge_feats.to(self.device), 
                edts.to(self.device), 
                non_act_edge_mask.to(self.device)
            ]

            raw_actions_results = [self.trainer[agent_idx].policy.get_raw_actions(
                                                                self.buffer[agent_idx].share_obs[step],
                                                                self.buffer[agent_idx].obs[step],
                                                                self.buffer[agent_idx].rnn_states[step],
                                                                self.buffer[agent_idx].rnn_states_critic[step],
                                                                self.buffer[agent_idx].masks[step],
                                                                self.buffer[agent_idx].available_actions[step])
                                            for agent_idx in range(self.num_agents)]
            node_feats, old_dist_entropy = zip(*raw_actions_results)
            node_feats, old_dist_entropy = torch.stack(node_feats, 1), _t2n(torch.stack(old_dist_entropy, 1))
            # node_feats = node_feats.permute(1, 0, 2)
            # old_dist_entropy = old_dist_entropy.permute(1, 0, 2)

            agent_id_feat = torch.eye(self.num_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1).to(self.device)
            node_feats = torch.cat([node_feats, agent_id_feat], -1) 

            temporal_neighbors = torch.stack([check(self.buffer[agent_idx].temporal_neighbors) for agent_idx in range(self.num_agents)], 1).to(self.device)
            neighbors_feats_mask = temporal_neighbors >= 0
            temporal_neighbors = temporal_neighbors * neighbors_feats_mask
            neighbors_feats = node_feats.view(-1, self.all_args.node_feat_size).index_select(0, temporal_neighbors.view(-1)) \
                                    .view(self.n_rollout_threads, self.num_agents, self.all_args.time_gap, self.all_args.node_feat_size)
            neighbors_feats = neighbors_feats * neighbors_feats_mask.unsqueeze(-1)
            node_feats = torch.mean(neighbors_feats, dim=-2) + node_feats

            adjs = self.graph_policy(temporal_inputs, node_feats.detach())

            orignal_adjs = adjs.clone()

            adjs = torch.stack([adj if is_acyclic(adj) else pruning(adj) for adj in adjs])
            # print("adjs:", adjs, adjs.grad_fn)
            execution_masks = adjs.clone().permute(0, 2, 1).contiguous().view(-1, self.num_agents)
            Gs = [ig.Graph.Weighted_Adjacency(adj.tolist()) for adj in adjs]
            ordered_vertices = np.stack([G.topological_sorting() for G in Gs])
        else:
            ordered_vertices = _t2n(self.agent_order)
            # ordered_vertices = np.stack([[i for i in range(self.num_agents)] for _ in range(self.n_rollout_threads)])
            execution_masks = generate_mask_from_order(
            self.agent_order.clone(), ego_exclusive=False).to(
                self.device).float().view(-1, self.num_agents)  # [bs, n_agents, n_agents]
            # execution_masks = torch.stack([torch.stack([torch.ones(self.n_rollout_threads)] * order +
            #                                 [torch.zeros(self.n_rollout_threads)] *
            #                                 (self.num_agents - order), -1) for order in range(self.num_agents)], -2).to(self.device).view(-1, self.num_agents)
            orignal_adjs = execution_masks.view(self.n_rollout_threads, self.num_agents, self.num_agents).permute(0, 2, 1)

        rollout_actors = np.array([self.policy for _ in range(self.n_rollout_threads)])
        rollout_actors = np.take_along_axis(rollout_actors, ordered_vertices, axis=1)

        if self.mix_actions:
            hard_actions = hard_actions.reshape(-1, self.continous_dim + 1)
        else:
            hard_actions = hard_actions.reshape(-1, 1)
        actions = actions.reshape(-1, self.action_dim)
        action_log_probs = action_log_probs.reshape(-1, 1)
        new_dist_entropies = new_dist_entropies.reshape(-1)
        values = values.reshape(-1, 1)
        rnn_states = rnn_states.reshape(-1, self.recurrent_N, self.hidden_size)
        rnn_states_critic = rnn_states_critic.reshape(-1, self.recurrent_N, self.hidden_size)
        
        for order in range(self.num_agents):
            raw_agent_indices = ordered_vertices[:, order] + np.array([self.num_agents*i for i in range(self.n_rollout_threads)])
            # all_actions = actions.view(-1, self.num_agents)
            ego_inclusive_action = actions.reshape(-1, self.num_agents, self.action_dim)
            tmp_execution_mask = execution_masks.index_select(0, torch.from_numpy(raw_agent_indices).to(self.device))
            
            inputs = [(rollout_actors[num, order], 
                np.expand_dims(self.buffer[ordered_vertices[num, order]].share_obs[step][num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].obs[step][num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].rnn_states[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].rnn_states_critic[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].masks[step, num], axis=0),
                np.expand_dims(ego_inclusive_action[num], axis=0),
                tmp_execution_mask[num].unsqueeze(0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].available_actions[step, num], axis=0)) for num in range(self.n_rollout_threads)]
            results \
                = [policy.get_actions(share_obs, obs, rnn_states, rnn_states_critic, 
                                    masks, one_hot_action, exe_mask, available_actions) 
                                    for policy, share_obs, obs, rnn_states, rnn_states_critic, masks, one_hot_action, exe_mask, available_actions in inputs]
            value, action, action_log_prob, rnn_state, rnn_state_critic, _, new_dist_entropy = zip(*results)
            value, action, action_log_prob, rnn_state, rnn_state_critic, new_dist_entropy= \
                torch.stack(value).squeeze(1), torch.stack(action).squeeze(1), torch.stack(action_log_prob).squeeze(1), torch.stack(rnn_state).squeeze(1), torch.stack(rnn_state_critic).squeeze(1), torch.stack(new_dist_entropy).squeeze(1)
            
            if self.mix_actions:
                con_action = action[:, :2]
                dis_action = torch.argmax(action[:, 2:], -1).unsqueeze(1)
                hard_action = torch.cat([con_action, dis_action], -1)
            else:
                hard_action = torch.argmax(action, -1).unsqueeze(1).to(torch.int)

            hard_actions[raw_agent_indices, :] = _t2n(hard_action)
            actions[raw_agent_indices, :] = _t2n(action)
            action_log_probs[raw_agent_indices, :] = _t2n(action_log_prob)
            new_dist_entropies[raw_agent_indices] = _t2n(new_dist_entropy)
            values[raw_agent_indices, :] = _t2n(value)
            rnn_states[raw_agent_indices, :] = _t2n(rnn_state)
            rnn_states_critic[raw_agent_indices, :] = _t2n(rnn_state_critic)
        
        if self.mix_actions:
            hard_actions = hard_actions.reshape(-1, self.num_agents, self.continous_dim + 1)
        else:
            hard_actions = hard_actions.reshape(-1, self.num_agents, 1)
        actions = actions.reshape(-1, self.num_agents, self.action_dim)
        action_log_probs = action_log_probs.reshape(self.n_rollout_threads, self.num_agents, 1)
        new_dist_entropies = new_dist_entropies.reshape(self.n_rollout_threads, self.num_agents)
        values = values.reshape(-1, self.num_agents, 1)
        rnn_states = rnn_states.reshape(-1, self.num_agents, self.recurrent_N, self.hidden_size)
        rnn_states_critic = rnn_states_critic.reshape(-1, self.num_agents, self.recurrent_N, self.hidden_size)

        neighbors_agents = []
        edges_agents = []
        edges_agents_ts = []
        if self.use_graph:
            diff_dist_entropies = new_dist_entropies - old_dist_entropy
            
            direction_type = np.eye(2, dtype=np.float32)
            src_dst_id = np.eye(self.num_agents, dtype=np.float32)
            adjs_t = adjs.clone().permute(0, 2, 1)
            for agent_idx in range(self.num_agents):

                adjs_per_agent = adjs[:, agent_idx]
                adjs_t_per_agent = adjs_t[:, agent_idx]

                neighbors_per_agent = []
                edges_per_agent = []
                edges_per_agent_ts = []

                for batch_idx in range(adjs_per_agent.shape[0]):
                    tmp_neighbors = []
                    tmp_edge_feat = []
                    tmp_edges_ts = []

                    adj_ = adjs_per_agent[batch_idx]
                    neighbors = adj_.nonzero(as_tuple=True)[0].tolist()
                    diff_dist_entropies_ = diff_dist_entropies[batch_idx]
                    tmp_neighbors.extend(neighbors)

                    for idx in range(len(neighbors)):
                        tmp_edge_feat.append(np.concatenate([np.expand_dims(diff_dist_entropies_[neighbors[idx]], 0), src_dst_id[agent_idx], src_dst_id[neighbors[idx]], direction_type[0]], -1))
                        tmp_edges_ts.append(int(step))

                    adj_t = adjs_t_per_agent[batch_idx]
                    neighbors_t = adj_t.nonzero(as_tuple=True)[0].tolist()
                    tmp_neighbors.extend(neighbors_t)

                    for idx in range(len(neighbors_t)):
                        tmp_edge_feat.append(np.concatenate([np.expand_dims(diff_dist_entropies_[neighbors_t[idx]], 0), src_dst_id[neighbors_t[idx]], src_dst_id[agent_idx], direction_type[1]], -1))
                        tmp_edges_ts.append(int(step))

                    neighbors_per_agent.append(tmp_neighbors)
                    edges_per_agent.append(tmp_edge_feat)
                    edges_per_agent_ts.append(tmp_edges_ts)

                neighbors_agents.append(neighbors_per_agent)
                edges_agents.append(edges_per_agent)
                edges_agents_ts.append(edges_per_agent_ts)

        return values, actions, hard_actions, action_log_probs, rnn_states, rnn_states_critic, \
                    execution_masks.view(self.n_rollout_threads, self.num_agents, self.num_agents), neighbors_agents, edges_agents, edges_agents_ts, orignal_adjs
    
    def collect_eval(self, step, eval_obs, eval_rnn_states_old, eval_masks):
        # with torch.autograd.set_detect_anomaly(True):
        actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim))
        if self.mix_actions:
            hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.continous_dim + 1))
        else:
            hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1))
        # hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int32)
        new_dist_entropies = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        # for agent_id in range(self.num_agents):
        #     self.trainer[agent_id].prep_rollout()
        if self.use_graph:
            edge_feats = torch.stack([check(self.eval_buffer[agent_idx].temporal_neighbors_edge_feat) for agent_idx in range(self.num_agents)], 1)
            edts = torch.stack([check(self.eval_buffer[agent_idx].temporal_neighbors_edge_timestamps) for agent_idx in range(self.num_agents)], 1)
            non_act_edge_mask = edts >= 0
            assert not torch.isinf(edts).any(), "InF!!"

            temporal_inputs = [
                edge_feats.to(self.device), 
                edts.to(self.device), 
                non_act_edge_mask.to(self.device)
            ]

            raw_actions_results = [self.trainer[agent_idx].policy.get_raw_actions(
                                                                None,
                                                                eval_obs,
                                                                eval_rnn_states_old[:, agent_idx],
                                                                None,
                                                                eval_masks[:, agent_idx],
                                                                deterministic=True)
                                            for agent_idx in range(self.num_agents)]
            node_feats, old_dist_entropy = zip(*raw_actions_results)
            node_feats, old_dist_entropy = torch.stack(node_feats, 1), _t2n(torch.stack(old_dist_entropy, 1))
            # node_feats = node_feats.permute(1, 0, 2)
            # old_dist_entropy = old_dist_entropy.permute(1, 0, 2)

            agent_id_feat = torch.eye(self.num_agents).unsqueeze(0).repeat(self.n_eval_rollout_threads, 1, 1).to(self.device)
            node_feats = torch.cat([node_feats, agent_id_feat], -1) 

            temporal_neighbors = torch.stack([check(self.eval_buffer[agent_idx].temporal_neighbors) for agent_idx in range(self.num_agents)], 1).to(self.device)
            neighbors_feats_mask = temporal_neighbors >= 0
            temporal_neighbors = temporal_neighbors * neighbors_feats_mask
            neighbors_feats = node_feats.view(-1, self.all_args.node_feat_size).index_select(0, temporal_neighbors.view(-1)) \
                                    .view(self.n_eval_rollout_threads, self.num_agents, self.all_args.time_gap, self.all_args.node_feat_size)
            neighbors_feats = neighbors_feats * neighbors_feats_mask.unsqueeze(-1)
            node_feats = torch.mean(neighbors_feats, dim=-2) + node_feats

            adjs = self.graph_policy(temporal_inputs, node_feats.detach())

            orignal_adjs = adjs.clone()

            adjs = torch.stack([adj if is_acyclic(adj) else pruning(adj) for adj in adjs])
            # print("adjs:", adjs, adjs.grad_fn)

            Gs = [ig.Graph.Weighted_Adjacency(adj.tolist()) for adj in adjs]
            ordered_vertices = np.stack([G.topological_sorting() for G in Gs])
            execution_masks = adjs.clone().permute(0, 2, 1).contiguous().view(-1, self.num_agents)
        else:
            eval_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_eval_rollout_threads, 1).to(self.device)
            ordered_vertices = _t2n(eval_order)
            execution_masks = generate_mask_from_order(
            eval_order, ego_exclusive=False).to(
                self.device).float().view(-1, self.num_agents)
            # ordered_vertices = np.stack([[i for i in range(self.num_agents)] for _ in range(self.n_eval_rollout_threads)])
            # execution_masks = torch.stack([torch.stack([torch.ones(self.n_eval_rollout_threads)] * order +
            #                                 [torch.zeros(self.n_eval_rollout_threads)] *
            #                                 (self.num_agents - order), -1) for order in range(self.num_agents)], -2).to(self.device).view(-1, self.num_agents)

        rollout_actors = np.array([self.policy for _ in range(self.n_eval_rollout_threads)])
        rollout_actors = np.take_along_axis(rollout_actors, ordered_vertices, axis=1)

        if self.mix_actions:
            hard_actions = hard_actions.reshape(-1, self.continous_dim + 1)
        else:
            hard_actions = hard_actions.reshape(-1, 1)
        actions = actions.reshape(-1, self.action_dim)
        new_dist_entropies = new_dist_entropies.reshape(-1)
        eval_rnn_states = eval_rnn_states_old.reshape(-1, self.recurrent_N, self.hidden_size)
        
        for order in range(self.num_agents):
            raw_agent_indices = ordered_vertices[:, order] + np.array([self.num_agents*i for i in range(self.n_eval_rollout_threads)])
            # all_actions = actions.view(-1, self.num_agents)
            ego_inclusive_action = actions.reshape(-1, self.num_agents, self.action_dim)
            tmp_execution_mask = execution_masks.index_select(0, torch.from_numpy(raw_agent_indices).to(self.device))

            inputs = [(rollout_actors[num, order], 
                np.expand_dims(eval_obs[num], axis=0),
                np.expand_dims(eval_rnn_states_old[num,ordered_vertices[num, order]], axis=0),
                np.expand_dims(eval_masks[num,ordered_vertices[num, order]], axis=0),
                np.expand_dims(ego_inclusive_action[num], axis=0),
                tmp_execution_mask[num].unsqueeze(0)) for num in range(self.n_eval_rollout_threads)]
            results \
                = [policy.act(obs, rnn_states, 
                                    masks, one_hot_action, exe_mask, deterministic=True) 
                                    for policy, obs, rnn_states, masks, one_hot_action, exe_mask in inputs]
            action, rnn_state = zip(*results)
            action, rnn_state= \
                torch.stack(action).squeeze(1), torch.stack(rnn_state).squeeze(1)

            if self.mix_actions:
                con_action = action[:, :2]
                dis_action = action[:, 2:].to(torch.int)
                hard_action = torch.cat([con_action, dis_action], -1)
                action = torch.cat([con_action, F.one_hot(action[:, 2:].long(), self.discrete_dim).squeeze(1)], -1)
            else:
                hard_action = action.to(torch.int)
                action = F.one_hot(action, self.action_dim).squeeze(1)
            hard_actions[raw_agent_indices, :] = _t2n(hard_action)
            actions[raw_agent_indices, :] = _t2n(action)
            eval_rnn_states[raw_agent_indices, :] = _t2n(rnn_state)

        if self.mix_actions:
            hard_actions = hard_actions.reshape(-1, self.num_agents, self.continous_dim + 1)
        else:
            hard_actions = hard_actions.reshape(-1, self.num_agents, 1)
        actions = actions.reshape(-1, self.num_agents, self.action_dim)
        new_dist_entropies = new_dist_entropies.reshape(self.n_eval_rollout_threads, self.num_agents)
        eval_rnn_states = eval_rnn_states.reshape(-1, self.num_agents, self.recurrent_N, self.hidden_size)

        neighbors_agents = []
        edges_agents = []
        edges_agents_ts = []
        if self.use_graph:
            diff_dist_entropies = new_dist_entropies - old_dist_entropy
            direction_type = np.eye(2, dtype=np.float32)
            src_dst_id = np.eye(self.num_agents, dtype=np.float32)
            adjs_t = adjs.clone().permute(0, 2, 1)
            for agent_idx in range(self.num_agents):

                adjs_per_agent = adjs[:, agent_idx]
                adjs_t_per_agent = adjs_t[:, agent_idx]

                neighbors_per_agent = []
                edges_per_agent = []
                edges_per_agent_ts = []

                for batch_idx in range(adjs_per_agent.shape[0]):
                    tmp_neighbors = []
                    tmp_edge_feat = []
                    tmp_edges_ts = []

                    adj_ = adjs_per_agent[batch_idx]
                    neighbors = adj_.nonzero(as_tuple=True)[0].tolist()
                    diff_dist_entropies_ = diff_dist_entropies[batch_idx]
                    tmp_neighbors.extend(neighbors)

                    for idx in range(len(neighbors)):
                        tmp_edge_feat.append(np.concatenate([np.expand_dims(diff_dist_entropies_[neighbors[idx]], 0), src_dst_id[agent_idx], src_dst_id[neighbors[idx]], direction_type[0]], -1))
                        tmp_edges_ts.append(int(step))

                    adj_t = adjs_t_per_agent[batch_idx]
                    neighbors_t = adj_t.nonzero(as_tuple=True)[0].tolist()
                    tmp_neighbors.extend(neighbors_t)

                    for idx in range(len(neighbors_t)):
                        tmp_edge_feat.append(np.concatenate([np.expand_dims(diff_dist_entropies_[neighbors_t[idx]], 0), src_dst_id[neighbors_t[idx]], src_dst_id[agent_idx], direction_type[1]], -1))
                        tmp_edges_ts.append(int(step))

                    neighbors_per_agent.append(tmp_neighbors)
                    edges_per_agent.append(tmp_edge_feat)
                    edges_per_agent_ts.append(tmp_edges_ts)

                neighbors_agents.append(neighbors_per_agent)
                edges_agents.append(edges_per_agent)
                edges_agents_ts.append(edges_per_agent_ts)

        return actions, hard_actions, eval_rnn_states, neighbors_agents, edges_agents, edges_agents_ts

    def insert(self, data):
        obs_raw, obs, share_obs, rewards, dones_env, infos, values, actions, hard_actions, action_log_probs, \
            rnn_states, rnn_states_critic, execution_masks, neighbors_agents, edges_agents, edges_agents_ts, adjs = data
        
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
                                        hard_actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        np.expand_dims(rewards[:, agent_id], -1),
                                        masks[:, agent_id],
                                        execution_masks[:, agent_id],
                                        neighbors_agents[agent_id] if self.use_graph else None,
                                        edges_agents[agent_id] if self.use_graph else None,
                                        edges_agents_ts[agent_id] if self.use_graph else None,
                                        adjs if self.use_graph else None,
                                        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_env_infos = defaultdict(list)
        eval_share_obs, eval_obs, eval_obs_raw = self.eval_envs.reset()
        eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.eval_buffer = []
        for agent_id in range(self.num_agents):
            bu = SeparatedReplayBufferEval(self.all_args)
            self.eval_buffer.append(bu)
        
        bot_agents = []
        for env_idx in range(self.n_eval_rollout_threads):
            bot_agent = []
            for player in range(self.all_args.player_num_per_team, self.all_args.team_num * self.all_args.player_num_per_team):
                bot_agent.append(BotAgent(player))
            bot_agents.append(bot_agent)
        
        eval_average_episode_rewards = []
        for eval_step in range(self.episode_length):
            _, hard_actions, eval_rnn_states, neighbors_agents, edges_agents, edges_agents_ts = self.collect_eval(eval_step, eval_share_obs, eval_rnn_states, eval_masks)
            for agent_id in range(self.num_agents):
                self.eval_buffer[agent_id].insert(
                                            eval_masks[:, agent_id],
                                            neighbors_agents[agent_id] if self.use_graph else None,
                                            edges_agents[agent_id] if self.use_graph else None,
                                            edges_agents_ts[agent_id] if self.use_graph else None,
                                            )

            env_actions = []
            trans_actions = self.transform_action(hard_actions)
            for env_idx in range(self.n_eval_rollout_threads):
                env_action = {i: [trans_actions[env_idx][i][0], trans_actions[env_idx][i][1], trans_actions[env_idx][i][2]] for i in range(self.num_agents)} 
                env_action.update({bot.name: bot.step(eval_obs_raw[env_idx][1][bot.name]) for bot in bot_agents[env_idx]})
                env_actions.append(env_action)
            
            # Obser reward and next obs
            eval_share_obs, eval_obs, eval_obs_raw, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(env_actions)
            eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    
            # share_obs = np.stack([default_collate_with_dim(share_obs[env_idx], device=self.device) for env_idx in range(self.n_rollout_threads)])

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