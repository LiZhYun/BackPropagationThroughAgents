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
            
            self.temperature = max(self.all_args.temperature - (self.all_args.temperature * (episode / float(episodes))), 1.0)
            self.agent_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            for step in range(self.episode_length):
                # Sample actions
                values, actions, hard_actions, action_log_probs, rnn_states, \
                    rnn_states_critic, execution_masks, neighbors_agents, edges_agents, edges_agents_ts, adjs = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(hard_actions)
                obs = np.stack(obs)
                total_num_steps += (self.n_rollout_threads)
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
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

    def collect(self, step):
        # with torch.autograd.set_detect_anomaly(True):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_dim))
        hard_actions = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
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
                                                                self.buffer[agent_idx].masks[step])
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

            adjs = self.graph_policy(temporal_inputs, node_feats.detach(), tau=self.temperature)

            orignal_adjs = adjs.clone()

            adjs = torch.stack([adj if is_acyclic(adj) else pruning(adj) for adj in adjs])
            # print("adjs:", adjs, adjs.grad_fn)
            Gs = [ig.Graph.Weighted_Adjacency(adj.tolist()) for adj in adjs]
            ordered_vertices = np.stack([G.topological_sorting() for G in Gs])
            execution_masks = adjs.clone().permute(0, 2, 1).contiguous().view(-1, self.num_agents)
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

        hard_actions = hard_actions.reshape(-1, *hard_actions.shape[2:])
        actions = actions.reshape(-1, self.action_dim)
        action_log_probs = action_log_probs.reshape(-1, *action_log_probs.shape[2:])
        new_dist_entropies = new_dist_entropies.reshape(-1)
        values = values.reshape(-1, *values.shape[2:])
        rnn_states = rnn_states.reshape(-1, *rnn_states.shape[2:])
        rnn_states_critic = rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:])
        
        for order in range(self.num_agents):
            raw_agent_indices = ordered_vertices[:, order] + np.array([self.num_agents*i for i in range(self.n_rollout_threads)])
            ego_inclusive_action = actions.reshape(-1, self.num_agents, self.action_dim)
            tmp_execution_mask = execution_masks.index_select(0, torch.from_numpy(raw_agent_indices).to(self.device))
            # #test
            # tmp_execution_mask = torch.stack([torch.ones(self.n_rollout_threads)] * order +
            #                             [torch.zeros(self.n_rollout_threads)] *
            #                             (self.num_agents - order), -1).to(self.device)
            
            inputs = [(rollout_actors[num, order], 
                np.expand_dims(self.buffer[ordered_vertices[num, order]].share_obs[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].obs[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].rnn_states[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].rnn_states_critic[step, num], axis=0),
                np.expand_dims(self.buffer[ordered_vertices[num, order]].masks[step, num], axis=0),
                np.expand_dims(ego_inclusive_action[num], axis=0),
                tmp_execution_mask[num].unsqueeze(0)) for num in range(self.n_rollout_threads)]
            results \
                = [policy.get_actions(share_ob, ob, rnn_state, rnn_state_critic, 
                                    mask, one_hot_action, exe_mask) 
                                    for policy, share_ob, ob, rnn_state, rnn_state_critic, mask, one_hot_action, exe_mask in inputs]
            value, action, action_log_prob, rnn_state, rnn_state_critic, _, new_dist_entropy = zip(*results)
            value, action, action_log_prob, rnn_state, rnn_state_critic, new_dist_entropy= \
                torch.stack(value).squeeze(1), torch.stack(action).squeeze(1), torch.stack(action_log_prob).squeeze(1), torch.stack(rnn_state).squeeze(1), torch.stack(rnn_state_critic).squeeze(1), torch.stack(new_dist_entropy).squeeze(1)
            
            hard_action = torch.argmax(action, -1).unsqueeze(1).to(torch.int)

            hard_actions[raw_agent_indices, :] = _t2n(hard_action)
            actions[raw_agent_indices, :] = _t2n(action)
            action_log_probs[raw_agent_indices, :] = _t2n(action_log_prob)
            new_dist_entropies[raw_agent_indices] = _t2n(new_dist_entropy)
            values[raw_agent_indices, :] = _t2n(value)
            rnn_states[raw_agent_indices, :] = _t2n(rnn_state)
            rnn_states_critic[raw_agent_indices, :] = _t2n(rnn_state_critic)

        hard_actions = hard_actions.reshape(self.n_rollout_threads, self.num_agents, 1)
        actions = actions.reshape(self.n_rollout_threads, self.num_agents, self.action_dim)
        action_log_probs = action_log_probs.reshape(self.n_rollout_threads, self.num_agents, 1)
        new_dist_entropies = new_dist_entropies.reshape(self.n_rollout_threads, self.num_agents)
        values = values.reshape(self.n_rollout_threads, self.num_agents, 1)
        rnn_states = rnn_states.reshape(self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size)
        rnn_states_critic = rnn_states_critic.reshape(self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size)

        self.test_rnn = rnn_states_critic.copy()

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
        actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim))
        hard_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int32)
        new_dist_entropies = np.zeros((self.n_eval_rollout_threads, self.num_agents))

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
                                                                eval_obs[:, agent_idx],
                                                                eval_rnn_states_old[:, agent_idx],
                                                                None,
                                                                eval_masks[:, agent_idx],
                                                                deterministic=True)
                                            for agent_idx in range(self.num_agents)]
            node_feats, old_dist_entropy = zip(*raw_actions_results)
            node_feats, old_dist_entropy = torch.stack(node_feats, 1), _t2n(torch.stack(old_dist_entropy, 1))

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
            #test
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

        hard_actions = hard_actions.reshape(-1, 1)
        actions = actions.reshape(-1, self.action_dim)
        new_dist_entropies = new_dist_entropies.reshape(-1)
        eval_rnn_states = eval_rnn_states_old.reshape(-1, self.recurrent_N, self.hidden_size)
        
        for order in range(self.num_agents):
            raw_agent_indices = ordered_vertices[:, order] + np.array([self.num_agents*i for i in range(self.n_eval_rollout_threads)])
            ego_inclusive_action = actions.reshape(-1, self.num_agents, self.action_dim)
            tmp_execution_mask = execution_masks.index_select(0, torch.from_numpy(raw_agent_indices).to(self.device))
            # #test
            # tmp_execution_mask = torch.stack([torch.ones(self.n_eval_rollout_threads)] * order +
            #                             [torch.zeros(self.n_eval_rollout_threads)] *
            #                             (self.num_agents - order), -1).to(self.device)
            inputs = [(rollout_actors[num, order], 
                np.expand_dims(eval_obs[num,ordered_vertices[num, order]], axis=0),
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
            
            hard_action = action.to(torch.int)
            action = F.one_hot(action.long(), self.action_dim).squeeze(1)

            hard_actions[raw_agent_indices, :] = _t2n(hard_action)
            actions[raw_agent_indices, :] = _t2n(action.clone())
            eval_rnn_states[raw_agent_indices, :] = _t2n(rnn_state)

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
        obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
            rnn_states, rnn_states_critic, execution_masks, neighbors_agents, edges_agents, edges_agents_ts, adjs = data

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
                                        hard_actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
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
        eval_obs, _, _ = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.eval_buffer = []
        for agent_id in range(self.num_agents):
            bu = SeparatedReplayBufferEval(self.all_args)
            self.eval_buffer.append(bu)
        
        eval_average_episode_rewards = []
        for eval_step in range(self.episode_length):
            _, hard_actions, eval_rnn_states, neighbors_agents, edges_agents, edges_agents_ts = self.collect_eval(eval_step, eval_obs, eval_rnn_states, eval_masks)
            for agent_id in range(self.num_agents):
                self.eval_buffer[agent_id].insert(
                                            eval_masks[:, agent_id],
                                            neighbors_agents[agent_id] if self.use_graph else None,
                                            edges_agents[agent_id] if self.use_graph else None,
                                            edges_agents_ts[agent_id] if self.use_graph else None,
                                            )
            eval_actions = hard_actions
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_average_episode_rewards.append(eval_rewards)
            
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        for eval_info in eval_infos:
            eval_env_infos['eval_ep_sparse_r_by_agent0'].append(eval_info['episode']['ep_sparse_r_by_agent'][0])
            eval_env_infos['eval_ep_sparse_r_by_agent1'].append(eval_info['episode']['ep_sparse_r_by_agent'][1])
            eval_env_infos['eval_ep_shaped_r_by_agent0'].append(eval_info['episode']['ep_shaped_r_by_agent'][0])
            eval_env_infos['eval_ep_shaped_r_by_agent1'].append(eval_info['episode']['ep_shaped_r_by_agent'][1])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])

        eval_env_infos['eval_average_episode_rewards'] = np.sum(eval_average_episode_rewards, axis=0)
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
            