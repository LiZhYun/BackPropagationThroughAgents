from collections import defaultdict, deque
from itertools import chain
import os
import time
import numpy as np
from functools import reduce
import torch
import torch.nn.functional as F
import wandb
import imageio
from bta.runner.temporal.base_runner import Runner
from bta.utils.util import update_linear_schedule, is_acyclic, pruning, generate_mask_from_order
from bta.algorithms.utils.util import check
from bta.algorithms.utils.distributions import FixedCategorical, FixedNormal
import igraph as ig


def _t2n(x):
    return x.detach().cpu().numpy()

class MatrixRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MatrixRunner, self).__init__(config)
        self.env_infos = defaultdict(list)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            self.temperature = max(self.all_args.temperature - (self.all_args.temperature * (episode / float(episodes))), 1.0)
            # self.agent_order = torch.randperm(self.num_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)
            self.agent_order = torch.tensor([i for i in range(self.num_agents)]).unsqueeze(0).repeat(self.n_rollout_threads, 1).to(self.device)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, hard_actions, action_log_probs, rnn_states, \
                    rnn_states_critic, joint_actions, joint_action_log_probs, joint_values = self.collect(step)
                    
                # Obser reward and next obs
                env_actions = joint_actions if joint_actions is not None else hard_actions
                obs, rewards, dones, infos = self.envs.step(np.squeeze(env_actions, axis=-1))
                share_obs = obs.copy()
                data = obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
                    rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, joint_values
                # insert data into buffer
                self.insert(data)
                
            # compute return and update network
            self.compute()
            if self.use_action_attention:
                self.joint_compute()
            
            train_infos = self.joint_train() if self.use_action_attention else [self.train_seq_agent, self.train_seq_epoch, self.train_sim][self.train_sim_seq]()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                for agent_id in range(self.num_agents):
                    train_infos[agent_id].update({"average_episode_rewards_by_eplength": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                print("average episode rewards of agent 0 is {}".format(train_infos[0]["average_episode_rewards_by_eplength"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps=total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # replay buffer
        # in GRF, we have full observation, so mappo is just ippo
        share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

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
                                                            tau=self.temperature)
            hard_actions[:, agent_idx] = _t2n(torch.argmax(action, -1, keepdim=True).to(torch.int))
            actions[:, idx] = _t2n(action)
            logits[:, idx] = logit.clone()
            obs_feats[:, idx] = obs_feat.clone()
            action_log_probs[:, agent_idx] = _t2n(action_log_prob)
            values[:, agent_idx] = _t2n(value)
            rnn_states[:, agent_idx] = _t2n(rnn_state)
            rnn_states_critic[:, agent_idx] = _t2n(rnn_state_critic)

        joint_actions, joint_action_log_probs, joint_values = None, None, None
        if self.use_action_attention:
            bias_, joint_values = self.action_attention(logits, obs_feats, tau=self.temperature)
            if self.discrete:
                joint_dist = FixedCategorical(logits=logits+bias_)
            else:
                action_mean = logits+bias_
                action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
                joint_dist = FixedNormal(action_mean, action_std)
            joint_actions = joint_dist.sample()
            joint_action_log_probs = joint_dist.log_probs_joint(joint_actions)
            joint_actions = _t2n(joint_actions)
            joint_action_log_probs = _t2n(joint_action_log_probs)
            joint_values = _t2n(joint_values)
            for agent_idx in range(self.num_agents):
                ego_exclusive_action = actions
                tmp_execution_mask = torch.stack([torch.zeros(self.n_rollout_threads)] * self.num_agents, -1).to(self.device)
                _, action_log_prob, _, _, _, _ = self.trainer[agent_idx].policy.actor.evaluate_actions(
                    self.buffer[agent_idx].obs[step],
                    self.buffer[agent_idx].rnn_states[step],
                    joint_actions[:,agent_idx],
                    self.buffer[agent_idx].masks[step],
                    ego_exclusive_action,
                    tmp_execution_mask,
                    tau=self.temperature
                )
                action_log_probs[:, agent_idx] = _t2n(action_log_prob)

        return values, actions, hard_actions, action_log_probs, rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, joint_values

    def collect_eval(self, step, eval_obs, eval_rnn_states, eval_masks):
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
                                                            deterministic=True)
            hard_actions[:, agent_idx] = _t2n(action.to(torch.int))
            actions[:, idx] = _t2n(F.one_hot(action.long(), self.action_dim).squeeze(1))
            eval_rnn_states[:, agent_idx] = _t2n(rnn_state)

        return actions, hard_actions, eval_rnn_states

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, hard_actions, action_log_probs, \
            rnn_states, rnn_states_critic, joint_actions, joint_action_log_probs, joint_values = data
        
        dones_env = np.all(dones, axis=-1)
        
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

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
                                        joint_actions=joint_actions,
                                        joint_action_log_probs=joint_action_log_probs,
                                        joint_value_preds=joint_values
                                        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            _, hard_actions, eval_rnn_states = self.collect_eval(eval_step, eval_obs, eval_rnn_states, eval_masks)
            eval_actions = hard_actions
            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(np.squeeze(eval_actions, axis=-1))
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_average_episode_rewards=eval_average_episode_rewards/self.episode_length*self.envs.observation_space[0].high[0]
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[
                    0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                # self.trainer.prep_rollout()
                # render_actions, render_rnn_states = self.trainer.policy.act(
                #     np.concatenate(render_obs),
                #     np.concatenate(render_rnn_states),
                #     np.concatenate(render_masks),
                #     deterministic=True
                # )

                # # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                # render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                # render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                # render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                eval_actions_collector = []
                eval_rnn_states_collector = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(render_obs[:, agent_id],
                                                          render_rnn_states[:,
                                                                            agent_id],
                                                          render_masks[:,
                                                                       agent_id],
                                                          deterministic=True)
                    render_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                    render_actions = np.array(
                        eval_actions_collector).transpose(1, 0, 2)
                    # eval_actions shape will be (100,3,1): batch_size*number_agents*action_dim
                    render_actions_env = [render_actions[idx, :, 0]
                                          for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(
                    render_actions_env)

                # append frame
                if self.all_args.save_gifs:
                    image = render_infos[0]["frame"]
                    frames.append(image)

            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(
                i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir),
                                                  i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

        print("expected goal: {}".format(np.mean(render_goals)))
