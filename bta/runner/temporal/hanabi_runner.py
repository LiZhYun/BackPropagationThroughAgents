import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F

from bta.utils.util import update_linear_schedule, is_acyclic, pruning, generate_mask_from_order, flatten
from bta.runner.temporal.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class HanabiRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Hanabi. See parent class for details."""
    def __init__(self, config):
        super(HanabiRunner, self).__init__(config)
        self.true_total_num_steps = 0
    
    def run(self):
        self.turn_obs = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].obs.shape[2:]), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].share_obs.shape[2:]), dtype=np.float32)
        self.turn_available_actions = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].available_actions.shape[2:]), dtype=np.float32)
        self.turn_values = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].value_preds.shape[2:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].actions.shape[2:]), dtype=np.float32)       
        self.turn_one_hot_actions = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].one_hot_actions.shape[3:]), dtype=np.float32)       
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].action_log_probs.shape[2:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents,*self.buffer[0].rnn_states.shape[2:]), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones((self.n_rollout_threads, self.num_agents,*self.buffer[0].masks.shape[2:]), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].rewards.shape[2:]), dtype=np.float32)

        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.scores = []
            for step in range(self.episode_length):
                self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
                # Sample actions
                self.collect(step) 

                if step == 0 and episode > 0:
                    # deal with the data of the last index in buffer
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].share_obs[-1] = self.turn_share_obs[:, agent_id].copy()
                        self.buffer[agent_id].obs[-1] = self.turn_obs[:, agent_id].copy()
                        self.buffer[agent_id].available_actions[-1] = self.turn_available_actions[:, agent_id].copy()
                        self.buffer[agent_id].active_masks[-1] = self.turn_active_masks[:, agent_id].copy()

                        # deal with rewards
                        # 1. shift all rewards
                        self.buffer[agent_id].rewards[0:self.episode_length-1] = self.buffer[agent_id].rewards[1:]
                        # 2. last step rewards
                        self.buffer[agent_id].rewards[-1] = self.turn_rewards[:, agent_id].copy()

                    # compute return and update network
                    self.compute()
                    train_infos = self.train()

                for agent_id in range(self.num_agents):
                    # insert turn data into buffer
                    self.buffer[agent_id].chooseinsert(self.turn_share_obs[:, agent_id],
                                            self.turn_obs[:, agent_id],
                                            self.turn_rnn_states[:, agent_id],
                                            self.turn_rnn_states_critic[:, agent_id],
                                            self.turn_actions[:, agent_id],
                                            self.turn_one_hot_actions,
                                            self.turn_action_log_probs[:, agent_id],
                                            self.turn_values[:, agent_id],
                                            self.turn_rewards[:, agent_id],
                                            self.turn_masks[:, agent_id],
                                            self.turn_bad_masks[:, agent_id],
                                            self.turn_active_masks[:, agent_id],
                                            self.turn_available_actions[:, agent_id])
                # env reset
                obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
                share_obs = share_obs if self.use_centralized_V else obs

                self.use_obs[self.reset_choose] = obs[self.reset_choose]
                self.use_share_obs[self.reset_choose] = share_obs[self.reset_choose]
                self.use_available_actions[self.reset_choose] = available_actions[self.reset_choose]
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0 and episode > 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.hanabi_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "Hanabi":
                    average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
                    print("average score is {}.".format(average_score))
                    if self.use_wandb:
                        wandb.log({'average_score': average_score}, step = self.true_total_num_steps)
                    else:
                        self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

                for agent_id in range(self.num_agents):
                    train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[0].rewards)
                
                self.log_train(train_infos, self.true_total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.true_total_num_steps)

    def warmup(self):
        # reset env
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)

        share_obs = share_obs if self.use_centralized_V else obs

        # replay buffer
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        one_hot_actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)
        for current_agent_id in range(self.num_agents):
            env_actions = np.ones((self.n_rollout_threads, *self.buffer[current_agent_id].actions.shape[2:]), dtype=np.float32)*(-1.0)
            choose = np.any(self.use_available_actions == 1, axis=1)
            if ~np.any(choose):
                self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
                break
            
            self.trainer[current_agent_id].prep_rollout()
            execution_mask = torch.stack([torch.ones(self.n_rollout_threads)] * current_agent_id +
                                        [torch.zeros(self.n_rollout_threads)] *
                                        (self.num_agents - current_agent_id), -1).to(self.device)
            ego_inclusive_action = one_hot_actions.copy()
            
            value, action, action_log_prob, rnn_state, rnn_state_critic, _, _ \
                = self.trainer[current_agent_id].policy.get_actions(self.use_share_obs[choose],
                                                self.use_obs[choose],
                                                self.turn_rnn_states[choose, current_agent_id],
                                                self.turn_rnn_states_critic[choose, current_agent_id],
                                                self.turn_masks[choose, current_agent_id],
                                                ego_inclusive_action[choose],
                                                execution_mask[choose],
                                                self.use_available_actions[choose],
                                                tau=self.temperature)
            
            self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
            self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
            self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
            self.turn_values[choose, current_agent_id] = _t2n(value)
            self.turn_actions[choose, current_agent_id] = _t2n(torch.argmax(action, -1).unsqueeze(1).to(torch.int))
            self.turn_one_hot_actions[choose, current_agent_id] = _t2n(action)
            env_actions[choose] = _t2n(torch.argmax(action, -1).unsqueeze(1).to(torch.int))
            one_hot_actions[choose, current_agent_id] = _t2n(action)
            self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
            self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
            self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)
            
            self.true_total_num_steps += (choose==True).sum()
            share_obs = share_obs if self.use_centralized_V else obs

            # truly used value
            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

            # rearrange reward
            # reward of step 0 will be thrown away.
            self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
            self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
            self.turn_rewards_since_last_action[choose] += rewards[choose]

            # done==True env

            # deal with reset_choose
            self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)

            # deal with all agents
            self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.buffer[current_agent_id].available_actions.shape[2:]), dtype=np.float32)
            self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer[current_agent_id].rnn_states_critic.shape[2:]), dtype=np.float32)

            # deal with the current agent
            self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)

            # deal with the left agents
            left_agent_id = current_agent_id + 1
            left_agents_num = self.num_agents - left_agent_id
            self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
            self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            # other variables use what at last time, action will be useless.
            self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            self.turn_obs[dones == True, left_agent_id:] = 0
            self.turn_share_obs[dones == True, left_agent_id:] = 0

            # done==False env
            # deal with current agent
            self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
            self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

            # done==None
            # pass

            for done, info in zip(dones, infos):
                if done:
                    if 'score' in info.keys():
                        self.scores.append(info['score'])
            
    def train(self):
        train_infos = []

        action_dim=self.buffer[0].one_hot_actions.shape[-1]
        factor = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        old_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        new_actions_probs = np.ones((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        action_grad = np.zeros((self.num_agents, self.num_agents, self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        ordered_vertices = [i for i in range(self.num_agents)] * self.agent_layer

        for idx, agent_id in enumerate(reversed(ordered_vertices)):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(np.prod(np.concatenate([factor[:agent_id], factor[agent_id+1:]],0), 0))

            # other agents' gradient to agent_id
            action_grad_per_agent = np.zeros((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
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
                if idx > (self.agent_layer - 1) * self.num_agents - 1:
                    execution_masks_batch = torch.stack([torch.ones(self.episode_length*self.n_rollout_threads)] * agent_id +
                                                    [torch.zeros(self.episode_length*self.n_rollout_threads)] *
                                                    (self.num_agents - agent_id), -1).to(self.device)
                else:
                    execution_masks_batch = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(list(reversed(ordered_vertices))[idx+1:idx+self.num_agents]), 1.0)\
                        .unsqueeze(0).repeat(self.episode_length*self.n_rollout_threads, 1).to(self.device)
            else:
                if idx != self.agent_layer * self.num_agents - 1:
                    execution_masks_batch = torch.zeros(self.num_agents).scatter_(-1, torch.tensor(list(reversed(ordered_vertices))[idx+1]), 1.0)\
                        .unsqueeze(0).repeat(self.episode_length*self.n_rollout_threads, 1).to(self.device)
                else:
                    execution_masks_batch = torch.stack([torch.zeros(self.episode_length*self.n_rollout_threads)] * self.num_agents, -1).to(self.device)

            if idx > (self.agent_layer - 1) * self.num_agents - 1:
                one_hot_actions = torch.from_numpy(self.buffer[agent_id].one_hot_actions[:,:,0:self.num_agents].reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])).to(self.device)
                old_one_hot_actions = self.buffer[agent_id].one_hot_actions[:,:,0:self.num_agents].reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])
            else:
                sorted_tuples = sorted(zip(ordered_vertices[((self.agent_layer * self.num_agents - 1)-(idx+self.num_agents-1)):((self.agent_layer * self.num_agents - 1)-(idx-1))], [((self.agent_layer * self.num_agents - 1)-(idx+self.num_agents-1-i)) for i in range(self.num_agents)]), key=lambda x: x[0])
                _, sorted_index_vector = zip(*sorted_tuples)
                one_hot_actions = torch.from_numpy(np.stack([self.buffer[agent_id].one_hot_actions[:,:,i] for i in sorted_index_vector], -2).reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])).to(self.device)
                old_one_hot_actions = np.stack([self.buffer[agent_id].one_hot_actions[:,:,i] for i in sorted_index_vector], -2).reshape(-1, self.num_agents, *self.buffer[agent_id].one_hot_actions.shape[3:])

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
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices, idx//self.num_agents],
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
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[:,idx//self.num_agents],
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            old_one_hot_actions,
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            old_actions_probs[agent_id] = _t2n(torch.exp(old_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,1)

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
                                                                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[indices,idx//self.num_agents],
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
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:])[:,idx//self.num_agents],
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            one_hot_actions,
                                                            execution_masks_batch,
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            tau=self.temperature)
            new_actions_probs[agent_id] = _t2n(torch.exp(new_actions_logprob)).reshape(self.episode_length,self.n_rollout_threads,1)

            self.trainer[agent_id].policy.actor_optimizer.zero_grad()
            if self.inner_clip_param == 0.:
                torch.sum(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob.detach()),dim=-1), dim=-1).mean().backward()
            else:
                torch.sum(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob.detach()), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1), dim=-1).mean().backward()
            for i in range(self.num_agents):
                action_grad[agent_id][i] = _t2n(one_hot_actions.grad[:,i]).reshape(self.episode_length,self.n_rollout_threads,action_dim)
            if self.inner_clip_param == 0.:
                factor[agent_id] = _t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            else:
                factor[agent_id] = _t2n(torch.prod(torch.clamp(torch.exp(new_actions_logprob-old_actions_logprob), 1.0 - self.inner_clip_param, 1.0 + self.inner_clip_param),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos
  
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_envs = self.eval_envs

        eval_scores = []

        eval_finish = False
        eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        
        eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, *self.buffer[0].rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            if eval_finish:
                break
            eval_one_hot_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)
            for agent_id in range(self.num_agents):
                eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
                eval_choose = np.any(eval_available_actions == 1, axis=1)

                if ~np.any(eval_choose):
                    eval_finish = True
                    break

                self.trainer[agent_id].prep_rollout()
                execution_mask = np.stack([np.ones((self.n_eval_rollout_threads))] * agent_id +
                                        [np.zeros((self.n_eval_rollout_threads))] *
                                        (self.num_agents - agent_id), -1)
                ego_inclusive_action = eval_one_hot_actions.copy()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(eval_obs[eval_choose],
                                                                eval_rnn_states[eval_choose, agent_id],
                                                                eval_masks[eval_choose, agent_id],
                                                                ego_inclusive_action[eval_choose],
                                                                execution_mask[eval_choose],
                                                                eval_available_actions[eval_choose],
                                                                deterministic=True)

                eval_actions[eval_choose] = _t2n(eval_action)
                eval_one_hot_actions[eval_choose, agent_id] = _t2n(F.one_hot(eval_action.long(), self.action_dim).squeeze(1))
                eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)
                
                eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer[agent_id].available_actions.shape[2:]), dtype=np.float32)

                for eval_done, eval_info in zip(eval_dones, eval_infos):
                    if eval_done:
                        if 'score' in eval_info.keys():
                            eval_scores.append(eval_info['score'])

        eval_average_score = np.mean(eval_scores)
        print("eval average score is {}.".format(eval_average_score))
        if self.use_wandb:
            wandb.log({'eval_average_score': eval_average_score}, step=total_num_steps)
        else:
            self.writter.add_scalars('eval_average_score', {'eval_average_score': eval_average_score}, total_num_steps)

    
    @torch.no_grad()
    def eval_100k(self, eval_games=100000):
        eval_envs = self.eval_envs
        trials = int(eval_games/self.n_eval_rollout_threads)

        eval_scores = []
        for trial in range(trials):
            print("trail is {}".format(trial))
            eval_finish = False
            eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            
            eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            while True:
                if eval_finish:
                    break
                for agent_id in range(self.num_agents):
                    eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
                    eval_choose = np.any(eval_available_actions == 1, axis=1)

                    if ~np.any(eval_choose):
                        eval_finish = True
                        break

                    self.trainer.prep_rollout()
                    eval_action, eval_rnn_state = self.trainer.policy.act(eval_obs[eval_choose],
                                                                    eval_rnn_states[eval_choose, agent_id],
                                                                    eval_masks[eval_choose, agent_id],
                                                                    eval_available_actions[eval_choose],
                                                                    deterministic=True)

                    eval_actions[eval_choose] = _t2n(eval_action)
                    eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

                    # Obser reward and next obs
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)
                    
                    eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)

                    for eval_done, eval_info in zip(eval_dones, eval_infos):
                        if eval_done:
                            if 'score' in eval_info.keys():
                                eval_scores.append(eval_info['score'])

        eval_average_score = np.mean(eval_scores)
        print("eval average score is {}.".format(eval_average_score))