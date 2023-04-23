from collections import defaultdict, deque
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb
import math

from bta.utils.util import update_linear_schedule, default_collate_with_dim
from bta.runner.mappo.base_runner import Runner
from gobigger.agents import BotAgent


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
        bot_agents = []
        for env_idx in range(self.n_rollout_threads):
            bot_agent = []
            for player in range(self.all_args.player_num_per_team, self.all_args.team_num * self.all_args.player_num_per_team):
                bot_agent.append(BotAgent(player))
            bot_agents.append(bot_agent)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

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

                data = obs_raw, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

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
                
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
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
        share_obs = np.stack([share_obs for _ in range(self.num_agents)], 1)
        
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = share_obs.copy()
        
        return obs_raw

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))


        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs_raw, obs, share_obs, rewards, dones_env, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        share_obs = np.stack([share_obs for _ in range(self.num_agents)], 1)
        
        # update env_infos if done
        if np.any(dones_env):
            for done, info, ob_raw in zip(dones_env, infos, obs_raw):
                if done:
                    sorted_leaderboard_ = sorted(ob_raw[0]["leaderboard"].items(), key=lambda item: item[1])
                    if sorted_leaderboard_[-1][0] == 0:
                        self.env_infos["win_rate"].append(1)
                    else:
                        self.env_infos["win_rate"].append(0)

        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(
            share_obs=share_obs,
            obs=share_obs,
            rnn_states=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=np.expand_dims(rewards, -1),
            masks=masks,
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_env_infos = defaultdict(list)
        eval_share_obs, eval_obs, eval_obs_raw = self.eval_envs.reset()
        eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    
        eval_share_obs = np.stack([eval_share_obs for _ in range(self.num_agents)], 1)
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
            self.trainer.prep_rollout()
            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            
            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            env_actions = []
            trans_actions = self.transform_action(eval_actions)
            for env_idx in range(self.n_eval_rollout_threads):
                env_action = {i: [trans_actions[env_idx][i][0], trans_actions[env_idx][i][1], trans_actions[env_idx][i][2]] for i in range(self.num_agents)} 
                env_action.update({bot.name: bot.step(eval_obs_raw[env_idx][1][bot.name]) for bot in bot_agents[env_idx]})
                env_actions.append(env_action)

            # step
            eval_share_obs, eval_obs, eval_obs_raw, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(env_actions)
            eval_share_obs = np.stack([default_collate_with_dim(eval_share_obs[env_idx], device=self.device) for env_idx in range(self.n_eval_rollout_threads)])    
            eval_share_obs = np.stack([eval_share_obs for _ in range(self.num_agents)], 1)
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
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                self.trainer.prep_rollout()
                render_actions, render_rnn_states = self.trainer.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

                # append frame
                if self.all_args.save_gifs:        
                    image = render_infos[0]["frame"]
                    frames.append(image)
            
            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
        
        print("expected goal: {}".format(np.mean(render_goals)))
