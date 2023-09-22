import numpy as np
import torch
from bta.algorithms.bta.algorithm.r_actor_critic import R_Actor, R_Critic
from bta.utils.util import update_linear_schedule


class TemporalPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, agent_id, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.attention_lr = args.attention_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.use_action_attention = args.use_action_attention

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space
        self.agent_id = agent_id

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.agent_id, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.act_space, self.device)
        # if self.use_action_attention:
        #     from bta.algorithms.utils.action_attention import Action_Attention
        #     self.action_attention = Action_Attention(args, act_space, self.agent_id, share_obs_space, device = self.device)
        #     self.action_attention_optimizer = torch.optim.Adam(self.action_attention.parameters(), lr=self.attention_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_mix_actions(self, logits, obs, rnn_states, masks, **kwargs):
        bias_, action_std, rnn_states = self.action_attention(logits, obs, rnn_states, masks)
        return bias_, action_std, rnn_states

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, onehot_action, execution_mask, available_actions=None, deterministic=False, task_id=None, tau=1.0, **kwargs):
        actions, action_log_probs, rnn_states_actor, logits, dist_entropy, obs_feat = self.actor(obs, rnn_states_actor, masks, onehot_action, execution_mask, available_actions, deterministic, tau=tau)
        values, rnn_states_critic, state_feat = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, logits, obs_feat

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None):
        values, _, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, onehot_action, execution_mask, available_actions=None, active_masks=None, task_id=None, tau=1.0, kl=False, joint_actions=None):
        train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, onehot_action, execution_mask, available_actions, active_masks, tau=tau, kl=kl, joint_actions=joint_actions)
        values, _, state_feat = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat
    
    def evaluate_actions_logprobs(self, obs, rnn_states_actor, action, masks, onehot_action, execution_mask, available_actions=None, active_masks=None, tau=1.0, kl=False, joint_actions=None):
        train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, onehot_action, execution_mask, available_actions, active_masks, tau=tau, kl=kl, joint_actions=joint_actions)
        return action_log_probs

    def act(self, obs, rnn_states_actor, masks, onehot_action, execution_mask, available_actions=None, deterministic=False, tau=1.0, **kwargs):
        actions, action_log_probs, rnn_states_actor, logits, dist_entropy, obs_feat = self.actor(obs, rnn_states_actor, masks, onehot_action, execution_mask, available_actions, deterministic, tau=tau)
        return actions, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if 'actor' in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if 'critic' in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()