import numpy as np
import time
import math

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bta.utils.util import get_gard_norm, huber_loss, mse_loss, cal_acyclic_loss, generate_mask_from_order
from bta.utils.valuenorm import ValueNorm
from bta.algorithms.utils.util import check
from .algorithm.arMAPPOPolicy import AutoRegressivePolicy

class AR_MAPPO():
    def __init__(self,
                 args,
                 policy: AutoRegressivePolicy,
                 agent_id,
                 action_space,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = args.num_agents
        self.agent_id = agent_id
        self.args = args

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.shaped_info_coef = getattr(args, "shaped_info_coef", 0.5)
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.share_policy = args.share_policy
        self.entropy_lr = args.entropy_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead
        self._predict_other_shaped_info = (args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False))
        self._policy_group_normalization = (args.env_name == "Overcooked" and getattr(args, "policy_group_normalization", False))
        self._use_task_v_out = getattr(args, "use_task_v_out", False)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
            if self._use_policy_vhead:
                self.policy_value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None
            if self._use_policy_vhead:
                self.policy_value_normalizer = None
        
        self.automatic_target_entropy_tuning = args.automatic_target_entropy_tuning
        self.log_entropy_coef = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        if action_space.__class__.__name__ == "Discrete":
            if self.automatic_target_entropy_tuning:
                self.log_entropy_coef = torch.tensor(np.log(np.e), requires_grad=True, device=self.device)
                self.target_entropy = (torch.log(torch.tensor(action_space.n))).to(self.device)
            else:
                self.target_entropy = (torch.log(torch.tensor(action_space.n))*0.2).to(self.device)
        elif action_space.__class__.__name__ == "Box":
            self.target_entropy = -torch.prod(torch.tensor(action_space.shape[0]).to(self.device)).item()
        self.entropy_coef = self.log_entropy_coef.exp()
        self.entropy_coef_optim = torch.optim.Adam([self.log_entropy_coef], lr=self.entropy_lr, eps=self.opti_eps, weight_decay=self.weight_decay)


    def cal_value_loss(self, value_normalizer, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        # with torch.autograd.set_detect_anomaly(True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)

        if self.args.env_name == "mujoco":
            ego_exclusive_action = torch.cat(
                [actions_batch[:, :self.agent_id], actions_batch[:, self.agent_id + 1:]],
                -2).to(**self.tpdv)
        else:
            ego_exclusive_action = torch.cat(
                [actions_batch[:, :self.agent_id], actions_batch[:, self.agent_id + 1:]],
                -2).squeeze(-1).to(**self.tpdv)
        agent_order = torch.stack(                     
            [torch.randperm(self.num_agents) for _ in range(actions_batch.shape[0])]).to(self.device)
        # agent_order = agent_order.view(-1, self.args.n_rollout_threads, self.num_agents)
        all_execution_mask = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()  # [bs, n_agents, n_agents]
        execution_mask = torch.cat([all_execution_mask[..., self.agent_id, :self.agent_id], all_execution_mask[..., self.agent_id,
                                    self.agent_id + 1:]], -1)
        if self.args.env_name == "mujoco":
            onehot_action = ego_exclusive_action
        else:
            onehot_action = F.one_hot(ego_exclusive_action.long(), self.args.action_dim).float()
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, policy_values, pred_shaped_info = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch[:, self.agent_id], 
                                                                            masks_batch, 
                                                                            onehot_action,
                                                                            execution_mask,
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            )
        # actor update
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            policy_value_loss = self.cal_value_loss(self.policy_value_normalizer, policy_values, value_preds_batch, return_batch, active_masks_batch)       
            policy_loss = policy_action_loss + policy_value_loss * self.policy_value_loss_coef
        else:
            policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        loss = policy_loss - dist_entropy * self.entropy_coef
        loss.backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(self.value_normalizer, values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        entropy_loss = -(self.log_entropy_coef * (action_log_probs + self.target_entropy).detach()).mean()

        self.entropy_coef_optim.zero_grad()
        entropy_loss.backward()
        self.entropy_coef_optim.step()

        self.entropy_coef = self.log_entropy_coef.exp()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio
    
    def update_actor(self):
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

    def compute_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        return advantages

    def train(self, buffer, turn_on=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = defaultdict(float)

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio \
                    = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['ratio'] += ratio.mean().item()
                
                if int(torch.__version__[2]) < 5:
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                else:
                    train_info['actor_grad_norm'] += actor_grad_norm.item()
                    train_info['critic_grad_norm'] += critic_grad_norm.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():             
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    def to(self, device):
        self.policy.to(device)
