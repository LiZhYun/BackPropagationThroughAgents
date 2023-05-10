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
from .algorithm.temporalPolicy import TemporalPolicy

class T_POLICY():
    def __init__(self,
                 args,
                 policy: TemporalPolicy,
                 agent_id,
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
        self.threshold = args.threshold

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
        self.use_graph = args.use_graph
        self._predict_other_shaped_info = (args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False))
        self._policy_group_normalization = (args.env_name == "Overcooked" and getattr(args, "policy_group_normalization", False))
        self._use_task_v_out = getattr(args, "use_task_v_out", False)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.
        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
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

    def graph_update(self, sample):
        # with torch.autograd.set_detect_anomaly(True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
        value_preds_batch, return_batch, masks_batch, execution_masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, adjs_batch, factor_batch, action_grad = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        #test
        execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
                                        [torch.zeros(actions_batch.shape[0])] *
                                        (self.num_agents - self.agent_id), -1).to(**self.tpdv)
        
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            one_hot_actions_batch,
                                                                            execution_masks_batch,
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            )
        # actor update
        imp_weights = action_log_probs

        surr = imp_weights * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(surr,
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # acyclic loss
        acyclic_loss = 0
        # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
        for i in range(adjs_batch.shape[0]):
            adjs_ = adjs_batch[i]
            acyclic_loss += cal_acyclic_loss(adjs_, self.num_agents)
        acyclic_loss = acyclic_loss / adjs_batch.shape[0]

        return policy_loss + acyclic_loss
    
    def ppo_update(self, sample, agent_order=None, tau=1.0):
        # with torch.autograd.set_detect_anomaly(True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
        value_preds_batch, return_batch, masks_batch, execution_masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, adjs_batch, factor_batch, action_grad = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)
        action_grad = check(action_grad).to(**self.tpdv)
        #test
        # execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
        #                                 [torch.zeros(actions_batch.shape[0])] *
        #                                 (self.num_agents - self.agent_id), -1).to(**self.tpdv)
        if agent_order is None:
            agent_order = torch.stack([torch.randperm(self.num_agents) for _ in range(actions_batch.shape[0])]).to(self.device)
        else:
            agent_order = torch.stack([agent_order for _ in range(actions_batch.shape[0])]).to(self.device)
        execution_masks_batch = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()[:, self.agent_id]  # [bs, n_agents, n_agents]
        
        # Reshape to do in a single forward pass for all steps
        values, train_actions, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            one_hot_actions_batch,
                                                                            execution_masks_batch,
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            tau=tau
                                                                            )
        # actor update
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)
        if self.args.env_name == "matrix":
            if self.agent_id == (self.num_agents - 1):
                surr1 = (imp_weights + self.threshold * (imp_weights * factor_batch + imp_weights.detach() * action_grad * train_actions - imp_weights)) * adv_targ
                surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) + self.threshold * (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * factor_batch \
                        + torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param) * action_grad * train_actions - torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param))) * adv_targ
            else:
                surr1 = (self.threshold * (imp_weights.detach() * action_grad * train_actions)) * adv_targ
                surr2 = (self.threshold * (torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param) * action_grad * train_actions)) * adv_targ
        else:
            surr1 = (imp_weights + self.threshold * (imp_weights * factor_batch + imp_weights.detach() * action_grad * train_actions - imp_weights)) * adv_targ
            surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) + self.threshold * (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * factor_batch \
                    + torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param) * action_grad * train_actions - torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param))) * adv_targ
            # surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * torch.clamp(factor_batch, 1.0 - self.clip_param / 2, 1.0 + self.clip_param / 2) \
            #          + torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param) * action_grad * train_actions) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        value_loss = value_loss * self.value_loss_coef
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def compute_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        return advantages

    def train(self, buffer, agent_order=None, tau=1.0):
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
        if self.use_graph:
            train_info['graphic_loss'] = 0
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

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, agent_order, tau)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['ratio'] += imp_weights.mean().item()
                
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
    
    def train_graph(self, buffer, turn_on=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self._use_recurrent_policy:
            data_generator = buffer.graph_recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.graph_naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.graph_feed_forward_generator(advantages, self.num_mini_batch)

        graph_loss = 0
        for sample in data_generator:

            loss = self.graph_update(sample)
            graph_loss += loss
 
        return graph_loss

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    def to(self, device):
        self.policy.to(device)
