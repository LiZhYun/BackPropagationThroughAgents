import numpy as np
import torch
import torch.nn as nn
from bta.utils.util import get_gard_norm, huber_loss, mse_loss
from bta.algorithms.utils.util import _h_A
from bta.utils.popart import PopArt
from bta.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.args = args
        self.discrete = args.discrete
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
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
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
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

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, father_action_batch, last_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              father_action_batch,
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # graph_actor update
        # share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        #         value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        #         adv_targ, available_actions_batch, father_action_batch = sample
        ### get the sequence
        if self.args.env_name == "Hanabi":
            loss_graph_actor = torch.zeros(1)
        else:
            self.episode_length = self.args.episode_length
            self.n_rollout_threads = self.args.n_rollout_threads
            self.n_agents = self.args.num_agents  ### self.n_rollout_threads, num_agents, act_shape
            self.n_actions = self.args.n_actions

            epi_roll = self.episode_length * self.n_rollout_threads // self.num_mini_batch
            if self.args.env_name == "GoBigger":
                for batch_idx in range(obs_batch.shape[0]):
                    for key in obs_batch[batch_idx].keys():
                        if 'Dict' in obs_batch[batch_idx][key].__class__.__name__.capitalize():
                            for sub_key in obs_batch[batch_idx][key].keys():
                                obs_batch[batch_idx][key][sub_key] = check(obs_batch[batch_idx][key][sub_key]).to(**self.tpdv)
                        else:
                            obs_batch[batch_idx][key] = check(obs_batch[batch_idx][key]).to(**self.tpdv)
                obs_batch_ = torch.stack([self.policy.actor.base(obs_batch[batch_idx]) for batch_idx in range(obs_batch.shape[0])])
                obs_batch_ = obs_batch_.reshape(epi_roll, self.n_agents, -1)
            else:
                obs_batch_ = check(obs_batch.reshape(epi_roll, self.n_agents, -1)).to(**self.tpdv)
            agent_id_graph_ = torch.eye(self.n_agents).unsqueeze(0).repeat(epi_roll, 1, 1).to(self.device)
            if self.discrete:
                action_ids = np.repeat(np.expand_dims(np.eye(self.args.n_actions), axis=0), epi_roll*self.n_agents, 0)  # 600.19.19

            if self.discrete:
                last_actions_batch = last_actions_batch[:,[0]]
                last_actions_batch_ =[action_ids[i][last_actions_batch.astype(np.int32)[i]] for i in range(len(action_ids))]  # 600.1.19
            else:
                last_actions_batch_ =last_actions_batch  # 600.1.19
            last_actions_batch_ = torch.tensor(last_actions_batch_).reshape(epi_roll, self.n_agents, -1).float().to(self.device)  # 25.4.19
            # if self.args.env_name == "GoBigger":
            #     inputs_graph = {'obs': obs_batch_, 'id_act': torch.cat((agent_id_graph_, last_actions_batch_), -1).float()}
            # else:
            inputs_graph = torch.cat((obs_batch_, agent_id_graph_, last_actions_batch_), -1).float() # 25.4.33

            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization = self.policy.graph_actor(inputs_graph)

            loss_graph_actor = -(torch.sum(log_softmax_logits_for_rewards, dim=(1, 2)) * torch.sum(
                adv_targ.reshape(epi_roll, self.n_agents, -1), dim=(1, 2))).mean()

            # ## DAG loss
            loss = 0
            # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
            for i in range(epi_roll):
                m_s = adj_prob[i]
                # sparse_loss = self.args.tau_A * torch.sum(torch.abs(m_s))
                h_A = _h_A(m_s, self.n_agents)
                loss += h_A
            loss_hA = loss / epi_roll
            loss_graph_actor = loss_graph_actor + loss_hA

            # x = torch.eye(d).double()+ torch.div(matrix, d)
            # def _h_A(A, m):
            #     expm_A = matrix_poly(A*A, m)
            #     h_A = torch.trace(expm_A) - m
            #     return h_A

            self.policy.graph_actor_optimizer.zero_grad()
            loss_graph_actor.backward()
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.graph_actor.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.graph_actor.parameters())
            self.policy.graph_actor_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, loss_graph_actor

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['loss_graph_actor'] = 0
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

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, loss_graph_actor \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['loss_graph_actor'] += loss_graph_actor.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.graph_actor.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.graph_actor.eval()
