import numpy as np
import torch
import torch.nn as nn
from bta.utils.util import get_gard_norm, huber_loss, mse_loss, generate_mask_from_order
from bta.utils.valuenorm import ValueNorm
from bta.algorithms.utils.util import check


class MacpfTrainer:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads

        self.args = args
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.gamma = args.gamma
        self.alpha = 0.01

        self.discrim_loss = nn.CrossEntropyLoss(reduction="none")
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
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

        if self._use_valuenorm:
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

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def train_actor(self, sample, dep_mode):
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
        share_obs_batch, target_share_obs_batch, obs_batch, target_obs_batch, rnn_states_batch, target_rnn_states_batch, \
            rnn_states_critic_batch, target_rnn_states_critic_batch, actions_batch, masks_batch, target_masks, active_masks_batch, \
                available_actions_batch, target_available_actions_batch, rewards_batch, old_action_log_probs_batch = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        # noise_batch = check(noise_batch).to(**self.tpdv)
        # target_noise_batch = check(target_noise_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)

        bs = share_obs_batch.shape[0]

        agent_order = torch.stack([check(np.arange(self.num_agents)) for _ in range(actions_batch.shape[0])]).to(self.device)
        execution_masks_batch = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()  # [bs, n_agents, n_agents]

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                obs_batch.reshape(bs*self.num_agents,-1), 
                                                rnn_states_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                                rnn_states_critic_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                                actions_batch.reshape(bs*self.num_agents,-1), 
                                                masks_batch.reshape(bs*self.num_agents,-1), 
                                                actions_batch.unsqueeze(1).repeat(1,self.num_agents,1,1).reshape(bs*self.num_agents,self.num_agents,-1), 
                                                execution_masks_batch.reshape(bs*self.num_agents,-1),
                                                dep_mode,
                                                available_actions_batch.reshape(bs*self.num_agents,-1) if available_actions_batch is not None else None,
                                                )
        # chosen_action_qvals = torch.gather(values, dim=-1, index=actions_batch.reshape(bs*self.num_agents,-1).long())  # Remove the last dim
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch.reshape(bs*self.num_agents,-1))

        pol_loss = (self.alpha * action_log_probs - values.detach()).mean()
        self.policy.a_optimizer.zero_grad()
        pol_loss.backward()
        agent_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.a_optimizer.step()

        return pol_loss, agent_grad_norm
        
    def train_critic(self, sample, dep_mode):
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
        share_obs_batch, target_share_obs_batch, obs_batch, target_obs_batch, rnn_states_batch, target_rnn_states_batch, \
            rnn_states_critic_batch, target_rnn_states_critic_batch, actions_batch, masks_batch, target_masks_batch, active_masks_batch, \
                available_actions_batch, target_available_actions_batch, rewards_batch, old_action_log_probs_batch = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        target_masks_batch = check(target_masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        # noise_batch = check(noise_batch).to(**self.tpdv)
        # target_noise_batch = check(target_noise_batch).to(**self.tpdv)

        bs = share_obs_batch.shape[0]

        agent_order = torch.stack([check(np.arange(self.num_agents)) for _ in range(actions_batch.shape[0])]).to(self.device)
        execution_masks_batch = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()  # [bs, n_agents, n_agents]

        target_actions_batch = torch.zeros_like(actions_batch)
        next_action_log_probs_batch = torch.zeros_like(old_action_log_probs_batch)
        for agent_idx in range(self.num_agents):
            next_actions, action_log_probs, rnn_states = self.policy.actor(
                target_obs_batch[:, agent_idx], 
                target_rnn_states_batch[:, agent_idx], 
                target_masks_batch[:, agent_idx], 
                target_actions_batch, 
                execution_masks_batch[:, agent_idx],
                dep_mode,
                target_available_actions_batch[:, agent_idx] if target_available_actions_batch is not None else None,
            )
            target_actions_batch[:, agent_idx] = next_actions.clone()
            next_action_log_probs_batch[:, agent_idx] = action_log_probs.clone()

        values, _ = self.policy.critic(
                                        obs_batch.reshape(bs*self.num_agents,-1), 
                                        # rnn_states_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                        rnn_states_critic_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                        actions_batch.reshape(bs*self.num_agents,-1), 
                                        masks_batch.reshape(bs*self.num_agents,-1), 
                                        actions_batch.unsqueeze(1).repeat(1,self.num_agents,1,1).reshape(bs*self.num_agents,self.num_agents,-1), 
                                        execution_masks_batch.reshape(bs*self.num_agents,-1),
                                        dep_mode,
                                        # available_actions_batch.reshape(bs*self.num_agents,-1), 
                                        )
        values = self.policy.mixer(values, share_obs_batch[:,0])

        next_values, _ = self.policy.target_critic(
                                            target_obs_batch.reshape(bs*self.num_agents,-1), 
                                            # target_rnn_states_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                            target_rnn_states_critic_batch.reshape(bs*self.num_agents,self._recurrent_N,-1), 
                                            target_actions_batch.reshape(bs*self.num_agents,-1), 
                                            target_masks_batch.reshape(bs*self.num_agents,-1), 
                                            target_actions_batch.unsqueeze(1).repeat(1,self.num_agents,1,1).reshape(bs*self.num_agents,self.num_agents,-1), 
                                            execution_masks_batch.reshape(bs*self.num_agents,-1),
                                            dep_mode,
                                            # target_available_actions_batch.reshape(bs*self.num_agents,-1) if target_available_actions_batch is not None else None,
                                            )
        next_values = self.policy.target_mixer(next_values, target_share_obs_batch[:,0])
        next_action_log_probs = next_action_log_probs_batch.reshape(bs, self.num_agents, -1).sum((1,2), keepdim=True)
        next_q_value = rewards_batch[:,0].squeeze() + target_masks_batch[:,0].squeeze() * self.gamma * (next_values.squeeze()-self.alpha*next_action_log_probs.squeeze())

        # Td-error
        td_error = (values.squeeze() - next_q_value.detach())

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).mean()

        # loss = loss + averaged_discrim_loss

        self.policy.c_optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.params, self.max_grad_norm)
        self.policy.c_optimizer.step()

        # if step % self.args.target_update_interval == 0:
        #     self._update_targets()

        return loss, grad_norm
    
    def _update_targets(self):
        self.policy.target_critic.load_state_dict(self.policy.critic.state_dict())
        self.policy.target_mixer.load_state_dict(self.policy.mixer.state_dict())
        self.policy.dep_target_mixer.load_state_dict(self.policy.dep_mixer.state_dict())

    def train(self, buffer, step):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(self.num_mini_batch)

        for sample in data_generator:
            
            value_loss, critic_grad_norm \
                = self.train_critic(sample, True)
            loss, grad_norm \
                = self.train_critic(sample, False)
            policy_loss, actor_grad_norm \
                = self.train_actor(sample, True)
            loss, grad_norm \
                = self.train_actor(sample, False)

            train_info['value_loss'] += value_loss.item()
            train_info['policy_loss'] += policy_loss.item()
            train_info['actor_grad_norm'] += actor_grad_norm
            train_info['critic_grad_norm'] += critic_grad_norm

        num_updates = self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        
        if step % self.args.target_update_interval == 0:
            self._update_targets()
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
