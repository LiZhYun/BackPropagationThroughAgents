import torch
from bta.algorithms.gcs.algorithm.r_actor_critic import R_Actor, R_Critic
from bta.utils.util import update_linear_schedule, get_shape_from_obs_space, get_shape_from_act_space
from bta.algorithms.utils.util import *
import igraph as ig
from bta.algorithms.gcs.algorithm.graph_net_trans import Actor_graph
from gym import spaces
from bta.algorithms.utils.cnn import CNNBase
from bta.algorithms.utils.mlp import MLPBase

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space

        if args.env_name == 'GRFootball':
            self.act_space = act_space
        else:
            self.act_space = act_space
        self.mix_action = False
        self.discrete = False
        if act_space.__class__.__name__ == "Discrete":
            self.discrete = True
            self.n_actions = act_space.n
        elif act_space.__class__.__name__ == "Box":
            self.n_actions = act_space.shape[0]
        else:
            self.mix_action = True
            self.continous_dim = act_space[0].shape[0]
            self.discrete_dim = act_space[1].n
            self.n_actions = self.continous_dim + self.discrete_dim

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.obs_shape = get_shape_from_obs_space(obs_space)

        if args.env_name == "GoBigger":
            self._mixed_obs = False
            self._nested_obs = True
            obs_dim = self.actor.base.output_size
        else:
            self._mixed_obs = False
            self._nested_obs = False
            if len(self.obs_shape) == 3:
                import functools
                obs_dim = functools.reduce(lambda a, b: a*b, self.obs_shape)
            else:
                obs_dim = self.obs_shape[0]
        graph_input_dim = obs_dim + self.n_actions + self.args.num_agents
        self.graph_actor = Actor_graph(args, graph_input_dim, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.graph_actor_optimizer = torch.optim.Adam(self.graph_actor.parameters(),
                                                 lr=self.lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _t2n(self, x):
        return x.detach().cpu().numpy()
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.graph_actor_optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, last_actions, available_actions=None,
                    deterministic=False, agent_id=None, one_hot_actions=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.0
        """

        ### get the sequence
        self.n_agents = last_actions.shape[-2]  ### self.n_rollout_threads, num_agents, act_shape
        self.n_rollout_threads = last_actions.shape[0]

        if self.args.env_name == "Hanabi":
            G_s = torch.stack([torch.ones(obs.shape[0])] * agent_id +
                                            [torch.zeros(obs.shape[0])] *
                                            (self.args.num_agents - agent_id), -1).to(self.device)
        else:
            agent_id_graph = torch.eye(self.n_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1).to(self.device)  # self.n_rollout_threads, num_agents, num_agents
            if self._nested_obs:
                for batch_idx in range(obs.shape[0]):
                    for key in obs[batch_idx].keys():
                        if 'Dict' in obs[batch_idx][key].__class__.__name__.capitalize():
                            for sub_key in obs[batch_idx][key].keys():
                                obs[batch_idx][key][sub_key] = check(obs[batch_idx][key][sub_key]).to(**self.tpdv)
                        else:
                            obs[batch_idx][key] = check(obs[batch_idx][key]).to(**self.tpdv)
                obs = torch.stack([self.actor.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
            elif self._mixed_obs:
                for key in obs.keys():
                    obs[key] = check(obs[key]).to(**self.tpdv)
            else:
                obs = check(obs).to(**self.tpdv)
            obs_ = obs
            if self.mix_action:
                discrete_ = np.squeeze(np.eye(self.discrete_dim)[last_actions[:, :, self.continous_dim:].astype(np.int32)], 2)
                last_actions = np.concatenate([last_actions[:, :, :self.continous_dim], discrete_], -1)
            elif self.discrete:
                # last_actions = np.eye(self.n_actions)[last_actions.astype(np.int32)]
                last_actions = np.squeeze(np.eye(self.n_actions)[last_actions.astype(np.int32)], 2)
            last_actions_ = check(last_actions).to(**self.tpdv).reshape(self.n_rollout_threads, self.n_agents, -1)

            # if self._nested_obs:
            #     obs_ = obs_.reshape(self.n_rollout_threads, self.n_agents)
            #     inputs_graph = {'obs': obs_, 'id_act': torch.cat((agent_id_graph, last_actions_), -1).float()}
            # else:
            obs_ = obs_.reshape(self.n_rollout_threads, self.n_agents, -1)
            inputs_graph = torch.cat((obs_, agent_id_graph), -1).float()  # 1. 4.33
            inputs_graph = torch.cat((inputs_graph, last_actions_), -1).float()  # 1. 4.33

            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization = self.graph_actor(inputs_graph)
            graph_A = samples.clone().cpu().numpy()

            ######## pruning
            G_s = []
            for i in range(graph_A.shape[0]):
                G = ig.Graph.Weighted_Adjacency(graph_A[i].tolist())
                if not is_acyclic(graph_A[i]):
                    G, new_A = pruning_1(G, graph_A[i])
                G_s.append(G)

        # if len(self.obs_shape) == 1:
        #     obs = obs.reshape(self.n_rollout_threads, self.n_agents, obs.shape[-1])
        # else:
        #     obs = obs.reshape(self.n_rollout_threads, self.n_agents, *obs.shape[-3:])
        # rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.n_agents, rnn_states_actor.shape[-2],
        #                                             rnn_states_actor.shape[-1])
        actions, action_log_probs, rnn_states_actor, father_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, G_s,
                                                                 available_actions,
                                                                 deterministic,
                                                                 one_hot_actions=one_hot_actions)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks) # 4.1 ，  4.1.64
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, father_actions
        # return values.reshape(self.n_rollout_threads, self.n_agents, -1), actions.reshape(self.n_rollout_threads, self.n_agents, -1), \
        #         action_log_probs.reshape(self.n_rollout_threads, self.n_agents, -1), \
        #         rnn_states_actor.reshape(self.n_rollout_threads, self.n_agents, self.args.recurrent_N, -1), \
        #         rnn_states_critic.reshape(self.n_rollout_threads, self.n_agents, self.args.recurrent_N, -1), \
        #         father_actions.reshape(self.n_rollout_threads, self.n_agents, -1)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, father_action, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     father_action,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, last_actions, available_actions=None, deterministic=True, agent_id=None, one_hot_actions=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        ### get the sequence
        self.n_agents = last_actions.shape[-2]  ### self.n_rollout_threads, num_agents, act_shape
        self.n_rollout_threads = last_actions.shape[0]

        if self.args.env_name == "Hanabi":
            G_s = torch.stack([torch.ones(obs.shape[0])] * agent_id +
                                            [torch.zeros(obs.shape[0])] *
                                            (self.args.num_agents - agent_id), -1).to(self.device)
        else:
            agent_id_graph = torch.eye(self.n_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1).to(self.device)  # self.n_rollout_threads, num_agents, num_agents
            if self._nested_obs:
                for batch_idx in range(obs.shape[0]):
                    for key in obs[batch_idx].keys():
                        if 'Dict' in obs[batch_idx][key].__class__.__name__.capitalize():
                            for sub_key in obs[batch_idx][key].keys():
                                obs[batch_idx][key][sub_key] = check(obs[batch_idx][key][sub_key]).to(**self.tpdv)
                        else:
                            obs[batch_idx][key] = check(obs[batch_idx][key]).to(**self.tpdv)
                obs = torch.stack([self.actor.base(obs[batch_idx]) for batch_idx in range(obs.shape[0])])
            elif self._mixed_obs:
                for key in obs.keys():
                    obs[key] = check(obs[key]).to(**self.tpdv)
            else:
                obs = check(obs).to(**self.tpdv)
            obs_ = obs
            if self.mix_action:
                discrete_ = np.squeeze(np.eye(self.discrete_dim)[last_actions[:, :, self.continous_dim:].astype(np.int32)], 2)
                last_actions = np.concatenate([last_actions[:, :, :self.continous_dim], discrete_], -1)
            elif self.discrete:
                # last_actions = np.eye(self.n_actions)[last_actions.astype(np.int32)]
                last_actions = np.squeeze(np.eye(self.n_actions)[last_actions.astype(np.int32)], 2)
            last_actions_ = check(last_actions).to(**self.tpdv).reshape(self.n_rollout_threads, self.n_agents, -1)

            # if self._nested_obs:
            #     obs_ = obs_.reshape(self.n_rollout_threads, self.n_agents)
            #     inputs_graph = {'obs': obs_, 'id_act': torch.cat((agent_id_graph, last_actions_), -1).float()}
            # else:
            obs_ = obs_.reshape(self.n_rollout_threads, self.n_agents, -1)
            inputs_graph = torch.cat((obs_, agent_id_graph), -1).float()  # 1. 4.33
            inputs_graph = torch.cat((inputs_graph, last_actions_), -1).float()  # 1. 4.33

            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization = self.graph_actor(inputs_graph)
            graph_A = samples.clone().cpu().numpy()

            ######## pruning
            G_s = []
            for i in range(graph_A.shape[0]):
                G = ig.Graph.Weighted_Adjacency(graph_A[i].tolist())
                if not is_acyclic(graph_A[i]):
                    G, new_A = pruning_1(G, graph_A[i])
                G_s.append(G)

        # if len(self.obs_shape) == 1:
        #     obs = obs.reshape(self.n_rollout_threads, self.n_agents, obs.shape[-1])
        # else:
        #     obs = obs.reshape(self.n_rollout_threads, self.n_agents, *obs.shape[-3:])
        # rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.n_agents, rnn_states_actor.shape[-2],
        #                                             rnn_states_actor.shape[-1])
        actions, action_log_probs, rnn_states_actor, father_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, G_s,
                                                                 available_actions,
                                                                 deterministic,
                                                                 one_hot_actions=one_hot_actions)
        return actions, rnn_states_actor
