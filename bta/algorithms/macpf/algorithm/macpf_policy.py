import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from bta.utils.util import update_linear_schedule
from bta.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from bta.algorithms.utils.util import check
from bta.algorithms.macpf.algorithm.noise_mix import QMixer as NoiseQMixer
from bta.algorithms.macpf.algorithm.noise_rnn_critic import R_Critic as Critic
from bta.algorithms.macpf.algorithm.noise_rnn_actor import R_Actor as Actor
from bta.algorithms.macpf.algorithm.uniform import Uniform as NoiseGen
from torch.distributions import Categorical

class MacpfPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.obs_space, self.act_space, self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.mixer = NoiseQMixer(args, self.share_obs_space, self.device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.dep_mixer = NoiseQMixer(args, self.share_obs_space, self.device)
        self.dep_target_mixer = copy.deepcopy(self.dep_mixer)

        if act_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            self.action_dim = act_space.n
        elif act_space.__class__.__name__ == "Box":
            self.continuous_action = True
            self.action_dim = act_space.shape[0]
        self.state_dim = get_shape_from_obs_space(share_obs_space)[0]
        # discrim_input = self.state_dim + self.args.num_agents * self.action_dim
        # self.discrim = Discrim(discrim_input, self.args.noise_dim, args)

        self.params = list(self.critic.parameters())
        self.params += list(self.mixer.parameters())
        self.params += list(self.dep_mixer.parameters())

        # self.epsilon = self.args.epsilon_start

        # self.noise_distrib = NoiseGen(self.args)

        self.a_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.c_optimizer = torch.optim.Adam(params=self.params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.to(device)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.a_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.c_optimizer, episode, episodes, self.lr)

    def epsilon_decay(self, episode, episodes):
        delta = (self.args.epsilon_start - self.args.epsilon_finish) / episodes
        self.epsilon = max(self.args.epsilon_finish, self.args.epsilon_start - delta * episode)

    def get_actions(self, obs, rnn_states, rnn_states_critic, masks, parents_actions, execution_mask, dep_mode, available_actions=None, **kwargs):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, parents_actions, execution_mask, dep_mode, available_actions, deterministic)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, actions, masks, parents_actions, execution_mask, dep_mode, task_id=task_id)
        return actions, rnn_states_actor, rnn_states_critic

    def evaluate_actions(self, obs, rnn_states, rnn_states_critic, action, masks, parents_actions, execution_mask, dep_mode, available_actions=None, **kwargs):
        action_log_probs, dist_entropy, policy_values, pred_shaped_info = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, parents_actions, execution_mask, dep_mode, available_actions, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, action, masks, parents_actions, execution_mask, dep_mode, task_id=task_id)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states, masks, parents_actions, execution_mask, dep_mode, available_actions=None, **kwargs):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, parents_actions, execution_mask, dep_mode, available_actions, deterministic)
        return actions, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if 'actor' in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if 'critic' in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        self.dep_mixer.to(device)
        self.dep_target_mixer.to(device)

    def train(self):
        self.actor.train()
        self.critic.train()
        self.target_critic.train()
        self.mixer.train()
        self.target_mixer.train()
        self.dep_mixer.train()
        self.dep_target_mixer.train()

    def eval(self):
        self.actor.eval()

class Discrim(nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [torch.nn.Linear(input_size, self.args.hidden_size), torch.nn.ReLU()]
        for _ in range(self.args.layer_N - 1):
            layers.append(torch.nn.Linear(self.args.hidden_size, self.args.hidden_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.args.hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
