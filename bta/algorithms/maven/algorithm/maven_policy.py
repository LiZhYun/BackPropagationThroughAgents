import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from bta.utils.util import update_linear_schedule
from bta.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from bta.algorithms.utils.util import check
from bta.algorithms.maven.algorithm.noise_mix import QMixer as NoiseQMixer
from bta.algorithms.maven.algorithm.noise_rnn_agent import RNNAgent as Agent
from bta.algorithms.maven.algorithm.uniform import Uniform as NoiseGen
from torch.distributions import Categorical

class MavenPolicy:
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

        self.actor = Agent(args, self.obs_space, self.act_space, self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.mixer = NoiseQMixer(args, self.share_obs_space, self.device)
        self.target_mixer = copy.deepcopy(self.mixer)

        if act_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            self.action_dim = act_space.n
        elif act_space.__class__.__name__ == "Box":
            self.continuous_action = True
            self.action_dim = act_space.shape[0]
        self.state_dim = get_shape_from_obs_space(share_obs_space)[0]
        discrim_input = self.state_dim + self.args.num_agents * self.action_dim
        self.discrim = Discrim(discrim_input, self.args.noise_dim, args)

        self.params = list(self.actor.parameters())
        self.params += list(self.mixer.parameters())
        self.params += list(self.discrim.parameters())

        self.epsilon = self.args.epsilon_start

        self.noise_distrib = NoiseGen(self.args)

        self.optimizer = torch.optim.Adam(params=self.params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.to(device)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def epsilon_decay(self, episode, episodes):
        delta = (self.args.epsilon_start - self.args.epsilon_finish) / episodes
        self.epsilon = max(self.args.epsilon_finish, self.args.epsilon_start - delta * episode)

    def get_actions(self, obs, rnn_states, masks, noise, available_actions=None, **kwargs):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        agent_outs, rnn_states = self.actor(obs, rnn_states, masks, noise, available_actions)
        if available_actions is not None:
            agent_outs[available_actions == 0] = -1e10
        agent_outs = nn.functional.softmax(agent_outs, dim=-1)
        # Epsilon floor
        if available_actions is not None:
            epsilon_action_num = available_actions.sum(dim=-1, keepdim=True).float()
        else:
            epsilon_action_num = agent_outs.size(-1)
        agent_outs = ((1 - self.epsilon) * agent_outs
                        + torch.ones_like(agent_outs) * self.epsilon/epsilon_action_num)
        if available_actions is not None:
            agent_outs[available_actions == 0] = 0.0

        random_numbers = torch.rand(1).to(self.device)
        pick_random = (random_numbers < self.epsilon).long()
        if available_actions is not None:
            random_actions = Categorical(available_actions.float()).sample().long()
        else:
            random_actions = Categorical(torch.ones_like(agent_outs) * self.epsilon/epsilon_action_num).sample().long()

        actions = pick_random * random_actions + (1 - pick_random) * agent_outs.max(dim=-1)[1]
        return actions.unsqueeze(-1), rnn_states

    def evaluate_actions(self, obs, target_obs, rnn_states, target_rnn_states, masks, noise, target_noise, available_actions=None, target_available_actions=None, **kwargs):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            target_available_actions = check(target_available_actions).to(**self.tpdv)
        agent_outs, rnn_states = self.actor(obs, rnn_states, masks, noise, available_actions)
        if available_actions is not None:
            agent_outs[available_actions == 0] = -1e10
        agent_outs = nn.functional.softmax(agent_outs, dim=-1)
        # Epsilon floor
        if available_actions is not None:
            epsilon_action_num = available_actions.sum(dim=-1, keepdim=True).float()
        else:
            epsilon_action_num = agent_outs.size(-1)
        agent_outs = ((1 - self.epsilon) * agent_outs
                        + torch.ones_like(agent_outs) * self.epsilon/epsilon_action_num)
        if available_actions is not None:
            agent_outs[available_actions == 0] = 0.0

        target_agent_outs, rnn_states = self.target_actor(target_obs, target_rnn_states, masks, target_noise, target_available_actions)
        if target_available_actions is not None:
            target_agent_outs[target_available_actions == 0] = -1e10
        target_agent_outs = nn.functional.softmax(target_agent_outs, dim=-1)
        # Epsilon floor
        target_agent_outs = ((1 - self.epsilon) * target_agent_outs
                        + torch.ones_like(target_agent_outs) * self.epsilon/epsilon_action_num)
        if target_available_actions is not None:
            target_agent_outs[target_available_actions == 0] = 0.0
        return agent_outs, target_agent_outs

    def act(self, obs, rnn_states, masks, noise, available_actions=None, **kwargs):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        agent_outs, rnn_states = self.actor(obs, rnn_states, masks, noise, available_actions)
        if available_actions is not None:
            agent_outs[available_actions == 0] = -1e10
        agent_outs = nn.functional.softmax(agent_outs, dim=-1)
        actions = agent_outs.max(dim=-1)[1]
        return actions.unsqueeze(-1), rnn_states

    def load_checkpoint(self, ckpt_path):
        if 'actor' in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if 'critic' in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))
    
    def to(self, device):
        self.actor.to(device)
        self.target_actor.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        self.discrim.to(device)

    def train(self):
        self.actor.train()
        self.target_actor.train()
        self.mixer.train()
        self.target_mixer.train()
        self.discrim.train()

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
