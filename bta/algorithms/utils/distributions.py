import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bta.algorithms.utils.util import init, relaxed_softmax
from bta.utils.util import get_weights, set_weights

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)
    
    def rsample(self, hard=True, tau=1.0):
        logits_2d = self.logits.reshape(*self.logits.shape[:-1], self._num_events)
        
        return F.gumbel_softmax(logits_2d, hard=hard, tau=tau)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def log_probs_joint(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
#         super(DiagGaussian, self).__init__()

#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
#         def init_(m): 
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.logstd = AddBias(torch.zeros(num_outputs))

#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)

#         #  An ugly hack for my KFAC implementation.
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()

#         action_logstd = self.logstd(zeros)
#         return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
    
class GaussianTorch(nn.Module):
    def __init__(self, mu, std, train=True, device=torch.device("cpu")):
        super(GaussianTorch, self).__init__()

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.std_x_coef = 1
        self.std_y_coef = 0.5
        self._mu = nn.Parameter(torch.as_tensor(mu), requires_grad=train)
        self.log_std = torch.nn.Parameter(torch.as_tensor(std) * self.std_x_coef, requires_grad=train) 

        self.to(device)

    def forward(self):
        action_mean = self._mu
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
    
    def set_weights(self, weights):
        set_weights([self._mu], weights[0], self.device)
        set_weights([self.log_std], weights[1], self.device)

    @staticmethod
    def from_weights(weights, device=torch.device("cpu")):
        mu = weights[0]
        log_std = weights[1]
        return GaussianTorch(mu, log_std, train=True, device=device)

    def get_weights(self):
        mu_weights = get_weights([self._mu])
        log_std = get_weights([self.log_std])

        return np.concatenate([mu_weights, log_std])


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 1:
            bias = self._bias.t().view(-1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class GumbleSoftmax(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)
    
    def rsample(self):
        logits_2d = self.logits.reshape(-1, self._num_events)
        
        return F.gumbel_softmax(logits_2d, hard=True)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class SoftCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(SoftCategorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        
        return GumbleSoftmax(logits=x)

if __name__ == "__main__":
    actionout = SoftCategorical(64, 6)
    x = torch.rand((32,64))
    action_logits = actionout(x)
    actions = action_logits.rsample()
    action_log_probs = action_logits.log_probs(torch.argmax(actions, -1))

    actionout2 = Categorical(64, 6)
    x2 = torch.rand((32,64))
    action_logits2 = actionout2(x2)
    actions2 = action_logits2.sample()
    action_log_probs2 = action_logits2.log_probs(actions2)

    actionout3 = DiagGaussian(64, 6)
    x3 = torch.rand((32,64))
    action_logits3 = actionout3(x3)
    actions3 = action_logits3.rsample()
    action_log_probs3 = action_logits3.log_probs(actions3)

    print(actions, action_log_probs)