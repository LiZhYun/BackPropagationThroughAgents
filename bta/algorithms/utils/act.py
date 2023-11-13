
from .distributions import Bernoulli, Categorical, DiagGaussian, SoftCategorical
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, use_attention_action=False):
        super(ACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False
        self.discrete_action = False
        self.single_discrete_action = False
        self.single_continuous_action = False
        self.use_attention_action = use_attention_action
        self.action_type = action_space.__class__.__name__
        if action_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        elif action_space.__class__.__name__ == "list":
            self.single_discrete_action = True
            self.action_outs = []
            action_dim = action_space[0].n
            for act_space in action_space:
                self.action_outs.append(Categorical(inputs_dim, act_space.n, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        elif action_space.__class__.__name__ == "tuple":
            self.single_continuous_action = True
            self.action_outs = []
            action_dim = 0
            for act_space in action_space:
                action_dim += act_space.shape[0]
                self.action_outs.append(DiagGaussian(inputs_dim, act_space.shape[0], use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), SoftCategorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])
        self.action_dim = action_dim
    
    def forward(self, x, available_actions=None, deterministic=False, rsample=True, tau=1.0, joint=False):
        if self.mixed_action :
            actions = []
            action_log_probs = []
            dist_entropy = []
            for idx, action_out in enumerate(self.action_outs):
                if idx == 0:
                    action_logit = action_out(x)
                    if deterministic:
                        action = action_logit.mode()
                    elif rsample:
                        action = action_logit.rsample() 
                    else: 
                        action = action_logit.sample()
                    # action = action_logit.mode() if deterministic else action_logit.rsample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action.float())
                else:
                    action_logit = action_out(x, available_actions)
                    if deterministic:
                        action = action_logit.mode()
                        action_log_prob = action_logit.log_probs(action)
                    elif rsample:
                        action = action_logit.rsample() 
                        action_log_prob = action_logit.log_probs(torch.argmax(action, -1))
                    else: 
                        action = action_logit.sample()
                        action_log_prob = action_logit.log_probs(action)
                    actions.append(action.int())
                action_log_probs.append(action_log_prob)
                dist_entropy.append(action_logit.entropy().mean())

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = (dist_entropy[0] * 0.0025 + dist_entropy[1] * 0.01).unsqueeze(0)

        elif self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                if deterministic:
                    action = action_logit.mode()
                elif rsample:
                    action = action_logit.rsample() 
                else: 
                    action = action_logit.sample()
                # action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        elif self.single_discrete_action:
            actions = []
            action_log_probs = []
            dist_entropy = []
            for idx, action_out in enumerate(self.action_outs):
                action_logit = action_out(x, available_actions[:, idx])
                if rsample:
                    if deterministic:
                        action = action_logit.mode()
                        action = F.one_hot(action.long(), self.action_dim).float().squeeze(1)
                    else:
                        action = action_logit.rsample(tau=tau) 
                        # if joint:
                        #     actions = action_logits.mode() 
                        #     action_log_probs = action_logits.log_probs(actions)
                        # else:
                    action_log_prob = action_logit.log_probs(torch.argmax(action, -1))
                    # dist_entropy = action_logit.entropy()
                    # return actions, action_log_probs, dist_entropy, action_logits.logits
                else:
                    if deterministic:
                        action = action_logit.mode()
                        action_log_prob = action_logit.log_probs(action)
                    else: 
                        action = action_logit.sample()
                        action_log_prob = action_logit.log_probs(action)
                # dist_entropy = action_logit.entropy()
                actions.append(action.int())
                action_log_probs.append(action_log_prob)
                dist_entropy.append(action_logit.entropy().mean())
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)

        elif self.single_continuous_action:
            actions = []
            action_log_probs = []
            dist_entropy = []
            for idx, action_out in enumerate(self.action_outs):
                action_logit = action_out(x)
                if rsample:
                    if deterministic:
                        action = action_logit.mode()
                    else:
                        action = action_logit.rsample(tau=tau) 
                        # if joint:
                        #     actions = action_logits.mode() 
                        #     action_log_probs = action_logits.log_probs(actions)
                        # else:
                    # action_log_prob = action_logit.log_probs(torch.argmax(action, -1))
                    # dist_entropy = action_logit.entropy()
                    # return actions, action_log_probs, dist_entropy, action_logits.logits
                else:
                    if deterministic:
                        action = action_logit.mode()
                        # action_log_prob = action_logit.log_probs(action)
                    else: 
                        action = action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                # dist_entropy = action_logit.entropy()
                actions.append(action)
                action_log_probs.append(action_log_prob)
                dist_entropy.append(action_logit.entropy().mean())
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)
        
        elif self.continuous_action:
            action_logits = self.action_out(x)
            if rsample:
                if deterministic:
                    actions = action_logits.mode() 
                else:
                    actions = action_logits.rsample() 
                # if joint:
                #     actions = action_logits.mode() 
                action_log_probs = action_logits.log_probs(actions)
                dist_entropy = action_logits.entropy()
                return actions, action_log_probs, dist_entropy, action_logits
            else:
                if deterministic:
                    actions = action_logits.mode()
                else: 
                    actions = action_logits.sample()
            # actions = action_logits.mode() if deterministic else action_logits.rsample() 
            action_log_probs = action_logits.log_probs(actions)
            dist_entropy = action_logits.entropy()
        
        elif self.discrete_action:
            action_logits = self.action_out(x, available_actions)
            if rsample:
                if deterministic:
                    actions = action_logits.mode()
                    actions = F.one_hot(actions.long(), self.action_dim).float().squeeze(1)
                else:
                    actions = action_logits.rsample(tau=tau) 
                    # if joint:
                    #     actions = action_logits.mode() 
                    #     action_log_probs = action_logits.log_probs(actions)
                    # else:
                action_log_probs = action_logits.log_probs(torch.argmax(actions, -1))
                dist_entropy = action_logits.entropy()
                return actions, action_log_probs, dist_entropy, action_logits.logits
            else:
                if deterministic:
                    actions = action_logits.mode()
                    action_log_probs = action_logits.log_probs(actions)
                else: 
                    actions = action_logits.sample()
                    action_log_probs = action_logits.log_probs(actions)
            dist_entropy = action_logits.entropy()

        else:
            action_logits = self.action_out(x, available_actions)
            if deterministic:
                actions = action_logits.mode()
            elif rsample:
                actions = action_logits.rsample() 
            else: 
                actions = action_logits.sample()
            # actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
            dist_entropy = action_logits.entropy()
        
        return actions, action_log_probs, dist_entropy

    def get_probs(self, x, available_actions=None):
        if self.mixed_action or self.multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_probs = action_logits.probs
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, rsample=False, tau=1.0, kl=False, joint_actions=None, joint=False):
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            sec = False
            for action_out, act in zip(self.action_outs, action):
                if sec:
                    action_logit = action_out(x, available_actions)
                    action_log_probs.append(action_logit.log_probs(act))
                    if active_masks is not None:
                        if len(action_logit.entropy().shape) == len(active_masks.shape):
                            dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                        else:
                            dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())
                else:
                    sec = True
                    action_logit = action_out(x)
                    action_log_probs.append(action_logit.log_probs(act))
                    if active_masks is not None:
                        if len(action_logit.entropy().shape) == len(active_masks.shape):
                            dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                        else:
                            dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)

        elif self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = sum(dist_entropy)/len(dist_entropy)
        
        elif self.single_discrete_action:
            action_log_probs = []
            dist_entropy = []
            for idx, action_out in enumerate(self.action_outs):
                action_logit = action_out(x, available_actions[:, idx])
                action_log_prob = action_logit.log_probs(action[:, idx])
                action_log_probs.append(action_log_prob)
                dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)

        elif self.single_continuous_action:
            action_log_probs = []
            dist_entropy = []
            for idx, action_out in enumerate(self.action_outs):
                action_logit = action_out(x)
                action_log_prob = action_logit.log_probs(action[:, idx])
                action_log_probs.append(action_log_prob)
                dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = sum(dist_entropy)/len(dist_entropy)

        elif self.continuous_action:
            action_logits = self.action_out(x)
            if joint:
                action_log_probs = action_logits.log_probs(action)
            else:
                action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()   
        elif self.discrete_action:
            action_logits = self.action_out(x, available_actions)
            if joint:
                action_log_probs = action_logits.log_probs_joint(action)
            else:
                action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        if rsample:
            if self.continuous_action:
                train_actions_soft = action_logits.mean
                train_actions = action - train_actions_soft.detach() + train_actions_soft
                # train_actions = train_actions_soft
            elif self.discrete_action:
                train_actions_soft = action_logits.rsample(hard=False, tau=tau)
                train_actions_soft_ = train_actions_soft.gather(1, action.long())
                index = action
                train_actions_hard = torch.zeros_like(train_actions_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index.long(), 1.0)
                train_actions_soft = torch.zeros_like(train_actions_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index.long(), train_actions_soft_)
                train_actions = train_actions_hard - train_actions_soft.detach() + train_actions_soft
                # train_actions = train_actions_soft
            if kl:
                action_log_probs_kl = action_logits.log_probs(joint_actions)
            else:
                action_log_probs_kl = None
            if self.continuous_action:
                return train_actions, action_log_probs, action_log_probs_kl, dist_entropy, action_logits
            elif self.discrete_action:
                return train_actions, action_log_probs, action_log_probs_kl, dist_entropy, action_logits.logits
            
        else:
            return action_log_probs, dist_entropy
    
    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector,-1)
            action_std = torch.cat(std_collector,-1)
            all_probs = torch.cat(probs_collector,-1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs