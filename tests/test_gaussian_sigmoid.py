"""Bounded local Gaussian scales: analytic map, privacy, gradients and safety."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from bta_repro.agentmixer_continuous import ContinuousAgentMixerPolicy
from bta_repro.recurrent import RecurrentContinuousAgentMixerPolicy
from bta_repro.simulator_train import Config,make_policy

RELEASED_STD=0.5/(1+math.exp(-1))

def example(recurrent=False,modifier='residual',seed=37):
    torch.manual_seed(seed)
    cls=RecurrentContinuousAgentMixerPolicy if recurrent else ContinuousAgentMixerPolicy
    policy=cls(3,4,2,2,8,initial_std=RELEASED_STD,local_scale='state_independent_sigmoid',modifier_scale=modifier)
    observation=torch.randn(5,2,3)
    if recurrent:observation=policy.pack_observations(observation,torch.randn(5,2,8))
    return policy,observation,torch.randn(5,4)

@pytest.mark.parametrize('recurrent',[False,True])
def test_released_initial_parameter_and_analytic_entropy_derivative(recurrent):
    policy,obs,_=example(recurrent)
    mean,std=policy.local_parameters(obs)
    for actor in policy.local_actors:assert torch.equal(actor.scale_score,torch.ones(2))
    torch.testing.assert_close(std,torch.full_like(std,RELEASED_STD),rtol=1e-7,atol=0)
    # H(N(mu,sigma)) = log(sigma)+constant, so dH/dscore=1-sigmoid(score).
    Normal(mean,std).entropy().sum().backward()
    expected=5/(1+math.exp(1))
    for actor in policy.local_actors:
        torch.testing.assert_close(actor.scale_score.grad,torch.full_like(actor.scale_score,expected))

@pytest.mark.parametrize('recurrent',[False,True])
@pytest.mark.parametrize('modifier',['residual','bounded'])
def test_means_are_exact_and_local_uncertainty_is_private_and_state_independent(recurrent,modifier):
    policy,obs,state=example(recurrent,modifier)
    nn.init.normal_(policy.modifier.log_std_head.weight,std=.2)
    local_mean,local_std=policy.local_parameters(obs)
    for changed_state in (state,state+100):
        assert torch.equal(local_mean,policy.joint_parameters(obs,changed_state)[0])
    changed=obs.clone();changed[:,1]+=20
    mean,std=policy.local_parameters(changed)
    assert torch.equal(mean[:,0],local_mean[:,0]) and torch.equal(std,local_std)
    with torch.no_grad():policy.local_actors[0].scale_score.add_(.7)
    new_std=policy.local_parameters(obs)[1]
    assert not torch.equal(new_std[:,0],local_std[:,0]) and torch.equal(new_std[:,1],local_std[:,1])
    assert torch.equal(new_std,new_std[:1].expand_as(new_std))

@pytest.mark.parametrize('recurrent',[False,True])
@pytest.mark.parametrize('score',[-1e30,1e30])
def test_extreme_scores_have_finite_positive_scales_and_defined_floor_gradient(recurrent,score):
    policy,obs,state=example(recurrent)
    with torch.no_grad():
        for actor in policy.local_actors:actor.scale_score.fill_(score)
    mean,std=policy.local_parameters(obs)
    assert torch.isfinite(std).all() and (std>=math.exp(-5)*(1-1e-6)).all() and (std<=.5).all()
    logp=Normal(mean,std).log_prob(mean.detach()+.1).sum(-1)
    logp.sum().backward()
    for actor in policy.local_actors:
        assert torch.isfinite(actor.scale_score.grad).all()
        assert torch.count_nonzero(actor.scale_score.grad)==0
    raw,scores=policy.sample(obs,state)
    assert torch.isfinite(raw).all() and torch.isfinite(scores).all()

@pytest.mark.parametrize('recurrent',[False,True])
@pytest.mark.parametrize('modifier',['residual','bounded'])
def test_log_likelihood_scale_gradient_matches_finite_difference(recurrent,modifier):
    policy,obs,state=example(recurrent,modifier)
    policy=policy.double();obs=obs.double();state=state.double()
    nn.init.normal_(policy.modifier.log_std_head.weight,std=.2)
    raw=policy.local_actions(obs).detach()+.1
    parameter=policy.local_actors[0].scale_score
    def objective():return policy.evaluate_actions(obs,state,raw)[0].sum()
    objective().backward();gradient=parameter.grad[0].clone()
    assert gradient.abs()>0
    with torch.no_grad():
        original=parameter[0].clone();parameter[0]=original+1e-5;plus=objective().item()
        parameter[0]=original-1e-5;minus=objective().item();parameter[0]=original
    torch.testing.assert_close(gradient,gradient.new_tensor((plus-minus)/2e-5),rtol=2e-5,atol=1e-8)

@pytest.mark.parametrize('recurrent',[False,True])
@pytest.mark.parametrize('modifier',['residual','bounded'])
def test_parameterization_control_preserves_mean_modifier_initialization_and_rng(recurrent,modifier):
    cls=RecurrentContinuousAgentMixerPolicy if recurrent else ContinuousAgentMixerPolicy
    policies=[];rng=[]
    for local_scale in ('state_independent','state_independent_sigmoid'):
        torch.manual_seed(719)
        policies.append(cls(3,4,2,2,8,initial_std=RELEASED_STD,modifier_scale=modifier,local_scale=local_scale))
        rng.append(torch.get_rng_state().clone())
    assert torch.equal(*rng)
    a,b=policies
    for name,value in a.state_dict().items():
        if not name.endswith('.log_std'):assert torch.equal(value,b.state_dict()[name]),name
    obs=torch.randn(5,2,3)
    if recurrent:obs=a.pack_observations(obs,torch.randn(5,2,8))
    ma,sa=a.local_parameters(obs);mb,sb=b.local_parameters(obs)
    assert torch.equal(ma,mb)
    # The inverse maps round differently; scale agreement is numerical,not bitwise.
    torch.testing.assert_close(sa,sb,rtol=2e-7,atol=0)

@pytest.mark.parametrize('initial_std',[.5,1.,math.exp(2)])
def test_unrepresentable_initial_scale_is_rejected_before_rng_changes(initial_std):
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='initial_std'):
        ContinuousAgentMixerPolicy(3,4,2,2,8,initial_std=initial_std,local_scale='state_independent_sigmoid')
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='initial_std'):
        Config(environment='mujoco',continuous_local_scale='state_independent_sigmoid',initial_std=initial_std).validate()

@pytest.mark.parametrize('environment,algorithm',[('smac','agentmixer'),('football','agentmixer'),('prey','agentmixer'),('mujoco','mappo'),('mujoco','arppo'),('mujoco','bppo')])
def test_scale_option_rejects_unsupported_families(environment,algorithm):
    cfg=Config(environment=environment,algorithm=algorithm,continuous_local_scale='state_independent_sigmoid')
    with pytest.raises(ValueError,match='local.scale|local_scale'):cfg.validate()
    env=SimpleNamespace(is_discrete=environment!='mujoco')
    with pytest.raises(ValueError,match='local.scale|local_scale'):make_policy(cfg,env)
