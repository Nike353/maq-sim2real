from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule
from humanoidverse.utils.running_mean_std import RunningMeanStd
class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std,
                module_dim_dict={}):
        super(PPOActor, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict, module_dim_dict)

        # Action noise
        if module_config_dict.get("freeze_std", False):
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.std.requires_grad = False
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def rollout(self, actor_obs, **kwargs):
        return self.act(actor_obs, **kwargs)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')
        
    def init_rollout(self):
        pass
    
    def clear_rollout(self):
        pass
    
    def eval_mode(self):
        pass
    

class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)
        if module_config_dict.get("running_mean_std", False):
            self.running_mean_std = RunningMeanStd((obs_dim_dict['critic_obs'], ), per_channel=True)
            self.critic_module =  nn.Sequential(self.running_mean_std, self.critic_module)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

# 
    
class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                module_config_dict,
                num_actions,
                module_dim_dict={}):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, module_config_dict, num_actions, 0.0, module_dim_dict)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = mean
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution
    
    @property
    def action_mean(self):
        return self.distribution

    @property
    def action_std(self):
        raise NotImplementedError