from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
from .modules import BaseModule


class Estimator(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(Estimator, self).__init__()
        self.module = BaseModule(obs_dim_dict, module_config_dict)

    # def estimate(self, obs_history):
    #     return self.module(obs_history)
    
    def forward(self, obs_history):
        return self.module(obs_history)

class Encoder(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, module_dim_dict={}):
        super(Encoder, self).__init__()
        
        self.module = BaseModule(obs_dim_dict, module_config_dict, module_dim_dict)

    def forward(self, encoder_obs):
        return self.module(encoder_obs)
    

class CenetModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, module_dim_dict={}):
        super(CenetModule, self).__init__()
        
        '''encoder module'''
        self.encoder_module = BaseModule(obs_dim_dict, module_config_dict.encoder, module_dim_dict, build_final_layer=False)
        
        # Build encoding heads for latent 
        latent_dim = module_config_dict.encoder['output_dim'][0] # taking the first outputdim in the list
        self.encode_mean_latent = nn.Linear(module_config_dict.encoder.layer_config.hidden_dims[-1], latent_dim)
        self.encode_logvar_latent = nn.Linear(module_config_dict.encoder.layer_config.hidden_dims[-1], latent_dim)

        '''incase velocity is needed'''
        # vel_dim = module_config_dict['output_dim'][1]  
        # self.encode_mean_vel = nn.Linear(vel_dim, 3)
        # self.encode_logvar_vel = nn.Linear(vel_dim, 3)
        

        
        '''decoder module'''
        self.decoder_module = BaseModule(obs_dim_dict, module_config_dict.decoder, module_dim_dict)
        
        
    @property
    def encoder(self):
        return self.encoder_module
    
    @property
    def decoder(self):
        return self.decoder_module
    
    
    def reset(self, dones=None):
        """Reset method for compatibility"""
        pass
    
    def forward(self):
        """Base forward method - should be overridden"""
        raise NotImplementedError
    
    def reparameterise(self, mean, logvar):
        """Reparameterization trick for VAE"""
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code
    
    def cenet_forward(self, cenet_obs):
        """Forward pass through the cenet architecture"""
        # Encode observation history
        distribution = self.encoder(cenet_obs)
        
        # Get latent and velocity parameters
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        
        # Reparameterize to get codes
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        
        # Decode
        decode = self.decoder(code_latent)
        
        return code_latent, decode, mean_latent, logvar_latent
    
class CenetModuleVelocity(CenetModule):
    def __init__(self, obs_dim_dict, module_config_dict, module_dim_dict={}):
        super(CenetModuleVelocity, self).__init__(obs_dim_dict, module_config_dict, module_dim_dict)
        
        vel_dim = module_config_dict.encoder['output_dim'][1]
        self.encode_mean_vel = nn.Linear(module_config_dict.encoder.layer_config.hidden_dims[-1], vel_dim)
        self.encode_logvar_vel = nn.Linear(module_config_dict.encoder.layer_config.hidden_dims[-1], vel_dim)
        
    def cenet_forward(self, cenet_obs):
        distribution = self.encoder(cenet_obs)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        decoder_input = torch.cat([code_latent, code_vel], dim=-1)
        decode = self.decoder(decoder_input)
        return code_latent, code_vel, decode, mean_latent, logvar_latent
        