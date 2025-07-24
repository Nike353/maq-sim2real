import torch
import torch.nn as nn
import inspect
from .modules import BaseModule



class GRU(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, module_dim_dict={}):
        super(GRU, self).__init__()
        self.module = BaseModule(obs_dim_dict, module_config_dict, module_dim_dict)
        self.input_dim = self.module.input_dim

    def forward(self, input, hidden_state):
        input = input.unsqueeze(1)
        hidden_state = hidden_state.unsqueeze(0)
        output, hidden_state = self.module(input, hidden_state)
        return output.squeeze(1), hidden_state.squeeze(0)
    