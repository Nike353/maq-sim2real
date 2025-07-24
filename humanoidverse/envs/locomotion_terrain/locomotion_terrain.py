from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from scipy.stats import vonmises


class LeggedRobotLocomotionTerrain(LeggedRobotLocomotion):
    def __init__(self, config, device):
        super().__init__(config, device)

    def _update_reset_buf(self):
        super()._update_reset_buf()

        # TODO: make if config
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        lowest_height = self.lowest_nearby_terrain_height[:, None]
        
        # feet stuck into the ground    
        self.reset_buf |= torch.any(feet_height < lowest_height, dim=1)

    def _get_obs_height_map(self):
        height_map =  self.simulator.get_height_map()
        # [batchsize, 9, 9] -> [batchsize, 81]
        return height_map.flatten(start_dim=1)  # flatten dimensions 1 and 2 (9x9) while preserving batch dimension
    

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.simulator.robot_root_states[:, 2] - self.ground_height
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2] - self.ground_height[:, None]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 

    def _reward_penalty_adaptive_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2] - self.ground_height[:, None]
        feet_height_target = self.highest_nearby_terrain_height[:, None] - self.ground_height[:, None]
        feet_height_target = torch.clamp(feet_height_target, min=0.0, max=0.55)
        # import ipdb; ipdb.set_trace()
        dif = torch.abs(feet_height - feet_height_target)
        # turn nan to 0.0
        # dif = torch.nan_to_num(dif, 0.0)
        # assert this is not nan
        # import ipdb; ipdb.set_trac
        # print("max feet height: ", torch.max(feet_height))
        # print("min feet height: ", torch.min(feet_height))
        # print("max feet height target: ", torch.max(feet_height_target))
        # print("min feet height target: ", torch.min(feet_height_target))
        assert torch.all(torch.isfinite(dif))
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 


    # fix bug: ground height is not updated
    @property
    def ground_height(self):
        height_map =  self.simulator.get_height_map()
        return height_map[:, 4, 4]
    
    @property
    def highest_nearby_terrain_height(self):
        height_map =  self.simulator.get_height_map()
        # extract inner 3x3 from the 9x9
        height_map = height_map[:, 2:7, 2:7].reshape(self.num_envs, -1)
        highest_height = height_map.max(dim=1).values
        # print("highest_height: ", highest_height.shape)
        return highest_height
    
    @property
    def lowest_nearby_terrain_height(self):
        height_map =  self.simulator.get_height_map()
        # extract inner 3x3 from the 9x9
        height_map = height_map[:, 2:7, 2:7].reshape(self.num_envs, -1)
        lowest_height = height_map.min(dim=1).values
        return lowest_height