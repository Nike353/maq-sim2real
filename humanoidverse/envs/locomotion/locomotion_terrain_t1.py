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
from humanoidverse.envs.locomotion.locomotion_t1 import LeggedRobotLocomotion
from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from scipy.stats import vonmises

# TODO: termination with respect height should handle the terrain height well 
class LeggedRobotLocomotionTerrain(LeggedRobotLocomotion):
    def __init__(self, config, device):
        super().__init__(config, device)

    def _init_buffers(self):
        super()._init_buffers()
        if self.config.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.height_samples = torch.tensor(self.simulator.terrain.heightsamples).view(self.simulator.terrain.tot_rows, self.simulator.terrain.tot_cols).to(self.device)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.config.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.config.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.config.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.config.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.simulator.robot_root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.simulator.robot_root_states[:, :3]).unsqueeze(1)

        points += self.simulator.terrain.cfg.border_size
        points = (points/self.simulator.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.simulator.terrain.cfg.vertical_scale
    
    def _update_reset_buf(self):
        super()._update_reset_buf()

        # # TODO: make if config
        # feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        # lowest_height = self.lowest_nearby_terrain_height[:, None]
        
        # # feet stuck into the ground    
        # self.reset_buf |= torch.any(feet_height < lowest_height, dim=1)

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        if self.config.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.terrain.measure_heights:
            self.measured_heights = self._get_heights()


    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return 
        
        distance  = torch.norm(self.simulator.robot_root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        # robots that walked far enough progress to harder terains
        move_up = distance > self.simulator.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def _reward_base_height(self):
        #find mean of measured heights around the robot, keep the dimension to be [num_envs, 1]
        heights = self._get_heights().flatten(start_dim=1)
        mean_heights = torch.mean(heights, dim=1)
        return torch.square(self.simulator.robot_root_states[:, 2] - mean_heights-self.config.rewards.desired_base_height)
    
    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        heights = self._get_heights().flatten(start_dim=1)
        mean_heights = torch.mean(heights, dim=1)   
        base_height_error = torch.square(self.simulator.robot_root_states[:, 2] - mean_heights-self.config.rewards.desired_base_height)
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height)

    
    def _get_obs_measured_heights(self):
        heights = torch.clip(self.simulator.robot_root_states[:, 2].unsqueeze(1)-0.5 - self.measured_heights, -1.0, 1.0)
        return heights.flatten(start_dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.simulator.robot_root_states[:, 2] - self.ground_height
        return torch.square(base_height - self.config.rewards.desired_base_height)

    

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


    