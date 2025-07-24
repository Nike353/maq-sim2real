
import sys
import os
import numpy as np
import torch
from rich.progress import Progress
# from humanoidverse.envs.env_utils.terrain import Terrain

import logging
from loguru import logger
from humanoidverse.utils.logging import capture_stdout_to_loguru
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from hydra.utils import instantiate, get_class

from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
from termcolor import colored
import gymnasium as gym

# Base class for RL tasks
class BaseTask(gym.Env):
    def __init__(self, config, device):
        self.config = config
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # self.simulator = instantiate(config=self.config.simulator, device=device)
        SimulatorClass = get_class(self.config.simulator._target_)
       
        self.simulator: BaseSimulator = SimulatorClass(config=self.config, device=device)
        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW
        self.num_envs = self.config.num_envs
        
        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.dim_obs = self.config.robot.algo_obs_dim_dict["actor_obs"]
        self.dim_critic_obs = self.config.robot.algo_obs_dim_dict["critic_obs"]
        self.dim_actions = self.config.robot.actions_dim
        
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_obs,))
        self.single_action_space = gym.spaces.Box(low=-self.config.robot.control.action_clip_value, high=self.config.robot.control.action_clip_value, shape=(self.dim_actions,))
        
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)
        self.setup_visualize_entities()

        # create envs, sim and viewer
        self._load_assets()
        self._get_env_origins()
        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()
        # self._create_sim()
        self.simulator.prepare_sim()
        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if self.headless == False:
            self.debug_viz = False #TODO
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer
        else:
            self.debug_viz=False #TODO
        self._init_buffers()
        self.all_obs_name = set(self.config.obs.obs_dict.actor_obs + self.config.obs.obs_dict.critic_obs)
        for key, dict in self.config.obs.obs_auxiliary.items():
            self.all_obs_name.update(list(dict.keys()))

        if self.headless == False:
            self.viewer = self.simulator.viewer


    def _init_buffers(self):
        self.obs_buf_dict = {}
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.log_dict = {}

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def reset_all(self):
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.robot_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        self._refresh_sim_tensors()
        
        self._pre_compute_observations_callback()
        self._compute_observations()
        self._post_compute_observations_callback()
        return self.obs_buf_dict
    
    # def _refresh_env_idx_tensors(self, env_ids):
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                 gymtorch.unwrap_tensor(self.all_root_states),
    #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                             gymtorch.unwrap_tensor(self.dof_state),
    #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def reset(self):
        self.reset_all()
        return self.obs_buf_dict, {}
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions
    ###########################################################################
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.config.terrain.max_init_terrain_level
            if not self.config.terrain.curriculum: max_init_level = self.config.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.config.terrain.num_rows
            if isinstance(self.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = torch.from_numpy(self.simulator.terrain.env_origins).to(self.device).to(torch.float)
            else:
                self.terrain_origins = self.simulator.terrain.env_origins.to(self.device).to(torch.float)  
            if self.num_envs == 1:
                self.env_origins[:] = self.env_origins
            else:
                self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            # import ipdb; ipdb.set_trace()
            # print(self.terrain_origins.shape)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _load_assets(self):

        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = self.simulator.num_dof, self.simulator.num_bodies, self.simulator.dof_names, self.simulator.body_names
        
        # check dimensions
        assert self.num_dof == self.dim_actions, "Number of DOFs must be equal to number of actions"

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = self.config.robot.init_state.pos + self.config.robot.init_state.rot + self.config.robot.init_state.lin_vel + self.config.robot.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # env_config = self.config
        self.simulator.create_envs(self.num_envs, 
                                    self.env_origins, 
                                    self.base_init_state)

    def _setup_robot_body_indices(self):
        feet_names = [s for s in self.body_names if self.config.robot.foot_name in s]
        knee_names = [s for s in self.body_names if self.config.robot.knee_name in s]
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(feet_names[i])
        
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.simulator.find_rigid_body_indice(knee_names[i])

        # # TODO: add hand indices after importing the dexterous hand
        # # hand_names = [s for s in self.body_names if self.config.robot.hand_name in s] 
        # self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(hand_names)):
        #     self.hand_indices[i] = self.simulator.find_rigid_body_indice(hand_names[i])
  
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            
            self.penalised_contact_indices[i] = self.simulator.find_rigid_body_indice(penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])
            
        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
            # import ipdb; ipdb.set_trace()
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
            self.waist_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.waist_dof_names]
        
        if hasattr(self.config.robot, 'waist_dof_names'):
            self.waist_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.waist_dof_names]
            self.waist_yaw_dof_indice = self.dof_names.index(self.config.robot.waist_yaw_dof_name) if hasattr(self.config.robot, 'waist_yaw_dof_name') else None
            self.waist_roll_dof_indice = self.dof_names.index(self.config.robot.waist_roll_dof_name) if hasattr(self.config.robot, 'waist_roll_dof_name') else None
            self.waist_pitch_dof_indice = self.dof_names.index(self.config.robot.waist_pitch_dof_name) if hasattr(self.config.robot, 'waist_pitch_dof_name') else None
        
        if hasattr(self.config.robot, 'arm_dof_names'):
            self.arm_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.arm_dof_names]
        
        if hasattr(self.config.robot, 'knee_dof_names'):
            self.knee_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.knee_dof_names]
            self.knee_joint_min_threshold = self.config.robot.get('knee_joint_min_threshold', 0.2)
        
        if hasattr(self.config.robot, 'left_arm_dof_names'):
            self.left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.left_arm_dof_names]
        
        if hasattr(self.config.robot, 'right_arm_dof_names'):
            self.right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.right_arm_dof_names]

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

        # print(self.config.robot)
        if not hasattr(self.config.robot, 'has_end_effector'):
            self.config.robot.has_end_effector = False
            
        if self.config.robot.has_end_effector:
            self.end_effector_name = self.config.robot.end_effector_name
            self.end_effector_index = self.simulator.find_rigid_body_indice(self.end_effector_name)

    def setup_visualize_entities(self):
        pass

    def get_env_state_dict(self):
        return {}
    
    def load_env_state_dict(self, state_dict):
        
        for key, value in state_dict.items():
            if key in self.__dict__:
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(self.device))
                elif isinstance(value, np.ndarray):
                    setattr(self, key, torch.from_numpy(value).to(self.device))
                else:
                    setattr(self, key, value)
            else:
                logger.warning(f"Key {key} not found in env state dict")
    