import os
import numpy as np
import torch
from loguru import logger
from pathlib import Path
import mujoco
import mujoco.viewer

from humanoidverse.utils.torch_utils import *
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator

# Assume BaseSimulator is defined elsewhere.
class MuJoCo(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_cfg = config.robot
        self.device = device
        self.visualize_viewer = False
        if config.save_rendering_dir is not None:
            self.save_rendering_dir = Path(config.save_rendering_dir)
    
    def setup(self):
        # Build the path to the MuJoCo model (MJCF/XML file)
        model_path = os.path.join(
            self.robot_cfg.asset.asset_root, 
            self.robot_cfg.asset.xml_file
        )
        model_path = "humanoidverse/data/robots/T1/T1_locomotion.xml"
        print(f"model: {model_path}")
        # model_path = "g1/scene_29dof_freebase.xml"
        # Load the MuJoCo model and create a simulation instance.
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_substeps = self.simulator_config.sim.substeps
        self.sim_dt = 1 / self.simulator_config.sim.fps  # MuJoCo timestep from the model options.

        self.model.opt.timestep = 1 / self.simulator_config.sim.fps
        # import ipdb;ipdb.set_trace()
        # self.model.opt.iterations = self.sim_substeps
        
        print(self.sim_dt)
        # MuJoCo does not support GPU acceleration, so we use CPU.
        
        # Optionally, set up a viewer for visualization.
        if self.visualize_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

    def setup_terrain(self, mesh_type):
        """Sets up the terrain based on the specified mesh type."""
        if mesh_type == 'plane':
            pass
            # self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognized. Allowed types are [None, plane, heightfield, trimesh]")

    def _create_ground_plane(self):
        """Creates a ground plane in MuJoCo by modifying the model's geom properties."""
        print("Creating plane terrain")

        # MuJoCo uses a geom of type "plane" for ground planes.
        plane_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane")
        if plane_id == -1:
            # Add a new geom for the plane
            new_geom_id = self.model.ngeom
            self.model.geom_pos = np.vstack([self.model.geom_pos, [0, 0, 0]])
            self.model.geom_size = np.vstack([self.model.geom_size, [1, 1, 0.01]])  # Plane size
            self.model.geom_friction = np.vstack([
                self.model.geom_friction,
                [self.simulator_config.terrain.static_friction, 
                 self.simulator_config.terrain.dynamic_friction, 0]
            ])
            self.model.geom_type = np.append(self.model.geom_type, mujoco.mjtGeom.mjGEOM_PLANE)

            print("Created plane terrain")
        else:
            print("Plane terrain already exists.")

    def _create_heightfield(self):
        """Creates a heightfield terrain in MuJoCo."""
        print("Creating heightfield terrain")

        heightfield_size = (self.simulator_config.terrain.width, self.simulator_config.terrain.length)
        height_samples = self._generate_heightfield_data(heightfield_size)

        # MuJoCo expects heightfields to be normalized between 0 and 1
        height_samples = (height_samples - np.min(height_samples)) / (np.max(height_samples) - np.min(height_samples))

        # Define the heightfield in the MuJoCo model
        hf_id = self.model.nhfield
        self.model.hfield_nrow[hf_id] = heightfield_size[0]
        self.model.hfield_ncol[hf_id] = heightfield_size[1]
        self.model.hfield_size[hf_id] = [self.simulator_config.terrain.horizontal_scale, 
                                         self.simulator_config.terrain.horizontal_scale, 
                                         self.simulator_config.terrain.vertical_scale]
        self.model.hfield_data[hf_id] = height_samples.flatten()

        print("Created heightfield terrain")

    def load_assets(self):
        # Extract degrees of freedom (DOFs) and bodies from the model.
        self.num_dof = self.model.nv - 6  # Number of generalized velocities.
        self.num_bodies = self.model.nbody -1
        
        # Retrieve joint and body names.
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(self.model.njnt)][1: ]
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)][1: ]
        # import ipdb; ipdb.set_trace()
        # Validate configuration consistency.
        # import ipdb; ipdb.set_trace()
        assert self.num_dof == len(self.robot_cfg.dof_names), "Number of DOFs must match the config."
        assert self.num_bodies == len(self.robot_cfg.body_names), "Number of bodies must match the config."
        assert self.dof_names == self.robot_cfg.dof_names, "DOF names must match the config."
        assert self.body_names == self.robot_cfg.body_names, "Body names must match the config."
    
    def create_envs(self, num_envs, env_origins, base_init_state):
        # MuJoCo does not support multiple environments in a single simulation.
        # import ipdb; ipdb.set_trace()
        self.num_envs = 1
        self.env_config = self.config
        self.env_origins = env_origins
        self.envs = [self.data]
        self.robot_handles = []  # Not applicable in MuJoCo.
        self.base_init_state = base_init_state
        # Set initial state using provided base_init_state.
        # import ipdb; ipdb.set_trace()
        self.data.qpos[0:3] = base_init_state[0:3].cpu()
        self.data.qpos[3:7] = base_init_state[3:7][..., [3, 0, 1, 2]].cpu()
        self.data.qvel[0:6] = base_init_state[7:13].cpu()
        # mujoco.mj_forward(self.model, self.data)
        self._body_list = self.body_names
        # dof_props_asset = self.get_dof_properties(self.model)
        # dof_props = self._process_dof_props(dof_props_asset, 0)
        return self.envs, self.robot_handles
    
    def prepare_sim(self):
        # In MuJoCo, forward simulation updates the state.
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        # import ipdb; ipdb.set_trace()
        self._rigid_body_pos = torch.tensor([self.data.xpos[1:, :]], device=self.device, dtype=torch.float32)
        self._rigid_body_rot = torch.tensor([self.data.xquat[1:, :]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
        self._rigid_body_vel = torch.tensor([self.data.cvel[1:, 3:6]], device=self.device, dtype=torch.float32) # Using xvelp for global velocity
        self._rigid_body_ang_vel = torch.tensor([self.data.cvel[1:, 0:3]], device=self.device, dtype=torch.float32)
        self._rigid_body_acc = torch.tensor([self.data.cacc[1:, 3:6]], device=self.device, dtype=torch.float32)
        self._rigid_body_ang_acc = torch.tensor([self.data.cacc[1:, 0:3]], device=self.device, dtype=torch.float32)

        self.base_quat = torch.tensor([self.data.qpos[3:7]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
        self.base_pos = torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32)
        self.base_lin_vel = torch.tensor([self.data.qvel[0:3]], device=self.device, dtype=torch.float32)
        self.base_ang_vel = torch.tensor([self.data.qvel[3:6]], device=self.device, dtype=torch.float32)
        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                quat_rotate(self.base_quat, self.base_ang_vel),
            ], dim=-1
        )
        self.robot_root_states = self.all_root_states

        self.dof_pos = torch.tensor([self.data.qpos[7:]], device=self.device, dtype=torch.float32)
        self.dof_vel = torch.tensor([self.data.qvel[6:]], device=self.device, dtype=torch.float32)
        # self.dof_state = torch.concat([self.dof_pos, self.dof_vel], dim=0).reshape(self.num_dof, -1).unsqueeze(0)

        self.contact_forces = torch.tensor([self.data.cfrc_ext[1:, 0:3]], device=self.device, dtype=torch.float32)

        self.dof_forces = torch.tensor([self.data.actuator_force[6:]], device=self.device, dtype=torch.float32)
        # import ipdb; ipdb.set_trace()
        # print(self.dof_forces)
        # self.contact_forces = torch.tensor(
        #     self.robot.get_links_net_contact_force(),
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )

    def refresh_sim_tensors(self):
        self._rigid_body_pos = torch.tensor([self.data.xpos[1:, :]], device=self.device, dtype=torch.float32)
        self._rigid_body_rot = torch.tensor([self.data.xquat[1:, :]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
        self._rigid_body_vel = torch.tensor([self.data.cvel[1:, 3:6]], device=self.device, dtype=torch.float32) # Using xvelp for global velocity
        self._rigid_body_ang_vel = torch.tensor([self.data.cvel[1:, 0:3]], device=self.device, dtype=torch.float32)
        self._rigid_body_acc = torch.tensor([self.data.cacc[1:, 3:6]], device=self.device, dtype=torch.float32)
        self._rigid_body_ang_acc = torch.tensor([self.data.cacc[1:, 0:3]], device=self.device, dtype=torch.float32)

        self.base_quat = torch.tensor([self.data.qpos[3:7]], device=self.device, dtype=torch.float32)[..., [1, 2, 3, 0]]
        self.base_pos = torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32)
        self.base_lin_vel = torch.tensor([self.data.qvel[0:3]], device=self.device, dtype=torch.float32)
        self.base_ang_vel = torch.tensor([self.data.qvel[3:6]], device=self.device, dtype=torch.float32)
        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                quat_rotate(self.base_quat, self.base_ang_vel),
            ], dim=-1
        )
        self.robot_root_states = self.all_root_states

        self.dof_pos = torch.tensor([self.data.qpos[7:]], device=self.device, dtype=torch.float32)
        self.dof_vel = torch.tensor([self.data.qvel[6:]], device=self.device, dtype=torch.float32)
        self.dof_acc = torch.tensor([self.data.qacc[6:]], device=self.device, dtype=torch.float32)
        # import ipdb; ipdb.set_trace()
        self.dof_forces = torch.tensor([self.data.sensordata[54:81]], device=self.device, dtype=torch.float32)
        # np.set_printoptions(precision=10, suppress=False)
        # print(self.data.sensordata[54:81] - self.data.ctrl[6:])
        # self.dof_state = torch.concat([self.dof_pos, self.dof_vel], dim=0).reshape(self.num_dof, -1).unsqueeze(0)
        self.contact_forces = torch.tensor([self.data.cfrc_ext[1:, 3:6]], device=self.device, dtype=torch.float32)

        # print(self.data.cfrc_ext[[7,13], 0:6])
        # import ipdb; ipdb.set_trace()        
    
    def apply_torques_at_dof(self, torques):
        # Convert torch tensor to numpy if needed.
        if isinstance(torques, torch.Tensor):
            torques = torques.cpu().numpy()
        self.data.ctrl[:] = torques
        # mujoco.mj_step(self.model, self.data)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        # In MuJoCo, the full state is given by qpos and qvel.
        root_states = self.robot_root_states[set_env_ids]
        if isinstance(root_states, torch.Tensor):
            root_states[:, 7:10] = quat_rotate_inverse(self.base_quat, root_states[:, 7:10])
            root_states = root_states.cpu().numpy()
        self.data.qpos[0:3] = root_states[0, 0:3]
        self.data.qpos[3:7] = root_states[0, 3:7][..., [3, 0, 1, 2]]
        self.data.qvel[0:6] = root_states[0, 7:13]
        # print(root_states.shape)
        # mujoco.mj_forward(self.model, self.data)
    
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        # Update joint positions and velocities.
        if isinstance(dof_states, torch.Tensor):
            dof_states = dof_states.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        # if len(dof_states.shape) == 2:
        #     self.data.qpos[7:] = dof_states[0]
        #     self.data.qvel[6:] *= 0
        # else:
        self.data.qpos[7:] = dof_states[0, :, 0]
        self.data.qvel[6:] = dof_states[0, :, 1]
        # mujoco.mj_forward(self.model, self.data)
    
    def apply_rigid_body_force_at_pos_tensor(self, force_tensor, force, pos):
        # In MuJoCo, external forces are applied via xfrc_applied.
        # xfrc_applied is a (nbody, 6) array: first three for force, last three for torque.
        self.data.xfrc_applied[:, 0:3] = force_tensor
        # mujoco.mj_step(self.model, self.data)
    
    def simulate_at_each_physics_step(self):
        # import ipdb;
        # ipdb.set_trace()
        mujoco.mj_step(self.model, self.data)
        # if self.viewer is not None:
        #     mujoco.viewer.sync(self.viewer, self.model, self.data)
        self.refresh_sim_tensors()
        if self.viewer is not None:
            self.viewer.sync()

    def get_dof_properties(self, model):
        """ Retrieves the DOF properties for a robot in a MuJoCo simulation.
            Retrieves joint position limits, velocity limits, and torque limits.

        Args:
            model (mujoco.MjModel): MuJoCo model containing the robot's assets.

        Returns:
            dict: A dictionary containing DOF properties like position limits, velocity limits, and torque limits.
        """
        dof_props = {}

        # Position limits (lower and upper)
        dof_props["lower"] = torch.tensor([model.jnt_range[1:][i, 0] for i in range(self.num_dof)])
        dof_props["upper"] = torch.tensor([model.jnt_range[1:][i, 1] for i in range(self.num_dof)])

        # Velocity limits (using model.dof_damping for approximation or another method)
        dof_props["velocity"] = torch.tensor([model.dof_damping[6:][i] for i in range(self.num_dof)])

        # Torque limits (from actuator control range)
        dof_props["effort"] = torch.tensor([model.actuator_ctrlrange[6:][i, 1] for i in range(self.num_dof)])

        return dof_props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_pos_limits_termination = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit

                self.dof_pos_limits_termination[i, 0] = m - 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
                self.dof_pos_limits_termination[i, 1] = m + 0.5 * r * self.env_config.termination_scales.termination_close_to_dof_pos_limit
        return props

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits



    def apply_commands(self, commands_description):
        if commands_description == "forward_command":
            self.commands[:, 4] = 1
            self.commands[:, 0] += 0.4
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "backward_command":
            self.commands[:, 0] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "left_command":
            self.commands[:, 1] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "right_command":
            self.commands[:, 1] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_left_command":
            self.commands[:, 3] -= 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "heading_right_command":
            self.commands[:, 3] += 0.1
            logger.info(f"Current Command: {self.commands[:, ]}")
        elif commands_description == "zero_command":
            self.commands[:, :4] = 0
            logger.info(f"Current Command: {self.commands[:, ]}")

    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        if body_name not in self.body_names:
            raise ValueError(f"Rigid body '{body_name}' not found in the model.")
        return self.body_names.index(body_name)

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        # self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        if self.viewer is None:
            raise RuntimeError("Viewer is not initialized. Call 'setup_viewer' first.")
        return
        # mujoco.mj_step(self.model, self.data)
        # self.viewer.sync()

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)