try:
    with open("./humanoidverse/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
        DEFAULT_ISAACSIM_VERSION = f.read().strip()
except FileNotFoundError:
    DEFAULT_ISAACSIM_VERSION = "4.5"


if DEFAULT_ISAACSIM_VERSION == "4.5":

    ### IsaacSim 4.5

    import sys
    import os
    from loguru import logger
    import importlib.util
    import torch
    from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
    import numpy as np
    from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
    # from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.utils.timer import Timer

    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
    from isaaclab.assets import ArticulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
    from isaaclab.terrains import TerrainGeneratorCfg
    import isaaclab.terrains as terrain_gen

    from isaaclab_assets import H1_CFG
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab.envs import ViewerCfg

    import isaaclab.sim as sim_utils

    from humanoidverse.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
    import builtins
    import inspect
    import copy
    from humanoidverse.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG

    from humanoidverse.simulator.isaacsim.event_cfg import EventCfg

    from isaaclab.managers import EventManager

    from isaaclab.managers import EventTermCfg as EventTerm

    from isaaclab.managers import SceneEntityCfg
    import isaaclab.envs.mdp as mdp
    from humanoidverse.simulator.isaacsim.events import randomize_body_com
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
    from isaaclab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg, FrameTransformerCfg, FrameTransformer
    from isaaclab.assets import ArticulationCfg, Articulation
    from isaaclab.assets import RigidObject, RigidObjectCfg
    
    # Common imports for all IsaacSim versions
    import torch.nn.functional as F
    import cv2
    import uuid
    from humanoidverse.utils.helpers import dict_to_true_list
    from pxr import UsdShade
    import omni

elif DEFAULT_ISAACSIM_VERSION == "4.2":
    #### IsaacSim 4.2

    import sys
    import os
    import importlib.util
    from loguru import logger
    import torch
    from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
    import numpy as np
    from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
    # from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg
    from omni.isaac.lab.sim import SimulationContext
    from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.scene import InteractiveScene
    from omni.isaac.lab.utils.timer import Timer
    from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers

    from omni.isaac.lab.assets import Articulation
    from omni.isaac.lab.assets.rigid_object import RigidObjectCfg, RigidObject
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.lab.sensors import ContactSensor, RayCaster, Camera, TiledCamera
    from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg, TiledCameraCfg
    from omni.isaac.lab.assets import ArticulationCfg
    from omni.isaac.lab.terrains import TerrainImporterCfg
    from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
    from omni.isaac.lab.terrains import TerrainGeneratorCfg
    import omni.isaac.lab.terrains as terrain_gen

    from omni.isaac.lab_assets import H1_CFG
    from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from omni.isaac.lab.envs import ViewerCfg

    import omni.isaac.lab.sim as sim_utils

    from humanoidverse.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
    import builtins
    import inspect
    import copy
    from humanoidverse.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG

    from humanoidverse.simulator.isaacsim.event_cfg import EventCfg

    from omni.isaac.lab.managers import EventManager

    from omni.isaac.lab.managers import EventTermCfg as EventTerm

    from omni.isaac.lab.managers import SceneEntityCfg
    import omni.isaac.lab.envs.mdp as mdp
    from humanoidverse.simulator.isaacsim.events import randomize_body_com
    import torch.nn.functional as F
    import cv2
    import uuid
    from humanoidverse.utils.helpers import dict_to_true_list
    from pxr import UsdShade
    import omni

else:
    raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")


class IsaacSim(BaseSimulator):
    def __init__(self, config, device, **kwargs):
        super().__init__(config, device)
        
        if "task" in config:
            self.task_type = config.task.type   # {"manipulation", "locomotion"}
            self.task_name = config.task.name
            self.task_config = config.task

        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand

        
        sim_config: SimulationCfg = SimulationCfg(dt=1./self.simulator_config.sim.fps, 
                                           render_interval=self.simulator_config.sim.render_interval, 
                                           device=self.sim_device,
                                           physx=PhysxCfg(solver_type=self.simulator_config.sim.physx.solver_type,
                                                          max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                                                          max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations))
        
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        # self.sim.set_camera_view([0.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        
        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1./self.simulator_config.sim.fps}")
        logger.info(f"\tRendering step-size   : {1./self.simulator_config.sim.fps * self.simulator_config.sim.substeps}")


        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)
        
        
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=self.simulator_config.scene.num_envs, env_spacing=self.simulator_config.scene.env_spacing, replicate_physics=self.simulator_config.scene.replicate_physics)
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)
    
        
        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
        
        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu")


        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(self.domain_rand_config["friction_range"]),
                    "operation": "scale",
                },
            )

        if self.domain_rand_config.get("randomize_base_com", False):
            
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            "torso_link",
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor([float(self.domain_rand_config["base_com_range"]["x"][0]), float(self.domain_rand_config["base_com_range"]["y"][0]), float(self.domain_rand_config["base_com_range"]["z"][0])]),
                        torch.tensor([float(self.domain_rand_config["base_com_range"]["x"][1]), float(self.domain_rand_config["base_com_range"]["y"][1]), float(self.domain_rand_config["base_com_range"]["z"][1])])
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )  

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
        
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
                
        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        
        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None


        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation


        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # print the environment information
        logger.info("Completed setting up the environment...")
        
    def _setup_scene(self):
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file
        # prapare to override the spawn configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
        else:
            raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")
        
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(asset_root, asset_path),
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions), solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            
        )
        
        # prepare to override the articulation configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )
       
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")    
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping
        
        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")


        # ImplicitActuatorCfg IdealPDActuatorCfg  
        # actuators = {
        #     dof_names_list[i]: IdealPDActuatorCfg(
        #         joint_names_expr=[dof_names_list[i]],
        #         effort_limit=dof_effort_limit_list[i],
        #         velocity_limit=dof_vel_limit_list[i],
        #         stiffness=0,
        #         damping=0,
        #         armature=dof_armature_list[i],
        #         friction=dof_joint_friction_list[i],
        #     ) for i in range(len(dof_names_list))
        # }
        actuators = dict()
        actuators["all"] = IdealPDActuatorCfg(
            joint_names_expr=[dof_names_list[i] for i in range(len(dof_names_list))],
            effort_limit={
                dof_names_list[i]: dof_effort_limit_list[i] for i in range(len(dof_names_list))
            },
            velocity_limit={
                dof_names_list[i]: dof_vel_limit_list[i] for i in range(len(dof_names_list))
            },
            stiffness=0,
            damping=0,
            armature={
                dof_names_list[i]: dof_armature_list[i] for i in range(len(dof_names_list))
            },
            friction={
                dof_names_list[i]: dof_joint_friction_list[i] for i in range(len(dof_names_list))
            }
        )
        
        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        

        # TODO: Haotian: add contact filter for manipulation tasks
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            update_period=0.005,
            track_air_time=True,
            debug_vis=False,
        )
        
        # # NOTE: Lingyun - 0408: Currently hard-code the fingertip contacts since isaaclab supports only one prim_path for each env
        # right_hand_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
        #     prim_path=f"/World/envs/env_.*/{self.task_config.target_obj}",
        #     debug_vis=True,
        #     history_length=3,
        #     update_period=0.005,
        #     track_air_time=True,
        #     filter_prim_paths_expr=["/World/envs/env_.*/Robot/right_hand_(thumb|index|middle)_[0-9]_link",]
        # )

        if "task_config" in self.__dict__ and hasattr(self.task_config, "target_obj"):
            right_hand_thumb_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                # NOTE: Lingyun - 0322 update: exclude the palm link
                # prim_path="/World/envs/env_.*/Robot/right_hand_thumb_2_link",
                prim_path=f"/World/envs/env_.*/{self.task_config.target_obj}",
                debug_vis=False,
                history_length=3,
                update_period=0.005,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Robot/right_hand_thumb_2_link",]
                # filter_prim_paths_expr=[f"/World/envs/env_.*/{self.task_config.target_obj}"]
            )
            
            right_hand_index_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                # prim_path="/World/envs/env_.*/Robot/right_hand_index_1_link",
                prim_path=f"/World/envs/env_.*/{self.task_config.target_obj}",
                debug_vis=False,
                history_length=3,
                update_period=0.005,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Robot/right_hand_index_1_link",]
                # filter_prim_paths_expr=[f"/World/envs/env_.*/{self.task_config.target_obj}"]
            )
            right_hand_middle_contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
                # prim_path="/World/envs/env_.*/Robot/right_hand_middle_1_link",
                prim_path=f"/World/envs/env_.*/{self.task_config.target_obj}",
                debug_vis=False,
                history_length=3,
                update_period=0.005,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Robot/right_hand_middle_1_link",]
                # filter_prim_paths_expr=[f"/World/envs/env_.*/{self.task_config.target_obj}"]
            )


        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True,
        )

        if hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                # Add a height scanner to the torso to detect the height of the terrain mesh
                height_scanner_config = RayCasterCfg(
                    prim_path="/World/envs/env_.*/Robot/pelvis",
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                    attach_yaw_only=True,
                    # Apply a grid pattern that is smaller than the resolution to only return one height value.
                    pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.6]),
                    debug_vis=True,
                    mesh_prim_paths=["/World/ground"],
                )

        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                self._height_scanner = RayCaster(height_scanner_config)
                self.scene.sensors["height_scanner"] = self._height_scanner
        
        # import ipdb; ipdb.set_trace()
        # print(self.simulator_config.cameras)


        # parse camera types:
        if hasattr(self.simulator_config, "cameras") and hasattr(self.simulator_config.cameras, "camera_types"):
            camera_types = dict_to_true_list(self.simulator_config.cameras.camera_types)
        else:
            camera_types = ["depth", "rgb"]

        print(f"camera_types: {camera_types}")

        if hasattr(self.simulator_config, "cameras") and getattr(self.simulator_config.cameras, "enable_cameras", False):
            ego_camera_config = TiledCameraCfg(
                prim_path="/World/envs/env_.*/ego_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
                data_types=camera_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=2.5, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
                ),
                width=self.simulator_config.cameras.camera_resolutions[0],
                height=self.simulator_config.cameras.camera_resolutions[1],
            )
            self.ego_camera = TiledCamera(ego_camera_config)
            self.scene.sensors["ego_camera"] = self.ego_camera
        else: 
            self.ego_camera = None

        



        if (self.terrain_config.mesh_type == "heightfield") or (self.terrain_config.mesh_type == "trimesh"):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                            # proportion=proportion, noise_range=(0.002, 0.02), noise_step=0.002, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.95, grid_height_range=(0.01, 0.15), platform_width=2.0
                        )
                    elif terrain_type == "stairs_up":
                        sub_terrains[terrain_type] = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                            proportion=proportion,
                            step_height_range=(0.01, self.terrain_config.stairs.max_height),
                            # step_height_range=(0.25, 0.25),
                            # step_height_range=(0.01, 0.40),
                            step_width=0.5,
                            platform_width=4.0,
                            border_width=0.25,
                            holes=False
                        )
                    elif terrain_type == "stairs_down":
                        sub_terrains[terrain_type] = terrain_gen.MeshPyramidStairsTerrainCfg(
                            proportion=proportion,
                            step_height_range=(0.01, self.terrain_config.stairs.max_height),
                            # step_height_range=(0.3, 0.3),
                            # step_height_range=(0.01, 0.40),
                            step_width=0.5,
                            platform_width=4.0,
                            border_width=0.25,
                            holes=False
                        )
                        

            # create a visual material
            visual_material_cfg = sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                # mdl_path="/home/wenli/materials/Architecture/Ceiling_Tiles.mdl",
                # mdl_path="/home/wenli/materials/Wood/Birch_Planks.mdl",
                project_uvw=True,
                # texture_scale=10.0
            )


            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                # visual_material=sim_utils.MdlFileCfg(
                #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                #     project_uvw=True,
                # ),
                visual_material=visual_material_cfg,
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            # terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
        
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot

        self._task = {}
        self.task_root_origin = {}
        if "task_type" in self.__dict__ and self.task_type=="manipulation":
            import_cfg_path = os.path.join("humanoidverse/data/tasks/", f"{self.task_name}/scenario_cfg/isaacsim.py")
            # Load the module from the file
            module_name = f"{self.task_name}"
            spec = importlib.util.spec_from_file_location(module_name, import_cfg_path)
            task_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = task_module
            spec.loader.exec_module(task_module)
            
            TaskObjCfgDict = task_module.TaskObjCfgDict

            for name, obj_cfg in TaskObjCfgDict.items():
                if name not in self.config.task.objects:
                    continue
                
                obj_cfg = obj_cfg.replace(prim_path=f"/World/envs/env_.*/{name}")
                self._task[name] = obj_cfg.class_type(obj_cfg)
                # import ipdb; ipdb.set_trace()
                if isinstance(self._task[name], Articulation):
                    self.scene.articulations[name] = self._task[name]
                elif isinstance(self._task[name], RigidObject):
                    self.scene.rigid_objects[name] = self._task[name]
                else:
                    raise NotImplementedError(f"Task object {name} is not supported.")
                
                self.task_root_origin[name] = torch.cat(
                    [torch.tensor(obj_cfg.init_state.pos, device=self.sim_device), 
                    torch.tensor(obj_cfg.init_state.rot, device=self.sim_device), 
                    torch.tensor(obj_cfg.init_state.lin_vel, device=self.sim_device),
                    torch.tensor(obj_cfg.init_state.ang_vel, device=self.sim_device)])
                
            self.task_root_origin["robot"] = torch.cat([torch.tensor(robot_articulation_config.init_state.pos, device=self.sim_device), 
                    torch.tensor(robot_articulation_config.init_state.rot, device=self.sim_device), 
                    torch.tensor(robot_articulation_config.init_state.lin_vel, device=self.sim_device),
                    torch.tensor(robot_articulation_config.init_state.ang_vel, device=self.sim_device)])
                
        self.contact_sensor = ContactSensor(contact_sensor_config)
        # self.left_hand_contact_sensor = ContactSensor(left_hand_contact_sensor_config)
        # self.right_hand_contact_sensor = ContactSensor(right_hand_contact_sensor_config)
        
        if hasattr(self, "task_config") and hasattr(self.task_config, "target_obj_contact_sensor"):
            self.right_hand_thumb_contact_sensor = ContactSensor(right_hand_thumb_contact_sensor_config)
            self.right_hand_index_contact_sensor = ContactSensor(right_hand_index_contact_sensor_config)
            self.right_hand_middle_contact_sensor = ContactSensor(right_hand_middle_contact_sensor_config)

        # self.scene.sensors["contact_sensor"] = self.contact_sensor
        # self.scene.sensors["left_hand_contact_sensor"] = self.left_hand_contact_sensor
        # self.scene.sensors["right_hand_contact_sensor"] = self.right_hand_contact_sensor
    
        if hasattr(self, "simulator_config") and hasattr(self.simulator_config, "heightmap"):
            if self.simulator_config.heightmap.enable_heightmap:
                self._height_scanner = RayCaster(height_scanner_config)
                self.scene.sensors["height_scanner"] = self._height_scanner
        
        
        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins


        if hasattr(self.simulator_config, "cameras")  and  getattr(self.simulator_config.cameras, "enable_cameras", False):
            ego_camera_config = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/ego_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0.0, 0.0, 0.0), convention="world"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self.simulator_config.cameras.camera_focal_length, 
                    focus_distance=self.simulator_config.cameras.camera_focus_distance, 
                    horizontal_aperture=self.simulator_config.cameras.camera_horizontal_aperture, 
                    clipping_range=eval(self.simulator_config.cameras.camera_clipping_range)
                ),
                width=self.simulator_config.cameras.camera_resolutions_width,
                height=self.simulator_config.cameras.camera_resolutions_height,
                debug_vis=True,
            )


            self.ego_camera = TiledCamera(ego_camera_config)
            self.scene.sensors["ego_camera"] = self.ego_camera
        else:
            self.ego_camera = None
        
        # import ipdb; ipdb.set_trace()

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])

        # add lights
        light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        light_config.func("/World/Light", light_config) # added for manipulation

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))
        
        self.vis_spheres = VisualizationMarkers(VisualizationMarkersCfg( prim_path="/Visuals/goal_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),),
            }))
        if self.config.simulator.get('enable_cameras', False):
            self.setup_rendering_cameras()
        
    def setup_keyboard(self):
        # TODO: add back
        # from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard
        # self.keyboard_interface = Se2Keyboard()
        pass
        
    def add_keyboard_callback(self, key, callback):
        # TODO: add back
        # self.keyboard_interface.add_callback(key, callback)
        pass
        
        
    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            if DEFAULT_ISAACSIM_VERSION == "4.5":   
                ### IsaacSim 4.5
                from isaacsim.util.debug_draw import _debug_draw
            elif DEFAULT_ISAACSIM_VERSION == "4.2":
                ### IsaacSim 4.2
                from omni.isaac.debug_draw import _debug_draw
            else:
                raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")
            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1. / self.simulator_config.sim.fps
        if not self.headless:
            self.setup_keyboard()
        
        
    def setup_terrain(self, mesh_type):
        pass


    def load_assets(self):
        '''
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        '''

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
    
        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True) 
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)
        
        # self.simulator._robot.find_bodies("d435_link", preserve_order=True)
        # if get the camera attached link, then get the body id
        if hasattr(self.simulator_config, "cameras") and self.simulator_config.cameras.enable_cameras:
            self.camera_body_id = self._robot.find_bodies(self.simulator_config.cameras.camera_attached_link, preserve_order=True)[0]
            logger.info(f"Camera attached link: {self.simulator_config.cameras.camera_attached_link}, Camera body id: {self.camera_body_id}")

        # import ipdb; ipdb.set_trace()
        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, humanoidverse config)
            # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

    
        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        '''
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19], ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        '''
        
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning("The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim.")
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"
       
        
        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names
        
    def set_ego_camera_pose(self, base_pos, base_quat_xyzw):
        if self.ego_camera is not None:
            # base_
            base_quat_wxyz = torch.cat([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, :3]], dim=1)
            self.ego_camera.set_world_poses(base_pos, base_quat_wxyz, convention="world")


    def create_envs(self, num_envs, env_origins, base_init_state):
        
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        
        return self.scene, self._robot
    
    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices
                
    def prepare_sim(self):
        self.refresh_sim_tensors() # initialize tensors

    def get_height_map(self):
        height_tensor = self.scene.sensors["height_scanner"].data.ray_hits_w[:, :, 2]
        batch_size = height_tensor.shape[0]
        height_map = height_tensor.view(batch_size, 9, 9) # TODO: config this resolution by heightmap config
        return height_map
    
    def get_rgb_image(self):
        # import ipdb; ipdb.set_trace()
        if self.ego_camera is not None:
            # Get RGB image from the ego camera
            # The image is typically in shape [batch_size, height, width, channels]
            rgb_image = self.scene.sensors["ego_camera"].data.output["rgb"]
            
            # Convert to float tensor if it's not already
            if rgb_image.dtype != torch.float:
                rgb_image = rgb_image.float()
                
            # If values are in 0-255 range, normalize to 0-1
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image / 255.0
                
            return rgb_image
        else:
            # Return empty tensor if camera is not available
            camera_resolution = self.simulator_config.cameras.camera_resolutions if hasattr(self.simulator_config, "cameras") else [0, 0]
            return torch.zeros((self.scene.cfg.num_envs, camera_resolution[0], camera_resolution[1], 3), 
                              device=self.sim_device, dtype=torch.float)
        
    def get_depth_image(self):
        if self.ego_camera is not None:
            # Get depth image from the ego camera
            depth_image = self.scene.sensors["ego_camera"].data.output["depth"]
            # Convert to float tensor if it's not already
            if depth_image.dtype != torch.float:
                depth_image = depth_image.float()
            # mask the inf to 0
            depth_image = torch.where(depth_image == float('inf'), torch.zeros_like(depth_image), depth_image)
            
            return depth_image
        else:
            return torch.zeros((self.scene.cfg.num_envs, self.simulator_config.cameras.camera_resolutions[0], self.simulator_config.cameras.camera_resolutions[1], 1), device=self.sim_device, dtype=torch.float)

    def compute_grad_norm(self):
        height_tensor = self.scene.sensors["height_scanner"].data.ray_hits_w[:, :, 2]
        batch_size = height_tensor.shape[0]
        height_map = height_tensor.view(batch_size, 9, 9)
        
        height_map_unsqueezed = height_map.unsqueeze(1)
        
        kernel_x = torch.tensor([[-0.5, 0, 0.5]], dtype=height_map.dtype, 
                                device=height_map.device).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], dtype=height_map.dtype, 
                                device=height_map.device).view(1, 1, 3, 1)

        grad_x = F.conv2d(height_map_unsqueezed, kernel_x, padding=(0,1))
        grad_y = F.conv2d(height_map_unsqueezed, kernel_y, padding=(1,0))
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2)
    
        # Remove the channel dimension: result shape is (batch_size, 9, 9)
        return grad_norm.squeeze(1)

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        ############################################################################################
        # TODO: currently, we only consider the robot root state, ignore other objects's root states
        # 2025.2.22: refrese all objects' root states, but only record robot's root state
        ############################################################################################
        # self.all_root_states = self._robot.data.root_state_w  # (num_envs, 13)
        # self.robot_root_states = self.all_root_states # (num_envs, 13)
        self.all_root_states = {}
        self.all_root_states["robot"] = self._robot.data.root_state_w # (num_envs, 13)
        self.robot_root_states = self.all_root_states["robot"] 

        for task_name, task_obj in self._task.items():
            self.all_root_states[task_name] = task_obj.data.root_state_w # (num_envs, 13)   
    
        self.base_quat = self.robot_root_states[:, [4, 5, 6, 3]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids] # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]
        
        self.contact_forces = self.contact_sensor.data.net_forces_w # (num_envs, num_bodies, 3)
        
        # NOTE: Lingyun - exclude the thumb contact force
        if  hasattr(self, "task_config") and hasattr(self.task_config, "target_obj_contact_sensor"):
            self.right_hand_contact_forces = torch.cat([self.right_hand_index_contact_sensor.data.net_forces_w, self.right_hand_middle_contact_sensor.data.net_forces_w], dim=1)
            self.left_hand_contact_forces = torch.zeros(self.num_envs, 3, 3, device=self.sim_device)
        
        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]

        # camera
        if hasattr(self.simulator_config, "cameras") and self.simulator_config.cameras.enable_cameras:
            self._camera_pos = self._robot.data.body_pos_w[:, self.camera_body_id, :]
            self._camera_quat = self._robot.data.body_quat_w[:, self.camera_body_id][:, :, [1, 2, 3, 0]]

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states): #TODO: only robot actor root state
        self._robot.write_root_state_to_sim(root_states[set_env_ids, :], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)

    def set_task_root_state_tensor(self, set_env_ids, root_states):  
        for name, task_obj in self._task.items():
            task_obj.write_root_state_to_sim(root_states[name][set_env_ids, :], set_env_ids)
            task_obj.reset() # reset buffer: is this necessary?
            
    def set_task_visual_state_tensor(self, set_env_ids):
        stage = omni.usd.get_context().get_stage()
        for env_enumerator, env_id in enumerate(set_env_ids):
            for name, task_obj in self._task.items():
                obj_path = f"/World/envs/env_{env_id}/{self.task_config.target_obj}"
                shader_path = f"{obj_path}/material/Shader"

                shader_prim = stage.GetPrimAtPath(shader_path)
                shader = UsdShade.Shader(shader_prim)

                # Set new color
                color = np.random.rand(3)
                diffuse_input = shader.GetInput("diffuseColor")
                if diffuse_input:
                    diffuse_input.Set((color[0], color[1], color[2]))
                
                
    
    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim 
        self.scene.update(dt=1./self.simulator_config.sim.fps)

        # Haoyang: update dof pos/vel for pd control
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids] # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]
    
    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller


    def render(self, sync_frame_time=True):
        pass

    def apply_commands(self, commands_description):
        if commands_description == "forward_command":
            self.commands[:, 0] += 0.1
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

     # debug visualization
    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)
        
    def draw_spheres_batch(self, pos, rot = None, scales = None):
        self.vis_spheres.visualize(pos, rot, scales)

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [(   start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)
    
    def adjust_camera_pos(self, pos):
        self.sim.set_camera_view(pos, [-0.5, 0.0, 0.5])
