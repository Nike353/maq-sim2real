try:
    with open("./humanoidverse/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
        DEFAULT_ISAACSIM_VERSION = f.read().strip()
except FileNotFoundError:
    DEFAULT_ISAACSIM_VERSION = "4.5"

if DEFAULT_ISAACSIM_VERSION == "4.5":
    ## IsaacSim 4.5
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

elif DEFAULT_ISAACSIM_VERSION == "4.2":
    ### IsaacSim 4.2
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
    from omni.isaac.lab.assets.articulation import ArticulationCfg, Articulation
    from omni.isaac.lab.assets.rigid_object import RigidObjectCfg, RigidObject
    from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

else:
    raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")


ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        usd_path="humanoidverse/data/robots/h1/h1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_joint": -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,        
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle_joint": 20.0},
            damping={".*_ankle_joint": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
            },
        ),
    },
)


# Haotian: moved task specific object configurations to "humanoidverse/data/tasks"
# TaskObjCfgDict = {
#     # "test_cone" : RigidObjectCfg(
#     #             prim_path="/World/envs/env_.*/additional_obj",
#     #             spawn=sim_utils.ConeCfg(
#     #                 radius=0.1,
#     #                 height=0.5,
#     #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#     #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
#     #                 collision_props=sim_utils.CollisionPropertiesCfg(),
#     #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
#     #             ),
#     #             init_state=RigidObjectCfg.InitialStateCfg(
#     #                 pos=(1.0, 0.0, 0.5),
#     #             ),
#     #         ),
#     "table": RigidObjectCfg(
#         class_type=RigidObject,
#         spawn=sim_utils.UsdFileCfg(
#             usd_path="humanoidverse/data/objects/simple/table.usd",
#             scale=(1.0, 1.0, 1.0),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 # disable_gravity=False,
#                 # retain_accelerations=False,
#                 # linear_damping=0.0,
#                 # angular_damping=0.0,
#                 # max_linear_velocity=1000.0,
#                 # max_angular_velocity=1000.0,
#                 # max_depenetration_velocity=1.0,
#             ),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=(-0.5, 0.0, 0.4),  # Initial position for the root of the articulation
#             #rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion rotation (w, x, y, z)
#         ),
#         prim_path="/World/Bar",
#     ),
#     "bottle": RigidObjectCfg(
#         class_type=RigidObject,
#         spawn=sim_utils.UsdFileCfg(
#             usd_path="humanoidverse/data/objects/simple/bottle.usd",
#             scale=(1.0, 1.0, 1.0),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg()
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=(-0.45, 0.0, 1.0),  # Initial position for the root of the articulation
#             #rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion rotation (w, x, y, z)
#             #lin_vel=(0.0, 0.0, 0.0),  # Linear velocity
#             #ang_vel=(0.0, 0.0, 0.0),  # Angular velocity
#         ),
#         prim_path="/World/Bar",
#     ),
#     # add more objects here
# }