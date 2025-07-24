try:
    with open("./humanoidverse/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
        DEFAULT_ISAACSIM_VERSION = f.read().strip()
except FileNotFoundError:
    DEFAULT_ISAACSIM_VERSION = "4.5"

if DEFAULT_ISAACSIM_VERSION == "4.5":
    ### IsaacSim 4.5
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass
    import isaaclab.envs.mdp as mdp

elif DEFAULT_ISAACSIM_VERSION == "4.2":
    from omni.isaac.lab.managers import EventTermCfg as EventTerm
    from omni.isaac.lab.managers import SceneEntityCfg
    from omni.isaac.lab.utils import configclass
    import omni.isaac.lab.envs.mdp as mdp
else:
    raise ValueError(f"Unsupported IsaacSim version: {DEFAULT_ISAACSIM_VERSION}")

# @configclass
# class EventCfg:
#     """Configuration for events."""

#     scale_body_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "mass_distribution_params": (0.8, 1.2),
#             "operation": "scale",
#         },
#     )
    
#     random_joint_friction = EventTerm(
#         func=mdp.randomize_joint_parameters,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "friction_distribution_params": (0.5, 1.25),
#             "operation": "scale",
#         },
#     )

@configclass
class EventCfg:
    """Configuration for events."""
    scale_body_mass = None
    random_joint_friction = None
    

    