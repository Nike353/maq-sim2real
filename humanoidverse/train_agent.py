import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger
import shutil



from utils.config_utils import *  # noqa: E402, F403


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator['_target_'].split('.')[-1]
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacSim':
        # read from config

        try:
            with open("./humanoidverse/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher # 4.2
            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        # args_cli, hydra_args = parser.parse_known_args()
        # sys.argv = [sys.argv[0]] + hydra_args
        # args_cli.num_envs = config.num_envs
        # args_cli.seed = config.seed
        # args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        # args_cli.output_dir = config.output_dir
        # args_cli.enable_cameras = config.simulator.get("enable_cameras", False)
        # args_cli.headless = config.headless
        
        # dest_path = os.environ.get('ISAACLAB_PATH') + "/apps"
        # current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        # if args_cli.enable_cameras and args_cli.headless:
        #     source_file = current_file_dir_path + "/../apps/phc.isaaclab.python.headless.rendering.kit"
        #     shutil.copy(source_file, dest_path)
        #     args_cli.experience = dest_path + "/phc.isaaclab.python.headless.rendering.kit"
        # elif args_cli.enable_cameras and not args_cli.headless:
        #     source_file = current_file_dir_path + "/../apps/phc.isaaclab.python.rendering.kit"
        #     shutil.copy(source_file, dest_path)
        #     args_cli.experience = dest_path + "/phc.isaaclab.python.rendering.kit"

        # app_launcher = AppLauncher(args_cli)
        # simulation_app = app_launcher.app  

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        args_cli.enable_cameras = config.simulator.config.cameras.enable_cameras
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401


    if config.multi_gpu:
        config.seed += app_launcher.global_rank
        config.simulator.config.sim_device = f"cuda:{app_launcher.local_rank}"
        config.global_rank = app_launcher.global_rank
        # ZL: not a complete list yet. 
        
    else:
        config.global_rank = 0
        config.algo.config.global_rank = 0
        config.algo.config.world_size = 1


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(project=project_name, 
                entity=config.wandb.wandb_entity,
                name=run_name,
                config=unresolved_conf,
                sync_tensorboard=True,
                dir=wandb_dir)
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)

    # torch.set_float32_matmul_precision("medium")

    # fabric: Fabric = instantiate(config.fabric)
    # fabric.launch()

    # if config.seed is not None:
    #     rank = fabric.global_rank
    #     if rank is None:
    #         rank = 0
    #     fabric.seed_everything(config.seed + rank)
    #     seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env: BaseEnv = instantiate(config=config.env, device=device)

    
    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo: BaseAlgo = instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)
    algo.setup()
    # import ipdb;    ipdb.set_trace()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)

    # handle saving config
    algo.learn()

    if simulator_type == 'IsaacSim':
        simulation_app.close()

if __name__ == "__main__":
    main()
