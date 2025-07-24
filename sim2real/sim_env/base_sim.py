import argparse
import sys
import threading
import time
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from loguru import logger
from loop_rate_limiters import RateLimiter

sys.path.append("../")

# from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

from sim2real.utils.sdk2py_bridge import ElasticBand, create_sdk2py_bridge


class BaseSimulator:
    def __init__(self, config):
        self.config = config
        self.init_config()
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()

        self.sim_thread = Thread(target=self.simulation_thread)

    def init_config(self):
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"])

    def init_factory(self):
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "booster":
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(str.format("SDK TYPE: {0}", self.sdk_type))

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
            self.band_attached_link = self.mj_model.body(band_attached_link_name).id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.config)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux":  # TODO [Yuanhang]: add other joystick support
                if self.config["SDK_TYPE"] == "unitree":
                    self.robot_bridge.SetupJoystick(
                        device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
                    )
                else:
                    self.logger.warning(f"Joystick is not supported for {self.config['SDK_TYPE']} yet.")
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[7 + i])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                    )
            except Exception as e:
                self.logger.error(str.format("Joint {0} not found in motor_cmd: {1}", i, e))
        # Set the torque limit
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        if self.robot_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

    def simulation_thread(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.logger.info(str.format("FPS: {0:.2f}", 100 / (end_time - start_time)))
                start_time = end_time
            self.rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = BaseSimulator(config)
    simulation.sim_thread.start()
