import sys

import mujoco
import mujoco.viewer
import numpy as np
import yaml
import argparse

sys.path.append("../")
sys.path.append("./sim2real")


from sim2real.sim_env.base_sim import BaseSimulator
from sim2real.utils.math import quat_rotate_numpy


class LocoManipSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)
        self.EE_xfrc = 0
        self.t = 0
        self.left_hand_link_name = self.config.get("left_hand_link_name", "left_hand_link")
        self.right_hand_link_name = self.config.get("right_hand_link_name", "right_hand_link")

    def init_scene(self):
        super().init_scene()
        NUM_FEET_SENSORS = 8
        # Assuming you know the order of the sensors in the XML
        self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3
        # Customize the viewer options
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )

        if self.elastic_band.estimate:
            self.EE_xfrc = self.elastic_band.apply_force
            # Yuanhang: Testing the estimated force
            base_target_axis = np.array([-1.0, 0.0, 0.0])
            body_quat = self.mj_data.qpos[3:7]
            force_in_global = quat_rotate_numpy(np.array([body_quat]), base_target_axis * self.EE_xfrc)
            self.mj_data.xfrc_applied[self.mj_model.body(self.left_hand_link_name).id, 0:3] = force_in_global
            self.mj_data.xfrc_applied[self.mj_model.body(self.right_hand_link_name).id, 0:3] = force_in_global
            self.t += self.dt

        self.compute_torques()
        if self.robot_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = LocoManipSimulator(config)
    simulation.sim_thread.start()