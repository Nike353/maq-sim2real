import mujoco
import mujoco.viewer
import threading
import numpy as np
import time
from loguru import logger
import argparse
import yaml
from threading import Thread
from loop_rate_limiters import RateLimiter

import sys
sys.path.append('../')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

# from std_msgs.msg import Float64MultiArray
from sim2real.utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from sim2real.tpg.tpg_general import TimedTPGManager
from sim2real.tpg.worldCC import parse_map_file

from loguru import logger

def euler_angles_to_quat(euler_angles):
    roll, pitch, yaw = euler_angles

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

class BaseSimulator:
    def __init__(self, config, map_file,tpg_file):
        self.config = config
        self._grid,self._resolution = parse_map_file(map_file)
        self._cell_size = self._resolution
        self._rows, self._cols = self._grid.shape
        self._timed_tpg_manager = TimedTPGManager()
        self._timed_tpg_manager.load_tpg(tpg_file)
        
        self.init_config()
        self.init_scene()
        self.init_unitree_bridge()

        # for more scenes
        self.init_subscriber()
        self.init_publisher()
        
        self.sim_thread = Thread(target=self.SimulationThread)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass
    
    def init_config(self):
        self.robot = Robot(self.config)
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.node = None
        if self.config.get("USE_ROS", False):
            import rclpy
            rclpy.init(args=None)
            self.node = rclpy.create_node("simulator")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(1/self.config["SIMULATE_DT"])
            thread = threading.Thread(target=rclpy.spin, args=(self.node, ), daemon=True)
            thread.start()
        else:
            self.logger = logger
            self.rate = RateLimiter(1/self.config["SIMULATE_DT"])

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        print(self._timed_tpg_manager.list_of_solutions[0].xythetas[0])

        
        # base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        # self.base_id = self.mj_model.body(base_body_name).id

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "h1" in self.config["ROBOT_TYPE"] or "g1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        start_xytheta = self._timed_tpg_manager.list_of_solutions[0].xythetas[0]
        self.mj_data.qpos[:2] = np.array([start_xytheta[0], start_xytheta[1]])
        starting_orientation = euler_angles_to_quat(np.array([0.0,0,start_xytheta[2]]))
        print(starting_orientation,"starting_orientation")
        self.mj_data.qpos[3:7] = starting_orientation
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print(self.mj_data.qpos[3:7])
        print(self.mj_data.qpos[:3])

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.mj_model, self.mj_data, self.config)
        # if self.config["PRINT_SCENE_INFORMATION"]:
        #     self.unitree_bridge.PrintSceneInformation()
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux":
                self.unitree_bridge.SetupJoystick(device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"])
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")

    def compute_torques(self):
        if self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_motor):
                if self.unitree_bridge.use_sensor:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.num_motor]
                        )
                    )
                else:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.qpos[7+i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[6+i]
                        )
                    )
        # Set the torque limit
        # self.torques = np.clip(self.torques, 
        #                        -self.unitree_bridge.torque_limit, 
        #                        self.unitree_bridge.torque_limit)

    def sim_step(self):
        self.unitree_bridge.PublishLowState()
        self.unitree_bridge.PublishBaseState()
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        if self.unitree_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else: self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def SimulationThread(self,):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            # self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                # self.logger.info(f"FPS: {100 / (end_time - start_time)}")
                start_time = end_time
            self.rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof_free.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("INTERFACE", None):
        if sys.platform == "linux":
            config["INTERFACE"] = "lo"
        elif sys.platform == "darwin":
            config["INTERFACE"] = "lo0"
        else: raise NotImplementedError("Only support Linux and MacOS.")
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    map_file = "/home/guanqihe/nikhil/multi_agent_quad/sim2real/maq-sim2real/sim2real/tpg/data/custom_map.map"
    tpg_file = "/home/guanqihe/nikhil/multi_agent_quad/sim2real/maq-sim2real/sim2real/tpg/data/solution_tpg.npz"

    simulation = BaseSimulator(config, map_file, tpg_file)
    simulation.sim_thread.start()