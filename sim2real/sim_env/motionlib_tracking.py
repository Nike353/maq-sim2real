import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
import threading
import numpy as np
import time
from loguru import logger
import argparse
import yaml
import torch
import onnxruntime

import sys
sys.path.append('../')

# motion lib
from roboverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c


from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize


from std_msgs.msg import Float64MultiArray

from base_sim import BaseSimulator

class MotionTrackingSimulator(BaseSimulator):
    def __init__(self, config, node):
        super().__init__(config, node)

        self.vr_3point_pos = np.zeros((3, 3))
        self.marker_ids = []
        for i in range(3):
            self.marker_ids.append(
                mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"marker{i}"
                )
            )

    def _init_motion_lib(self):
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        
        else:
            self._motion_lib.load_motions(random_sample=True)
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def _init_buffers(self):
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

    def init_scene(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def sim_step(self):
        self.mj_data.ctrl[:] = self.torques
        self.publish_low_state()
        self.publish_vr_obs()
        mujoco.mj_step(self.mj_model, self.mj_data)

        for i in range(3):
            pos = self.vr_3point_pos[i]
            # Add sphere marker directly to the scene
            self.mj_data.mocap_pos[i] = pos

    def init_subscriber(self):
        self.avp_sub = self.node.create_subscription(Float64MultiArray, 'vision_pro_data', self.avp_callback, 10)

    def init_publisher(self):
        self.vr_obs_pub = self.node.create_publisher(Float64MultiArray, 'vr_obs', 10)

    def avp_callback(self, msg):
        self.vr_3point_pos = np.array(msg.data[:9]).reshape(3, 3)

    def publish_vr_obs(self):
        robot_pos = self.mj_data.qpos[:3].reshape(1,3)
        robot_quat = self.mj_data.qpos[3:7].reshape(1, 4)
        local_vr_3point_pos = quat_rotate_inverse_numpy(robot_quat, self.vr_3point_pos - robot_pos)
        vr_obs = Float64MultiArray()
        vr_obs.data = local_vr_3point_pos.flatten().tolist()
        self.vr_obs_pub.publish(vr_obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])

    rclpy.init(args=None)
    node = rclpy.create_node('sim_mujoco')

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    simulation = MotionTrackingSimulator(config, node)

    start_time = time.time()
    sim_cnt = 0
    try:
        while rclpy.ok():
            simulation.sim_loop()
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                node.get_logger().info(f"Average Sim FPS: {100/(time.time()-start_time)}")
                start_time = time.time()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()