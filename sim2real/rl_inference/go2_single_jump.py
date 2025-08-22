import rclpy
from rclpy.node import Node
import numpy as np
import time
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
import torch
import onnxruntime
import threading

import argparse
import yaml
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_inference')

from rl_policy import RLPolicy
from sim2real.utils.key_cmd import KeyboardPolicy

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

class LocomotionPolicy(RLPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        super().__init__(config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        self.phase_len_command = np.array([[500.0]])
        self.phase = np.array([[0.0]])
    
    def clock(self):
        clock = np.vstack((2*np.sin(self.phase/self.phase_len_command),2*np.cos(self.phase/self.phase_len_command))).T
        return clock

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state [:2]: timestamps
        # robot_state [2:5]: robot base pos
        # robot_state [5:9]: robot base orientation
        # robot_state [9:9+dof_num]: joint angles 
        # robot_state [9+dof_num: 9+dof_num+3]: base linear velocity
        # robot_state [9+dof_num+3: 9+dof_num+6]: base angular velocity
        # robot_state [9+dof_num+6: 9+dof_num+6+dof_num]: joint velocities

        # RL observation preparation

        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]


        dof_pos_minus_default = dof_pos - self.default_dof_angles

        v = np.array([[0, 0, -1]])

        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)

        # import ipdb; ipdb.set_trace()   
        # print(base_ang_vel)
        obs = np.concatenate([self.last_policy_action, 
                                base_ang_vel*0.25,
                                self.clock(),
                                # self.ang_vel_command, 
                                # self.lin_vel_command, 
                                # self.phase_len_command,
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                projected_gravity
                                ], axis=1)
        # examine obs
        # print("last_policy_action", self.last_policy_action)
        # print("base_ang_vel", base_ang_vel)
        # print("ang_vel_command", self.ang_vel_command)
        # print("lin_vel_command", self.lin_vel_command)
        # print("dof_pos_minus_default", dof_pos_minus_default)
        # print("dof_vel", dof_vel)
        # print("projected_gravity", projected_gravity)
        return obs.astype(np.float32)
    
    def rl_inference(self):
        super().rl_inference()
        self.phase +=1


class LocomotionPolicyKeyboard(LocomotionPolicy, KeyboardPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config,
                            node,
                            model_path,
                            use_jit,
                            rl_rate,
                            policy_action_scale,
                            decimation)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    
    rate = node.create_rate(50)

    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_172529-H1_19dof_sim2real_-0.5actionrate_0.9_1.25mass-locomotion-h1/exported/model_800.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_213247-H1_19dof_sim2real_IsaacGym_noTerrain_noDR_actionrate-0.5_mass0.9_1.25-locomotion-h1/model_61800.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_180031-H1_19dof_sim2real_-0.5actionrate_0.9_1.25mass_delay0_20-locomotion-h1/model_86300.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241119_175059-TairanMotions_H1_dm_A6lift_nolinvel_addpush5_16_1.5_actionrate0.5_mass0.9_1.2_maxheight-300_initnoise-motion_tracking-h1/exported/model_1400.onnx"
    # h1_dof_nums = 19
    

    locomotion_policy = LocomotionPolicyKeyboard(config=config, 
                                        node=node, 
                                        model_path=args.model_path, 
                                        use_jit=args.use_jit,
                                        rl_rate=50, 
                                        decimation=4)

    time.sleep(1)
    start_time = time.time()
    total_inference_cnt = 0

    try:
        while rclpy.ok():
            locomotion_policy.rl_inference()
            rate.sleep()
            end_time = time.time()
            total_inference_cnt += 1
            if total_inference_cnt % 100 == 0:
                node.get_logger().info(f"Average inference FPS: {100/(end_time - start_time)}")
                start_time = end_time
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()