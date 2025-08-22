import rclpy
from rclpy.node import Node
import numpy as np
import time
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
import torch
import onnxruntime
import threading
from pynput import keyboard
import argparse
import yaml
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_inference')

from rl_policy import RLPolicy

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

class MotionTracking(RLPolicy):
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

        self.frame_idx = 0
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        self.frame_start_time = start_time
        self.frame_last_time = start_time

        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()


    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(key):
            try:
                if key.char == "p":
                    self.use_policy_action = True
                    self.get_ready_state = False
                    self.node.get_logger().info("Using policy actions")

                    self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
                    self.phase = 0.0
                elif key.char == "o":
                    self.use_policy_action = False
                    self.get_ready_state = False
                    self.node.get_logger().info("Actions set to zero")

                elif key.char == "i":
                    self.get_ready_state = True
                    self.init_count = 0
                    self.node.get_logger().info("Setting to init state")

            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive


    def get_frame_encoding(self):
        # 11 bins for 11 seconds, if (current_time-self.frame_start_time) > 1, increment frame_idx
        # the frame encoding is maped to 0-1

        current_time = self.node.get_clock().now().nanoseconds / 1e9
        # import ipdb; ipdb.set_trace()
        # if (current_time-self.frame_last_time) > 1:
        #     self.frame_idx += 1
        #     self.frame_last_time = current_time

        motion_length_s = 11.07
        self.phase = (current_time - self.frame_start_time) / motion_length_s
        if self.phase >= 1.0:
            self.frame_start_time = current_time
            self.phase = 0.0
        # print("current_s", current_time)
        # print("phase", self.phase)
        
        
        # print(f"Frame encoding: {self.frame_encoding}")

    def prepare_obs_for_rl(self, robot_state_data):

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

        self.get_frame_encoding()
        # print(self.phase)
        obs = np.concatenate([self.last_policy_action, 
                                base_ang_vel*0.25, 
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                projected_gravity,
                                np.array([[self.phase]])
                                ], axis=1)
        
        # import ipdb; ipdb.set_trace()
        # examine obs
        # print("last_policy_action", self.last_policy_action)
        # print("base_ang_vel", base_ang_vel)
        # print("ang_vel_command", self.ang_vel_command)
        # print("lin_vel_command", self.lin_vel_command)
        # print("dof_pos_minus_default", dof_pos_minus_default)
        # print("dof_vel", dof_vel)
        # print("projected_gravity", projected_gravity)
        return obs.astype(np.float32)

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
    
    
    # deepmimic
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241118_210923-TairanMotions_H1_dm_A6lift_nolinvel_addpush5_16_1.5_actionrate0.4_mass0.9_1.2_maxheight-100-motion_tracking-h1/model_1400.onnx"

    

    rl_policy = MotionTracking(config=config,
                                node=node, 
                               model_path=args.model_path, 
                               rl_rate=50, 
                               decimation=4,
                               use_jit=args.use_jit)

    time.sleep(1)
    start_time = time.time()
    count = 0
    
    try:

        while rclpy.ok():
            rl_policy.rl_inference()
            rate.sleep()
            count += 1
            if count % 100 == 0:
                print(f"Average command_cnt FPS: {count/(time.time()-start_time)}")
                print(rl_policy.phase)
    except KeyboardInterrupt:
        pass

    # rl_policy.destroy_node()
    rclpy.shutdown()