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

class VRMotionTracking(RLPolicy):
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

        # self.frame_idx = 0
        # start_time = self.node.get_clock().now().nanoseconds / 1e9
        # self.frame_start_time = start_time
        # self.frame_last_time = start_time

        self.avp_sub = self.node.create_subscription(Float64MultiArray, 'vr_obs', self.avp_callback, 10)
        self.vr_3point_pos = np.zeros((3, 3))
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()

    def avp_callback(self, msg):
        self.vr_3point_pos = np.array(msg.data[:9]).reshape(3, 3)


    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(key):
            try:
                if key.char == "[":
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

        vr_3point_pos = self.vr_3point_pos.reshape(1, -1)
        # print(self.phase)
        obs = np.concatenate([self.last_policy_action, 
                                base_ang_vel*0.25, 
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                projected_gravity,
                                vr_3point_pos], axis=1)

        
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

    rl_policy = VRMotionTracking(config=config,
                                node=node, 
                               model_path=args.model_path, 
                               rl_rate=50, 
                               decimation=4,
                               use_jit=args.use_jit)

    time.sleep(1)
    start_time = time.time()
    inference_count = 0
    
    try:
        while rclpy.ok():
            rl_policy.rl_inference()
            rate.sleep()
            inference_count += 1
            if inference_count % 100 == 0:
                print(f"Average inference FPS: {100/(time.time()-start_time)}")
                start_time = time.time()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
