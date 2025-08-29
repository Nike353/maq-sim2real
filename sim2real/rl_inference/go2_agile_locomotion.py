import rclpy
from rclpy.node import Node
import numpy as np
import time
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import threading
from pynput import keyboard
import argparse
import yaml

# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_inference')

from rl_policy import RLPolicy
from sim2real.utils.key_cmd import KeyboardPolicy
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_ as Pose_go

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
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        # self.epi_len = np.array([[0.0]])
        # self.last_last_action = np.zeros((1, 12))
        self.epi_len = np.array([[0.0]])
        self.ref_pub = self.node.create_publisher(Odometry, '/reference_position', 10 )
        self.epi_done_pub = self.node.create_publisher(Int32, '/epi_done', 1)
        self.epi_done = 2
        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        # self.gen_comm_profile()
        ChannelFactoryInitialize(0,"lo")
        self.base_state_subscriber = ChannelSubscriber("rt/base_state", Pose_go)
        self.base_state_subscriber.Init(self.BaseStateHandler, 1)
        self.base_state = None


    def BaseStateHandler(self, msg):
        self.base_pos = msg.position
        self.base_quat = msg.orientation
    
    def get_pose(self):
        return np.array([self.base_pos.x, self.base_pos.y, self.base_pos.z]), np.array([self.base_quat.w, self.base_quat.x, self.base_quat.y, self.base_quat.z])
        



    
        
        



    def prepare_obs_for_rl(self, robot_state_data,command):
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

        
        # print(self.pitch_ref(self.epi_len), self.epi_len)
        # if self.epi_len>250:
        #     exit()
        # print(self.rp_command)

        
        self.lin_vel_command[0][0] = command[0]
        self.lin_vel_command[0][1] = command[1]
        self.ang_vel_command[0][0] = command[2]
        obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.lin_vel_command, 
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    projected_gravity
                                    ], axis=1)

        
        
        return obs.astype(np.float32)

    def rl_inference(self,command):
        # print(f"command: {command}")
        
        start_time = time.time()
        command_cnt = 0

        if self.state_msg is None:
            print("No message received yet")

        # Get States
        
        state_msg = self.state_msg
        # print("state_msg", state_msg)
        time_and_state_data = np.array(state_msg.data, dtype=np.float64).reshape(1, -1)
        timestamp_data = time_and_state_data[:, :self.timestamp_digit].squeeze()
        robot_state_data = time_and_state_data[:, self.timestamp_digit:]

        self.timestamp_message[0] = float(timestamp_data[0])
        self.timestamp_message[1] = float(timestamp_data[1])
        self.timestamp_message[2] = float(self.receive_state_timestep)

        obs = self.prepare_obs_for_rl(robot_state_data,command)
        np.set_printoptions(precision=3, suppress=True)

        # print("epi_len", self.epi_len)
        # print("obs", obs)

        policy_action = self.policy(obs)

        # print("policy_action", policy_action)

        # if self.epi_len >= 10:
        #     exit()

        policy_action = np.clip(policy_action, -100, 100)

        # if not self.use_policy_action:
        #     policy_action *= 0.0  # Zero the actions if "e" was pressed

        self.last_policy_action = policy_action.copy()  
        scaled_policy_action = policy_action * self.policy_action_scale
        if self.get_ready_state:
            # import ipdb; ipdb.set_trace()
            self.epi_len = np.array([[0.0]])
            q_target = self.get_init_target(robot_state_data)
            if self.init_count > 100:
                self.init_count = 100
                
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7:7+self.num_dofs]
        else:
            # print(command)
            if scaled_policy_action.shape[1] == self.num_dofs:
                self.epi_len+=1
                self.epi_done = 0
                # if self.epi_len>150:
                #     self.epi_len = np.array([[0.0]])
            else:
                scaled_policy_action = np.concatenate([scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1)
            q_target = scaled_policy_action + self.default_dof_angles

        # if self.epi_len>450:
        #     self.epi_done = 1
        #     self.get_ready_state = True
        #     q_target = self.get_init_target(robot_state_data)
            

        


        commands_msg = Float64MultiArray()
        self.send_cmd_timestep = self.node.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[3] = self.send_cmd_timestep
        commands_msg.data = self.timestamp_message + [float(not self.use_policy_action and not self.get_ready_state)] + q_target.flatten().tolist()
        # print(f"Time in command0: {commands_msg.data[1] - commands_msg.data[0]}")
        # print(f"Time in command1: {commands_msg.data[2] - commands_msg.data[1]}")
        # print(f"Time in command2: {commands_msg.data[3] - commands_msg.data[2]}")
        # print("q_target", q_target)
        epi_done_msg = Int32()
        epi_done_msg.data = self.epi_done
        self.commands_pub.publish(commands_msg)
        self.epi_done_pub.publish(epi_done_msg)

    

    def odometry_callback(self, msg):
        # Extract current position from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.mocap_pos = np.array([x, y, z])
        
        # Convert orientation from quaternion to Euler angles
        quat = msg.pose.pose.orientation
        self.mocap_quat = np.array([quat.x, quat.y, quat.z, quat.w])
        rot = R.from_quat(self.mocap_quat)
        self.mocap_euler = rot.as_euler('xyz')
        twist = msg.twist.twist
        lin_vel = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        lin_vel = rot.inv().apply(lin_vel)
        self.mocap_lin_vel = lin_vel
        # print(f"Current position: {self.mocap_pos}, Current orientation: {self.mocap_euler}")



   


class LocomotionPolicyKeyboard(LocomotionPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit=False,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        LocomotionPolicy.__init__(self, config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation, 
                         use_mocap)
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        

    def start_key_listener(self):
        """Start a key listener using pynput."""
        print("Starting key listener")
        def on_press(key):
            try:
                print(f"Key pressed: {key.char}")
                if key.char == "]":
                    self.use_policy_action = True
                    self.get_ready_state = False
                    self.node.get_logger().info("Using policy actions")
                elif key.char == "o":
                    self.use_policy_action = False
                    self.get_ready_state = False
                    self.node.get_logger().info("Actions set to zero")
                elif key.char == "i":
                    self.get_ready_state = True
                    self.init_count = 0
                    self.node.get_logger().info("Setting to init state")
                elif key.char == "w":
                    # print("w pressed")
                    # import ipdb; ipdb.set_trace()
                    self.lin_vel_command[0, 0]+=0.1
                    # print(f"Linear velocity command -----: {self.lin_vel_command}")
                elif key.char == "s":
                    self.lin_vel_command[0, 0]-=0.1
                elif key.char == "a":
                    self.lin_vel_command[0, 1]+=0.1 
                elif key.char == "d":
                    self.lin_vel_command[0, 1]-=0.1
                elif key.char == "q":
                    self.ang_vel_command[0, 0]-=0.1
                elif key.char == "e":
                    self.ang_vel_command[0, 0]+=0.1
                elif key.char == "l":
                    self.rp_command[0, 1] += 0.05
                elif key.char == "k":
                    self.rp_command[0, 1] -= 0.05
                elif key.char == "h":
                    self.rp_command[0, 0] += 0.05
                elif key.char == "g":
                    self.rp_command[0, 0] -= 0.05
                elif key.char == "z":
                    self.ang_vel_command[0, 0] = 0.
                    self.lin_vel_command[0, 0] = 0.
                    self.lin_vel_command[0, 1] = 0.
                print(f"rp_command: {self.rp_command}")
            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
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
                                        decimation=4,
                                        use_mocap=args.use_mocap)

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
                # node.get_logger().info(f"Average inference FPS: {100/(end_time - start_time)}")
                start_time = end_time
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()