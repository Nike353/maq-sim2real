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
from utils.history_handler import HistoryHandler
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_inference')

from rl_policy import RLPolicy
from utils.joystick import JoystickController
from utils.key_cmd import KeyboardPolicy

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

def parse_observation(cls,
                      key_list, 
                      buf_dict, 
                      obs_scales,
                      ) -> None:
    """ Parse observations for the legged_robot_base class
    """

    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4]
       
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().copy()
        obs_scale = obs_scales[obs_key]
        buf_dict[obs_key] = actor_obs * obs_scale 

class LocomotionPolicy(RLPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False,
                 policy_config=None,
                 command_ref=None):
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
        
        # self.obs_config = policy_config['obs']
        # self.obs_config['obs_dict'].pop('critic_obs')
        # each_dict_obs_dims = {k: v for d in self.obs_config['obs_dims'] for k, v in d.items()}
        # self.history_handler = HistoryHandler(self.obs_config['obs_auxiliary'], each_dict_obs_dims)
        self.clock_inputs = np.zeros((1, 4))
        
        self.gait_indices = np.zeros(1)
        self.dt = 0.02
        self.command_ref = command_ref
        self.commands = self.command_ref[0].reshape(1, -1)
        self.joystick = JoystickController()
        self.commands = self.joystick.commands.copy()

        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        
        # self.gen_comm_profile()
        




    
        

        
        

    def _pre_compute_observations_callback(self, robot_state_data):
        # prepare quantities
        self.base_quat = robot_state_data[:, 3:7]
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        # print("self.base_ang_vel", self.base_ang_vel)
        v = np.array([[0, 0, -1]])
        self.projected_gravity = quat_rotate_inverse_numpy(self.base_quat, v)
        self.dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        self.dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
    
    def compute_observations(self):
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        # compute Algo observations
        for obs_key, obs_config in self.obs_config['obs_dict'].items():
            self.obs_buf_dict_raw[obs_key] = dict()
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.obs_config['obs_scales'])
        
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.obs_config['obs_scales'])
        self._post_config_observation_callback()

    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()
        for obs_key, obs_config in self.obs_config['obs_dict'].items():
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)
            self.obs_buf_dict[obs_key] = np.concatenate([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], axis=-1)

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state [:2]: timestamps
        # robot_state [2:5]: robot base pos
        # robot_state [5:9]: robot base orientation
        # robot_state [9:9+dof_num]: joint angles 
        # robot_state [9+dof_num: 9+dof_num+3]: base linear velocity
        # robot_state [9+dof_num+3: 9+dof_num+6]: base angular velocity
        # robot_state [9+dof_num+6: 9+dof_num+6+dof_num]: joint velocities

        # RL observation preparation
        # self._pre_compute_observations_callback(robot_state_data)
        # self.compute_observations()
    
        
        # obs = self.obs_buf_dict['actor_obs']
        self.commands = self.joystick.update_commands()
        # print("self.commands", self.commands)
        
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
        
        dof_pos_minus_default = dof_pos - self.default_dof_angles
        v = np.array([[0, 0, -1]])
        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)


        command_lin_vel = self.commands[:, :2]
        command_ang_vel = self.commands[:, 2:3]
        command_body_height = self.commands[:, 3:4]
        command_gait_freq = self.commands[:, 4:5]
        command_gait_phase = self.commands[:, 5:9]
        command_footswing_height = self.commands[:, 9:10]
        command_body_attitude = self.commands[:, 10:12]
        command_stance = self.commands[:, 12:14]
        command_aux_reward = self.commands[:, 14:]
        clock_inputs = self.clock_inputs
        history = self._get_obs_history()
        history*=self.obs_scales['history']

        obs = np.concatenate([self.last_policy_action*self.obs_scales['actions'],
                                base_ang_vel*self.obs_scales['base_ang_vel'],
                                clock_inputs*self.obs_scales['clock_inputs'],
                                command_ang_vel*self.obs_scales['command_ang_vel'],
                                command_aux_reward*self.obs_scales['command_aux_reward'],
                                command_body_attitude*self.obs_scales['command_body_attitude'],
                                command_body_height*self.obs_scales['command_body_height'],
                                command_footswing_height*self.obs_scales['command_footswing_height'],
                                command_gait_freq*self.obs_scales['command_gait_freq'],
                                command_gait_phase*self.obs_scales['command_gait_phase'],
                                command_lin_vel*self.obs_scales['command_lin_vel'],
                                command_stance*self.obs_scales['command_stance'],
                                dof_pos_minus_default*self.obs_scales['dof_pos'],
                                dof_vel*self.obs_scales['dof_vel'],
                                projected_gravity*self.obs_scales['projected_gravity'],
                                history*self.obs_scales['history']],axis=1)
        if self.history_handler:
            self.history_handler.add("actions", self.last_policy_action*self.obs_scales['actions'])
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales['base_ang_vel'])
            self.history_handler.add("clock_inputs", clock_inputs*self.obs_scales['clock_inputs'])
            self.history_handler.add("command_ang_vel", command_ang_vel*self.obs_scales['command_ang_vel'])
            self.history_handler.add("command_aux_reward", command_aux_reward*self.obs_scales['command_aux_reward'])
            self.history_handler.add("command_body_attitude", command_body_attitude*self.obs_scales['command_body_attitude'])
            self.history_handler.add("command_body_height", command_body_height*self.obs_scales['command_body_height'])
            self.history_handler.add("command_footswing_height", command_footswing_height*self.obs_scales['command_footswing_height'])
            self.history_handler.add("command_gait_freq", command_gait_freq*self.obs_scales['command_gait_freq'])
            self.history_handler.add("command_gait_phase", command_gait_phase*self.obs_scales['command_gait_phase'])
            self.history_handler.add("command_lin_vel", command_lin_vel*self.obs_scales['command_lin_vel'])
            self.history_handler.add("command_stance", command_stance*self.obs_scales['command_stance'])
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales['dof_pos'])
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales['dof_vel'])
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales['projected_gravity'])

        # ref_msg.pose.pose.position.x = 0.0
        # ref_msg.pose.pose.position.y = 0.0
        # ref_msg.pose.pose.orientation.x = 0.0
        # ref_msg.pose.pose.orientation.y = 0.0
        # ref_msg.pose.pose.orientation.z = 0.0
        # ref_msg.pose.pose.orientation.w = 0.0
        # ref_msg.twist.twist.linear.x = self.lin_vel_command[0][0]
        # ref_msg.twist.twist.linear.y = self.lin_vel_command[0][1]
        # ref_msg.twist.twist.angular.z =self.ang_vel_command[0][0]
        # ref_msg.child_frame_id = 'base_link'
        # ref_msg.header.frame_id = 'world'
        # ref_msg.header.stamp = self.node.get_clock().now().to_msg()
        # self.ref_pub.publish(ref_msg)

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

        obs = self.prepare_obs_for_rl(robot_state_data)
        
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
            if scaled_policy_action.shape[1] == self.num_dofs:
                self._step_contact_targets_numpy()
                self.epi_len+=1
                print("epi_len", self.epi_len)
                self.epi_done = 0
                index = int(self.epi_len//(8.0/self.dt))
                # self.commands = self.command_ref[index].reshape(1, -1)
                # if self.epi_len%(6.0/self.dt) == 0 or self.epi_len == 4:
                #     self.commands[:, 0] = np.random.uniform(-1.0, 1.0)
                #     self.commands[:, 2] = np.random.uniform(-1.0, 1.0)
                #     self.commands[:, 5] = np.random.uniform(0.0, 0.5)
                # if self.epi_len>150:
                #     self.epi_len = np.array([[0.0]])
            else:
                scaled_policy_action = np.concatenate([scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1)
            q_target = scaled_policy_action + self.default_dof_angles

        if self.epi_len>2500:
            self.epi_done = 1
            self.get_ready_state = True
            q_target = self.get_init_target(robot_state_data)
            

        


        commands_msg = Float64MultiArray()
        self.send_cmd_timestep = self.node.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[3] = self.send_cmd_timestep
        commands_msg.data = self.timestamp_message + [float(not self.use_policy_action and not self.get_ready_state)] + q_target.flatten().tolist()
        # print(f"Time in command0: {commands_msg.data[1] - commands_msg.data[0]}")
        # print(f"Time in command1: {commands_msg.data[2] - commands_msg.data[1]}")
        # print(f"Time in command2: {commands_msg.data[3] - commands_msg.data[2]}")
        # print("q_target", q_target)
        # epi_done_msg = Int32()
        # epi_done_msg.data = self.epi_done
        self.commands_pub.publish(commands_msg)
        # self.epi_done_pub.publish(epi_done_msg)

    

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
    def _step_contact_targets_numpy(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]

        self.gait_indices = np.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases
        ]


        

        self.clock_inputs[:, 0] = np.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = np.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = np.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = np.sin(2 * np.pi * foot_indices[3])

        

        

        

        

    def _get_obs_actions(self):
        return self.last_policy_action

    def _get_obs_base_ang_vel(self):
        return self.base_ang_vel
   
    def _get_obs_clock_inputs(self):
        return self.clock_inputs

    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
    
    def _get_obs_command_aux_reward(self):
        return self.commands[:, 14:]
    
    def _get_obs_command_body_attitude(self):
        return self.commands[:,10:12]

    def _get_obs_command_body_height(self):
        # print(self.commands)
        return self.commands[:,3:4]
    
    def _get_obs_command_footswing_height(self):
        return self.commands[:,9:10]
    
    def _get_obs_command_gait_freq(self):
        return self.commands[:,4:5]
    
    def _get_obs_command_gait_phase(self):
        return self.commands[:,5:9]
    
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    def _get_obs_command_stance(self):
        return self.commands[:,12:14]

    def _get_obs_dof_pos(self):
        return self.dof_pos - self.default_dof_angles
    
    def _get_obs_dof_vel(self):
        return self.dof_vel
    
    def _get_obs_projected_gravity(self):
        return self.projected_gravity

    def _get_obs_short_history(self,):
        assert "short_history" in self.obs_config['obs_auxiliary'].keys()
        history_config = self.obs_config['obs_auxiliary']['short_history']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return np.concatenate(history_tensors, axis=1)
    

    
class LocomotionPolicyKeyboard(LocomotionPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False,
                 policy_config=None,
                 command_ref=None):
        LocomotionPolicy.__init__(self, config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation, 
                         use_mocap,
                         policy_config,
                         command_ref)
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
    model_path = args.model_path
    config_path = model_path.replace("/exported/model_1500.onnx", "/config.yaml")
    print(config_path)
    with open(config_path) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
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
    
    with open("/home/guanqihe/rvgym/RoboVerse/logs/Go2_sim2real_wtw_test/20250327_170930-active_sysid_no_dr_policy_random-sys_id-go2/best_commands.yaml", "r") as file:
        data = yaml.safe_load(file)
        command_ref = np.array(data)

    locomotion_policy = LocomotionPolicyKeyboard(config=config, 
                                        node=node, 
                                        model_path=args.model_path, 
                                        use_jit=args.use_jit,
                                        rl_rate=50, 
                                        decimation=4,
                                        use_mocap=args.use_mocap,
                                        policy_config=policy_config,
                                        command_ref=command_ref)

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