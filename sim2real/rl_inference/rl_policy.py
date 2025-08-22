import rclpy
from rclpy.node import Node
import numpy as np
import time
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
import torch
import onnxruntime
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('../')

from sim2real.utils.robot import Robot
from sim2real.utils.history_handler import HistoryHandler

class RLPolicy:
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit=False,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        self.config = config
        self.robot = Robot(config)
        self.node = node
        self.states_sub = self.node.create_subscription(
            Float64MultiArray, "robot_state", self.state_callback, 1
        )
        self.commands_pub = self.node.create_publisher(Float64MultiArray, "robot_command", 1)
        self.state_msg = None

        # load onnx policy
        if not use_jit:
            self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
            def policy_act(obs):
                return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        else:
            self.jit_policy = torch.jit.load(model_path)
            def policy_act(obs):
                obs = torch.tensor(obs)
                action_10dof = self.jit_policy(obs)
                action_19dof = torch.cat([action_10dof, torch.zeros(1, 9)], dim=1)
                return action_19dof.detach().numpy()
        self.policy = policy_act

        self.num_dofs = self.robot.NUM_JOINTS
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.default_dof_angles = self.robot.DEFAULT_DOF_ANGLES
        self.policy_action_scale = policy_action_scale

        # Keypress control state
        self.use_policy_action = False

        self.receive_state_timestep = 0.0
        self.send_cmd_timestep = 0.0

        self.period = 1.0 / rl_rate  # Calculate period in seconds
        self.last_time = time.time()

        self.decimation = decimation

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        self.timestamp_digit = 6
        self.timestamp_message = [0.0] * self.timestamp_digit

        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        self.use_history = self.config["USE_HISTORY"]
        self.obs_scales = self.config["obs_scales"]
        self.history_handler = None
        self.current_obs = None
        if self.use_history:
            print("history_config", self.config["history_config"])
            print("obs_dims", self.config["obs_dims"])
            self.history_handler = HistoryHandler(self.config["history_config"], self.config["obs_dims"])
            self.current_obs = {key: np.zeros((1, self.config["obs_dims"][key])) for key in self.config["obs_dims"].keys()}

    def state_callback(self, msg):
        self.state_msg = msg
        self.receive_state_timestep = self.node.get_clock().now().nanoseconds / 1e9

    def prepare_obs_for_rl(self, robot_state_data):
        raise NotImplementedError


    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        if self.get_ready_state:
            # interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 100)
            self.init_count += 1
            return q_target
        else:
            return dof_pos

    def _get_obs_history(self,):
        assert "history_config" in self.config.keys()
        history_config = self.config["history_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)


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

        policy_action = self.policy(obs)

        policy_action = np.clip(policy_action, -100, 100)

        # if not self.use_policy_action:
        #     policy_action *= 0.0  # Zero the actions if "e" was pressed

        self.last_policy_action = policy_action.copy()  
        scaled_policy_action = policy_action * self.policy_action_scale
        if self.get_ready_state:
            # import ipdb; ipdb.set_trace()
            q_target = self.get_init_target(robot_state_data)
            if self.init_count > 100:
                self.init_count = 100
                
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7:7+self.num_dofs]
        else:
            if scaled_policy_action.shape[1] == self.num_dofs:
                pass
            else:
                scaled_policy_action = np.concatenate([scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1)
            q_target = scaled_policy_action + self.default_dof_angles

        # clip q target for g1_29dof
        if self.config["ROBOT_TYPE"] == "g1_29dof":
            # clip ankle joints qtarget
            # print(q_target.shape)

            dof_pos_lower_limit_list= np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
                                        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, 
                                        -2.618, -0.52, -0.52,
                                        -3.0892, -1.5882, -2.618, -1.0472, 
                                        -1.972222054, -1.61443, -1.61443,
                                        -3.0892, -2.2515, -2.618, -1.0472, 
                                        -1.972222054, -1.61443, -1.61443])
            dof_pos_upper_limit_list= np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                                        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 
                                        2.618, 0.52, 0.52,
                                        2.6704, 2.2515, 2.618, 2.0944, 
                                        1.972222054, 1.61443, 1.61443,
                                        2.6704, 1.5882, 2.618, 2.0944, 
                                        1.972222054, 1.61443, 1.61443])
            
            q_target[0] = np.clip(q_target[0], dof_pos_lower_limit_list, dof_pos_upper_limit_list)

            # q_target[0][4] = np.clip(q_target[0][4], -0.87267, 0.5236)
            # q_target[0][5] = np.clip(q_target[0][5], -0.2618, 0.2618)

            # q_target[0][10] = np.clip(q_target[0][10], -0.87267, 0.5236)
            # q_target[0][11] = np.clip(q_target[0][11], -0.2618, 0.2618)


        commands_msg = Float64MultiArray()
        self.send_cmd_timestep = self.node.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[3] = self.send_cmd_timestep
        commands_msg.data = self.timestamp_message + [float(not self.use_policy_action and not self.get_ready_state)] + q_target.flatten().tolist()
        # print(f"Time in command0: {commands_msg.data[1] - commands_msg.data[0]}")
        # print(f"Time in command1: {commands_msg.data[2] - commands_msg.data[1]}")
        # print(f"Time in command2: {commands_msg.data[3] - commands_msg.data[2]}")
        # print("q_target", q_target)

        self.commands_pub.publish(commands_msg)
