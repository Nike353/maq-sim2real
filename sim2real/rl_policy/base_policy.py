
import numpy as np
import time
import onnxruntime
from sim2real.utils.robot import Robot
import sys
sys.path.append(".././")
from sim2real.utils.robot_interface.base_interface import BaseInterface
from loop_rate_limiters import RateLimiter

class RLPolicy:
    def __init__(self, 
                 config,  
                 model_path, 
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        self.config = config
        self.robot = Robot(config)
        self.num_dofs = self.robot.NUM_JOINTS
        self.default_dof_angles = np.array(self.robot.DEFAULT_DOF_ANGLES)

        # load onnx policy
        
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        
        self.policy = policy_act

        self.last_policy_action = np.zeros((1, self.num_dofs))
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

        

        
        
        self.obs_scales = self.config["obs_scales"]
        self.current_obs = None
        self._init_command_components()

    

    def _init_command_components(self):
        self.use_policy_action = False
        self.get_ready_state = False
        self.init_count = 0
        
    
    def _init_rate_limiter(self):
        from loguru import logger
        self.logger = logger
        self.rate = RateLimiter(self.config["rl_rate"])

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

    


    def rl_inference(self, robot_state_data, command):
        start_time = time.time()
        command_cnt = 0

        


        

        obs = self.prepare_obs_for_rl(robot_state_data, command)

        policy_action = self.policy(obs)

        policy_action = np.clip(policy_action, -100, 100)

        # if not self.use_policy_action:
        #     policy_action *= 0.0  # Zero the actions if "e" was pressed

        self.last_policy_action = policy_action.copy()  
        scaled_policy_action = policy_action * self.policy_action_scale
        if self.get_ready_state:
            # import ipdb; ipdb.set_trace()
            print(self.get_ready_state,"get_ready_state")
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
        return q_target

        


        
