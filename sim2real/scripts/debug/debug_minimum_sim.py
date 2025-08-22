import mujoco
import mujoco.viewer
import numpy as np
import time
from loguru import logger
import argparse
import yaml
import torch
import onnxruntime

from utils.robot import Robot

def quat_rotate_inverse(q, v):
    # q is a quaternion in the form [w, x, y, z], different from the Rotation class
    q_w = q[0]
    q_vec = q[1:]

    # Calculate the different terms
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0

    return a - b + c

class Simulator:
    def __init__(self, config, ckpt_path):
        self.config = config
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.robot = Robot(self.config)
        self.kp = self.robot.JOINT_KP
        self.kd = self.robot.JOINT_KD
        self.onnx_policy_session = onnxruntime.InferenceSession(ckpt_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act

        self.num_dofs = self.robot.NUM_JOINTS
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.default_dof_angles = self.robot.DEFAULT_DOF_ANGLES
        self.policy_action_scale = 0.25

        self.lin_vel_command = np.array([0.5, 0])
        self.ang_vel_command = np.array([0])
    
    def get_obs(self):
        quat_robot_wxyz = self.mj_data.qpos[3:7]
        dof_pos = self.mj_data.qpos[7:7+self.num_dofs]
        dof_vel = self.mj_data.qvel[6:6+self.num_dofs].reshape(1, -1)
        base_ang_vel = self.mj_data.qvel[3:6].reshape(1, -1)
        dof_pos_minus_default = (dof_pos - self.default_dof_angles).reshape(1, -1)
        projected_gravity = quat_rotate_inverse(quat_robot_wxyz, np.array([0, 0, -1])).reshape(1, -1)
        lin_vel_command = self.lin_vel_command.reshape(1, -1)
        ang_vel_command = self.ang_vel_command.reshape(1, -1)
        obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    ang_vel_command, 
                                    lin_vel_command, 
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    projected_gravity
                                    ], axis=1)
        obs = obs.astype(np.float32)
        return obs
    
    def actions_to_joint_torque(self, actions):
        desired_dof_pos = actions * self.policy_action_scale
        dof_pos = self.mj_data.qpos[7:7+self.num_dofs]
        dof_vel = self.mj_data.qvel[6:6+self.num_dofs]
        torques = self.kp * (desired_dof_pos + self.default_dof_angles - dof_pos) - self.kd * dof_vel
        return torques
    
    def step(self):
        obs = self.get_obs()
        actions = self.policy(obs)
        self.last_policy_action = actions
        actions = actions.reshape(1, -1)
        torques = self.actions_to_joint_torque(actions)
        self.mj_data.ctrl[:] = torques
        mujoco.mj_step(self.mj_model, self.mj_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof.yaml', help='config file')
    args = parser.parse_args()

    ckpt_path = "/home/jiawei/Research/humanoid/RoboVerse/logs/Sim2Sim/20241126_141208-trail_0_resume0.95_linkmassRand-locomotion-g1_29dof/exported/model_500.onnx"

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    sim = Simulator(config, ckpt_path)

    while True:
        sim.step()
        sim.viewer.sync()