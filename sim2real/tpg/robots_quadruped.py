from typing import List, Tuple
import os
import math

import numpy as np
import torch
import sys

sys.path.append("../")
sys.path.append("./")


from sim2real.tpg.tpg_general import TPGInterfaceWithRobot
from sim2real.rl_inference.go2_agile_locomotion import LocomotionPolicyKeyboard
from sim2real.tpg.worldCC import XYThetaTimeSolution, getXYThetaAtTimes

def interpolate_pose(start_pos,start_yaw,goal_pos,goal_yaw,alpha):
    interp_pos = (1 - alpha) * start_pos + alpha * goal_pos
    interp_yaw = (1 - alpha) * start_yaw + alpha * goal_yaw
    return interp_pos, interp_yaw



def quat_to_euler_angles(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    Convention: ZYX (yaw-pitch-roll).
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90° if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


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

def get_yaw(orientation):
    print(orientation,"orientation")
    yaw = quat_to_euler_angles(orientation)[-1]
    # if yaw>np.pi/2:
    #     return yaw-2*np.pi
    # else:
    #     return yaw
    if yaw<-0.02:
        return yaw
    else:
        return yaw

def wrap_angle(angle):
    # return (angle + 1.5 * np.pi) % (2 * np.pi) - 1.5 * np.pi
    return angle

def convert_theta(raw_theta: float) -> float:
    """Required to map the solution xytheta (in degrees and with a different x-axis) 
    to the robot's xytheta (in radians and with a different x-axis).
    This can be checked by looking at the starting position and comparing to the start of rectangle_visualization.gif"""
    # return np.deg2rad(-raw_theta+89.9)
    return raw_theta
class QuadrupedRobot(TPGInterfaceWithRobot):
    def __init__(self, agent_idx: int, config, model_path,node,rl_rate=50,policy_action_scale=0.25) -> None:
        self._agent_idx = agent_idx
        self._name = f"quadruped_{agent_idx}"
        self.robot = LocomotionPolicyKeyboard(config, node, model_path)
        self.pos_tol = 0.08
        self.yaw_tol = 0.06
        self.intermediate_goals: List[Tuple[np.ndarray, float]] = None # (K, 2)
        self.intermediate_times: np.ndarray = None # (K,)
        self.intermediate_index: int = 0
        self._physics_ready: bool = False
        self.reset_height = 0.8
        self.heading_translation = 0.6
    
    def set_solution_path(self, solution: XYThetaTimeSolution) -> None:
        self._solution = solution
        self._current_time = 0
        self._current_xytheta = self._solution.xythetas[0]
        self._target_time = self._solution.times[0] # Note: We only have a target_time, not a target_xytheta
        
        self.intermediate_goals = [] # Starts empty
        self.intermediate_times = np.array([self._current_time]) # Need to start with current time otherwise errors
        self.intermediate_index = 0
        self._physics_ready = False
        
    def set_new_time_cleared(self, time: float) -> None:
        self._cleared_time = time
    
    def get_current_time(self) -> float:
        return self._current_time
    
    def get_current_xytheta(self) -> np.ndarray:
        return self._current_xytheta

    def physics_step(self) -> None:
        if self.robot.use_policy_action:
        
            cur_pos, cur_orientation = self.robot.get_pose()
            # print(cur_pos,cur_orientation)
            cur_pos_xy = cur_pos[:2]
            cur_yaw = get_yaw(cur_orientation)
            # exit()
            # print(f"Agent {self._agent_idx} current time: {self._current_time}, current xytheta: {self._current_xytheta}")
            # print(f"Agent {self._agent_idx} target time: {self._target_time}")
            
            # Get the next waypoint
            new_target_xytheta, new_target_time = self._solution.get_next_waypoint(self._current_time, self._cleared_time)
            # print(f"cleared_time {self._cleared_time}, new_target_time: {new_target_time}")
            
            # Replan if the target waypoint / time has changed
            if new_target_time != self._target_time: # Replan by recalculating the intermediate goals and times if the target time has changed
                self._target_time = new_target_time
                delta_time = new_target_time - self._current_time
                num_interp_steps = int(np.ceil(delta_time / 0.3)+1) # We should have a waypoint every 0.1 seconds
                # print(new_target_xytheta[2],convert_theta(new_target_xytheta[2]),cur_yaw)
                self.intermediate_goals = generate_bezier_path_with_yaw(
                    start_pos=cur_pos_xy,
                    start_yaw=cur_yaw,
                    goal_pos=new_target_xytheta[:2],
                    goal_yaw=convert_theta(new_target_xytheta[2]), # Note -np.deg2rad because the yaw is in degrees
                    scale=0.6,
                    N=num_interp_steps)
                
                self.intermediate_times = np.linspace(self._current_time, new_target_time, num_interp_steps)
                self.intermediate_index = 0
                if self._agent_idx == 0:
                    print(self._agent_idx,self.intermediate_index,new_target_xytheta,num_interp_steps,self._current_time,self._target_time)
                # print(f"Agent {self._agent_idx} intermediate goals: {self.intermediate_goals}, intermediate times: {self.intermediate_times}")


            # Execute to the next intermediate goal
            if self.intermediate_index < len(self.intermediate_goals):
                wp_pos, wp_yaw = self.intermediate_goals[self.intermediate_index]
                command = compute_command_bezier(
                        cur_pos_xy, cur_yaw, wp_pos, wp_yaw
                        )
                
                
                if np.linalg.norm(cur_pos_xy - wp_pos) < 0.1 and abs(wrap_angle(wp_yaw-cur_yaw))<0.1:
                    self.intermediate_index += 1
                    # print(f"Agent {self._agent_idx} intermediate index: {self.intermediate_index}")
            else:
                command = [0.0, 0.0, 0.0]
            if self.intermediate_index < len(self.intermediate_times):
                self._current_time = self.intermediate_times[self.intermediate_index]
            self._current_xytheta = np.array([cur_pos_xy[0], cur_pos_xy[1], cur_yaw]) # Note -np.rad2deg because want yaw in degrees
            # print(f"Agent {self._agent_idx} command: {command}")
            # Move the robot
            # command = [1.0, 0.0, 0.0]
            self.robot.rl_inference(command)
        else:
            self.robot.rl_inference([0.0, 0.0, 0.0])
        


class Go2Quadruped(QuadrupedRobot):
    def __init__(self, agent_idx: int, config, model_path,rl_rate=50,policy_action_scale=0.25) -> None:
        super().__init__(agent_idx,config,model_path,rl_rate,policy_action_scale)
        self._name = f"go2_{agent_idx}"
        
    
    
        
        
   
        
    
    
    
    
    

    
    
   


    



##############################################################
# region Linear Interpolation
def linear_interp(start_pos, start_yaw, goal_pos, goal_yaw, N):
    path = []
    for t in np.linspace(0, 1, N):
        pos = (1 - t) * start_pos + t * goal_pos
        yaw = (1 - t) * start_yaw + t * goal_yaw
        path.append((pos, yaw))
    return path

def compute_command_linear(start_pos, start_yaw, goal_pos, goal_yaw):
    # Transform goal position into robot's local frame
    dx = goal_pos[0] - start_pos[0] 
    dy = goal_pos[1] - start_pos[1]
    
    # Rotate the delta vector by -start_yaw to get it in robot's frame
    dx_local = dx * np.cos(-start_yaw) - dy * np.sin(-start_yaw)
    dy_local = dx * np.sin(-start_yaw) + dy * np.cos(-start_yaw)
    
    dtheta = goal_yaw - start_yaw
    
    return [dx_local/(abs(dx_local)*0.1+1e-5), 
            dy_local/(abs(dy_local)*0.1+1e-5), 
            dtheta/(abs(dtheta)*0.01+1e-5)]
# endregion Linear Interpolation
##############################################################

##############################################################
# region Bezier Interpolation
def bezier_interp(P0, P1, P2, P3, t):
    return ((1 - t)**3) * P0 + 3 * ((1 - t)**2) * t * P1 + 3 * (1 - t) * (t**2) * P2 + (t**3) * P3

def bezier_tangent(P0, P1, P2, P3, t):
    return (
        3 * (1 - t)**2 * (P1 - P0) +
        6 * (1 - t) * t * (P2 - P1) +
        3 * t**2 * (P3 - P2)
    )

def generate_bezier_path_with_yaw(start_pos, start_yaw, goal_pos, goal_yaw, scale=0.6, N=30) -> List[Tuple[np.ndarray, float]]:
    P0 = np.array(start_pos)
    P3 = np.array(goal_pos)
    P1 = P0 + scale * np.array([np.cos(start_yaw), np.sin(start_yaw)])
    P2 = P3 - scale * np.array([np.cos(goal_yaw), np.sin(goal_yaw)])


    if np.allclose(P0, P3, atol=1e-1):
        yaws = np.linspace(start_yaw, goal_yaw, N)
        return [(P0.copy(), yaw) for yaw in yaws]
    
    # path = []
    # for t in np.linspace(0, 1, N):
    #     pos = bezier_interp(P0, P1, P2, P3, t)
    #     tangent = bezier_tangent(P0, P1, P2, P3, t)
    #     yaw = math.atan2(tangent[1], tangent[0])
    #     path.append((pos, yaw))
    # return path  

    # 

    path = []
    yaws = np.linspace(start_yaw,goal_yaw,N)
    for i,t in enumerate(np.linspace(0, 1, N)):
        pos = bezier_interp(P0, P1, P2, P3, t)
        tangent = bezier_tangent(P0, P1, P2, P3, t)
        yaw = yaws[i]
        path.append((pos, yaw))
    return path 

def compute_command_bezier(
    cur_pos, cur_yaw, goal_pos, goal_yaw,
    k1=5.0, k2=5.0, k3=10.0,
    max_v=1.5,
    max_w=1.0,
    pos_tol=0.04,
    yaw_tol=0.05,
):
    dx = goal_pos[0] - cur_pos[0]
    dy = goal_pos[1] - cur_pos[1]

    e_x = math.cos(cur_yaw) * dx + math.sin(cur_yaw) * dy
    e_y = -math.sin(cur_yaw) * dx + math.cos(cur_yaw) * dy
    e_theta = wrap_angle(goal_yaw - cur_yaw)

    if np.linalg.norm([dx, dy]) < pos_tol and abs(e_theta) < yaw_tol:
        return [0.0, 0.0, 0.0]

    v_x = k1 * e_x
    v_y = k2 * e_y
    w_z = k3 * e_theta

    v_x = np.clip(v_x, -max_v, max_v)
    v_y = np.clip(v_y, -max_v, max_v)
    w_z = np.clip(w_z, -max_w, max_w)

    return [v_x, v_y, w_z]
# endregion Bezier Interpolation
##############################################################
