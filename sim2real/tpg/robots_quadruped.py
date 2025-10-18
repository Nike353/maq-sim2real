from typing import List, Tuple
import os
import math

import numpy as np
import sys

sys.path.append("../")
sys.path.append("./")

import time
from sim2real.tpg.tpg_general import TPGInterfaceWithRobot
from sim2real.tpg.worldCC import XYThetaTimeSolution, getXYThetaAtTimes
from sim2real.utils.robot_interface.go1_interface import Go1Interface
from sim2real.utils.robot_interface.go2_interface import Go2Interface
# from sim2real.tpg.tpg_ext_main import transform_pose_to_new_frame,wrap_to_pi

def interpolate_pose(start_pos,start_yaw,goal_pos,goal_yaw,alpha):
    interp_pos = (1 - alpha) * start_pos + alpha * goal_pos
    interp_yaw = (1 - alpha) * start_yaw + alpha * goal_yaw
    return interp_pos, interp_yaw

import zmq
import threading

def wrap_to_pi(angle):
    """Wrap angle from [0, 2π) to [-π, π] with π included."""
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if np.isclose(angle, -np.pi):  # ensure π included instead of -π
        angle = np.pi
    return angle
def transform_pose_to_new_frame(x, y, yaw):
    """
    Transform pose (x, y, yaw) from original frame to new frame.
    New frame: rotated 90° CCW and translated by (4, 4) in original frame.
    """
    # 1. Define transform from new frame to old frame
    theta = np.pi / 2  # 90 degrees in radians
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = np.array([4, 4])

    # 2. Compute robot pose in old frame
    p_old = np.array([x, y])

    # 3. Convert to new frame coordinates
    p_new = R.T @ (p_old - t)  # inverse of (R, t)

    # 4. Adjust yaw (subtract frame rotation)
    yaw_new = wrap_to_pi(yaw - theta)

    return p_new[0], p_new[1], yaw_new

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
    # print(orientation,"orientation")
    yaw = quat_to_euler_angles(orientation)[0]
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
    return (angle+1*np.pi)%(2*np.pi) - np.pi

def convert_theta(raw_theta: float) -> float:
    """Required to map the solution xytheta (in degrees and with a different x-axis) 
    to the robot's xytheta (in radians and with a different x-axis).
    This can be checked by looking at the starting position and comparing to the start of rectangle_visualization.gif"""
    # return np.deg2rad(-raw_theta+89.9)
    return raw_theta
class QuadrupedRobot(TPGInterfaceWithRobot):
    def __init__(self, agent_idx: int, config) -> None:
        self._agent_idx = agent_idx
        self._name = f"quadruped_{agent_idx}"
        self.robot = None
        self.pos_tol = 0.08
        self.yaw_tol = 0.06
        self.intermediate_goals: List[Tuple[np.ndarray, float]] = None # (K, 2)
        self.intermediate_times: np.ndarray = None # (K,)
        self.intermediate_index: int = 0
        self.reset_height = 0.8
        self.heading_translation = 0.6
    
    def set_solution_path(self, solution: XYThetaTimeSolution) -> None:
        self._solution = solution
        # print(
        self._current_time = 0
        self._current_xytheta = self._solution.xythetas[0]
        self._target_time = self._solution.times[0] # Note: We only have a target_time, not a target_xytheta
        # actual_pos, actual_orientation = self.robot.get_pose()
        # transformed_xythetas = self.transform_xythetas(solution.xythetas,actual_pos[:2],get_yaw(actual_orientation))
        # self._solution.xythetas = transformed_xythetas
        self.intermediate_goals = [] # Starts empty
        self.intermediate_times = np.array([self._current_time]) # Need to start with current time otherwise errors
        self.intermediate_index = 0
        
    
    def transform_xythetas(self, xythetas: np.ndarray, actual_pos: np.ndarray, actual_yaw: float) -> np.ndarray:
        transformed_xythetas = np.zeros_like(xythetas)
        x_p,y_p,yaw_p = xythetas[0]
        x_a,y_a,yaw_a = actual_pos[0],actual_pos[1],actual_yaw
        dtheta = yaw_a - yaw_p
        R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                      [np.sin(dtheta), np.cos(dtheta)]])
        t = np.array([x_a,y_a]) - R @ np.array([x_p,y_p])
        for i in range(len(xythetas)):
            pos = R@xythetas[i][:2] + t
            yaw = wrap_angle(xythetas[i][2] + dtheta)
            transformed_xythetas[i] = np.array([pos[0],pos[1],yaw])
        return transformed_xythetas
    
    def set_new_time_cleared(self, time: float) -> None:
        self._cleared_time = time
    
    def get_current_time(self) -> float:
        return self._current_time
    
    def get_current_xytheta(self) -> np.ndarray:
        return self._current_xytheta

    def physics_step(self) -> List[float]:
        if self.robot.physics_ready:
        
            cur_pos = self.global_pose[0]
            cur_orientation = self.global_pose[1]
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
                # self.intermediate_goals = generate_bezier_path_with_yaw(
                #     start_pos=cur_pos_xy,
                #     start_yaw=cur_yaw,
                #     goal_pos=new_target_xytheta[:2],
                #     goal_yaw=convert_theta(new_target_xytheta[2]), # Note -np.deg2rad because the yaw is in degrees
                #     scale=0.6,
                #     N=num_interp_steps)
                
                self.intermediate_goals = generate_linear_path_with_yaw(
                    start_pos=cur_pos_xy,
                    start_yaw=cur_yaw,
                    goal_pos=new_target_xytheta[:2],
                    goal_yaw=convert_theta(new_target_xytheta[2]), # Note -np.deg2rad because the yaw is in degrees
                    N=num_interp_steps)
                if self._agent_idx==2:
                    print(self.intermediate_goals,self._agent_idx,cur_yaw,convert_theta(new_target_xytheta[2]))
                self.intermediate_times = np.linspace(self._current_time, new_target_time, num_interp_steps)
                self.intermediate_index = 0
                if self._agent_idx == 0:
                    # print(self._agent_idx,self.intermediate_index,new_target_xytheta,num_interp_steps,self._current_time,self._target_time)
                    pass
                # print(f"Agent {self._agent_idx} intermediate goals: {self.intermediate_goals}, intermediate times: {self.intermediate_times}")


            # Execute to the next intermediate goal
            if self.intermediate_index < len(self.intermediate_goals):
                wp_pos, wp_yaw = self.intermediate_goals[self.intermediate_index]
                command = compute_command_bezier(
                        cur_pos_xy, cur_yaw, wp_pos, wp_yaw
                        )
                
                
                if np.linalg.norm(cur_pos_xy - wp_pos) < 0.05 and abs(wrap_angle(wp_yaw-cur_yaw))<0.1:
                    self.intermediate_index += 1
                    # print(f"Agent {self._agent_idx} intermediate index: {self.intermediate_index}")
            else:
                command = [0.0, 0.0, 0.0]
            if self.intermediate_index < len(self.intermediate_times):
                self._current_time = self.intermediate_times[self.intermediate_index]
            self._current_xytheta = np.array([cur_pos_xy[0], cur_pos_xy[1], cur_yaw]) # Note -np.rad2deg because want yaw in degrees
            # print(f"Agent {self._agent_idx} command: {command}")
            # print(cur_yaw,"cur_yaw",wp_yaw,"wp_yaw")
            # print(f"Agent {self._agent_idx} command: {command}")
            # print(cur_yaw,"cur_yaw",wp_yaw,"wp_yaw")
            # Move the robot
            return command
        else:
            # print("robot not ready")
            # pass
            if self.global_pose:
                actual_pos = self.global_pose[0]
                actual_orientation = self.global_pose[1]
                # print(get_yaw(actual_orientation),self._agent_idx)
                transformed_xythetas = self.transform_xythetas(self._solution.xythetas,actual_pos[:2],get_yaw(actual_orientation))
                # print(transformed_xythetas,self._agent_idx)
                # exit()
                self._solution.xythetas = transformed_xythetas
                return [0.0,0.0,0.0]
            else:
                print("pose not yet received")               
            # self.robot.send_velocity_cmd([0.0, 0.0, 0.0])
        


class Go2Quadruped(QuadrupedRobot):
    def __init__(self, agent_idx: int, config) -> None:
        super().__init__(agent_idx,config)
        self._name = f"go2_{agent_idx}"
        self.robot = Go2Interface(config)
        print("init go2 robot", agent_idx)
        
        
    
class Go1Quadruped(QuadrupedRobot):
    def __init__(self, agent_idx: int, config) -> None:
        super().__init__(agent_idx,config)
        self._name = f"go1_{agent_idx}"
        self.robot = Go1Interface(config)
        print("init go1 robot",agent_idx)
        
    

    
    
   
        
    
    
    
    
    

    
    
   


    





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

def generate_linear_path_with_yaw(start_pos, start_yaw, goal_pos, goal_yaw, N=30) -> List[Tuple[np.ndarray, float]]:
    """
    Generate a linear path with linearly interpolated yaw between start and goal positions.
    
    Args:
        start_pos: Starting position [x, y]
        start_yaw: Starting yaw angle in radians
        goal_pos: Goal position [x, y]
        goal_yaw: Goal yaw angle in radians
        N: Number of waypoints to generate
        
    Returns:
        List of tuples (position, yaw) representing the linear path
    """
    start_pos = np.array(start_pos)
    goal_pos = np.array(goal_pos)
    
    # If start and goal are very close, just interpolate yaw
    # if np.allclose(start_pos, goal_pos, atol=1e-1):
    #     yaws = np.linspace(start_yaw, goal_yaw, N)
    #     return [(start_pos.copy(), yaw) for yaw in yaws]
    
    # wrapped_goal_yaw = (goal_yaw - 2*np.pi + np.pi) % 2*np.pi - np.pi
    # if abs(wrapped_goal_yaw - start_yaw) < abs(goal_yaw - start_yaw):
    #     goal_yaw = wrapped_goal_yaw
    # wrapped_goal_yaw = (goal_yaw + 2*np.pi + np.pi) % 2*np.pi - np.pi
    # if abs(wrapped_goal_yaw - start_yaw) < abs(goal_yaw - start_yaw):
    #     goal_yaw = wrapped_goal_yaw
    wrapped_goal_yaw = goal_yaw % 2*np.pi # Gets positive value
    if abs(wrapped_goal_yaw - start_yaw) < abs(goal_yaw - start_yaw):
        # print(f"Initial goal: {goal_yaw}, wrapped: {wrapped_goal_yaw}")
        goal_yaw = wrapped_goal_yaw
    wrapped_goal_yaw = goal_yaw % 2*np.pi - 2*np.pi # Gets negative value
    if abs(wrapped_goal_yaw - start_yaw) < abs(goal_yaw - start_yaw):
        # print(f"Initial goal: {goal_yaw}, wrapped: {wrapped_goal_yaw}")
        goal_yaw = wrapped_goal_yaw
        
    path = []
    for i in range(N):
        alpha = i / (N - 1)  # Interpolation parameter from 0 to 1
        # Linear interpolation for position
        pos = (1 - alpha) * start_pos + alpha * goal_pos
        # Linear interpolation for yaw
        yaw = (1 - alpha) * start_yaw + alpha * goal_yaw
        path.append((pos, wrap_angle(yaw)))
    
    return path

def compute_command_bezier(
    cur_pos, cur_yaw, goal_pos, goal_yaw,
    # k1=5.0, k2=5.0, k3=10.0,
    k1=2.0, k2=2.0, k3=5.0,
    max_v=0.7,
    max_w=0.7,
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
