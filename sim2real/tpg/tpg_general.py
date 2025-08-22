from typing import Optional
from abc import ABC, abstractmethod
import pickle
import pdb
import argparse
import sys
import numpy as np

sys.path.append("../")

from sim2real.tpg.worldCC import XYThetaTimeSolution


class TimedTPGNode:
    def __init__(self, time: float, dependencies: np.ndarray):
        self.time = time
        self.dependencies = dependencies # (N-1,)
    
    def __repr__(self):
        return f"TimedTPGNode(time={self.time}, dependencies={self.dependencies})"


class TimedTPGManager:
    def __init__(self):
        self.num_agents: int = None
        self.list_of_solutions: list[XYThetaTimeSolution] = None
        self.agent_to_tpg_nodes: list[list[TimedTPGNode]] = None # [[] for _ in range(self.num_agents)]
        self.current_times: np.ndarray = None # np.zeros(self.num_agents) # (N,), current time for each agent
            
    def load_tpg(self, path: str):
        """Loads TPG data from a numpy file.
        
        Expected format is a .npz file containing:
        - tpg_times: Array of times for each agent's TPG nodes
        - tpg_deps: Array of dependencies for each agent's TPG nodes
        - tpg_mask: Boolean mask indicating valid TPG nodes
        - xythetas: Array of xytheta waypoints for each agent
        - times: Array of waypoint times for each agent
        - waypoint_mask: Boolean mask indicating valid waypoints
        - costs: Array of costs for each agent
        """
        data = np.load(path)
        
        # Reconstruct TPG nodes
        self.agent_to_tpg_nodes = []
        num_agents = data['tpg_times'].shape[0]
        
        for i in range(num_agents):
            mask = data['tpg_mask'][i]
            agent_nodes = [TimedTPGNode(t, d) for t, d in 
                         zip(data['tpg_times'][i][mask], 
                             data['tpg_deps'][i][mask])]
            self.agent_to_tpg_nodes.append(agent_nodes)
            
        # Load trajectory data
        self.num_agents = num_agents
        self.current_times = np.zeros(self.num_agents)
        self.list_of_solutions = []
        
        for i in range(num_agents):
            mask = data['waypoint_mask'][i]
            xytheta = data['xythetas'][i][mask]
            times = data['times'][i][mask]
            cost = data['costs'][i]
            print(f"Agent {i} has {len(xytheta)} waypoints")
            print(xytheta)
            self.list_of_solutions.append(
                XYThetaTimeSolution(xytheta, times, cost)
            )
                
    def get_next_time_target(self, agent_idx: int, cur_time_reached: float):
        self.current_times[agent_idx] = cur_time_reached # Update the current time
        
        if len(self.agent_to_tpg_nodes[agent_idx]) == 0:
            return "done", self.current_times[agent_idx]
        
        cur_target_tpg_node = self.agent_to_tpg_nodes[agent_idx][0] # Get the current target
        if cur_time_reached == cur_target_tpg_node.time: # If we reached the current target and are waiting
            # while np.all(self.current_times > cur_target_tpg_node.dependencies):
            #     print(f"Agent {agent_idx} passed {self.agent_to_tpg_nodes[agent_idx][0]}")
            #     self.agent_to_tpg_nodes[agent_idx].pop(0)
            #     if len(self.agent_to_tpg_nodes[agent_idx]) == 0:
            #         return "done", None
            #     cur_target_tpg_node = self.agent_to_tpg_nodes[agent_idx][0]
            # if cur_time_reached == cur_target_tpg_node.time: # Still waiting at current target
            #     return "waiting", None
            # else:
            #     return "time_target", cur_target_tpg_node.time # We got assigned a new target
            
            dependencies = cur_target_tpg_node.dependencies # These are the times
            # Check if all dependencies are reached
            if np.all(self.current_times > dependencies): # All other agents have passed the dependency
                # print(f"Agent {agent_idx} passed {self.agent_to_tpg_nodes[agent_idx][0]}")
                self.agent_to_tpg_nodes[agent_idx].pop(0) # Remove the current TPG node
                if len(self.agent_to_tpg_nodes[agent_idx]) > 0: # If there are more targets, return the next one
                    return "time_target", self.agent_to_tpg_nodes[agent_idx][0].time
                else:
                    print(f"Agent {agent_idx} done at {cur_time_reached}")
                    return "done", self.current_times[agent_idx]
            else:
                return "waiting", self.current_times[agent_idx]
        else:
            return "time_target", cur_target_tpg_node.time
                

class TPGInterfaceWithRobot(ABC):
    @abstractmethod
    def set_solution_path(self, solution: XYThetaTimeSolution) -> None:
        """Sets the solution path that the robot will follow."""
        raise NotImplementedError("Subclasses must implement set_solution_path")
    
    @abstractmethod
    def set_new_time_cleared(self, time: float) -> None:
        """Sets the time that the robot is cleared to move until without having a collision."""
        raise NotImplementedError("Subclasses must implement set_new_time_cleared")
    
    @abstractmethod
    def physics_step(self) -> None:
        """Updates the robot's position and orientation based on the current time."""
        raise NotImplementedError("Subclasses must implement physics_step")
    
    @abstractmethod
    def get_current_time(self) -> float:
        """Returns the current time of the robot.""" # Note we called this "alpha" in our discussion but I'm refeering to it as "current_time" here
        raise NotImplementedError("Subclasses must implement get_current_time")

    @abstractmethod
    def get_current_xytheta(self) -> np.ndarray:
        """Returns the current position and orientation of the robot."""
        raise NotImplementedError("Subclasses must implement get_current_xytheta")
    
    


class TimeTPGController:
    def __init__(self, agent_idx: int, tpg_manager: TimedTPGManager, robot: TPGInterfaceWithRobot):
        self.agent_idx: int = agent_idx
        self.tpg_manager: TimedTPGManager = tpg_manager
        self.robot: TPGInterfaceWithRobot = robot
        
        self.current_time = 0
        self.current_target_time = None
        self.current_status = "waiting"
        self.current_xytheta = self.robot.get_current_xytheta()
        
    def physics_step(self):
        self.current_status, self.current_target_time = self.tpg_manager.get_next_time_target(self.agent_idx, self.current_time)
        
        self.robot.set_new_time_cleared(self.current_target_time) # Set the new time cleared that the robot could move until
        self.robot.physics_step() # Update the robot's position and orientation
        self.current_time = self.robot.get_current_time() # Get the current time that the robot is at
        self.current_xytheta = self.robot.get_current_xytheta() # Get the current position and orientation of the robot
        return self.current_status
        # print(f"Agent {self.agent_idx} current time: {self.current_time}, current status: {self.current_status}, current target: {self.current_target}")
        # if self.current_status == "waiting": # If waiting, query TPG for the next target
            # self.current_status, self.current_target = self.tpg_manager.get_next_time_target(self.agent_idx, self.current_time)
        # if self.current_status == "waiting":
        
        # if self.current_status == "time_target": # If time target, move to the target
        #     self.robot.set_new_time_cleared(self.current_target_time)
        #     self.robot.physics_step(step_size, scene)
        #     self.current_time = self.robot.get_current_time()
        #     self.current_xytheta = self.robot.get_current_xytheta()
        # elif self.current_status == "waiting": # If waiting, do nothing
        #     return
        # elif self.current_status == "done": # If done, do nothing
        #     return
        # else:
        #     raise ValueError(f"Invalid status: {self.current_status}")

