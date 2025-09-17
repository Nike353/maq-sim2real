
import threading
import numpy as np
import argparse
import yaml
import sys


from loop_rate_limiters import RateLimiter

# from sim2real.sim_env.base_sim import BaseSimulator
NOT_RUN_GO2 = True
import time
sys.path.append(".././")
from sim2real.tpg.tpg_general import TimedTPGManager, TimeTPGController
from sim2real.tpg.robots_quadruped import Go2Quadruped, Go1Quadruped
from sim2real.tpg.worldCC import parse_map_file


class TPGRunner():
    def __init__(self, configs) -> None:
        OPTIONS = ["rectangle", "quadruped"][1]
        self.configs = configs
        self.timer = time.time()
        self.print_flag  = True
        self.use_sim = True
        ### Create tpg
        tpg_file = "/home/nikhil/nikhil/maq/maq-sim2real/sim2real/tpg/data/solution_2_agent.npz"
        self._timed_tpg_manager = TimedTPGManager()
        self._timed_tpg_manager.load_tpg(tpg_file)
        
        ### Create robots
        ROBOT_TYPES = ["go1","go2"]
        if OPTIONS == "rectangle":
            ROBOT_PROBS = [0.0, 1.0]
        elif OPTIONS == "quadruped":
            ROBOT_PROBS = [0.5, 0.5]
        else:
            raise ValueError(f"Invalid option: {OPTIONS}")
        # self._num_agents = 5 #10
        self._num_agents = self._timed_tpg_manager.num_agents
        self._robot_distribution = np.random.choice(ROBOT_TYPES, p=ROBOT_PROBS, size=self._num_agents) # (N,)
        # self._robot_distribution = ["go2","go2","anymal","spot"]
        # self._robot_distribution = ["spot","go2","anymal","spot","go2","anymal","go2","spot","anymal","go2","spot","spot"]
        # self._robot_distribution = ["spot","spot","go2","spot","go2","spot","anymal","spot",]
        self._robot_distribution = ["go2","go1"]#,"spot","spot","spot","spot"

        self._has_spot = np.sum(self._robot_distribution == "go2") > 0
        # if not self._has_spot:
        #     self._cell_size = 0.25
        # else:
        self._cell_size = 1.0
            
        ### Update the solution paths to be in respect to the cell size
        for i in range(self._num_agents):
            self._timed_tpg_manager.list_of_solutions[i].xythetas[:, :2] *= self._cell_size
        self._init_rate_handler()
       
        self.setup_scene()

    

    def _init_rate_handler(self):
        ## amogn the config, find the minimum rl_rate
        from loguru import logger
        self.logger = logger
        min_rl_rate = float("inf")
        for config in self.configs:
            if config.get("control_rate", 0) < min_rl_rate:
                min_rl_rate = config["control_rate"]
        self._rate_handler = RateLimiter(min_rl_rate)

    
    def setup_scene(self):
        
        
        

        
        ### Create robots
        # self._raw_robots: List[Union[CustomJetbot, CustomSpot]] = []
        self._raw_robots = []
        for i in range(self._num_agents):
            robot_type = self._robot_distribution[i]
            if robot_type == "jetbot":
                raise NotImplementedError("Jetbot not implemented")
            
            elif robot_type == "go2":
                robot = Go2Quadruped(i, self.configs[i])
                
            elif robot_type == "go1":
                robot = Go1Quadruped(i, self.configs[i])
            
            else:
                raise ValueError(f"Invalid robot type: {robot_type}")
            robot.set_solution_path(self._timed_tpg_manager.list_of_solutions[i])
            self._raw_robots.append(robot)
        
        self._tpg_controllers  = []
        for i in range(self._num_agents):
            tpg_controller = TimeTPGController(agent_idx=i, tpg_manager=self._timed_tpg_manager, robot=self._raw_robots[i])
            self._tpg_controllers.append(tpg_controller)
        self.timer = time.time()
        
        
        


    
    
    
    
    def run(self):
        while True:
            for i,tpg_controller in enumerate(self._tpg_controllers):
                if NOT_RUN_GO2 and i==0:
                    continue
                tpg_controller.physics_step()
             
            self._rate_handler.sleep()
            
        return 

    

if __name__ == "__main__":
    config_files = ["config/go2.yaml","config/go1.yaml"]
    configs = []
    
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        configs.append(config)
    


    
    tpg_runner = TPGRunner(configs)
    tpg_runner.run()


    
    
    

    

    

    
    