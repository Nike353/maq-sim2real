
import threading
import numpy as np
import argparse
import yaml
import sys
import rclpy


sys.path.append("../")
sys.path.append("./")

# from sim2real.sim_env.base_sim import BaseSimulator

import time

from sim2real.tpg.tpg_general import TimedTPGManager, TimeTPGController
from sim2real.tpg.robots_quadruped import Go2Quadruped
from sim2real.tpg.worldCC import parse_map_file



class TPGRunner():
    def __init__(self, configs, model_paths,node) -> None:
        OPTIONS = ["rectangle", "quadruped"][1]
        self.configs = configs
        self.model_paths = model_paths
        self.timer = time.time()
        self.print_flag  = True
        self.use_sim = True
        self.node = node
        ### Create tpg
        tpg_file = "/home/guanqihe/nikhil/multi_agent_quad/sim2real/maq-sim2real/sim2real/tpg/data/solution_tpg.npz"
        self._timed_tpg_manager = TimedTPGManager()
        self._timed_tpg_manager.load_tpg(tpg_file)
        
        ### Create robots
        ROBOT_TYPES = ["jetbot", "spot","anymal", "go2", "rectangle"]
        if OPTIONS == "rectangle":
            ROBOT_PROBS = [0.0, 0.0, 0.0, 0.0, 1.0]
        elif OPTIONS == "quadruped":
            ROBOT_PROBS = [0.0, 0.33, 0.33, 0.34, 0.0]
        else:
            raise ValueError(f"Invalid option: {OPTIONS}")
        # self._num_agents = 5 #10
        self._num_agents = self._timed_tpg_manager.num_agents
        self._robot_distribution = np.random.choice(ROBOT_TYPES, p=ROBOT_PROBS, size=self._num_agents) # (N,)
        # self._robot_distribution = ["go2","go2","anymal","spot"]
        # self._robot_distribution = ["spot","go2","anymal","spot","go2","anymal","go2","spot","anymal","go2","spot","spot"]
        # self._robot_distribution = ["spot","spot","go2","spot","go2","spot","anymal","spot",]
        self._robot_distribution = ["go2"]#,"spot","spot","spot","spot"]

        self._has_spot = np.sum(self._robot_distribution == "go2") > 0
        # if not self._has_spot:
        #     self._cell_size = 0.25
        # else:
        self._cell_size = 1.0
            
        ### Update the solution paths to be in respect to the cell size
        for i in range(self._num_agents):
            self._timed_tpg_manager.list_of_solutions[i].xythetas[:, :2] *= self._cell_size
        self.setup_scene()

    def setup_scene(self):
        
        
        

        
        ### Create robots
        # self._raw_robots: List[Union[CustomJetbot, CustomSpot]] = []
        self._raw_robots = []
        for i in range(self._num_agents):
            robot_type = self._robot_distribution[i]
            if robot_type == "jetbot":
                raise NotImplementedError("Jetbot not implemented")
            
            elif robot_type == "go2":
                robot = Go2Quadruped(i, self.configs[0], self.model_paths[0],self.node)
                # if i == 0:
                    # robot = Go2Quadruped(i, color=np.array([0.0,0.0,1.0]), end_xytheta=self._timed_tpg_manager.list_of_solutions[i].xythetas[-1])
                # else:
                    # robot = Go2Quadruped(i, color=np.array([1.0,0.8,0.0]), end_xytheta=self._timed_tpg_manager.list_of_solutions[i].xythetas[-1])
            
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
        
        for tpg_controller in self._tpg_controllers:
            tpg_controller.physics_step()
             
       
            
        return 

    

if __name__ == "__main__":
    config_files = ["config/go2.yaml"]
    configs = []
    model_paths = ["/home/guanqihe/nikhil/multi_agent_quad/sim2real/maq-sim2real/logs/go2_locomotion/20250724_193147-v2-locomotion-go2/exported/model_2600.onnx"]
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        configs.append(config)
    

    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    
    rate = node.create_rate(50)
    tpg_runner = TPGRunner(configs, model_paths,node)
    


    
    
    

    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_172529-H1_19dof_sim2real_-0.5actionrate_0.9_1.25mass-locomotion-h1/exported/model_800.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_213247-H1_19dof_sim2real_IsaacGym_noTerrain_noDR_actionrate-0.5_mass0.9_1.25-locomotion-h1/model_61800.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241117_180031-H1_19dof_sim2real_-0.5actionrate_0.9_1.25mass_delay0_20-locomotion-h1/model_86300.onnx"
    # onnx_model_path = "/Users/tairanhe/Workspace/RoboVerse/roboverse/logs/macOS_deploy/20241119_175059-TairanMotions_H1_dm_A6lift_nolinvel_addpush5_16_1.5_actionrate0.5_mass0.9_1.2_maxheight-300_initnoise-motion_tracking-h1/exported/model_1400.onnx"
    # h1_dof_nums = 19
    

    

    time.sleep(1)
    start_time = time.time()
    total_inference_cnt = 0

    try:
        while rclpy.ok():
            tpg_runner.run()
            rate.sleep()
            end_time = time.time()
            total_inference_cnt += 1
            if total_inference_cnt % 100 == 0:
                # node.get_logger().info(f"Average inference FPS: {100/(end_time - start_time)}")
                start_time = end_time
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()