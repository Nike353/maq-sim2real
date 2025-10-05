
import threading
import numpy as np
import argparse
import yaml
import sys


from loop_rate_limiters import RateLimiter

# from sim2real.sim_env.base_sim import BaseSimulator
import time
sys.path.append(".././")
from sim2real.tpg.tpg_general import TimedTPGManager, TimeTPGController
from sim2real.tpg.robots_quadruped import Go2Quadruped, Go1Quadruped
from sim2real.tpg.worldCC import parse_map_file
import zmq
import threading

class TPGRunner():
    def __init__(self, configs) -> None:
        OPTIONS = ["rectangle", "quadruped"][1]
        self.configs = configs
        self.timer = time.time()
        self.print_flag  = True
        self.use_sim = True
        self.msg_sub=None
        ### Create tpg
        tpg_file = "/home/rishi/Desktop/CMU/Research/maq-sim2real/sim2real/tpg/data/solution_tpg_new.npz"
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
        self._robot_distribution = ["go1","go1"]#,"spot","spot","spot","spot"

        self._has_spot = np.sum(self._robot_distribution == "go2") > 0
        # if not self._has_spot:
        #     self._cell_size = 0.25
        # else:
        self._cell_size = 1.0
        self.msg = None
        ### Update the solution paths to be in respect to the cell size
        for i in range(self._num_agents):
            self._timed_tpg_manager.list_of_solutions[i].xythetas[:, :2] *= self._cell_size
        self._init_zmq()

        # Start background subscriber thread
        self.sub_thread = threading.Thread(target=self._pose_listener, daemon=True)
        self.sub_thread.start()
        self._init_rate_handler()
        # exit()
        self.setup_scene()

    def _init_zmq(self, sub_ip="127.0.0.1", sub_port=6000, pub_ip="127.0.0.1", pub_port=6001):
        """Initialize ZMQ subscriber for mocap pose."""
        self.ctx = zmq.Context()
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{sub_ip}:{sub_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        print(f"Subscribed to mocap ZMQ at tcp://{sub_ip}:{sub_port}")
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{pub_ip}:{pub_port}")
        print(f"Published to mocap ZMQ at tcp://{pub_ip}:{pub_port}")
    
    def _pose_listener(self):
        """Background thread: listen for ZMQ messages and update pose cache."""
        while True:
            try:
                self.msg_sub = self.sub_socket.recv_pyobj()
                # print("hi")
                # if msg and msg.get("name") == "go1_base":
                #     pose_data = msg["pose"]
                #     pos = pose_data["position"]
                #     quat_xyzw = pose_data["orientation"]
                #     quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                #     self.global_pose = [pos, quat_wxyz]
            except Exception as e:
                print(f"Pose listener error: {e}")
                time.sleep(0.01)  # small backoff if something goes wrong
    

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
            command_dict = {}
            for i,tpg_controller in enumerate(self._tpg_controllers):
                #get the ith key from the msg
                if self.msg_sub:
                    key = list(self.msg_sub.keys())[i]
                    #ensure key ends with i
                    assert key.endswith(str(i))
                    pose = [self.msg_sub[key]['position'],self.msg_sub[key]['orientation']]
                    # print(pose[1],key)
                    tpg_controller.robot.global_pose = pose
                    # print(self.msg_sub[key],key)
                    _,command = tpg_controller.physics_step()
                    # print(command)
                    command_dict[key] = command
                self.msg_pub = command_dict
                # print(command_dict)
                self.pub_socket.send_pyobj(self.msg_pub)
            self._rate_handler.sleep()
            
        return 

    

if __name__ == "__main__":
    config_files = ["config/go1_0.yaml","config/go1_1.yaml"]
    configs = []
    
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        configs.append(config)
    


    
    tpg_runner = TPGRunner(configs)
    tpg_runner.run()


    
    
    

    

    

    
    