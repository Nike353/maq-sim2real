import sys
sys.path.append(".././")
from sim2real.utils.robot_interface.base_interface import BaseInterface
from sim2real.go1_sdk.lib.python.amd64 import robot_interface as sdk
import zmq
import threading
import json
import numpy as np
from sim2real.utils.robot import Robot
from loguru import logger

HIGHLEVEL = 0xee
LOWLEVEL  = 0xff

class Go1Interface(BaseInterface):
    def __init__(self, config, agent_name=None):
        
        
        # # Initialize ZMQ subscriber
        # if agent_name:
            
            
            
        self.config = config
        self.robot = Robot(config)

        self.num_dof = self.robot.NUM_JOINTS
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.physics_ready = False
        self.get_ready_state = False
        self.pose = None
        self.name = "go1_base"
        self.agent_name = agent_name
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        self.logger = logger
        if agent_name:
            
            self._init_sdk_components()
            self._init_level_components()
            self._init_zmq_subscriber()
            self._velocity_command = None  # Store the latest velocity command
            self._command_lock = threading.Lock()
       

    def _init_zmq_subscriber(self):
        """Initialize ZMQ subscriber for velocity commands on port 6001."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://127.0.0.1:6001")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        print(f"Go1Interface: Connected to ZMQ subscriber on port 6001 for agent {self.agent_name}")
        
        # Start the ZMQ listener thread
        self.zmq_thread = threading.Thread(target=self._zmq_listener, daemon=True)
        self.zmq_thread.start()

    def _zmq_listener(self):
        """Background thread to listen for ZMQ velocity commands."""
        while True:
            try:
                # Receive message
                message = self.zmq_socket.recv_pyobj()
                
                # Check if message is for this agent
                if message:
                    velocity_cmd = message.get(self.agent_name, None)
                    if velocity_cmd is not None:
                        with self._command_lock:
                            self._velocity_command = velocity_cmd
                        print(f"Go1Interface: Received velocity command for {self.agent_name}: {velocity_cmd}")
                        
            except Exception as e:
                print(f"Go1Interface: ZMQ listener error: {e}")
                import time
                time.sleep(0.01)  # Small delay on error

    def _init_sdk_components(self):
        self.level = self.config.get("LEVEL", "HIGHLEVEL")
        
        if self.level == "HIGHLEVEL":
            self.udp = sdk.UDP(HIGHLEVEL, 8080, self.config.get("high_level_ip"), 8082)
            self.cmd = sdk.HighCmd()
            self.state = sdk.HighState()
            self.udp.InitCmdData(self.cmd)
            print("Go1Interface: HIGHLEVEL")
        elif self.level == "LOWLEVEL":   
            self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.safe = sdk.Safety(sdk.LeggedType.Go1)
            self.cmd = sdk.LowCmd()
            self.state = sdk.LowState()
            self.udp.InitCmdData(self.cmd)
            print("Go1Interface: LOWLEVEL")

    def _get_robot_state(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        return self.state
    
    def _send_cmd_to_robot(self):
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def get_state(self):
        return self._get_robot_state()
    
    def get_latest_velocity_command(self):
        """Get the latest velocity command from ZMQ if available."""
        if not hasattr(self, '_velocity_command'):
            return None
            
        with self._command_lock:
            return self._velocity_command
    
    def clear_velocity_command(self):
        """Clear the current velocity command after using it."""
        if hasattr(self, '_velocity_command'):
            with self._command_lock:
                self._velocity_command = None

    def send_high_level_cmd(self, se2_vel):
        # print(f"send_high_level_cmd: {se2_vel}")
        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.speedLevel = 0
        self.cmd.velocity = [se2_vel[0], se2_vel[1]]
        self.cmd.yawSpeed = se2_vel[2]
        self.cmd.footRaiseHeight = 0.1
        self.cmd.bodyHeight = 0
        self.cmd.euler = [0, 0, 0]
        self.cmd.reserve = 0
        self._send_cmd_to_robot()

    def process_zmq_commands(self):
        """Main method to process ZMQ commands and send to robot."""
        if not self.agent_name:
            return
       
        # Get latest velocity command from ZMQ
        velocity_cmd = self.get_latest_velocity_command()
        
        if velocity_cmd is not None:
            # Convert to expected format (assume velocity_cmd is [vx, vy, yaw_rate])
            if len(velocity_cmd) >= 3:
                se2_vel = [velocity_cmd[0], velocity_cmd[1], velocity_cmd[2]]
                self.send_high_level_cmd(se2_vel)
                self.clear_velocity_command()  # Clear after using
            else:
                print(f"Go1Interface: Invalid velocity command format: {velocity_cmd}")


def main():
    """Main function to demonstrate usage with agent_name parameter."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Go1Interface with ZMQ Subscriber')
    parser.add_argument('--agent_name', type=str, required=True, 
                       help='Name of the agent to listen for commands')
    parser.add_argument('--config', type=str, default='config/go1.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
   
     # Load config (assuming YAML format)
    import yaml
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using default config")
        config = {"LEVEL": "LOWLEVEL"}
    
    # Create Go1Interface with agent name
    go1_interface = Go1Interface(config, agent_name=args.agent_name)
    
    print(f"Go1Interface started for agent: {args.agent_name}")
    print("Waiting for ZMQ velocity commands on port 6001...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Process ZMQ commands
            go1_interface.process_zmq_commands()
            time.sleep(0.01)  # 100Hz loop
    except KeyboardInterrupt:
        print("Shutting down Go1Interface...")


if __name__ == "__main__":
    main()
    
    

