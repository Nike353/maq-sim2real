import sys
sys.path.append(".././")
from sim2real.utils.robot_interface.base_interface import BaseInterface
from sim2real.rl_policy.go2_locomotion import LocomotionPolicy
import numpy as np
import time
import zmq
import threading
import json

class Go2Interface(BaseInterface):
    def __init__(self, config, agent_name=None):
        super().__init__(config)
        self.locomotion_policy = LocomotionPolicy(config, config.get("MODEL_PATH"))
        self.agent_name = agent_name
        
        # Initialize ZMQ subscriber
        if agent_name:
            self._init_zmq_subscriber()
            self._velocity_command = None  # Store the latest velocity command
            self._command_lock = threading.Lock()

    def _init_zmq_subscriber(self):
        """Initialize ZMQ subscriber for velocity commands on port 6001."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://127.0.0.1:6001")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        print(f"Go2Interface: Connected to ZMQ subscriber on port 6001 for agent {self.agent_name}")
        
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
                if message and message.get("agent_name") == self.agent_name:
                    velocity_cmd = message.get("agent_name", None)
                    if velocity_cmd is not None:
                        with self._command_lock:
                            self._velocity_command = velocity_cmd
                        print(f"Go2Interface: Received velocity command for {self.agent_name}: {velocity_cmd}")
                        
            except Exception as e:
                print(f"Go2Interface: ZMQ listener error: {e}")
                time.sleep(0.01)  # Small delay on error

    def _init_sdk_components(self):
        from unitree_sdk2py.core.channel import ChannelPublisher,ChannelSubscriber,ChannelFactoryInitialize
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ 
        ChannelFactoryInitialize(0,"enp5s0")
        self.low_cmd = unitree_go_msg_dds__LowCmd_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.robot_lowstate_subscriber.Init(self.LowStateHandler, 20)
        self.InitUnitreeLowCmd()
        self.low_state = None
        self.crc = CRC()

    def InitUnitreeLowCmd(self):
        """Initialize Unitree low-level command."""
        
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        for i in range(self.robot.NUM_MOTOR):
            
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

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

    def send_low_level_cmd(self, se2_vel):
        """Send command to Unitree robot."""
        # print("hi")
        rl_qtarget = self.locomotion_policy.rl_inference(self.get_state(), se2_vel)[0]
        cmd_q = rl_qtarget[0:self.num_dof]
        cmd_dq = 0.0 * np.ones(self.num_dof)
        cmd_tau = 0.0 * np.ones(self.num_dof)
        self._fill_motor_commands(self.low_cmd.motor_cmd, cmd_q, cmd_dq, cmd_tau)
        # Add CRC and send
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd) 

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
                self.send_low_level_cmd(se2_vel)
                self.clear_velocity_command()  # Clear after using
            else:
                print(f"Go2Interface: Invalid velocity command format: {velocity_cmd}")

    def LowStateHandler(self, msg):
        self.robot_low_state = msg
        # time.sleep(0.001)
        if self.physics_ready:
            self.locomotion_policy.use_policy_action = True
        else:
            self.locomotion_policy.use_policy_action = False
        if self.get_ready_state:
            self.locomotion_policy.get_ready_state = True
        else:
            self.locomotion_policy.get_ready_state = False

    def _prepare_low_state(self):
        imu_state = self.robot_low_state.imu_state
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion
        self.dq[3:6] = imu_state.gyroscope
        unitree_joint_state = self.robot_low_state.motor_state
        
        
        for i in range(self.num_dof):
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq

            error_code = unitree_joint_state[self.robot.JOINT2MOTOR[i]].reserve[0]
            if error_code != 0:
                print(f"joint {i} error code: {error_code}")
                self.q[7+i] = self.robot.DEFAULT_DOF_ANGLES[i]
                self.dq[6+i] = 0.0

    def get_state(self):
        self._prepare_low_state()
        return np.array(self.q.tolist()+self.dq.tolist(), dtype=np.float64).reshape(1, -1)


def main():
    """Main function to demonstrate usage with agent_name parameter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Go2Interface with ZMQ Subscriber')
    parser.add_argument('--agent_name', type=str, required=True, 
                       help='Name of the agent to listen for commands')
    parser.add_argument('--config', type=str, default='config/go2.yaml',
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
    
    # Create Go2Interface with agent name
    go2_interface = Go2Interface(config, agent_name=args.agent_name)
    
    print(f"Go2Interface started for agent: {args.agent_name}")
    print("Waiting for ZMQ velocity commands on port 6001...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Process ZMQ commands
            go2_interface.process_zmq_commands()
            time.sleep(0.01)  # 100Hz loop
    except KeyboardInterrupt:
        print("Shutting down Go2Interface...")


if __name__ == "__main__":
    main()