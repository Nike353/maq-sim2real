from abc import ABC, abstractmethod
import numpy as np
import threading
from sim2real.utils.robot import Robot
from loguru import logger
from pynput import keyboard
class BaseInterface(ABC):
    def __init__(self, config):
        self.config = config
        self.robot = Robot(config)

        self.num_dof = self.robot.NUM_JOINTS
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.physics_ready = False
        self.get_ready_state = False
        
        self._init_sdk_components()
        self._init_level_components()
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        self.logger = logger

    def _init_sdk_components(self):
        pass

    def _init_level_components(self):
        self.level = self.config.get("LEVEL", "HIGHLEVEL")
        pass

    @abstractmethod
    def get_state(self):
        pass

    
    def send_velocity_cmd(self, se2_vel):
        if self.level == "HIGHLEVEL":
            self.send_high_level_cmd(se2_vel)
        elif self.level == "LOWLEVEL":
            self.send_low_level_cmd(se2_vel)
    
    def send_low_level_cmd(self, se2_vel):
        pass
    
    def send_high_level_cmd(self, se2_vel):
        pass
    def _set_motor_command(self, motor_cmd, motor_id, joint_id, cmd_q, cmd_dq, cmd_tau):
        """Set motor command for a specific motor."""
        
        
        
    
        motor_cmd.q = cmd_q[joint_id]
        motor_cmd.dq = cmd_dq[joint_id]
        motor_cmd.tau = cmd_tau[joint_id]
        motor_cmd.kp = self.robot.JOINT_KP[joint_id] 
        motor_cmd.kd = self.robot.JOINT_KD[joint_id] 
    
    def _fill_motor_commands(self, motor_cmd, cmd_q, cmd_dq, cmd_tau):
        """Fill motor commands for all motors."""
        joint2motor = self.robot.JOINT2MOTOR
        motor2joint = self.robot.MOTOR2JOINT
        for i in range(self.robot.NUM_MOTOR):
            m_id = joint2motor[i]
            j_id = motor2joint[i]
            self._set_motor_command(motor_cmd[i], m_id, j_id, cmd_q, cmd_dq, cmd_tau) 

    def start_key_listener(self):
        """Start a key listener using pynput."""
        print("Starting key listener")
        def on_press(key):
            try:
                print(f"Key pressed: {key.char}")
                if key.char == "]":
                    self.physics_ready = True
                    self.get_ready_state = False
                    self.logger.info("Ready to run controller")
                elif key.char == "o":
                    self.physics_ready = False
                    self.get_ready_state = False
                    self.logger.info("Actions set to zero")
                elif key.char == "i":
                    self.get_ready_state = True
                    self.init_count = 0
                    self.logger.info("Setting to init state")
            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive
                