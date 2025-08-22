import time
import sys
import signal
import numpy as np
from collections import deque
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go, LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg, LowCmd_ as LowCmd_hg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DataLogger(Node):
    def __init__(self, save_path='data_log.npz', log_interval=1.0, buffer_rate=200.0, buffer_size=200*300, ignore_ref=False, robot_type='go2'):
        """
        Initializes the DataLogger node.

        Args:
            save_path (str): Path to save the .npz log file.
            log_interval (float): Interval in seconds for throttled logging.
            buffer_rate (float): Frequency in Hz to update the observation buffer.
            buffer_size (int): Maximum number of entries to store in the buffer.
        """
        super().__init__('data_logger')

        # Initialize buffer for high-frequency data (200 Hz)
        self.obs_buffer = deque(maxlen=buffer_size)  # Buffer size is 200*300 = 60000

        self.ignore_ref = ignore_ref

        # Initialize subscribers for mocap data
        self.create_subscription(Bool, '/reset_robot', self.reset_robot_callback, 10)
        self.create_subscription(Odometry, "/odometry", self.mocap_callback, 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
        self.create_subscription(Odometry, "/reference_position", self.ref_callback, 10)
        self.create_subscription(Int32, "/epi_done", self.epi_done_callback, 10)
        

        # Initialize subscribers for Unitree data
        if robot_type == 'go2':
            LowState_ = LowState_go
            LowCmd_ = LowCmd_go
        elif robot_type == 'g1':
            LowState_ = LowState_hg
            LowCmd_ = LowCmd_hg
        else:
            raise NotImplementedError

        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_state.Init(self.LowStateHandler, 10)
                
        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_cmd.Init(self.LowCmdHandler, 10)

        # Set up the signal handler
        signal.signal(signal.SIGINT, self.sigINT_handler)

        # Throttled logging setup
        self.log_interval = log_interval  # seconds
        self.last_log_time = time.time()
        self.log_timer = self.create_timer(0.1, self.throttled_log_callback)  # High-frequency timer for checking log interval

        # Buffer update timer at 200 Hz
        self.buffer_rate = buffer_rate  # Hz
        self.buffer_timer = self.create_timer(1.0 / self.buffer_rate, self.buffer_update_callback)

        self.get_logger().info("DataLogger node initialized.")

        # Initialize latest data holders
        self.latest_low_state = None
        self.latest_low_cmd = None
        self.latest_mocap = None
        self.latest_cmd = None
        self.latest_ref = None
        self.latest_reset = None
        self.latest_epi_done = None
        
        # Save path for the .npz file
        self.save_path = save_path
        

    def reset_robot_callback(self, msg):
        self.latest_reset = {
            'timestamp': time.time(),
            'reset_robot': msg.data
        }


    # Handler for Unitree low state messages
    def LowStateHandler(self, msg):
        """
        Handles incoming LowState_ messages and stores the latest data.
        """
        timestamp = time.time()
        self.latest_low_state = {
            'timestamp': timestamp,
            'motor_positions': np.array([motor.q for motor in msg.motor_state[:12]]),
            'motor_velocities': np.array([motor.dq for motor in msg.motor_state[:12]]),
            'motor_torques': np.array([motor.tau_est for motor in msg.motor_state[:12]]),
            'imu_quaternion': np.array(msg.imu_state.quaternion),
            'imu_gyroscope': np.array(msg.imu_state.gyroscope),
            'imu_accelerometer': np.array(msg.imu_state.accelerometer),
            'foot_force': np.array([msg.foot_force[i] for i in range(4)]),
            'foot_force_est': np.array([msg.foot_force_est[i] for i in range(4)])
        }

    # Handler for Unitree low command messages
    def LowCmdHandler(self, msg):
        """
        Handles incoming LowCmd_ messages and stores the latest data.
        """
        timestamp = time.time()
        self.latest_low_cmd = {
            'timestamp': timestamp,
            'motor_commands': {
                "q": np.array([motor_cmd.q for motor_cmd in msg.motor_cmd[:12]]),
                "dq": np.array([motor_cmd.dq for motor_cmd in msg.motor_cmd[:12]]),
                "kp": np.array([motor_cmd.kp for motor_cmd in msg.motor_cmd[:12]]),
                "kd": np.array([motor_cmd.kd for motor_cmd in msg.motor_cmd[:12]]),
                "tau": np.array([motor_cmd.tau for motor_cmd in msg.motor_cmd[:12]])
            }
        }
    
    def cmd_vel_callback(self, msg):
        self.latest_cmd = {
            'timestamp': time.time(),
            'linear': [msg.linear.x, msg.linear.y],
            'angular': [0, 0, msg.angular.z]
        }
    
    def ref_callback(self, msg):
        self.latest_ref = {
            'timestamp': time.time(),
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y],
            'orientation': [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z],
            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        }

    def epi_done_callback(self, msg):
        self.latest_epi_done = {
            'timestamp': time.time(),
            'epi_done': msg.data
        }

    # Callback function for mocap data
    def mocap_callback(self, msg):
        """
        Handles incoming Mocap data and stores the latest data.
        """
        timestamp = time.time()
        self.latest_mocap = {
            'timestamp': timestamp,
            'position': [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ],
            'orientation': [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z
            ],
            'linear_velocity': [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ],
            'angular_velocity': [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
        }

    # Throttled log callback
    def throttled_log_callback(self):
        """
        Logs the status at specified intervals to avoid excessive logging.
        """
        current_time = time.time()
        if (current_time - self.last_log_time) >= self.log_interval:
            buffer_length = len(self.obs_buffer)
            self.get_logger().info(
                f"Running... Buffer Size: {buffer_length}"
            )
            self.last_log_time = current_time

    # Buffer update callback at 200 Hz
    def buffer_update_callback(self):
        """
        Updates the observation buffer with the latest synchronized data.
        This function runs at a high frequency (e.g., 200 Hz).
        """
        # Fetch the latest data from each category if available
        if self.ignore_ref:
            cond = self.latest_low_state and self.latest_low_cmd and self.latest_mocap
        else:
            cond = self.latest_low_state and self.latest_low_cmd and self.latest_mocap and self.latest_cmd and self.latest_ref and self.latest_reset
        if cond:
            obs = {
                'timestamp': time.time(),
                'low_state': self.latest_low_state,
                'low_cmd': self.latest_low_cmd,
                'mocap': self.latest_mocap,
                'cmd': self.latest_cmd,
                'ref': self.latest_ref,
                'reset': self.latest_reset,
                'epi_done': self.latest_epi_done,
            }
            self.obs_buffer.append(obs)
        else:
            # print(self.latest_low_cmd,self.latest_low_state,self.latest_mocap)
            # If any data is missing, log a warning (throttled)
            self.get_logger().warn(" ".join([str(i) for i in [type(self.latest_low_cmd),type(self.latest_low_state),type(self.latest_mocap),type(self.latest_cmd),type(self.latest_ref)]]), throttle_duration_sec=1.0)
            self.get_logger().warn("Incomplete data for buffer update.", throttle_duration_sec=1.0)

    # Signal handler to save data upon interruption
    def sigINT_handler(self, sig, frame):
        """
        Handles SIGINT (e.g., Ctrl+C) to gracefully save data and exit.
        """
        self.get_logger().info("SIGINT received. Saving data...")
        print("Saving data to ", self.save_path)
        print("Buffer samples: ", len(self.obs_buffer))
        
        # Process and save all collected data
        self.process_and_save_data()
        
        self.get_logger().info("Data saved successfully.")
        sys.exit(0)

    def process_and_save_data(self):
        """
        Processes the observation buffer and saves it into a single .npz file.
        """
        # Initialize lists for processed data
        buffer_timestamps = []
        buffer_low_state_motor_positions = []
        buffer_low_state_motor_velocities = []
        buffer_low_state_motor_torques = []
        buffer_low_state_imu_quaternion = []
        buffer_low_state_imu_gyroscope = []
        buffer_low_state_imu_accelerometer = []
        buffer_low_state_foot_force = []
        buffer_low_state_foot_force_est = []

        buffer_low_cmd_q = []
        buffer_low_cmd_dq = []
        buffer_low_cmd_kp = []
        buffer_low_cmd_kd = []
        buffer_low_cmd_tau = []

        buffer_mocap_positions = []
        buffer_mocap_orientations = []
        buffer_mocap_linear_velocities = []
        buffer_mocap_angular_velocities = []
        buffer_ref_orientations = []
        buffer_ref_lin_vel = []
        buffer_ref_ang_vel = []
        buffer_epi_done = []

        # Iterate through the buffer and extract data
        for obs in self.obs_buffer:
            buffer_timestamps.append(obs['timestamp'])

            # Low State
            low_state = obs['low_state']
            buffer_low_state_motor_positions.append(low_state['motor_positions'])
            buffer_low_state_motor_velocities.append(low_state['motor_velocities'])
            buffer_low_state_motor_torques.append(low_state['motor_torques'])
            
            buffer_low_state_imu_quaternion.append(low_state['imu_quaternion'])
            buffer_low_state_imu_gyroscope.append(low_state['imu_gyroscope'])
            buffer_low_state_imu_accelerometer.append(low_state['imu_accelerometer'])
            buffer_low_state_foot_force.append(low_state['foot_force'])
            buffer_low_state_foot_force_est.append(low_state['foot_force_est'])

            # Low Command
            low_cmd = obs['low_cmd']
            buffer_low_cmd_q.append(low_cmd['motor_commands']['q'])
            buffer_low_cmd_dq.append(low_cmd['motor_commands']['dq'])
            buffer_low_cmd_kp.append(low_cmd['motor_commands']['kp'])
            buffer_low_cmd_kd.append(low_cmd['motor_commands']['kd'])
            buffer_low_cmd_tau.append(low_cmd['motor_commands']['tau'])

            # Mocap
            mocap = obs['mocap']
            buffer_mocap_positions.append(mocap['position'])
            buffer_mocap_orientations.append(mocap['orientation'])
            buffer_mocap_linear_velocities.append(mocap['linear_velocity'])
            buffer_mocap_angular_velocities.append(mocap['angular_velocity'])

            #reference pos/ori
            ref = obs['ref']
            buffer_ref_lin_vel.append(ref['linear_velocity'])
            buffer_ref_ang_vel.append(ref['angular_velocity'])
            # buffer_ref_orientations.append(ref['orientation'])

            #epi_done
            epi_done = obs['epi_done']
            buffer_epi_done.append(epi_done['epi_done'])
        # Convert lists to NumPy arrays
        buffer_timestamps = np.array(buffer_timestamps)

        buffer_low_state_motor_positions = np.array(buffer_low_state_motor_positions)
        buffer_low_state_motor_velocities = np.array(buffer_low_state_motor_velocities)
        buffer_low_state_motor_torques = np.array(buffer_low_state_motor_torques)
        buffer_low_state_imu_quaternion = np.array(buffer_low_state_imu_quaternion)
        buffer_low_state_imu_gyroscope = np.array(buffer_low_state_imu_gyroscope)
        buffer_low_state_imu_accelerometer = np.array(buffer_low_state_imu_accelerometer)
        buffer_low_state_foot_force = np.array(buffer_low_state_foot_force)
        buffer_low_state_foot_force_est = np.array(buffer_low_state_foot_force_est)

        buffer_low_cmd_q = np.array(buffer_low_cmd_q)
        buffer_low_cmd_dq = np.array(buffer_low_cmd_dq)
        buffer_low_cmd_kp = np.array(buffer_low_cmd_kp)
        buffer_low_cmd_kd = np.array(buffer_low_cmd_kd)
        buffer_low_cmd_tau = np.array(buffer_low_cmd_tau)

        buffer_mocap_positions = np.array(buffer_mocap_positions)
        buffer_mocap_orientations = np.array(buffer_mocap_orientations)
        buffer_mocap_linear_velocities = np.array(buffer_mocap_linear_velocities)
        buffer_mocap_angular_velocities = np.array(buffer_mocap_angular_velocities)

        buffer_ref_orientations = np.array(buffer_ref_orientations)
        buffer_ref_lin_vel = np.array(buffer_ref_lin_vel)
        buffer_ref_ang_vel = np.array(buffer_ref_ang_vel)
        buffer_epi_done = np.array(buffer_epi_done)
        import os
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
            
            
        # Save data into npz file
        np.savez(
            self.save_path, 
            time=buffer_timestamps, 
            joint_pos=buffer_low_state_motor_positions, 
            joint_vel=buffer_low_state_motor_velocities, 
            tau_est=buffer_low_state_motor_torques,
            IMU_quaternion=buffer_low_state_imu_quaternion, 
            IMU_gyro=buffer_low_state_imu_gyroscope, 
            IMU_acc=buffer_low_state_imu_accelerometer,
            foot_force=buffer_low_state_foot_force,
            foot_force_est=buffer_low_state_foot_force_est,
            joint_pos_cmd=buffer_low_cmd_q,
            joint_vel_cmd=buffer_low_cmd_dq,
            kp=buffer_low_cmd_kp,
            kd=buffer_low_cmd_kd,
            tau_cmd=buffer_low_cmd_tau,
            pos=buffer_mocap_positions,
            quat=buffer_mocap_orientations,
            lin_vel=buffer_mocap_linear_velocities,
            ang_vel=buffer_mocap_angular_velocities,
            epi_done=buffer_epi_done,
            ref_ori=buffer_ref_orientations,
            ref_lin_vel=buffer_ref_lin_vel,
            ref_ang_vel=buffer_ref_ang_vel
        )

        self.get_logger().info(f"Data saved to {self.save_path}.")
        # print first 10 timestamps for all data
        data = np.load(self.save_path, allow_pickle=True)
        print("First 10 timestamps for all data:")
        for key in data.files:
            print(key, data[key][:10])


def main(args=None):
    # Parse command-line arguments for network interface and save path

    import argparse
    parser = argparse.ArgumentParser(description='Data Logger')
    parser.add_argument('--interface', type=str, default='lo', help='network interface')
    parser.add_argument('--save_path', type=str, default='data_log.npz', help='save path')
    parser.add_argument('--ignore_ref', type=int, default=0, help='ignore reference position')
    parser.add_argument('--robot_type', type=str, default='go2', help='robot type')
    args_par = parser.parse_args()

    network_interface = args_par.interface
    save_path = args_par.save_path
    ignore_ref = bool(args_par.ignore_ref)
    robot_type = args_par.robot_type
    
    print("ignoring ref: ", ignore_ref)

    # Initialize Unitree SDK
    ChannelFactoryInitialize(0)
    

    # Initialize rclpy and create the DataLogger node
    rclpy.init(args=args)
    data_logger = DataLogger(save_path=save_path, ignore_ref=ignore_ref, robot_type=robot_type)

    try:
        rclpy.spin(data_logger)
    except KeyboardInterrupt:
        data_logger.sigINT_handler(None, None)
    finally:
        data_logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()