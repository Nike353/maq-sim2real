import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter

# Import Vicon DataStream SDK
from pyvicon_datastream import tools

import argparse
import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32


class Vicon(Node):
    def __init__(self, cfg):
        
        super().__init__('vicon_publisher')
        
        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = cfg['vicon_tracker_ip']
        self.vicon_object_name = cfg['vicon_object_name']
        self.odom_frame_id = cfg['odom_frame_id']
        self.odom_child_frame_id = cfg['odom_child_frame_id']
        self.fs = cfg['frequency']
        
        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize previous values for velocity computation
        self.prev_time = None
        self.prev_position = None
        self.prev_quaternion = None

        # Low-pass filter parameters
        self.cut_off_freq =  config['cut_off_freq']
        self.filter_order =  config['filter_order']
        self.filter_window_size = config['filter_window_size']
        self.b, self.a = butter(self.filter_order, self.cut_off_freq / (0.5 * self.fs), btype='low')

        # Initialize data buffers for filtering
        self.vel_buffer = []
        self.omega_buffer = []
                
        # Odometry publisher
        self.odom_pub = self.create_publisher(Odometry, 'odometry', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'pose', 10)
        
        self.logger = self.get_logger()

        # Frequency counter
        self.freq_counter = 0
        self.create_timer(1.0, self.log_frequency)
        
    def get_vicon_data(self):
        position = self.tracker.get_position(self.vicon_object_name)
        
        if not position:
            print(f"Cannot get the pose of `{self.vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            position = np.array([x, y, z])/1000. # Convert to meters
            position = np.array([x, y, z])/1000. # Convert to meters
            rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, quaternion
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None

    def compute_velocities(self, current_time, position, quaternion):
        # Initialize velocities
        linear_velocity = np.zeros(3)
        angular_velocity = np.zeros(3)

        if (
            self.prev_time is not None
            and self.prev_position is not None
            and self.prev_quaternion is not None
        ):
            dt = current_time - self.prev_time
            if dt > 0:
                # Linear velocity
                dp = position - self.prev_position
                linear_velocity = dp / dt

                # Angular velocity
                prev_rot = R.from_quat(self.prev_quaternion)
                curr_rot = R.from_quat(quaternion)
                delta_rot = curr_rot * prev_rot.inv()
                delta_angle = delta_rot.as_rotvec()
                angular_velocity = delta_angle / dt
        else:
            # First data point; velocities remain zero
            pass

        # Update previous values
        self.prev_time = current_time
        self.prev_position = position
        self.prev_quaternion = quaternion

        return linear_velocity, angular_velocity

    def low_pass_filter(self, data_buffer, new_data):
        # Append new data to the buffer
        data_buffer.append(new_data)
        # Keep only the last N samples (buffer size)
        buffer_size = self.filter_window_size
        if len(data_buffer) > buffer_size:
            data_buffer.pop(0)
        # Apply low-pass filter if enough data points are available
        if len(data_buffer) >= self.filter_order + 1:
            data_array = np.array(data_buffer)
            filtered_data = lfilter(self.b, self.a, data_array, axis=0)[-1]
            return filtered_data
        else:
            return new_data  # Not enough data to filter; return the new data as is

    def log_frequency(self):
        # Log the frequency information
        # print(f"Vicon data acquisition frequency: {self.freq_counter} Hz")
        self.logger.info(f"Vicon data acquisition frequency: {self.freq_counter} Hz")
        self.freq_counter = 0  # Reset the counter for the next second

    def main_loop(self):
        print("Starting Vicon data acquisition...")
        try:
            while True:
                # Get Vicon data
                current_time, position, quaternion = self.get_vicon_data()
                if position is None:
                    print("Failed to get Vicon data.")
                    continue

                # Compute velocities
                linear_velocity, angular_velocity = self.compute_velocities(
                    current_time, position, quaternion
                )

                # Apply low-pass filter to velocities
                filtered_linear_velocity = self.low_pass_filter(
                    self.vel_buffer, linear_velocity
                )
                filtered_angular_velocity = self.low_pass_filter(
                    self.omega_buffer, angular_velocity
                )

                # Create and populate Odometry message
                msg = Odometry()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.odom_frame_id
                msg.child_frame_id = self.odom_child_frame_id
                msg.pose.pose.position.x = position[0]
                msg.pose.pose.position.y = position[1]
                msg.pose.pose.position.z = position[2]
                msg.pose.pose.orientation.x = quaternion[0]
                msg.pose.pose.orientation.y = quaternion[1]
                msg.pose.pose.orientation.z = quaternion[2]
                msg.pose.pose.orientation.w = quaternion[3]
                msg.twist.twist.linear.x = filtered_linear_velocity[0]
                msg.twist.twist.linear.y = filtered_linear_velocity[1]
                msg.twist.twist.linear.z = filtered_linear_velocity[2]
                msg.twist.twist.angular.x = filtered_angular_velocity[0]
                msg.twist.twist.angular.y = filtered_angular_velocity[1]
                msg.twist.twist.angular.z = filtered_angular_velocity[2]
                
                # Publish the message
                self.odom_pub.publish(msg)
                
                msg = PoseWithCovarianceStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.odom_frame_id
                msg.pose.pose.position.x = position[0]
                msg.pose.pose.position.y = position[1]
                msg.pose.pose.position.z = position[2]
                msg.pose.pose.orientation.x = quaternion[0]
                msg.pose.pose.orientation.y = quaternion[1]
                msg.pose.pose.orientation.z = quaternion[2]
                msg.pose.pose.orientation.w = quaternion[3]
                self.pose_pub.publish(msg)
                
                # Increment frequency counter
                self.freq_counter += 1
                
                # self.logger.info(f"Publishing Odometry message: {msg}", throttle_duration_sec=1.0)
                # print(f"frequency: {self.freq_counter}")
                rclpy.spin_once(self, timeout_sec=1.0 / self.fs - 2e-3)
                # Sleep to mimic sampling rate
                # time.sleep(1.0 / self.fs - 1e-3) # Subtract a small value to ensure the sleep time is less than the desired period

        except KeyboardInterrupt:
            print("Exiting Vicon data acquisition.")

def main(config):
    rclpy.init(args=None)

    vicon = Vicon(config)
    try:
        vicon.main_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mocap Publisher')
    parser.add_argument('--config', type=str, default='config/mocap_wean_go2.yaml', help='motion capture configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    main(config=config)
