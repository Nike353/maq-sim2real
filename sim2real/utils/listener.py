import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from datetime import datetime


class DataLoggerNode(Node):
    def __init__(self):
        super().__init__('data_logger_node')
        
        # Subscriptions
        self.create_subscription(Bool, '/reset_robot', self.reset_robot_callback, 10)
        self.create_subscription(Float64MultiArray, '/robot_command', self.robot_command_callback, 10)
        self.create_subscription(Float64MultiArray, '/robot_state', self.robot_state_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Odometry, '/odometry', self.odometry_callback, 10)
        self.create_subscription(Odometry, '/reference_position', self.reference_position_callback, 10)
        
        # Data storage
        self.data = {
            'reset_robot': [],
            'reset_robot_timestamps': [],
            'robot_command': [],
            'robot_command_timestamps': [],
            'robot_state': [],
            'robot_state_timestamps': [],
            'cmd_vel': [],
            'cmd_vel_timestamps': [],
            'odometry': [],
            'odometry_timestamps': [],
            'reference_position': [],
            'reference_position_timestamps': []
        }

    def reset_robot_callback(self, msg):
        self.data['reset_robot'].append(msg.data)
        self.data['reset_robot_timestamps'].append(self.get_timestamp())

    def robot_command_callback(self, msg):
        self.data['robot_command'].append(msg.data)
        self.data['robot_command_timestamps'].append(self.get_timestamp())

    def robot_state_callback(self, msg):
        self.data['robot_state'].append(msg.data)
        self.data['robot_state_timestamps'].append(self.get_timestamp())

    def cmd_vel_callback(self, msg):
        cmd_vel_data = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z]
        self.data['cmd_vel'].append(cmd_vel_data)
        self.data['cmd_vel_timestamps'].append(self.get_timestamp())

    def odometry_callback(self, msg):
        odom_data = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        }
        self.data['odometry'].append(odom_data)
        self.data['odometry_timestamps'].append(self.get_timestamp())

    def reference_position_callback(self, msg):
        ref_data = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        }
        self.data['reference_position'].append(ref_data)
        self.data['reference_position_timestamps'].append(self.get_timestamp())

    def get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def save_data(self, file_name):
        np.savez(file_name, **self.data)


def main(args=None):

    rclpy.init(args=args)
    
    bag_file = '10kg.npz'

    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Saving data and shutting down...')
        node.save_data(bag_file)
        keys = node.data.keys()
        for key in keys:
            node.get_logger().info(f'Saved {key} data, length: {len(node.data[key])}')
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
