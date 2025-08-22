import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np
import scipy.spatial.transform
import time
import yaml
import argparse
from scipy.spatial.transform import Rotation as R, Slerp
import pickle as pkl
import matplotlib.pyplot as plt
import os


np.set_printoptions(precision=3, suppress=True)

class PosControlNode(Node):
    def __init__(self, config):
        super().__init__('poscontrol_node')
        self.fs = 50.0

        self.traj_path = config['traj_path']
        self.traj = pkl.load(open(self.traj_path, 'rb'))
        
        self.xy, self.yaw, self.xy_vel, self.yaw_vel = self.traj[:, :2], self.traj[:, 2], self.traj[:, 3:5], self.traj[:, 5]


        self.kp_lin = 0.3  # Linear position gain
        self.kp_ang = 0.3  # Angular position gain
        self.lin_vel_filter_alpha = 0.05


        self.current_index = 0
        self.current_position = np.array([0.0, 0.0])
        self.current_orientation = 0.0
        self.current_velocity = np.array([0.0, 0.0])
        self.current_angular_velocity = 0.0

        # Subscribers
        self.create_subscription(Odometry, '/odometry', self.odometry_callback, 10)

        # Publishers
        self.ref_pos_pub = self.create_publisher(Odometry, '/reference_position', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.init_pub = self.create_publisher(Bool, '/reset_robot', 10)

        # Timer for error computation and control, 50 Hz
        self.create_timer(1.0 / self.fs, self.control_callback)


        self.pos_hist = np.zeros((len(self.traj), 2))
        self.ori_hist = np.zeros((len(self.traj), 1))
        self.lin_vel_hist = np.zeros((len(self.traj), 2))
        self.ang_vel_hist = np.zeros((len(self.traj), 1))
        self.lin_vel_fb = np.zeros((len(self.traj), 2))
        self.ang_vel_fb = np.zeros((len(self.traj), 1))


    def reset_robot(self):
        msg = Bool()
        msg.data = True
        self.init_pub.publish(msg)

    def odometry_callback(self, msg):
        # Extract current position from odometry
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        # Convert orientation from quaternion to Euler angles
        orientation_q = msg.pose.pose.orientation
        self.current_orientation = scipy.spatial.transform.Rotation.from_quat([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ]).as_euler('xyz')[2]

        self.current_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self.current_angular_velocity = msg.twist.twist.angular.z
        # print(f"Current position: {self.current_position}, Current orientation: {self.current_orientation}")

    def control_callback(self):
        # Get the reference position and orientation
        if self.current_index >= len(self.traj) - 1:
            self.get_logger().info("Reached the end of the reference trajectory")
            traj = np.concatenate([self.pos_hist, self.ori_hist, self.lin_vel_hist, self.ang_vel_hist], axis=1)
            traj_ref = np.concatenate([self.xy, self.yaw.reshape(-1, 1), self.xy_vel, self.yaw_vel.reshape(-1, 1)], axis=1)
            self.vis_traj(traj_ref[:, :3], traj_ref[:, 3:], traj[:, :3], traj[:, 3:], "results")
            import matplotlib.pyplot as plt
            


            raise Exception("Reached the end of the reference trajectory")
        
        ref_xy = self.xy[self.current_index]
        ref_yaw = self.yaw[self.current_index]

        # Compute errors
        perr = ref_xy - self.current_position
        R_ref = R.from_euler('z', ref_yaw)
        R_cur = R.from_euler('z', self.current_orientation)
        R_err = R_ref.inv() * R_cur
        yaw_err = -R_err.as_euler('xyz')[2]


        xy_vel_fb_world = self.kp_lin * perr + self.xy_vel[self.current_index]
        xy_vel_fb_world_3d = np.array([xy_vel_fb_world[0], xy_vel_fb_world[1], 0.0])
        xy_vel_fb_body_3d = R_cur.inv().as_matrix() @ xy_vel_fb_world_3d
        xy_vel_fb_body = xy_vel_fb_body_3d[:2]
        xy_vel_fb_body = np.clip(xy_vel_fb_body, -2.5, 2.5)
        
        yaw_vel_fb = self.kp_ang * yaw_err + self.yaw_vel[self.current_index]
        yaw_vel_fb = np.clip(yaw_vel_fb, -1.5, 1.5)


        self.get_logger().info(f"lin_vel_residual: {self.kp_lin * perr}, ang_vel_residual: {self.kp_ang * yaw_err}, lin_vel_ref: {self.xy_vel[self.current_index]}, ang_vel_ref: {self.yaw_vel[self.current_index]}")


        # Publish twist command
        twist = Twist()
        twist.linear.x = xy_vel_fb_body[0]
        twist.linear.y = xy_vel_fb_body[1]
        twist.angular.z = yaw_vel_fb

        self.cmd_vel_pub.publish(twist)

        # Publish reference position
        ref_msg = Odometry()
        ref_msg.pose.pose.position.x = ref_xy[0]
        ref_msg.pose.pose.position.y = ref_xy[1]
        ref_quat = R_ref.as_quat()
        ref_msg.pose.pose.orientation.x = ref_quat[0]
        ref_msg.pose.pose.orientation.y = ref_quat[1]
        ref_msg.pose.pose.orientation.z = ref_quat[2]
        ref_msg.pose.pose.orientation.w = ref_quat[3]
        ref_msg.twist.twist.linear.x = twist.linear.x
        ref_msg.twist.twist.linear.y = twist.linear.y
        ref_msg.twist.twist.angular.z = twist.angular.z
        ref_msg.child_frame_id = 'base_link'
        ref_msg.header.frame_id = 'world'
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        self.ref_pos_pub.publish(ref_msg)

        # Update index for the reference trajectory
        self.current_index += 1
        
        # # Clamp to maximum index
        # if self.current_index >= len(self.pos_traj):
        #     self.current_index = len(self.pos_traj) - 1
        #     raise Exception("Reached the end of the reference trajectory")

        self.pos_hist[self.current_index] = self.current_position
        self.ori_hist[self.current_index] = self.current_orientation
        self.lin_vel_hist[self.current_index] = self.current_velocity
        self.ang_vel_hist[self.current_index] = self.current_angular_velocity
        self.lin_vel_fb[self.current_index] = xy_vel_fb_world
        self.ang_vel_fb[self.current_index] = yaw_vel_fb


    def vis_traj(self, traj_ref, traj_vel_ref, traj, traj_vel, folder):
        os.makedirs(folder, exist_ok=True)

        t = np.arange(len(traj)) * 0.02
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # plot xy yaw on the 2d figure, yaw as the arrow direction
        ax.plot(traj[:, 0], traj[:, 1], label="ref pos")
        ax.plot(traj_ref[:, 0], traj_ref[:, 1], label="pos")
        ax.set_title("XY")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        for i in range(0, len(traj), 50):
            l = 0.05
            ax.arrow(traj[i, 0], traj[i, 1], l*np.cos(traj[i, 2]), l*np.sin(traj[i, 2]), head_width=0.01)
            ax.arrow(traj_ref[i, 0], traj_ref[i, 1], l*np.cos(traj_ref[i, 2]), l*np.sin(traj_ref[i, 2]), head_width=0.01)

        
        plt.tight_layout()
        fig.savefig(os.path.join(folder, "xy.png"))
        plt.close(fig)
        
        vel_fb = np.concatenate([self.lin_vel_fb, self.ang_vel_fb], axis=1)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            ax[i].plot(t, traj[:, i], label="pos")
            ax[i].plot(t, traj_ref[:, i], label="ref pos")
            ax[i].plot(t, traj_vel[:, i], label="vel")
            ax[i].plot(t, traj_vel_ref[:, i], label="ref vel")
            ax[i].plot(t, vel_fb[:, i], label="vel fb")
            ax[i].set_title(f"{['X', 'Y', 'Yaw'][i]}")
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Value")
            ax[i].legend()
            ax[i].grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(folder, "traj.png"))
        plt.close(fig)


def main(args=None):
    config_file = "controller_config.yaml"
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    
    rclpy.init(args=args)
    node = PosControlNode(config)
    node.reset_robot()

    input("Press Enter to start the control loop")
    
    print("Waiting for the robot to reset")
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(3.0)
    rclpy.spin_once(node, timeout_sec=1.0)
    print("Starting the control loop")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        traj = np.concatenate([node.pos_hist, node.ori_hist, node.lin_vel_hist, node.ang_vel_hist], axis=1)
        traj_ref = np.concatenate([node.xy, node.yaw.reshape(-1, 1), node.xy_vel, node.yaw_vel.reshape(-1, 1)], axis=1)
        node.vis_traj(traj_ref[:, :3], traj_ref[:, 3:], traj[:, :3], traj[:, 3:], "results")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
