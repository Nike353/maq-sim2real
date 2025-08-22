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


np.set_printoptions(precision=3, suppress=True)

class PosControlNode(Node):
    def __init__(self, config):
        super().__init__('poscontrol_node')
        self.fs = 50.0

        self.traj_name = config["traj_name"]
        self.traj_config = config[self.traj_name]
        self.traj_type = self.traj_config["traj_type"]

        if self.traj_type == "line":
            self.default_xy = np.array(self.traj_config["default_xy"])
            self.default_theta = self.traj_config["default_theta"]
            T = self.traj_config["T"]
            Dx = self.traj_config["Dx"]
            Dy = self.traj_config["Dy"]
            Dtheta = self.traj_config["Dtheta"]
            repeat = self.traj_config["repeat"]
            self.pos_traj = self.line_traj(T, Dx, Dy, Dtheta, repeat)

        elif self.traj_type == "rotating":
            self.default_xy = np.array(self.traj_config["default_xy"])
            self.default_theta = self.traj_config["default_theta"]
            self.pos_traj = self.rotating_traj()
        
        elif self.traj_type == "sine":
            self.default_xy = np.array(self.traj_config["default_xy"])
            self.default_theta = self.traj_config["default_theta"]
            self.pos_traj = self.sine_traj()
        
        elif self.traj_type == "open_loop":
            self.pos_traj = self.open_loop_traj()

        vel_traj = np.diff(self.pos_traj, axis=0, prepend=self.pos_traj[0].reshape(1, -1))
        vel_traj[:, 2] = np.zeros_like(vel_traj[:, 2])
        vel_traj = vel_traj * self.fs
        self.vel_traj = vel_traj

        

        self.kp_lin = 0.8  # Linear position gain
        self.kp_ang = 0.8  # Angular position gain
        self.current_index = 0
        self.current_position = np.array([0.0, 0.0])
        self.current_orientation = 0.0

        # Subscribers
        self.create_subscription(Odometry, '/odometry', self.odometry_callback, 10)

        # Publishers
        self.ref_pos_pub = self.create_publisher(Odometry, '/reference_position', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.init_pub = self.create_publisher(Bool, '/reset_robot', 10)

        # Timer for error computation and control, 50 Hz
        self.create_timer(1.0 / self.fs, self.control_callback)

    def sine(self, t, A, w, b):
        return A * np.sin(w * t) + b

    def line(self, t, x0, x1, T):
        return x0 + (x1 - x0) / T * t
    
    def sine_traj(self, T, A, w, b):
        fs = self.fs
        T = 20.0
        default_x = self.default_xy[0]
        default_y = self.default_xy[1]
        default_theta = self.default_theta

        # x sine
        Ax, Ay, Ayaw = A
        wx, wy, wyaw = w
        bx, by, byaw = b

        t = np.linspace(0, T, int(fs * T))
        trajx = default_x + self.sine(t, Ax, wx, bx)
        trajy = default_y + self.sine(t, Ay, wy, by)
        traj_yaw = default_theta + self.sine(t, Ayaw, wyaw, byaw)

        traj = np.vstack([trajx, trajy, traj_yaw]).T
        
        return traj

    def line_traj(self, T, Dx, Dy, Dtheta, repeat):
        t_wait = 0.1
        fs = self.fs

        x0 = self.default_xy[0]
        y0 = self.default_xy[1]
        theta0 = self.default_theta

        t = np.linspace(0, T, int(fs * T))
        x = self.line(t, x0, x0 + Dx, T)
        x_wait = np.ones(int(fs * t_wait)) * x[-1]
        trajx = np.concatenate([x, x_wait, x[::-1]])
        trajx = np.tile(trajx, repeat)

        y = self.line(t, y0, y0 + Dy, T)
        y_wait = np.ones(int(fs * t_wait)) * y[-1]
        trajy = np.concatenate([y, y_wait, y[::-1]])
        trajy = np.tile(trajy, repeat)
        
        theta = self.line(t, theta0, theta0 + Dtheta, T)
        theta_wait = np.ones(int(fs * t_wait)) * theta[-1]
        traj_theta = np.concatenate([theta, theta_wait, theta[::-1]])
        traj_theta = np.tile(traj_theta, repeat)

        traj = np.vstack([trajx, trajy, traj_theta]).T
        return traj
    
    def slerp_yaw(t, yaws, t_new, degrees=False):
        t = np.asarray(t, float)
        t_new = np.asarray(t_new, float)
        rots = R.from_euler('z', yaws, degrees=degrees)
        slerp = Slerp(t, rots)
        rots_new = slerp(t_new)
        mats_new = rots_new.as_matrix()
        angular_vel = np.zeros((len(t_new)-1, 3))
        for i in range(len(t_new)-1):
            dt = t_new[i+1] - t_new[i]
            R_delta = mats_new[i].T @ mats_new[i+1]
            rotvec_delta = R.from_matrix(R_delta).as_rotvec()
            angular_vel[i] = rotvec_delta / dt
        xyz_interp = rots_new.as_euler('xyz', degrees=degrees).squeeze()
        yaws_interp = xyz_interp[:, 2]
        yaw_vel = angular_vel[:, 2]
        return yaws_interp, yaw_vel
    
    def rotating_traj(self):
        T = 6.0
        repeat = 1
        fs = self.fs
        vel = 1.5 # 0.7 1.5
        t = np.linspace(0, T, int(fs * T))
        trajx = np.zeros_like(t) + self.default_xy[0]
        trajy = np.zeros_like(t) + self.default_xy[1]
        traj_theta = vel * np.ones_like(t)

        trajx = np.concatenate([trajx, trajx[::-1]])
        trajx = np.tile(trajx, repeat)

        trajy = np.concatenate([trajy, trajy[::-1]])
        trajy = np.tile(trajy, repeat)

        traj_theta = np.concatenate([traj_theta, -traj_theta])
        traj_theta = np.tile(traj_theta, repeat)

        # moving avg traj_theta
        traj_theta = np.convolve(traj_theta, np.ones(10) / 10, mode='same')

        traj = np.vstack([trajx, trajy, traj_theta]).T
        return traj
    
    def open_loop_traj(self):
        T = 4.0
        fs = self.fs
        repeat = 1

        vx = 0.5
        vy = 0.0
        omega = 0.0

        t = np.linspace(0, T, int(fs * T))
        trajx = vx * np.ones_like(t)
        trajy = vy * np.ones_like(t)
        traj_theta = omega * np.ones_like(t)

        traj = np.vstack([trajx, trajy, traj_theta]).T
        return traj


    def init_reference_trajectory(self):
        pos_traj = None
        if self.traj_type == "sine":
            pos_traj = self.sine_traj()
        elif self.traj_type == "line":
            pos_traj = self.line_traj()
        elif self.traj_type == "rotating":
            pos_traj = self.rotating_traj()
        elif self.traj_type == "open_loop":
            pos_traj = self.open_loop_traj()
        
        vel_traj = np.diff(pos_traj, axis=0, prepend=pos_traj[0].reshape(1, -1))
        
        vel_traj[:, 2] = np.zeros_like(vel_traj[:, 2])
        vel_traj = vel_traj * self.fs

        return pos_traj, vel_traj

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
        # print(f"Current position: {self.current_position}, Current orientation: {self.current_orientation}")

    def control_callback(self):
        # Get the reference position and orientation
        if self.current_index >= len(self.pos_traj):
            self.get_logger().info("Reached the end of the reference trajectory")
            return
        
        reference = self.pos_traj[self.current_index]
        reference_position = np.array([reference[0], reference[1]])
        reference_orientation = reference[2]

        # Compute errors
        position_error = reference_position - self.current_position
        orientation_error = reference_orientation - self.current_orientation

        # Normalize orientation error to be within [-pi, pi]
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi
        orientation_error = np.array([orientation_error])

        linear_velocity = self.kp_lin * position_error + self.vel_traj[self.current_index, :2]


        if self.traj_type == "rotating":
            angular_velocity = self.pos_traj[self.current_index, 2:3]
        elif self.traj_type == "open_loop":
            angular_velocity = self.pos_traj[self.current_index, 2:3]
            linear_velocity = self.pos_traj[self.current_index, :2]
        else:
            # Compute control velocities
            angular_velocity = self.kp_ang * orientation_error + self.vel_traj[self.current_index, 2]

        self.get_logger().log(f"lin_vel_residual: {self.kp_lin * position_error},      \
                                ang_vel_residual: {self.kp_ang * orientation_error},   \
                                lin_vel_ref: {self.vel_traj[self.current_index, :2]},  \
                                ang_vel_ref: {self.vel_traj[self.current_index, 2]}", 0, throttle_duration_sec=0.5)

        rotation_matrix = scipy.spatial.transform.Rotation.from_euler('z', self.current_orientation).as_matrix()
        
        if self.traj_type != "open_loop":
            linear_velocity = rotation_matrix[:2, :2].T @ linear_velocity
        
        if self.traj_type == "rotating":
            linear_velocity = np.array([0.0, 0.0])
            reference_position += np.inf

        # Publish twist command
        twist = Twist()
        twist.linear.x = linear_velocity[0]
        twist.linear.y = linear_velocity[1]
        twist.angular.z = angular_velocity[0]
        
        twist.linear.x = np.clip(twist.linear.x, -5.0, 5.0)
        twist.linear.y = np.clip(twist.linear.y, -5.0, 5.0)
        twist.angular.z = np.clip(twist.angular.z, -10.0, 10.0)

        self.cmd_vel_pub.publish(twist)

        # Publish reference position
        ref_msg = Odometry()
        ref_msg.pose.pose.position.x = reference_position[0]
        ref_msg.pose.pose.position.y = reference_position[1]
        ref_quat = scipy.spatial.transform.Rotation.from_euler('xyz', [0.0, 0.0, reference_orientation]).as_quat()
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
        
        # Clamp to maximum index
        if self.current_index >= len(self.pos_traj):
            self.current_index = len(self.pos_traj) - 1
            raise Exception("Reached the end of the reference trajectory")


def main(args=None):
    parser = argparse.ArgumentParser(description='Position control')
    parser.add_argument('--config', type=str, default='controller_config.yaml', help='config file')
    args = parser.parse_args()
    config_file = args.config
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    
    rclpy.init(args=args)
    node = PosControlNode(config)
    node.reset_robot()

    input("Press Enter to start the control loop")
    
    print("Waiting for the robot to reset")
    time.sleep(3.0)
    print("Starting the control loop")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
