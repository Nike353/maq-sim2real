import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# h1, go2:
# 

# g1:




from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import argparse
import yaml
from utils.robot import Robot

class StatePublisher(Node):
    def __init__(self, config):
        super().__init__("StatePublisher")
        # subscribe to the robot state by unitree sdk
        self.config = config
        self.robot = Robot(config)
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        else:
            raise NotImplementedError
        self.robot_low_state = None
        self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.robot_lowstate_subscriber.Init(self.LowStateHandler, 10)

        # `states_pub` is publishing the robot state get by unitree sdk to ROS2
        self.states_pub = self.create_publisher(Float64MultiArray, "robot_state", 1)

        self.could_start_loop = False
        
        self.num_dof = self.robot.NUM_JOINTS

        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)

        self.receive_from_sdk_timestep = 0.0
        self.publish_to_ros_timestep = 0.0

        self.timestamp_digit = 6
        self.timestamp_message = [0.0] * self.timestamp_digit

    def LowStateHandler(self, msg):

        self.robot_low_state = msg
        

        # print("mode machine", self.robot_low_state.mode_machine)
        self.receive_from_sdk_timestep = self.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[0] = self.receive_from_sdk_timestep
        self.could_start_loop = True

    def _prepare_low_state(self):
        imu_state = self.robot_low_state.imu_state
        # base quaternion
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        unitree_joint_state = self.robot_low_state.motor_state

        for i in range(self.num_dof):
            # import ipdb; ipdb.set_trace()
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq

            # error code
            error_code = unitree_joint_state[self.robot.JOINT2MOTOR[i]].reserve[0]
            if error_code != 0:
                print(f"joint {i} error code: {error_code}")
            
        
        # print("low_state_big_flag", self.robot_low_state.bit_flag)
          


    def main_loop(self):
        total_publish_cnt = 0
        start_time = time.time()
        rclpy.spin_once(self, timeout_sec=0.001)
        while not self.could_start_loop:
            rclpy.spin_once(self, timeout_sec=0.001)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            self._prepare_low_state()
            # send control
            state_msg = Float64MultiArray()
            self.publish_to_ros_timestep = self.get_clock().now().nanoseconds / 1e9
            self.timestamp_message[1] = self.publish_to_ros_timestep
            state_msg.data = self.timestamp_message + self.q.tolist() + self.dq.tolist()
            self.states_pub.publish(state_msg)

            # print(f"Send state to ROS2: {self.publish_to_ros_timestep - self.receive_from_sdk_timestep}")
            # print(f"Time in message: {state_msg.data[1] - state_msg.data[0]}")

            # FPS
            total_publish_cnt += 1
            if total_publish_cnt % 100 == 0:
                end_time = time.time()
                # self.get_logger().info(f"state sent {state_msg.data}")
                self.get_logger().info(f"FPS: {100/(end_time - start_time)}")
                start_time = end_time


def main(config):
    rclpy.init(args=None)

    ChannelFactoryInitialize(0,"lo")
    state_publisher = StatePublisher(config)

    try:
        state_publisher.main_loop()
    except KeyboardInterrupt:
        pass
    state_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    main(config=config)