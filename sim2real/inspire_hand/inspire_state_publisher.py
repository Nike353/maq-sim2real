import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_ as MotorStates



from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import argparse
import yaml
from loguru import logger

import sys
sys.path.append('../')
from utils.robot import Robot

class StatePublisher(Node):
    def __init__(self, config):
        super().__init__("StatePublisher")
        # subscribe to the robot state by unitree sdk
        self.config = config
        self.robot = Robot(config)
        self.robot_low_state = None

        # `states_pub` is publishing the robot state get by unitree sdk to ROS2
        self.states_pub = self.create_publisher(Float64MultiArray, "inspire_state", 1)

        self.could_start_loop = False
        
        self.num_dof = self.robot.NUM_JOINTS

        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(self.num_dof)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)

        self.receive_from_sdk_timestep = 0.0
        self.publish_to_ros_timestep = 0.0

        self.timestamp_digit = 6
        self.timestamp_message = [0.0] * self.timestamp_digit
        if self.config["ROBOT_TYPE"] == "unitree_inspire":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/inspire/state", MotorStates)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_inspire, 10)
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported")
        
        self.send_joint_pos = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_pos_pub = self.create_publisher(Float64MultiArray, "inspire_joint_pos", 1)



    def LowStateHandler_inspire(self, msg): # Yuanhang: the msg type MUST be declared explicitly for simulation
        self.robot_low_state = msg

        self.receive_from_sdk_timestep = self.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[0] = self.receive_from_sdk_timestep
        self.could_start_loop = True

    def _prepare_low_state(self):


        unitree_joint_state = self.robot_low_state

        for i in range(self.num_dof):
            # import ipdb; ipdb.set_trace()
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq
            self.tau_est[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].tau_est
            self.temp_first[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].temperature[0]
            self.temp_second[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].temperature[1]
            # print("joint", i, "temp", self.temp_first[i], self.temp_second[i])
            # error code
            error_code = unitree_joint_state[self.robot.JOINT2MOTOR[i]].reserve[0]
            if error_code != 0:
                print(f"joint {i} error code: {error_code}")
            
        
        # print("low_state_big_flag", self.robot_low_state.bit_flag)

    def main_loop(self):
        total_publish_cnt = 0
        start_time = time.time()
        logger.info("Start loop")
        rclpy.spin_once(self, timeout_sec=0.001)
        while not self.could_start_loop:
            rclpy.spin_once(self, timeout_sec=0.001)
            logger.warning("Waiting for low state")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            self._prepare_low_state()
            # send control
            state_msg = Float64MultiArray()
            self.publish_to_ros_timestep = self.get_clock().now().nanoseconds / 1e9
            self.timestamp_message[1] = self.publish_to_ros_timestep
            # print(len(self.timestamp_message))
            # print(len(self.q))
            # print(len(self.dq))
            state_msg.data = self.timestamp_message + self.q.tolist() + self.dq.tolist()
            self.states_pub.publish(state_msg)

            # print(f"Send state to ROS2: {self.publish_to_ros_timestep - self.receive_from_sdk_timestep}")
            # print(f"Time in message: {state_msg.data[1] - state_msg.data[0]}")
            self.send_joint_pos.data = self.q[7:].tolist()
            self.send_joint_pos_pub.publish(self.send_joint_pos)


            # FPS
            total_publish_cnt += 1
            if total_publish_cnt % 100 == 0:
                end_time = time.time()
                # self.get_logger().info(f"state sent {state_msg.data}")
                self.get_logger().info(f"FPS: {100/(end_time - start_time)}")
                start_time = end_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/unitree_inspire.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    rclpy.init(args=None)

    ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])

    state_publisher = StatePublisher(config)

    try:
        state_publisher.main_loop()
    except KeyboardInterrupt:
        pass
    state_publisher.destroy_node()
    rclpy.shutdown()
