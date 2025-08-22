import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize




from unitree_sdk2py.utils.crc import CRC
import threading
import argparse
import yaml
from utils.robot import Robot

# from utils.h1_constant import UNITREE_LEGGED_CONST



class CommandSender:
    def __init__(self, config, node):
        self.config = config 
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        else:
            raise NotImplementedError


        # import ipdb; ipdb.set_trace()
        self.robot = Robot(config)
        for i in range(len(self.robot.JOINT_KP)):
            self.robot.JOINT_KP[i] *= 1.0

        for i in range(len(self.robot.JOINT_KD)):
            self.robot.JOINT_KD[i] *= 1.0

        self.weak_motor_joint_index = []
        # import ipdb; ipdb.set_trace()
        for key, value in self.robot.WeakMotorJointIndex.items():
            self.weak_motor_joint_index.append(value)

        # print(self.weak_motor_joint_index)

        self.node = node

        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # subscribe to the robot command by rl inference
        self.commands_sub = self.node.create_subscription(
            Float64MultiArray, "robot_command", self.command_callback, 1
        )
        self.command_msg = None

        self.send_command_q = np.zeros(self.robot.NUM_MOTOR)

        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError

        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

        self.timestamp_digit = 6
        self.timestamp_message = [0.0] * self.timestamp_digit

        self.get_command_timestep = 0.0
        self.send_command_timestep = 0.0

        self.delay_timestamp_publisher = self.node.create_publisher(Float64MultiArray, "delay_timestamp", 1)
        



    def InitLowCmd(self):

        # h1/go2:
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        # g1:

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(self.robot.NUM_MOTOR):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01 
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A 
            self.low_cmd.motor_cmd[i].q= self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            if self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
            else:
                pass

    def is_weak_motor(self, motor_index):
        return motor_index in self.weak_motor_joint_index
        
        # return motor_index in {
        #     self.robot.WeakMotorJointIndex.kLeftAnkle,
        #     self.robot.WeakMotorJointIndex.kRightAnkle,
        #     self.robot.WeakMotorJointIndex.kRightShoulderPitch,
        #     self.robot.WeakMotorJointIndex.kRightShoulderRoll,
        #     self.robot.WeakMotorJointIndex.kRightShoulderYaw,
        #     self.robot.WeakMotorJointIndex.kRightElbow,
        #     self.robot.WeakMotorJointIndex.kLeftShoulderPitch,
        #     self.robot.WeakMotorJointIndex.kLeftShoulderRoll,
        #     self.robot.WeakMotorJointIndex.kLeftShoulderYaw,
        #     self.robot.WeakMotorJointIndex.kLeftElbow,
        # }

    def command_callback(self, msg):
        self.command_msg = msg
        self.get_command_timestep = self.node.get_clock().now().nanoseconds / 1e9


    def _prepare_low_cmd(self):
        self.timestamp_message[0] = self.command_msg.data[0]
        self.timestamp_message[1] = self.command_msg.data[1]
        self.timestamp_message[2] = self.command_msg.data[2]
        self.timestamp_message[3] = self.command_msg.data[3]
        self.timestamp_message[4] = self.get_command_timestep

        self.emergence_stop = self.command_msg.data[self.timestamp_digit]

        rl_actions = self.command_msg.data[self.timestamp_digit+1:]
        # print(rl_actions)
        assert len(rl_actions) == self.robot.NUM_JOINTS
       
        self.send_command_q = rl_actions

    def send_command(self):
        if self.emergence_stop:
            for i in range(self.robot.NUM_JOINTS):
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].q = self.send_command_q[i]
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].dq = 0.0
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kp = 0.0
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kd = self.robot.JOINT_KD[i]
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].tau = 0.0
        else:
            for i in range(self.robot.NUM_JOINTS):
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].q = self.send_command_q[i]
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].dq = 0.0
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kp = self.robot.JOINT_KP[i]  # * 0.85 for g1_29dof
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kd = self.robot.JOINT_KD[i]
                self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].tau = 0.0 
                # print(self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].q)
                # print(self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kp)
                # print(self.low_cmd.motor_cmd[self.robot.JOINT2MOTOR[i]].kd)

        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def main_loop(self):
        self._prepare_low_cmd()
        self.send_command_timestep = self.node.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[5] = self.send_command_timestep
        self.delay_timestamp_publisher.publish(Float64MultiArray(data=self.timestamp_message))

        self.send_command()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--onnx_model_path', type=str, default=None, help='onnx model path')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    rclpy.init(args=None)
    node = rclpy.create_node('command_sender')

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    rate = node.create_rate(200)

    ChannelFactoryInitialize(0,"lo")
    command_sender = CommandSender(config, node)
    time.sleep(1)

    start_time = time.time()
    command_cnt = 0

    try:
        while rclpy.ok():
            command_sender.main_loop()
            command_cnt += 1
            if command_cnt % 100 == 0:
                node.get_logger().info(f"Average command_cnt FPS: {100/(time.time()-start_time)}")
                start_time = time.time()
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()