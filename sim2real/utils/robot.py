import yaml 
import os
import argparse

class Robot:
    def __init__(self, config):
        self.ROBOT_TYPE = config['ROBOT_TYPE']
        self.MOTOR2JOINT = config['MOTOR2JOINT']
        self.JOINT2MOTOR = config['JOINT2MOTOR']
        self.UNITREE_LEGGED_CONST = config['UNITREE_LEGGED_CONST']
        self.JOINT_KP = config['JOINT_KP']
        self.JOINT_KD = config['JOINT_KD']
        self.WeakMotorJointIndex = config['WeakMotorJointIndex']
        self.NUM_MOTOR = config['NUM_MOTOR']
        self.NUM_JOINTS = config['NUM_JOINTS']
        self.DEFAULT_DOF_ANGLES = config['DEFAULT_DOF_ANGLES']
        self.USE_SENSOR = config['USE_SENSOR']
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    robot = Robot(config)

    print(robot.config)