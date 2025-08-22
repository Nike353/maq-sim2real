import time
import numpy as np
# from scipy.spatial.transform import Rotation as R
# from scipy.signal import butter, lfilter
from avp_stream import VisionProStreamer

import argparse
import yaml
from loguru import logger
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

def quat_rotate_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a + b + c  # Changed from a - b + c

class VisionPro(Node):
    def __init__(self, cfg, node):
        
        super().__init__('vision_pro_publisher')
        self.cfg = cfg
        self.node = node
        self.avp_ip = self.cfg['avp_ip']
        self.avp_frequency = self.cfg['avp_frequency']

        # Connect to Vision Pro
        logger.info(f"Connecting to Vision Pro at {self.avp_ip} with frequency {self.avp_frequency} Hz")
        self.avp_streamer = VisionProStreamer(self.avp_ip, self.avp_frequency)

        self.avp_pub = self.node.create_publisher(Float64MultiArray, 'vision_pro_data', 10)

        self.prev_ref_body_pos = np.zeros((3, 3))
        self.prev_pose = np.zeros(9)

    def main_loop(self):
        last_timestamp = time.time()
        avp_latest = self.avp_streamer.latest

        
        left_wrist = avp_latest['left_wrist'][:, :3, -1][0]
        right_wrist = avp_latest['right_wrist'][:, :3, -1][0]
        # right_wrist = avp_latest['right_fingers'][13, :3, -1]
        # left_wrist = avp_latest['left_fingers'][13, :3, -1]
        head = avp_latest['head'][:, :3, -1][0]

        # Adjust position and orientations
        # left_wrist[0] += 0.2
        # right_wrist[0] += 0.2
        # head[0] += 0.1
        # head[2] += 0.1

        rotate_quat = np.array([0.707107, 0, 0, -0.707107]).reshape(1,4)
        transition = np.array([0.0, 0, 0.3])

        # rotate 90 degrees around z axis
        left_wrist = quat_rotate_numpy(rotate_quat, left_wrist)
        right_wrist = quat_rotate_numpy(rotate_quat, right_wrist)
        head = quat_rotate_numpy(rotate_quat, head)

        left_wrist = left_wrist + transition
        right_wrist = right_wrist + transition
        head = head + transition

        avp_pose_new = np.concatenate([left_wrist, right_wrist, head])
        print(avp_pose_new)
        # rgb_pos = rgb_pos - rgb_pos[0]
        avp_msg = Float64MultiArray()
        smooth_pose = 0.8 * self.prev_pose + 0.2 * avp_pose_new.flatten()
        now_timestamp = time.time()
        velocity = (np.gradient(np.concatenate([self.prev_pose[None, ], smooth_pose[None, ]], axis=0), axis=-2) * (1/(now_timestamp - last_timestamp)))[0]
        #velocity = [1]
        last_timestamp = now_timestamp
        self.prev_pose = smooth_pose.copy()
        smooth_pose = np.concatenate([smooth_pose, velocity])

        avp_msg.data = smooth_pose.tolist()
        self.avp_pub.publish(avp_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Pro Publisher')
    parser.add_argument('--config', type=str, default='config/vision_pro.yaml', help='vision pro configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    
    rate = node.create_rate(config['avp_frequency'])
    avp_counter = 0
    start_time = time.time()
    vision_pro = VisionPro(config, node)
    try:
        while rclpy.ok():
            vision_pro.main_loop()
            rate.sleep()
            end_time = time.time()
            avp_counter += 1
            if avp_counter % 100 == 0:
                node.get_logger().info(f"Average inference FPS: {100/(end_time - start_time)}")
                start_time = end_time
    except KeyboardInterrupt:
        pass
