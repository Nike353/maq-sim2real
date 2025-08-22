import rclpy
import threading
from std_msgs.msg import Float64MultiArray, Float32
import numpy as np
import time

class VisDelay:
    def __init__(self, node):
        self.node = node 
        self.delay_sub = self.node.create_subscription(
            Float64MultiArray, "delay_timestamp", self.delay_callback, 1
        )
        self.foxglove_pubs = {
            'state_processing': self.node.create_publisher(Float32, "foxglove/delay/state_processing", 1),
            'state_to_rl': self.node.create_publisher(Float32, "foxglove/delay/state_to_rl", 1),
            'rl_inference': self.node.create_publisher(Float32, "foxglove/delay/rl_inference", 1),
            'action_to_cmd': self.node.create_publisher(Float32, "foxglove/delay/action_to_cmd", 1),
            'cmd_processing': self.node.create_publisher(Float32, "foxglove/delay/cmd_processing", 1),
            'total_latency': self.node.create_publisher(Float32, "foxglove/delay/total_latency", 1),
        }
        self.delay_msg = None
        
    def delay_callback(self, msg):
        self.delay_msg = msg

    def publish_delay(self, print_delay=False):
        state_processing_time = (self.delay_msg.data[1] - self.delay_msg.data[0]) * 1e3
        state_to_rl_time = (self.delay_msg.data[2] - self.delay_msg.data[1]) * 1e3
        rl_inference_time = (self.delay_msg.data[3] - self.delay_msg.data[2]) * 1e3
        action_to_cmd_time = (self.delay_msg.data[4] - self.delay_msg.data[3]) * 1e3
        cmd_processing_time = (self.delay_msg.data[5] - self.delay_msg.data[4]) * 1e3
        total_latency = (self.delay_msg.data[5] - self.delay_msg.data[0]) * 1e3

        self.foxglove_pubs['state_processing'].publish(Float32(data=state_processing_time))
        self.foxglove_pubs['state_to_rl'].publish(Float32(data=state_to_rl_time))
        self.foxglove_pubs['rl_inference'].publish(Float32(data=rl_inference_time))
        self.foxglove_pubs['action_to_cmd'].publish(Float32(data=action_to_cmd_time))
        self.foxglove_pubs['cmd_processing'].publish(Float32(data=cmd_processing_time))
        self.foxglove_pubs['total_latency'].publish(Float32(data=total_latency))

        if print_delay:
            self.node.get_logger().info(f"state_processing_time: {state_processing_time:.2f} ms")
            self.node.get_logger().info(f"state_to_rl_time: {state_to_rl_time:.2f} ms")
            self.node.get_logger().info(f"rl_inference_time: {rl_inference_time:.2f} ms")
            self.node.get_logger().info(f"action_to_cmd_time: {action_to_cmd_time:.2f} ms")
            self.node.get_logger().info(f"cmd_processing_time: {cmd_processing_time:.2f} ms")
            self.node.get_logger().info(f"total_latency: {total_latency:.2f} ms")

if __name__ == "__main__":
    rclpy.init(args=None)
    node = rclpy.create_node('delay_vis_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    vis_delay = VisDelay(node)

    time.sleep(1)

    total_count = 0

    try:
        while rclpy.ok():
            time.sleep(0.01)
            if total_count % 10000 == 0:
                vis_delay.publish_delay(print_delay=True)
            else:
                vis_delay.publish_delay()
            total_count += 1
    except KeyboardInterrupt:
        pass