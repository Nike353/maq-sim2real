"""
This script listens to the Vicon data and publishes the object poses via ZeroMQ.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List
from threading import Thread

import zmq
from pyvicon_datastream import tools


class ViconZMQPublisher:
    def __init__(
        self,
        vicon_object_names: List[str],
        publish_names: List[str],
        frequency: int = 200,
        vicon_tracker_ip: str = "128.2.184.3",
        zmq_pub_port: int = 6000,
    ):
        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = vicon_tracker_ip
        self.freq = frequency
        self.vicon_object_names = vicon_object_names
        self.publish_names = publish_names

        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            raise RuntimeError(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize ZMQ publisher
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{zmq_pub_port}")
        print(f"Publishing mocap data on tcp://*:{zmq_pub_port}")

        # Frequency counter
        self.freq_counter = 0

        # Start publishing thread
        self.publish_rate = self.freq
        self.state_thread = Thread(target=self.state_publisher_thread, daemon=True)
        self.state_thread.start()

    def get_vicon_data(self, vicon_object_name):
        position = self.tracker.get_position(vicon_object_name)
        if not position:
            return None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            pos = np.array([x, y, z]) / 1000.0  # Convert to meters
            quat = R.from_euler("XYZ", [roll, pitch, yaw], degrees=False).as_quat()
            return {
                "timestamp": current_time,
                "position": pos.tolist(),
                "orientation": quat.tolist(),  # [x, y, z, w]
            }
        except Exception as e:
            print(f"Error retrieving Vicon data for {vicon_object_name}: {e}")
            return None

    def log_frequency(self):
        print(f"Vicon data publishing frequency: {self.freq_counter} Hz")
        self.freq_counter = 0

    def state_publisher_thread(self):
        print("Starting Vicon → ZMQ publisher thread")
        last_log_time = time.time()

        while True:
            try:
                for vicon_object_name, publish_name in zip(
                    self.vicon_object_names, self.publish_names
                ):
                    data = self.get_vicon_data(vicon_object_name)
                    if data is None:
                        continue

                    # Wrap with topic
                    message = {
                        "name": publish_name,
                        "pose": data,
                    }
                    self.socket.send_pyobj(message)

                self.freq_counter += 1

                now = time.time()
                if now - last_log_time >= 1.0:
                    self.log_frequency()
                    last_log_time = now

                time.sleep(1.0 / self.publish_rate)

            except Exception as e:
                print(f"Error in publisher loop: {str(e)}")
                time.sleep(0.1)

    def main_loop(self):
        print("Running main loop… Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting ViconZMQPublisher…")


if __name__ == "__main__":
    publish_names = ["go1_base","go2_base"]
    object_names = ["go1_base","go2_base"]

    vicon_pub = ViconZMQPublisher(
        vicon_object_names=object_names,
        publish_names=publish_names,
        frequency=200,
        zmq_pub_port=6000,
    )
    vicon_pub.main_loop()
