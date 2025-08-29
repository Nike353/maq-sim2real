"""This script listens to the Vicon data and publishes the object poses via unitreesdk2"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import List

# Import Vicon DataStream SDK
from pyvicon_datastream import tools

# Import unitreesdk2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_ as Pose_go
from threading import Thread

class ViconUnitree:
    def __init__(
        self,
        vicon_object_names: List[str],
        publish_names: List[str],
        frequency: int = 200,
        vicon_tracker_ip: str = "128.2.184.3",
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
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize Unitree publishers
        self.init_publisher()

        # Frequency counter
        self.freq_counter = 0
        
    def init_publisher(self):
        # Initialize unitreesdk2 channel factory
        ChannelFactoryInitialize(0, "lo")
        
        # Initialize Unitree publishers
        self.pose_publishers = {}
        
        for vicon_obj, publish_name in zip(self.vicon_object_names, self.publish_names):
            # Create publisher for each object with unique topic name
            topic_name = f"rt/{publish_name}_state"  # e.g., "rt/base_state", "rt/door_state", etc.
            publisher = ChannelPublisher(topic_name, Pose_go)
            publisher.Init()
            self.pose_publishers[publish_name] = publisher
            print(f"Publishing {publish_name} (vicon: {vicon_obj}) poses on topic {topic_name}")

        # Give time for publishers to initialize
        time.sleep(1)

        # Start state publishing thread
        self.publish_rate = self.freq
        self.state_thread = Thread(target=self.state_publisher_thread, daemon=True)
        self.state_thread.start()
        
    def get_vicon_data(self, vicon_object_name):
        position = self.tracker.get_position(vicon_object_name)
        
        if not position:
            print(f"Cannot get the pose of `{vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            position = np.array([x, y, z]) / 1000. # Convert to meters
            rotation = R.from_euler('XYZ', [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, quaternion
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None

    def log_frequency(self):
        # Log the frequency information
        print(f"Vicon data acquisition frequency: {self.freq_counter} Hz")
        self.freq_counter = 0  # Reset the counter for the next second

    def state_publisher_thread(self):
        print("Starting Vicon state publisher thread")
        
        # Timer for frequency logging
        last_log_time = time.time()
        
        while True:
            try:
                for vicon_object_name, publish_name in zip(self.vicon_object_names, self.publish_names):
                    current_time, position, quaternion = self.get_vicon_data(vicon_object_name)
                    if position is None:
                        print(f"Failed to get Vicon data for {vicon_object_name}.")
                        continue

                    # Create Pose_go message
                    pose_msg = Pose_go()
                    
                    # Set position (x, y, z)
                    pose_msg.position.x = float(position[0])
                    pose_msg.position.y = float(position[1])
                    pose_msg.position.z = float(position[2])
                    
                    # Set quaternion (x, y, z, w format from scipy)
                    pose_msg.orientation.x = float(quaternion[0])
                    pose_msg.orientation.y = float(quaternion[1])
                    pose_msg.orientation.z = float(quaternion[2])
                    pose_msg.orientation.w = float(quaternion[3])
                    
                    # Set timestamp if available in the message
                    # pose_msg.timestamp = current_time  # Uncomment if Pose_go has timestamp field
                    
                    # Publish the message
                    self.pose_publishers[publish_name].Write(pose_msg)
                    
                # Increment frequency counter
                self.freq_counter += 1
                
                # Log frequency every second
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    self.log_frequency()
                    last_log_time = current_time
                
                time.sleep(1.0 / self.publish_rate)
                
            except Exception as e:
                print(f"Error in Vicon state publisher thread: {str(e)}")
                time.sleep(0.1)

    def main_loop(self):
        print("Starting Vicon data acquisition...")
        try:
            while True:
                time.sleep(1)  # Main thread just waits, publishing happens in background thread
        except KeyboardInterrupt:
            print("Exiting Vicon data acquisition.")

if __name__ == "__main__":
    # For base_state specifically, you might want just one object
    publish_names = ["base"]  # This will publish to "rt/base_state"
    object_names = ["haoyang_pelvis"]  # Assuming pelvis represents the base
    
    # Or for multiple objects:
    # publish_names = ["base", "door", "wall"]
    # object_names = ["haoyang_pelvis", "haoyang_Door", "haoyang_Wall"]
    
    vicon_unitree = ViconUnitree(vicon_object_names=object_names, publish_names=publish_names)

    try:
        vicon_unitree.main_loop()
    except KeyboardInterrupt:
        pass