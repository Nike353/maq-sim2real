import mujoco
import rclpy
import threading
import mujoco.viewer
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
from loguru import logger

class VisLowState:
    def __init__(self, node):
        self.node = node 
        self.state_sub = self.node.create_subscription(
            Float64MultiArray, "robot_state", self.state_callback, 1
        )
        self.num_dof = 27
        self.state_msg = None
        self.q = np.zeros(3 + 4 + self.num_dof)
        self.dq = np.zeros(3 + 3 + self.num_dof)
    def state_callback(self, msg):
        # print("Received state message")
        # print(msg.data)
        self.q = np.array(msg.data[6: 6 + 3+4+self.num_dof], dtype=np.float64)
        self.dq = np.array(msg.data[6 + 3+4+self.num_dof: 6 + 3+4+self.num_dof+3+3+self.num_dof], dtype=np.float64)
        # print("-----------q-----------")
        # print(self.q)
        # print("-----------dq-----------")
        # print(self.dq)
        self.state_msg = msg

def main(args=None):
    # Load the model
    mj_xml = "roboverse/data/robots/g1/g1_29dof.xml"
    mj_model = mujoco.MjModel.from_xml_path(mj_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Run the simulation

    rclpy.init(args=args)
    node = rclpy.create_node('mujoco_vis_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    
    rate = node.create_rate(50)

    low_state = VisLowState(node)

    logger.info("Starting simulation")

    # time.sleep(1)
    try:
        while rclpy.ok():
            mj_data.qpos[:] = low_state.q
            mj_data.qpos[0:3] = [0, 0, 1.5]
            mj_data.qvel[:] = low_state.dq
            mujoco.mj_forward(mj_model, mj_data)
            mj_viewer.sync()
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == "__main__":
    main()