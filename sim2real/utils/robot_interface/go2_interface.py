from sim2real.utils.robot_interface.base_interface import BaseInterface
from sim2real.rl_policy.go2_locomotion import LocomotionPolicy
from numpy import np
class Go2Interface(BaseInterface):
    def __init__(self, config):
        super().__init__(config)
        self.locomotion_policy = LocomotionPolicy(config, config.get("MODEL_PATH"))

    def _init_sdk_components(self):
        from unitree_sdk2py.core.channel import ChannelPublisher,ChannelSubscriber,ChannelFactoryInitialize
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ 
        ChannelFactoryInitialize(0,"en0")
        self.low_cmd = unitree_go_msg_dds__LowCmd_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.robot_lowstate_subscriber.Init(self.LowStateHandler, 1)
        self.InitUnitreeLowCmd()
        self.low_state = None
        self.crc = CRC()

    

    def InitUnitreeLowCmd(self):
        """Initialize Unitree low-level command."""
        
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        for i in range(self.robot.NUM_MOTORS):
            
            self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            
    

    def send_low_level_cmd(self, se2_vel):
        """Send command to Unitree robot."""

        rl_qtarget = self.locomotion_policy.rl_inference(self.get_state(), se2_vel)
        cmd_q = rl_qtarget[0:self.num_dof]
        cmd_dq = 0.0 * np.ones(self.num_dof)
        cmd_tau = 0.0 * np.ones(self.num_dof)
        self._fill_motor_commands(self.low_cmd.motor_cmd, cmd_q, cmd_dq, cmd_tau)
        
        # Add CRC and send
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd) 

    def LowStateHandler(self, msg):
        self.robot_low_state = msg
    
    def _prepare_low_state(self):
        imu_state = self.robot_low_state.imu_state
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion
        self.dq[3:6] = imu_state.gyroscope
        unitree_joint_state = self.robot_low_state.motor_state
        
        
        for i in range(self.num_dof):
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq

            error_code = unitree_joint_state[self.robot.JOINT2MOTOR[i]].reserve[0]
            if error_code != 0:
                print(f"joint {i} error code: {error_code}")
                self.q[7+i] = self.robot.DEFAULT_MOTOR_ANGLES[i]
                self.dq[6+i] = 0.0

    def get_state(self):
        self._prepare_low_state()
        return np.array(self.q.tolist()+self.dq.tolist(), dtype=np.float64).reshape(1, -1)