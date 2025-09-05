import time
import sys
import os
import numpy as np
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, 'lib', 'python', 'amd64')
sys.path.append(lib_path)

import robot_interface as sdk
# import go1_interface 
#define a class for high level cmd sender
HIGHLEVEL = 0xee
LOWLEVEL  = 0xff
class Go1SDKInterface:

    def __init__(self,level=0xee):
        
        if level == HIGHLEVEL:
            self.udp = sdk.UDP(level, 8080, "192.168.123.161", 8082)
            self.cmd = sdk.HighCmd()
            self.state = sdk.HighState()
            self.udp.InitCmdData(self.cmd)
        elif level == LOWLEVEL:
            udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.safe = sdk.Safety(sdk.LeggedType.Go1)
            self.cmd = sdk.LowCmd()
            self.state = sdk.LowState()
            self.udp.InitCmdData(self.cmd)
        
    def _get_robot_state(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        return self.state
    
    def _send_robot_cmd(self):
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def send_velocity_cmd(self,se2_vel:np.ndarray):
        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.speedLevel = 0
        self.cmd.velocity[0] = se2_vel[0]
        self.cmd.velocity[1] = se2_vel[1]
        self.cmd.yawSpeed = se2_vel[2]
        self.cmd.footRaiseHeight = 0.1
        self.cmd.bodyHeight = 0
        self.cmd.euler = [0, 0, 0]
        self.cmd.reserve = 0
        self._send_robot_cmd()


if __name__ == '__main__':

    

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)
        
        # print(motiontime)
        # print(state.imu.rpy[0])
        # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        # print(state.imu.rpy[0])

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        # cmd.mode = 2
        # cmd.gaitType = 1
        # # cmd.position = [1, 0]
        # # cmd.position[0] = 2
        # cmd.velocity = [-0.2, 0] # -1  ~ +1
        # cmd.yawSpeed = 0
        # cmd.bodyHeight = 0.1

        if(motiontime > 0 and motiontime < 1000):
            cmd.mode = 1
            cmd.euler = [-0.3, 0, 0]
        
        if(motiontime > 1000 and motiontime < 2000):
            cmd.mode = 1
            cmd.euler = [0.3, 0, 0]
        
        if(motiontime > 2000 and motiontime < 3000):
            cmd.mode = 1
            cmd.euler = [0, -0.2, 0]
        
        if(motiontime > 3000 and motiontime < 4000):
            cmd.mode = 1
            cmd.euler = [0, 0.2, 0]
        
        if(motiontime > 4000 and motiontime < 5000):
            cmd.mode = 1
            cmd.euler = [0, 0, -0.2]
        
        if(motiontime > 5000 and motiontime < 6000):
            cmd.mode = 1
            cmd.euler = [0.2, 0, 0]
        
        if(motiontime > 6000 and motiontime < 7000):
            cmd.mode = 1
            cmd.bodyHeight = -0.2
        
        if(motiontime > 7000 and motiontime < 8000):
            cmd.mode = 1
            cmd.bodyHeight = 0.1
        
        if(motiontime > 8000 and motiontime < 9000):
            cmd.mode = 1
            cmd.bodyHeight = 0.0
        
        if(motiontime > 9000 and motiontime < 11000):
            cmd.mode = 5
        
        if(motiontime > 11000 and motiontime < 13000):
            cmd.mode = 6
        
        if(motiontime > 13000 and motiontime < 14000):
            cmd.mode = 0
        
        if(motiontime > 14000 and motiontime < 18000):
            cmd.mode = 2
            cmd.gaitType = 2
            cmd.velocity = [0.4, 0] # -1  ~ +1
            cmd.yawSpeed = 2
            cmd.footRaiseHeight = 0.1
            # printf("walk\n")
        
        if(motiontime > 18000 and motiontime < 20000):
            cmd.mode = 0
            cmd.velocity = [0, 0]
        
        if(motiontime > 20000 and motiontime < 24000):
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0.2, 0] # -1  ~ +1
            cmd.bodyHeight = 0.1
            # printf("walk\n")
            

        udp.SetSend(cmd)
        udp.Send()


    