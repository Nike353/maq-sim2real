import sys
sys.path.append(".././")
from sim2real.utils.robot_interface.base_interface import BaseInterface
from sim2real.go1_sdk.lib.python.amd64 import robot_interface as sdk
HIGHLEVEL = 0xee
LOWLEVEL  = 0xff

class Go1Interface(BaseInterface):
    def __init__(self, config):
        super().__init__(config)
        self.pose = None
        self.name = "go1_base"


    def _init_sdk_components(self):
        self.level = self.config.get("LEVEL", "HIGHLEVEL")
        
        if self.level == "HIGHLEVEL":
            self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
            self.cmd = sdk.HighCmd()
            self.state = sdk.HighState()
            self.udp.InitCmdData(self.cmd)
            print("Go1Interface: HIGHLEVEL")
        elif self.level == "LOWLEVEL":   
            self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.safe = sdk.Safety(sdk.LeggedType.Go1)
            self.cmd = sdk.LowCmd()
            self.state = sdk.LowState()
            self.udp.InitCmdData(self.cmd)
            print("Go1Interface: LOWLEVEL")
    
    

    def _get_robot_state(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        return self.state
    
    def _send_cmd_to_robot(self):
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def get_state(self):
        return self._get_robot_state()
    
    def send_high_level_cmd(self,se2_vel):
        # print(f"send_high_level_cmd: {se2_vel}")
        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.speedLevel = 0
        self.cmd.velocity = [se2_vel[0], se2_vel[1]]
        self.cmd.yawSpeed = se2_vel[2]
        self.cmd.footRaiseHeight = 0.1
        self.cmd.bodyHeight = 0
        self.cmd.euler = [0, 0, 0]
        self.cmd.reserve = 0
        self._send_cmd_to_robot()
    
    

