import numpy as np




from rl_policy.base_policy import RLPolicy


def quat_rotate_inverse_numpy(q, v):
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

    return a - b + c


class LocomotionPolicy(RLPolicy):
    def __init__(self, 
                 config, 
                 model_path):
        super().__init__(config, 
                         model_path)
        
        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        # self.gen_comm_profile()
        


   
        



    
        
        



    def prepare_obs_for_rl(self, robot_state_data, command):
        # robot_state [:2]: timestamps
        # robot_state [2:5]: robot base pos
        # robot_state [5:9]: robot base orientation
        # robot_state [9:9+dof_num]: joint angles 
        # robot_state [9+dof_num: 9+dof_num+3]: base linear velocity
        # robot_state [9+dof_num+3: 9+dof_num+6]: base angular velocity
        # robot_state [9+dof_num+6: 9+dof_num+6+dof_num]: joint velocities

        # RL observation preparation

        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]


        dof_pos_minus_default = dof_pos - self.default_dof_angles

        v = np.array([[0, 0, -1]])

        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)

        

       

        
        self.lin_vel_command[0][0] = command[0]
        self.lin_vel_command[0][1] = command[1]
        self.ang_vel_command[0][0] = command[2]
        obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.lin_vel_command, 
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    projected_gravity
                                    ], axis=1)

        
        
        return obs.astype(np.float32)

    
        
            

        


        

    

    



   


