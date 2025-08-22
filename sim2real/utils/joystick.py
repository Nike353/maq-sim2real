import pygame
import numpy as np

class JoystickController:
    def __init__(self):
        pygame.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected. Please plug in a USB joystick.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"🎮 Joystick connected: {self.joystick.get_name()}")

        # Default command of size 15 (e.g., lin_vel x/y, yaw rate, etc.)
        self.default_command = np.array([[0.0,
                                         0.0,
                                         0.0,
                                         0.0,
                                         2.5,
                                         0.5, ##trot
                                         0.0,
                                         0.0,
                                         0.5,
                                         0.08,
                                         0.0,
                                         0.0,
                                         0.25,
                                         0.40,
                                         0.0
                                         ]])
        self.commands = self.default_command.copy()
        self.gaits = [[0.5,0.0,0.0],[0.0,0.0,0.0],[0.0,0.5,0.0]]


    def update_commands(self):
        pygame.event.pump()

        # Example joystick mapping:
        axis_0 = self.joystick.get_axis(0)  # Left stick horizontal → lin_vel_y
        axis_1 = self.joystick.get_axis(1)  # Left stick vertical → lin_vel_x
        axis_2 = self.joystick.get_axis(3)  # Right stick horizontal → yaw
        button_a = self.joystick.get_button(0)  # A button → trot
        button_b = self.joystick.get_button(1)  # B button → jump
        button_x = self.joystick.get_button(2)  # X button → bound

        # Update commands vector
        self.commands[0, 0] = -axis_1  # command_lin_vel_x
        self.commands[0, 1] = axis_0   # command_lin_vel_y
        self.commands[0, 2] = axis_2   # command_ang_vel_yaw

        

        # Example: press A to zero everything
        if button_a and not button_b and not button_x:
            self.commands[:,5:8] = self.gaits[0]
        elif button_b and not button_a and not button_x:
            self.commands[:,5:8] = self.gaits[1]
        elif button_x and not button_a and not button_b:
            self.commands[:,5:8] = self.gaits[2]

        return self.commands
