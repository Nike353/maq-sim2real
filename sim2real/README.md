# FALCON Deployment Guide

This folder provides seamless sim2sim/sim2real deployment scripts for both Unitree G1 (H1, H1-2) and Booster T1. The key idea is to simulate the real robot's state publisher and command receiver in Mujoco:

<table>
  <tr>
    <td style="text-align: center;">
      <img src="../assets/deploy.png" style="width: 99%;"/>
    </td>
  </tr>
</table>

## Table of Contents

- [Pre-Configuration](#pre-configuration)
- [Installation](#installation)
- [Deployment](#deployment)
  - [G1 29DoF Locomotion](#g1-29dof-locomotion)
  - [G1 29DoF FALCON](#g1-29dof-falcon)
  - [T1 29DoF FALCON](#t1-29dof-falcon)
- [Sim2Real Tips](#sim2real-tips)

## Pre-Configuration
Here, we use `config/g1/g1_29dof.yaml`. Before testing sim2sim/sim2real, check the `ROBOT_TYPE`, `SDK_TYPE`, and `INTERFACE` in the `.yaml` file:
```yaml
ROBOT_TYPE: 'g1_29dof' # Robot name, "t1_29dof", "g1_29dof"...

ROBOT_SCENE: "../humanoidverse/data/robots/g1/scene_29dof_freebase.xml" # Robot scene, for Sim2Sim

ASSET_ROOT: "../humanoidverse/data/robots/g1" # Robot Asset Root

DOMAIN_ID: 0 # Domain id
# IP Interface 
# For sim2sim, 'lo' is for linux and 'lo0' is for mac.
# For sim2real, this needs to be specified according to your host ip, e.g., 'en0'
INTERFACE: "lo"

SDK_TYPE: "unitree" # SDK type, "unitree", "booster"
MOTOR_TYPE: "serial" # Motor type, "serial" or "parallel"

USE_JOYSTICK: 0 # Simulate Unitree WirelessController using a gamepad (0: disable, 1: enable)
JOYSTICK_TYPE: "xbox" # support "xbox" and "switch" gamepad layout; Unitree WirelessController is "xbox" layout.
```

# Installation
## Prebuild environment
* OS  (Ubuntu 22.04 LTS)  
* CPU  (aarch64 and x86_64)   
* Compiler  (gcc version 11.4.0) 

## Create a conda env
```
conda create -n fcreal python=3.10
conda activate fcreal
```
## Install Pinocchio for Inverse Kinematics
```
conda install pinocchio=3.2.0 -c conda-forge
```
## Install unitree_sdk2_python for Unitree G1 deployment
```
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```
## Install booster_robotics_sdk for Booster T1 deployment
Note that the official [booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) does not provide state publisher and command receiver, so I improve the repo a bit and add them in my [forked repo](https://github.com/hang0610/booster_robotics_sdk). Also booster sdk is NOT supported on Mac OS yet.
```bash
git clone https://github.com/hang0610/booster_robotics_sdk
# Install python package for building python binding locally
pip3 install pybind11
pip3 install pybind11-stubgen
# Build & Install
mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDING=on
make
sudo make install
```
## Install others
```bash
cd sim2real
pip install -r requirements.txt
```

# Deployment
> [!IMPORTANT]
> For sim2sim, you need to start Mujoco and then launch the policy, but for sim2real, you **only** need to launch the policy.
> Make sure you read the keyboard and joystick control protocol in `sim2real/rl_policy/base_policy.py`.
> All the deployment scripts are running under `sim2real`, so do `cd sim2real` first.

<details>
<summary>TEST with G1_29DoF Locomotion</summary>

## G1 29DoF Locomotion
  
Here, we fix the upper body target joint angles to the default, and the policy only outputs the lower body action.
### 1. Start Mujoco Env (ONLY for Sim2Sim)

```bash
python sim_env/base_sim.py \
--config=config/g1/g1_29dof.yaml
```

### 2. Launch the Policy

```bash
python rl_policy/dec_loco/dec_loco.py \
--config=config/g1/g1_29dof.yaml \
--model_path=models/dec_loco/g1_29dof.onnx 
```

https://github.com/user-attachments/assets/dc2d8821-6361-49a8-93cd-fb443bd63c39

</details>

## G1 29DoF FALCON

### 1. Start Mujoco Env (ONLY for Sim2Sim)

```bash
python sim_env/loco_manip.py \
--config=config/g1/g1_29dof_falcon.yaml
```

### 2. Launch the Policy

```bash
python rl_policy/loco_manip/loco_manip.py \
--config=config/g1/g1_29dof_falcon.yaml \
--model_path=models/falcon/g1_29dof.onnx 
```

https://github.com/user-attachments/assets/273b52c1-0248-40e5-b218-e078e74b322d

## T1 29DoF FALCON
### 1. Start Mujoco Env (ONLY for Sim2Sim)
```bash
python sim_env/loco_manip.py \
--config=config/t1/t1_29dof_falcon.yaml 
```
### 2. Luanch the Policy
```bash
python rl_policy/loco_manip/loco_manip.py \
--config=config/t1/t1_29dof_falcon.yaml \
--model_path=models/falcon/t1_29dof.onnx
```

https://github.com/user-attachments/assets/e35ff90e-428b-41ea-8cac-64d9906c78e8

## Sim2Real Tips
> [!CAUTION]
> **FALCON is a strong policy trained for robust locomotion and manipulation.** Before deploying to real robots, ensure:

### Network Configuration
- Set correct `INTERFACE` in config file (e.g., 'en0', 'eth0')
- Verify network connectivity between computer and robot
- Check firewall settings if using specific ports

### Testing Protocol
1. Always do sim2sim before real-robot deployment
2. Start with small kp, kd gains
3. Ensure robot feet touch the ground before running falcon policies

### Emergency Control
- **Keyboard**: Press 'o' to stop policy actions
- **Joystick**: Press 'B+Y' to stop policy actions

# Acknowledgement
We thank the following open-sourced repos that we build upon:
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [booster_robotics_python](https://github.com/BoosterRobotics/booster_robotics_sdk)
