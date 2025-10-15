"""
Simple script to load a TPG, compute transformed initial poses, and sequentially move each robot from
a random start to its transformed initial pose using a PD controller.

- Subscribes to mocap on tcp://127.0.0.1:6000 (to read current poses)
- Publishes velocity commands to tcp://127.0.0.1:6001 as a dict {"agent_name": [vx, vy, yaw_rate]}
- Moves robots one-by-one to their targets

This script is intentionally minimal and uses the same transform and wrap utilities
as in `tpg_ext_main.py`.
"""

import time
import numpy as np
import zmq
import os
import argparse
import sys
sys.path.append(".././")
import threading
# reuse the transform logic
from sim2real.tpg.tpg_ext_main import transform_pose_to_new_frame, wrap_to_pi
from sim2real.tpg.tpg_general import TimedTPGManager


def quat_to_euler_angles(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    Convention: ZYX (yaw-pitch-roll).
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90° if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def get_yaw(orientation):
    # print(orientation,"orientation")
    yaw = quat_to_euler_angles(orientation)[0]
    # if yaw>np.pi/2:
    #     return yaw-2*np.pi
    # else:
    #     return yaw
    if yaw<-0.02:
        return yaw
    else:
        return yaw

def load_tpg_and_targets(tpg_file, cell_size=1.0):
    manager = TimedTPGManager()
    manager.load_tpg(tpg_file)
    num_agents = manager.num_agents
    targets = []
    for i in range(num_agents):
        sol = manager.list_of_solutions[i]
        if hasattr(sol, 'xythetas') and sol.xythetas.shape[0] > 0:
            x0, y0, yaw0 = sol.xythetas[0].tolist()
            x0 *= cell_size
            y0 *= cell_size
            tx, ty, tyaw = transform_pose_to_new_frame(x0, y0, yaw0)
            targets.append((tx, ty, tyaw))
        else:
            targets.append((np.nan, np.nan, np.nan))
    return manager, targets


class MocapListener:
    def __init__(self, sub_ip='127.0.0.1', sub_port=6000):
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{sub_ip}:{sub_port}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._latest = None
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def _listener(self):
        # blocking recv loop running in background thread
        while self._running:
            try:
                msg = self.sub.recv_pyobj()
                with self._lock:
                    self._latest = msg
            except Exception as e:
                # on error, sleep briefly and continue
                print(f"Mocap listener error: {e}")
                time.sleep(0.01)

    def start(self):
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._listener, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.1)
            self._thread = None

    def poll(self, timeout=100):
        """Return the most recent mocap message (thread-safe)."""
        with self._lock:
            return self._latest


class VelocityPublisher:
    def __init__(self, pub_ip='127.0.0.1', pub_port=6001):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://{pub_ip}:{pub_port}")
        # small sleep to let subscribers connect
        time.sleep(0.1)

    def send(self, name, vel):
        # send as a dict with agent name
        self.pub.send_pyobj({name: vel})

    def send_dict(self, cmd_dict):
        """Send a full dictionary mapping agent_name -> [vx,vy,yaw_rate]."""
        self.pub.send_pyobj(cmd_dict)


def se2_error(target_xyyaw, current_pos_quat):
    # current_pos_quat: [position, orientation_xyzw]
    
    pos = current_pos_quat[0]
    quat = current_pos_quat[1]
    # position is likely [x,y,z]
    cx, cy = pos[0], pos[1]
    # derive yaw from quaternion (xyzw)
    cyaw = get_yaw(quat)

    tx, ty, tyaw = target_xyyaw
    # compute world-frame error
    dx_w = tx - cx
    dy_w = ty - cy
    # rotate world-frame error into robot body frame (forward=x, lateral=y)
    # body_error = R_body^T * world_error where R_body rotates body->world by cyaw
    cos_y = np.cos(cyaw)
    sin_y = np.sin(cyaw)
    # inverse rotation (world -> body): [cos, sin; -sin, cos] * [dx_w; dy_w]
    dx_body =  cos_y * dx_w + sin_y * dy_w
    dy_body = -sin_y * dx_w + cos_y * dy_w
    yaw_err = wrap_to_pi(tyaw - cyaw)
    return dx_body, dy_body, yaw_err, (cx, cy, cyaw)


def pd_velocity_from_error(dx, dy, yaw_err, Kp_lin=0.5, Kd_lin=0.1, Kp_yaw=0.5, Kd_yaw=0.1, max_lin=0.6, max_yaw=1.0):
    # simple P controller (no derivative state kept for simplicity)
    vx = np.clip(Kp_lin * dx, -max_lin, max_lin)
    vy = np.clip(Kp_lin * dy, -max_lin, max_lin)
    yaw_rate = np.clip(Kp_yaw * yaw_err, -max_yaw, max_yaw)
    return [float(vx), float(vy), float(yaw_rate)]


def move_agent_to_target(agent_idx, agent_name, target, mocap, pub, all_agent_names, timeout=50.0, pos_tol=0.15, yaw_tol=0.1):
    print(f"Moving {agent_name} to target {target}")
    t0 = time.time()
    last_pub = 0
    while True:
        mocap.poll(50)
        if mocap._latest is None:
            time.sleep(0.01)
            if time.time() - t0 > timeout:
                print(f"Timeout waiting for mocap for {agent_name}")
                return False
            continue
        # pick the key for this agent
        if agent_name not in mocap._latest:
            # wait a bit for correct message
            time.sleep(0.01)
            if time.time() - t0 > timeout:
                print(f"Mocap didn't contain {agent_name} within timeout")
                return False
            continue
        cur = mocap._latest[agent_name]
        pos = cur['position']
        quat = cur['orientation']
        dx, dy, yaw_err, (cx, cy, cyaw) = se2_error(target, [pos,quat])
        dist = np.hypot(dx, dy)
        # control strategy: first reach position (no rotation), then correct yaw in-place
        if dist >= pos_tol:
            # still far from target: move in robot frame, avoid rotating while moving
            # compute linear velocities only, zero yaw error for controller
            lin_vel = pd_velocity_from_error(dx, dy, 0.0)
            # ensure yaw rate is zero while moving
            vel = [lin_vel[0], lin_vel[1], 0.0]
        else:
            # within position tolerance: correct orientation only
            if abs(yaw_err) < yaw_tol:
                # reached target pose; send zeros for all agents
                zeros = {n: [0.0, 0.0, 0.0] for n in all_agent_names}
                pub.send_dict(zeros)
                print(f"{agent_name} reached target: pos ({cx:.3f},{cy:.3f}) yaw {cyaw:.3f}")
                return True
            # rotate in place to correct yaw
            rot = pd_velocity_from_error(0.0, 0.0, yaw_err)
            vel = [0.0, 0.0, rot[2]]
        # publish at ~20Hz
        if time.time() - last_pub > 1.0 / 20.0:
            # construct full command dict (zeros for inactive agents)
            cmd = {n: [0.0, 0.0, 0.0] for n in all_agent_names}
            cmd[agent_name] = vel
            pub.send_dict(cmd)
            print(vel)
            last_pub = time.time()
        # small sleep
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            print(f"Timeout moving {agent_name} to target")
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpg', type=str, default='data/real_world_empty@20.scen_n4_s7_tpg.npz')
    parser.add_argument('--cell_size', type=float, default=1.0)
    parser.add_argument('--agent_prefix', type=str, default='go1')
    args = parser.parse_args()

    tpg_file = os.path.join(os.path.dirname(__file__), args.tpg)
    manager, targets = load_tpg_and_targets(tpg_file, cell_size=args.cell_size)
    num_agents = manager.num_agents

    mocap = MocapListener(sub_port=6000)
    mocap.start()
    pub = VelocityPublisher(pub_port=6001)

    # Agent name convention: provide unique names per agent; default will be agent_prefix + i
    agent_names = [f"{args.agent_prefix}_{i}" for i in range(num_agents)]

    print("Targets:")
    for i, t in enumerate(targets):
        print(i, t)
    
    
    

    # Move robots one-by-one
    for i, name in enumerate(agent_names):
        target = targets[i]
        if np.isnan(target[0]):
            print(f"Skipping agent {i} (no target)")
            continue
        # Wait briefly to let mocap stream stabilize
        time.sleep(0.5)
        ok = move_agent_to_target(i, name, target, mocap, pub, agent_names, timeout=1000.0)
        if not ok:
            print(f"Failed to move {name} to target")
        # wait before next agent
        time.sleep(0.5)

    print("All done")


if __name__ == '__main__':
    main()
