import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import scipy.interpolate as si
import yaml
from scipy.interpolate import interp1d
import argparse
import os
import pickle as pkl



def slerp_yaw(t, yaws, t_new, degrees=False):
    t = np.asarray(t, float).flatten()
    t_new = np.asarray(t_new, float).flatten()
    rots = R.from_euler('z', yaws, degrees=degrees)
    slerp = Slerp(t, rots)
    rots_new = slerp(t_new)
    mats_new = rots_new.as_matrix()
    angular_vel = np.zeros((len(t_new), 3))
    for i in range(len(t_new)-1):
        dt = t_new[i+1] - t_new[i]
        R_delta = mats_new[i].T @ mats_new[i+1]
        rotvec_delta = R.from_matrix(R_delta).as_rotvec()
        angular_vel[i] = rotvec_delta / dt
    angular_vel[-1] = angular_vel[-2]
    xyz_interp = rots_new.as_euler('xyz', degrees=degrees).squeeze()
    yaws_interp = xyz_interp[:, 2]
    yaw_vel = angular_vel[:, 2]
    return yaws_interp, yaw_vel



def interp_xy(t, xys, t_new):
    t = np.asarray(t, float).flatten()
    t_new = np.asarray(t_new, float).flatten()
    xys = np.asarray(xys, float)
    interp = np.zeros((len(t_new), 2))
    for i in range(2):
        interp[:, i] = np.interp(t_new, t, xys[:, i])
    vel = np.diff(interp, axis=0) / np.diff(t_new, axis=0)[:, np.newaxis]
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    return interp, vel


def sine_traj(dt, T, A, W, Phi, B):
    W = W.reshape(-1, 1)
    t = np.arange(0, T, dt).reshape(1, -1)
    Phi = Phi.reshape(-1, 1)
    A = A.reshape(-1, 1)
    B = B.reshape(-1, 1)
    pos = A * np.sin(W * t + Phi) + B
    t = t.flatten()
    return t, pos


def bspline_traj(dt, T, T_sample, start, max_speed, pos_max, pos_min):
    R = T_sample * max_speed
    N = int(T / T_sample)

    last_p = start.reshape(1, -1)
    key_points = [np.copy(last_p)] * 1
    for i in range(N):
        box_max = last_p + np.array([R, R])
        box_min = last_p - np.array([R, R])
        sample_max = np.minimum(box_max, pos_max)
        sample_min = np.maximum(box_min, pos_min)
        sample = np.random.uniform(sample_min, sample_max)
        key_points.append(sample)
        last_p = np.copy(sample)

    key_points.extend([np.copy(last_p)] * 1)
    key_points = np.concatenate(key_points)

    steps = int(T / dt)
    pos = bspline(key_points, n=steps, degree=5, periodic=False)

    t = np.arange(0, T, dt).reshape(1, -1)
    
    return t, pos


def bspline(cv, n=100, degree=3, periodic=False):
    """Calculate n samples on a bspline

    cv :      Array ov control vertices
    n  :      Number of samples to return
    degree:   Curve degree
    periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count + degree + 1)
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count - 1)
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

    # Return samples
    max_param = count - (degree * (1 - periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))

def yaw_as_xy_vel(xy_vel):
    yaw = np.arctan2(xy_vel[:, 1], xy_vel[:, 0])
    return yaw



def traj_gen(config):
    traj_type = config["traj_type"]
    traj_config = config[traj_type]
    dt = 0.02 # 50 Hz
    if traj_type == "xy_bspline":
        T = traj_config["T"]
        T_sample = traj_config["T_sample"]
        start = np.array(traj_config["start"])
        max_speed = np.array(traj_config["max_speed"])
        pos_max = np.array(traj_config["pos_max"])
        pos_min = np.array(traj_config["pos_min"])
        t, traj = bspline_traj(dt, T, T_sample, start, max_speed, pos_max, pos_min)
    
    elif traj_type == "sine":
        T = traj_config["T"]
        A = np.array(traj_config["A"])
        W = np.array(traj_config["W"])
        Phi = np.array(traj_config["Phi"])
        B = np.array(traj_config["B"])
        t, traj = sine_traj(dt, T, A, W, Phi, B)
        traj = traj.T
    else:
        raise ValueError("Unknown traj_type: %s" % traj_type)
    
    wait_time = config["wait_time"]
    t_wait = np.arange(0, wait_time, dt)
    wait_traj = np.zeros((len(t_wait), traj.shape[1])) + traj[0]
    t = np.concatenate([t_wait, t + wait_time])
    traj = np.concatenate([wait_traj, traj], axis=0)


    pos = traj
    xy = pos[:, :2]
    yaw = pos[:, 2]
    
    xy, xy_vel = interp_xy(t, xy, t)
    
    if config["yaw_as_xy_vel"]:
        yaw = yaw_as_xy_vel(xy_vel)
    
    yaw, yaw_vel = slerp_yaw(t, yaw, t, degrees=True)
    
    
    print("Trajectory shape: ", xy.shape, "t shape: ", t.shape, "yaw shape: ", yaw.shape, "yaw_vel shape: ", yaw_vel.shape, "xy_vel shape: ", xy_vel.shape)
    traj = np.concatenate([xy, yaw[:, None], xy_vel, yaw_vel[:, None]], axis=1)
    
    return traj



def vis_traj(traj, traj_vel, folder):
    t = np.arange(len(traj)) * 0.02
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # plot xy yaw on the 2d figure, yaw as the arrow direction
    ax.plot(traj[:, 0], traj[:, 1])
    ax.set_title("XY")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    for i in range(0, len(traj), 50):
        l = 0.05
        ax.arrow(traj[i, 0], traj[i, 1], l*np.cos(traj[i, 2]), l*np.sin(traj[i, 2]), head_width=0.01)
    
    plt.tight_layout()
    fig.savefig(os.path.join(folder, "xy.png"))
    plt.close(fig)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax[i].plot(t, traj[:, i], label="pos")
        ax[i].plot(t, traj_vel[:, i], label="vel")
        ax[i].set_title(f"{['X', 'Y', 'Yaw'][i]}")
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Value")
        ax[i].legend()
        ax[i].grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, "traj.png"))
    plt.close(fig)



def main():
    
    config_path = "traj_gen.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    output_file = config["output"]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    traj = traj_gen(config)
    
    print("Trajectory shape: ", traj.shape)
    
    vis_traj(traj[:, :3], traj[:, 3:], os.path.dirname(output_file))
    
    print("Trajectory saved to ", output_file)
    
    txt_path = output_file.replace(".pkl", ".txt")
    np.savetxt(txt_path, traj, fmt="%.6f", delimiter=",", header="X, Y, X_vel, Y_vel, Yaw, Yaw_vel")
    pkl.dump(traj, open(output_file, "wb"))


if __name__ == "__main__":
    main()