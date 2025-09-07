import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

sns.set_theme(style="whitegrid")

# 1. Get all MuJoCo log files
mujoco_files = glob.glob("./**/log_error/MuJoCo.npz", recursive=True)

policy_names = []
normalized_errors = []
##sort the mujoco_file by the name in alphabetical order
# mujoco_files.sort()
for mujoco_path in mujoco_files:
    try:
        # 2. Extract timestamp directory
        path_parts = mujoco_path.split('/')
        timestamp_dir = path_parts[-3]  # e.g., '20250711_235522-T1_loco_dr_v1-locomotion-T1_locomotion'
        
        # Extract policy name from timestamp_dir
        tokens = timestamp_dir.split('-')
        if len(tokens) >= 3:
            policy_name = f"{tokens[1]}-{tokens[2]}"  # e.g., T1_loco_dr_v1-locomotion
            ##REMOVE T1_loco
            policy_name = policy_name.replace("T1_loco_", "")
        else:
            policy_name = "unknown"

        # 3. Find corresponding Isaac path
        isaac_path = mujoco_path.replace("MuJoCo.npz", "IsaacGym.npz")
        if not os.path.exists(isaac_path):
            print(f"⚠️ Isaac file not found for: {policy_name}, skipping...")
            continue

        # 4. Load data
        mujoco_data = np.load(mujoco_path)
        isaac_data = np.load(isaac_path)

        mujoco_errors = mujoco_data["tracking_errors"]
        isaac_mean = isaac_data["mean_tracking_error"]

        # 5. Normalize each Mujoco error
        # norm_degradation = (mujoco_errors - isaac_mean) / isaac_mean
        norm_degradation = mujoco_errors

        # 6. Append for plotting
        policy_names.extend([policy_name] * len(norm_degradation))
        normalized_errors.extend(norm_degradation)

    except Exception as e:
        print(f"❌ Failed to process {mujoco_path}: {e}")

# 7. Plot
import pandas as pd

df = pd.DataFrame({
    "Policy": policy_names,
    "Normalized Degradation": normalized_errors
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Policy", y="Normalized Degradation", inner='box', scale='width', palette='Set2')

plt.axhline(0, linestyle='--', color='gray', label='No Degradation')
plt.ylabel("Normalized Degradation")
plt.xlabel("Policy")
plt.title("Sim2Sim Tracking Error: Normalized Degradation in MuJoCo")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
# plt.savefig("violin_normalized_degradation.png")
plt.show()
