import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd

sns.set_theme(style="whitegrid")

# 1. Find all IsaacGym log files
isaac_files = glob.glob("./logs/go2_locomotion/**/log_error/IsaacGym.npz", recursive=True)

policy_names = []
isaac_errors = []

for isaac_path in isaac_files:
    try:
        # 2. Extract policy name from path
        path_parts = isaac_path.split('/')
        timestamp_dir = path_parts[-3]

        tokens = timestamp_dir.split('-')
        if len(tokens) >= 3:
            policy_name = f"{tokens[1]}-{tokens[2]}"
            policy_name = policy_name.replace("T1_loco_", "")
        else:
            policy_name = "unknown"

        # 3. Load IsaacGym error array
        data = np.load(isaac_path)
        if "tracking_errors" in data:
            errors = data["tracking_errors"]
        elif "mean_tracking_error" in data:
            errors = [data["mean_tracking_error"]]  # will show as a dot
        else:
            print(f"⚠️ No tracking error in {isaac_path}, skipping...")
            continue

        policy_names.extend([policy_name] * len(errors))
        isaac_errors.extend(errors)

    except Exception as e:
        print(f"❌ Failed to process {isaac_path}: {e}")

# 4. Plot
df = pd.DataFrame({
    "Policy": policy_names,
    "Isaac Tracking Error": isaac_errors
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Policy", y="Isaac Tracking Error", inner='box', scale='width', palette='Set2')

plt.ylabel("Tracking Error (IsaacGym)")
plt.xlabel("Policy")
plt.title("Tracking Error Distribution in IsaacGym")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
