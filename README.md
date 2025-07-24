# MAQ_sim2real
Repository for sim2real for multi-agent quadrupeds 
This is adaptated based on humanoidverse




# Installation

## IsaacGym Conda Env

Create mamba/conda environment, in the following we use conda for example, but you can use mamba as well.

```bash
conda create -n hvgym python=3.8
conda activate hvgym
```
### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
    ```

### Install HumanoidVerse

Install dependencies:
```bash
conda install pinocchio -c conda-forge
pip install -e .
pip install -e isaac_utils

brew install swig
```

Test with:
```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=1 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco \
headless=False
```

Training basic locomotion for t1:
```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion_t1 \         
+domain_rand=NO_domain_rand \
+rewards=loco/reward_t1_locomotion_isaaclab \
+robot=t1/t1_12dof \
+terrain=terrain_locomotion_plane \
+obs=loco/legged_loco_t1 \         
num_envs=4096 \
project_name=T1_locomotion \         
experiment_name=T1_loco_vanilla_v1 \         
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.0003 \
rewards.reward_scales.penalty_dof_pos_l1=-0.00 \
rewards.reward_scales.base_height=-10.0 \
rewards.reward_scales.feet_air_time_single_stance=3.0 \
rewards.reward_scales.penalty_orientation=-0.75 \
+opt=wandb

```

