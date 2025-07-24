user=${1:-yey}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cache_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/cache/$user"
export TORCH_HOME=$cache_dir
export HUGGINGFACE_HUB_CACHE=$cache_dir
export XDG_CACHE_HOME=$cache_dir

export runapp=/isaac-sim/runapp.sh
export runheadless=/isaac-sim/runheadless.sh
export ISAACLAB_PATH=/workspace/isaaclab
export isaaclab=/workspace/isaaclab/isaaclab.sh
export python=/workspace/isaaclab/_isaac_sim/python.sh
export python3=/workspace/isaaclab/_isaac_sim/python.sh
export pip='/workspace/isaaclab/_isaac_sim/python.sh -m pip'
export pip3='/workspace/isaaclab/_isaac_sim/python.sh -m pip'
export tensorboard='/workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/_isaac_sim/tensorboard'
export TZ=UTC

export DISPLAY=:1
export PATH=/isaac-sim/kit/python/bin:$PATH

export SCRIPT_DIR="/workspace/isaaclab/_isaac_sim"
export CARB_APP_PATH=$SCRIPT_DIR/kit
export ISAAC_PATH=$SCRIPT_DIR
export EXP_PATH=$SCRIPT_DIR/apps
source ${SCRIPT_DIR}/setup_python_env.sh
export RESOURCE_NAME="IsaacSim"
export LD_PRELOAD=$SCRIPT_DIR/kit/libcarb.so

if [[ -n "$SLURM_PROCID" && "$SLURM_PROCID" -eq 0 ]]; then
    echo "create & link folders since SLURM_PROCID is 0"

    if [ ! -d "$cache_dir" ]; then
        mkdir -p $cache_dir
    fi

    results_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/genmo/humanoid_tracking/results/$user"
    mkdir -p $results_dir
    ln -s $results_dir ./logs
    ln -s /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/genmo/humanoid_tracking/data ./data
fi


if [[ -n "$SLURM_LOCALID" && "$SLURM_LOCALID" -ne 0 ]]; then
    echo "skip installation since SLURM_LOCALID is not 0"
    # Check if the total number of SLURM nodes used is more than 4
    if [ "$SLURM_JOB_NUM_NODES" -gt 4 ]; then
        echo "sleep 60s since SLURM_JOB_NUM_NODES is more than 4"
        sleep 60
    else
        echo "sleep 60s since SLURM_JOB_NUM_NODES is less than 4"
        sleep 60
    fi
else
    echo "run installation since SLURM_LOCALID is 0"

    $pip install trl loguru
    $pip install -e isaac_utils
    $pip install -e .
fi
