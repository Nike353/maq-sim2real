user=${1:-yey}
branch=${2:-main}
env_var=${3}
python_cmd=${@:4}


if [ ! -v LOCAL_RANK ]; then
    export LOCAL_RANK=$SLURM_LOCALID
fi

export NUM_PROCESSES=$((SLURM_NNODES * SUBMIT_GPUS))

echo "slurm_job_id: $SLURM_JOB_ID"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "env_var: $env_var"
echo "python_cmd: $python_cmd"
echo "user: $user"
echo "branch: $branch"
echo "SUBMIT_GPUS: $SUBMIT_GPUS"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "RANK: $RANK"
echo "NODE_RANK: $NODE_RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SUBMIT_SAVE_ROOT: $SUBMIT_SAVE_ROOT"

source scripts/slurm_init.sh $user

echo "==========="
echo "pwd:"
pwd
echo "cmd:"
echo "$python_cmd"

if [ -n "$env_var" ]; then
    export $env_var
fi

set -x
accelerate launch --num_processes $NUM_PROCESSES --machine_rank $NODE_RANK --num_machines $SLURM_NNODES --rdzv_backend static --main_process_port $MASTER_PORT --main_process_ip $MASTER_ADDR $python_cmd
set +x