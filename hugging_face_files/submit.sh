#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --account=leemg-jinlongphd
#SBATCH --time=240:0
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:2

set -e

module purge
module load bluebear
module load bear-apps/2022a
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

pip install --user -r ../requirements.txt

# Set the number of processes based on the number of GPUs
export WORLD_SIZE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
export RANK=$SLURM_PROCID

# Get the master node's address
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

# Set the master port (replace 12345 with your desired port number)
export MASTER_PORT=12345

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

python fine-tune.py
python storytelling.py
