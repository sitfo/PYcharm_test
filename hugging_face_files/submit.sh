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

# Set the necessary environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=2
export RANK=$SLURM_PROCID
# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=all  # Use all available GPUs

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 fine-tune.py
python storytelling.py