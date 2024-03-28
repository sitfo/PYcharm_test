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

nvidia-smi

# Path to requirements.txt
REQUIREMENTS_FILE="../requirements.txt"

# Read each line (package) in requirements.txt
while IFS= read -r package; do
    # Check if the package is installed
    if ! pip show "$package" &>/dev/null; then
        # Install the package silently
        pip install --user "$package" &>/dev/null
    fi
done < "$REQUIREMENTS_FILE"

torchrun --nnodes 1 --nproc_per_node 2 fine-tune.py
python storytelling.py