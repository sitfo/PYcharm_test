#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --qos=bbgpu
#SBATCH --mail-type=ALL

set -e

module purge; module load bluebear
module load bear-apps/2022a
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

pip install --user -r requirements.txt

python ./hugging_face_files/model_train_longformer.py