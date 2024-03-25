#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=240:0
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:2

set -e

module purge; module load bluebear
module load bear-apps/2022a
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

pip install --user -r requirements.txt&&python /rds/homes/j/jxl1872/PYcharm_test/hugging_face_files/model_train_longform.py