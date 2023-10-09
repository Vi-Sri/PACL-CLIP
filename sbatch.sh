#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=pacl_coco
#SBATCH --output=pacl.out
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=100

nvidia-smi
source ~/.bashrc
cd /home/cap6411.student34/cap6411-pacl

CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate vit

CUDA_VISIBLE_DEVICES=0 python3 train.py
