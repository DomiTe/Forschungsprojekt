#!/bin/bash

#SBATCH --job-name=imagenet100
#SBATCH --output=Forschungsprojekt/results/logs/job_%j.out  # %A = Array ID, %a = Task ID, %j = SLURM JOB ID
#SBATCH --error=Forschungsprojekt/results/logs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04-00:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=normal

source $HOME/.bashrc

# Load the C Compiler for Triton
# module load gnu12/12.4.0

conda activate /scratch/dominic/Forschungsprojekt/.conda

export PATH=/opt/ohpc/pub/compiler/gcc/12.4.0/bin:/bin:/usr/bin:$PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPLUS_INCLUDE_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RUN_ID=$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID

#known good single GPU:
# python Forschungsprojekt/src/main.py name=test arch=bert-base train=bert-base data=sanity-check-2 impl.microbatch_size=2

#known good multi GPU torchrun --nproc_per_node=1 --standalone Forschungsprojekt/src/main.py name=multi_GPU_2_cramming  arch=bert-base train=bert-base data=sanity-check-2
cd /scratch/dominic/Forschungsprojekt
# --deploy-int8-only --load-run-id 20260624_100106_23827
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 --standalone -m src.main --skip-training --load-run-id 20260810_104157_31209