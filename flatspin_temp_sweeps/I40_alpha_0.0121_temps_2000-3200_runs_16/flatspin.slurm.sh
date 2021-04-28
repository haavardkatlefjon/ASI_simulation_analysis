#!/bin/bash
#SBATCH --job-name=flatspin
#SBATCH --partition=GPUQ
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=I40_alpha_0.0121_temps_2000-3200_runs_16/flatspin.slurm.sh.slurm-%j.log
#SBATCH --account=share-ie-idi

set -e

# module load GCC/10.2.0 CUDA/11.1.1-GCC-10.2.0
# module load Python/3.8.6

set -x

env

# SLURM will set CUDA_VISIBLE_DEVICES for us which automatically selects the allocated GPU
# OpenCL will always see the allocated GPU as device 0
flatspin-run -r worker -o I40_alpha_0.0121_temps_2000-3200_runs_16 --worker-id ${SLURM_ARRAY_TASK_ID} --num-workers $((SLURM_ARRAY_TASK_MAX+1))

