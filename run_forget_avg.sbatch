#!/bin/bash

#SBATCH --partition=default_partition
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=avx2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjh337@cornell.edu
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --get-user-env
#SBATCH --job-name=forget_avg

set -euo pipefail

mkdir -p logs

# Initialize conda in non-interactive shell and activate your env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adapt-forget-clean-clean

# Run from repository root (directory from which sbatch is called).
cd "${SLURM_SUBMIT_DIR}"

export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false"
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
python -u experiment/forget/forget_avg.py
