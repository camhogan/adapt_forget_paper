#!/bin/bash

#SBATCH --partition=default_partition
#SBATCH --time=0-24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=avx2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjh337@cornell.edu
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --get-user-env
#SBATCH --job-name=acc_avg

set -euo pipefail

mkdir -p logs

# Initialize conda in non-interactive shell and activate your env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adapt-forget

# Run from repository root (directory from which sbatch is called).
cd "${SLURM_SUBMIT_DIR}"

python -u experiment/accuracy/acc_avg.py
