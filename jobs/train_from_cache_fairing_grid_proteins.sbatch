#!/bin/bash
#SBATCH --job-name=mrgnn_fair_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=aidin.attar@phd.unipd.it
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

export SINGULARITY_TMPDIR="$HOME/singularity_tmp"
export SINGULARITY_CACHEDIR="$HOME/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

SIF_PATH="/scratch/$USER/images/mrgnn_gpu_openmp_mpi.sif"

singularity exec --nv --bind "$PWD":/workspace "$SIF_PATH" \
  python /workspace/experiments/run_master_fairing_cache_pipeline.py \
    --phase train \
    --dataset-root /workspace/Dataset/PROTEINS \
    --dataset-name PROTEINS \
    --output-root /workspace/Dataset/Reservoir_MPI_Cache \
    --n-classes 2 \
    --runs 0 1 2 3 \
    --adjacency-matrices D A L \
    --max-k-list 3 4 5 \
    --n-units-list 25 50 \
    --lr-list 1e-4 5e-4 \
    --drop-prob-list 0.3 0.5 \
    --weight-decay-list 5e-3 5e-4 \
    --batch-size-list 32 \
    --readout-list funnel \
    --n-epochs 100 \
    --n-folds 10 \
    --test-epoch 1 \
    --log-dir /workspace/test_log/fairing_cache_grid
