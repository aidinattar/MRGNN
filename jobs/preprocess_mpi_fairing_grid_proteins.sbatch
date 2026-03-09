#!/bin/bash
#SBATCH --job-name=mrgnn_fair_pre
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=aidin.attar@phd.unipd.it
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1

export SINGULARITY_TMPDIR="$HOME/singularity_tmp"
export SINGULARITY_CACHEDIR="$HOME/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

SIF_PATH="/scratch/$USER/images/mrgnn_gpu_openmp_mpi.sif"

singularity exec --bind "$PWD":/workspace "$SIF_PATH" \
  python /workspace/experiments/run_master_fairing_cache_pipeline.py \
    --phase preprocess \
    --preprocess-script Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
    --dataset-root /workspace/Dataset/PROTEINS \
    --dataset-name PROTEINS \
    --output-root /workspace/Dataset/Reservoir_MPI_Cache \
    --n-classes 2 \
    --runs 0 1 2 3 \
    --adjacency-matrices D A L \
    --max-k-list 3 4 5 \
    --n-units-list 25 50 \
    --mpi-launcher "mpirun" \
    --mpi-size-flag -np \
    --mpi-np "$SLURM_NTASKS" \
    --log-every 100
