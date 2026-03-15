#!/bin/bash
#SBATCH --job-name=mrgnn_fair_pre_enz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

SIF_PATH="${SIF_PATH:-/scratch/$USER/images/mrgnn_gpu_openmp_mpi.sif}"

singularity exec --bind "$PWD":/workspace "$SIF_PATH" \
  python /workspace/experiments/run_master_fairing_cache_pipeline.py \
    --phase preprocess \
    --preprocess-script Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
    --dataset-root /workspace/Dataset/ENZYMES \
    --dataset-name ENZYMES \
    --output-root /workspace/Dataset/Reservoir_MPI_Cache \
    --n-classes 6 \
    --use-node-attr \
    --runs 0 \
    --adjacency-matrices D A L \
    --max-k-list 2 6 10 \
    --n-units-list 50 200 \
    --mpi-launcher "mpirun" \
    --mpi-size-flag=-np \
    --mpi-np "$SLURM_NTASKS" \
    --log-every 100
