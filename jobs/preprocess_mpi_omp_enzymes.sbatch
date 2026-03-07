#!/bin/bash
#SBATCH --job-name=mrgnn_pre_enz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=aidin.attar@phd.unipd.it
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1

export SINGULARITY_TMPDIR=$HOME/singularity_tmp
export SINGULARITY_CACHEDIR=$HOME/singularity_cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" logs

srun -n "$SLURM_NTASKS" singularity exec --bind "$PWD":/workspace /scratch/$USER/images/mrgnn_gpu_openmp_mpi.sif \
  python /workspace/Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
  --dataset-root /workspace/Dataset/ENZYMES \
  --dataset-name ENZYMES \
  --output-root /workspace/Dataset/Reservoir_OMP_Cache \
  --n-units 50 \
  --n-classes 6 \
  --max-k 4 \
  --runs 0 \
  --use-node-attr \
  --omp-threads $SLURM_CPUS_PER_TASK
