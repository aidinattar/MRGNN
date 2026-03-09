#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch non trovato nel PATH."
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-REDDIT-MULTI-12K}"
DATASET_TAG="${DATASET_NAME//[^A-Za-z0-9_]/_}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/Dataset/${DATASET_NAME}}"
SIF_PATH="${SIF_PATH:-/scratch/$USER/images/mrgnn_gpu_openmp_mpi.sif}"

PARTITION="${PARTITION:-allgroups}"
MAX_K="${MAX_K:-8}"
ADJACENCY_MATRIX="${ADJACENCY_MATRIX:-D}"
NUM_GRAPHS_OMP="${NUM_GRAPHS_OMP:-4096}"
NUM_GRAPHS_MPI="${NUM_GRAPHS_MPI:-4096}"
REPEATS_OMP="${REPEATS_OMP:-3}"
REPEATS_MPI="${REPEATS_MPI:-3}"

# Isolate MPI scaling by default: 1 OMP thread per rank.
MPI_OMP_THREADS="${MPI_OMP_THREADS:-1}"
USE_NODE_ATTR="${USE_NODE_ATTR:-0}"

if [[ "${USE_NODE_ATTR}" == "1" ]]; then
  NODE_ATTR_FLAG="--use-node-attr"
else
  NODE_ATTR_FLAG=""
fi

OMP_THREADS_LIST=(1 2 4 8 16 32)
MPI_RANKS_LIST=(1 2 4 8 16)

cd "${REPO_ROOT}"
mkdir -p logs benchmark_results/hpc

echo "Submitting OpenMP scaling jobs for ${DATASET_NAME}..."
OMP_JOB_IDS=()
for t in "${OMP_THREADS_LIST[@]}"; do
  job_id="$(sbatch --parsable \
    --job-name="bm_omp_${DATASET_TAG}_${ADJACENCY_MATRIX}_t${t}" \
    --output="logs/%x_%j.out" \
    --error="logs/%x_%j.err" \
    --partition="${PARTITION}" \
    --time="01:30:00" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${t}" \
    --mem="24G" \
    --mail-user="aidin.attar@phd.unipd.it" \
    --mail-type="END,FAIL" \
    --wrap="set -euo pipefail; \
      cd '${REPO_ROOT}'; \
      export SINGULARITY_TMPDIR=\$HOME/singularity_tmp; \
      export SINGULARITY_CACHEDIR=\$HOME/singularity_cache; \
      mkdir -p \"\$SINGULARITY_TMPDIR\" \"\$SINGULARITY_CACHEDIR\" benchmark_results/hpc; \
      singularity exec --bind '${REPO_ROOT}':/workspace '${SIF_PATH}' \
        python /workspace/benchmarks/benchmark_omp_preprocess_threads.py \
          --dataset-root '${DATASET_ROOT}' \
          --dataset-name '${DATASET_NAME}' \
          --adjacency-matrix '${ADJACENCY_MATRIX}' \
          --max-k ${MAX_K} \
          --threads ${t} \
          --num-graphs ${NUM_GRAPHS_OMP} \
          --repeats ${REPEATS_OMP} \
          --select largest \
          ${NODE_ATTR_FLAG} \
          --output-json /workspace/benchmark_results/hpc/${DATASET_TAG}_${ADJACENCY_MATRIX}_omp_t${t}.json" \
  )"
  OMP_JOB_IDS+=("${job_id}")
  echo "  thread ${t} -> job ${job_id}"
done

echo
echo "Submitting MPI scaling jobs for ${DATASET_NAME}..."
MPI_JOB_IDS=()
for n in "${MPI_RANKS_LIST[@]}"; do
  mpi_cpus="$(( n * MPI_OMP_THREADS ))"
  job_id="$(sbatch --parsable \
    --job-name="bm_mpi_${DATASET_TAG}_${ADJACENCY_MATRIX}_n${n}" \
    --output="logs/%x_%j.out" \
    --error="logs/%x_%j.err" \
    --partition="${PARTITION}" \
    --time="01:30:00" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${mpi_cpus}" \
    --mem="32G" \
    --mail-user="aidin.attar@phd.unipd.it" \
    --mail-type="END,FAIL" \
    --wrap="set -euo pipefail; \
      cd '${REPO_ROOT}'; \
      export OMP_NUM_THREADS=${MPI_OMP_THREADS}; \
      export MKL_THREADING_LAYER=SEQUENTIAL; \
      export MKL_NUM_THREADS=1; \
      export SINGULARITY_TMPDIR=\$HOME/singularity_tmp; \
      export SINGULARITY_CACHEDIR=\$HOME/singularity_cache; \
      mkdir -p \"\$SINGULARITY_TMPDIR\" \"\$SINGULARITY_CACHEDIR\" benchmark_results/hpc; \
      singularity exec --bind '${REPO_ROOT}':/workspace '${SIF_PATH}' \
        mpirun -np ${n} python /workspace/benchmarks/benchmark_mpi_preprocess_ranks.py \
          --dataset-root '${DATASET_ROOT}' \
          --dataset-name '${DATASET_NAME}' \
          --adjacency-matrix '${ADJACENCY_MATRIX}' \
          --max-k ${MAX_K} \
          --num-graphs ${NUM_GRAPHS_MPI} \
          --repeats ${REPEATS_MPI} \
          --select largest \
          ${NODE_ATTR_FLAG} \
          --omp-threads ${MPI_OMP_THREADS} \
          --output-json /workspace/benchmark_results/hpc/${DATASET_TAG}_${ADJACENCY_MATRIX}_mpi_n${n}.json" \
  )"
  MPI_JOB_IDS+=("${job_id}")
  echo "  ranks ${n} -> job ${job_id}"
done

echo
echo "Benchmark submissions completed."
echo "OpenMP jobs: ${OMP_JOB_IDS[*]}"
echo "MPI jobs:    ${MPI_JOB_IDS[*]}"
echo
echo "Monitor:"
echo "  squeue -u $USER"
echo "  sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode"
