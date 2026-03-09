#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREPROCESS_JOB="${SCRIPT_DIR}/preprocess_mpi_fairing_grid_enzymes.sbatch"
TRAIN_JOB="${SCRIPT_DIR}/train_from_cache_fairing_grid_enzymes.sbatch"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH."
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

echo "Submitting fairing preprocess ENZYMES job..."
PREPROCESS_ID="$(sbatch --parsable "${PREPROCESS_JOB}")"
echo "  preprocess job id: ${PREPROCESS_ID}"

echo "Submitting fairing train ENZYMES job with dependency afterok:${PREPROCESS_ID}..."
TRAIN_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_ID} "${TRAIN_JOB}")"
echo "  train job id:      ${TRAIN_ID}"

echo
echo "Pipeline submitted."
echo "Monitor:"
echo "  squeue -j ${PREPROCESS_ID},${TRAIN_ID}"
echo "  sacct -j ${PREPROCESS_ID},${TRAIN_ID} --format=JobID,JobName,State,ExitCode,Elapsed"
