# MRGNN

Repository for Multiresolution Reservoir Graph Neural Network (MRGNN) with:

- original reservoir pipeline,
- MPI preprocessing cache,
- MPI + OpenMP sparse preprocessing (CSR + normalized diffusion),
- training from cached reservoir features.

Paper: [Multiresolution Reservoir Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/9476188)

## Quick setup

Main files:

- `Reservoir_dataset_creation/mpi_reservoir_cache_preprocess.py`
- `Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py`
- `experiments/train_from_cache.py`
- `benchmarks/benchmark_omp_preprocess_threads.py`

## Preprocess (MPI + OpenMP)

```bash
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1

mpirun -np 8 python Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
  --dataset-root ~/Dataset/PROTEINS \
  --dataset-name PROTEINS \
  --output-root ~/Dataset/Reservoir_OMP_Cache \
  --n-units 50 \
  --n-classes 2 \
  --max-k 4 \
  --runs 0 \
  --omp-threads 4
```

Output cache (per run):

- `metadata.json`
- `graphs/graph_XXXXXXXX.npz`
- `shards/rank_XXXXX.json`

## Train from cache

```bash
python experiments/train_from_cache.py \
  --cache-dir ~/Dataset/Reservoir_OMP_Cache/run_0_OMP_GCN_4_n_units_50_PROTEINS \
  --dataset-name PROTEINS \
  --n-epochs 200 \
  --n-folds 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --drop-prob 0.5 \
  --weight-decay 5e-4 \
  --output funnel
```

Optional parallel folds on multiple GPUs:

```bash
python experiments/train_from_cache.py \
  --cache-dir ~/Dataset/Reservoir_OMP_Cache/run_0_OMP_GCN_4_n_units_50_PROTEINS \
  --dataset-name PROTEINS \
  --n-epochs 200 \
  --n-folds 10 \
  --batch-size 64 \
  --output funnel \
  --parallel-folds \
  --gpu-ids 0 1 2 3
```

## Benchmark preprocess scaling

OpenMP threads:

```bash
python benchmarks/benchmark_omp_preprocess_threads.py \
  --dataset-root ~/Dataset/REDDIT-MULTI-12K \
  --dataset-name REDDIT-MULTI-12K \
  --max-k 8 \
  --threads 1 2 4 8 16 32 \
  --num-graphs 4096 \
  --repeats 3 \
  --select largest \
  --output-json ./benchmark_results/reddit_omp_threads.json
```

MPI ranks:

```bash
mpirun -np 8 python benchmarks/benchmark_mpi_preprocess_ranks.py \
  --dataset-root ~/Dataset/REDDIT-MULTI-12K \
  --dataset-name REDDIT-MULTI-12K \
  --max-k 8 \
  --num-graphs 4096 \
  --repeats 3 \
  --omp-threads 1 \
  --output-json ./benchmark_results/reddit_mpi_ranks_n8.json
```

## SLURM jobs

Ready scripts are in `jobs/`:

- pipeline preprocess+train for PROTEINS and ENZYMES,
- benchmark submission script for OpenMP/MPI scaling.
