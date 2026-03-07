# Container (Singularity)

Definition file:

- `mrgnn_gpu_openmp_mpi.def`

## Build

From repo root:

```bash
singularity build mrgnn_gpu_openmp_mpi.sif container/mrgnn_gpu_openmp_mpi.def
```

## Smoke test

```bash
singularity exec --nv --bind "$PWD":/workspace mrgnn_gpu_openmp_mpi.sif \
  python -c "import torch, torch_geometric, mpi4py; print(torch.__version__, torch.cuda.is_available())"
```

## Preprocess (MPI + OpenMP)

```bash
export OMP_NUM_THREADS=4
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1

singularity exec --bind "$PWD":/workspace mrgnn_gpu_openmp_mpi.sif \
  mpirun -np 8 python /workspace/Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
    --dataset-root /workspace/Dataset/PROTEINS \
    --dataset-name PROTEINS \
    --output-root /workspace/Dataset/Reservoir_OMP_Cache \
    --n-units 50 \
    --n-classes 2 \
    --max-k 4 \
    --runs 0 \
    --omp-threads 4
```

## Train from cache

```bash
singularity exec --nv --bind "$PWD":/workspace mrgnn_gpu_openmp_mpi.sif \
  python /workspace/experiments/train_from_cache.py \
    --cache-dir /workspace/Dataset/Reservoir_OMP_Cache/run_0_OMP_GCN_4_n_units_50_PROTEINS \
    --dataset-name PROTEINS \
    --n-epochs 200 \
    --n-folds 10
```
