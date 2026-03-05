# Multiresolution Reservoir Graph Neural Network

Graph neural networks are receiving increasing
attention as state-of-the-art methods to process graph-structured
data. However, similar to other neural networks, they tend
to suffer from a high computational cost to perform training.
Reservoir computing (RC) is an effective way to define neural
networks that are very efficient to train, often obtaining compa-
rable predictive performance with respect to the fully trained
counterparts. Different proposals of reservoir graph neural
networks have been proposed in the literature. However, their
predictive performances are still slightly below the ones of fully
trained graph neural networks on many benchmark datasets,
arguably because of the oversmoothing problem that arises when
iterating over the graph structure in the reservoir computation.
In this work, we aim to reduce this gap defining a multiresolution
reservoir graph neural network (MRGNN) inspired by graph
spectral filtering. Instead of iterating on the nonlinearity in
the reservoir and using a shallow readout function, we aim to
generate an explicit k-hop unsupervised graph representation
amenable for further, possibly nonlinear, processing. Experiments
on several datasets from various application areas show that our
approach is extremely fast and it achieves in most of the cases
comparable or even higher results with respect to state-of-the-art
approaches.

Paper: https://ieeexplore.ieee.org/abstract/document/9476188

If you find this code useful, please cite the following:
> @article{pasa2021multiresolution,  
  title={Multiresolution Reservoir Graph Neural Network},  
  author={Pasa, Luca and Navarin, Nicol{\`o} and Sperduti, Alessandro},   
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2021},  
  publisher={IEEE}  
}

## Code map

- `model/MRGNN.py`: core model. It has:
  - reservoir feature builders (`get_TANH_resevoir_A`, `get_TANH_resevoir_L`, `*_PROTEINS`),
  - readout forward (`readout_fw`) used during supervised training.
- `Reservoir_dataset_creation/ReservoirExtraction.py`: original preprocessing based on `TUDataset(pre_transform=...)`.
- `data_reader/cross_validation_reader.py`: original split and dataloader generation on TUDataset.
- `impl/binGraphClassifier.py`: training/evaluation loop for graph classification.
- `experiments/*/*.py`: exhaustive hyperparameter sweeps for each dataset.

## New: MPI coarse-grained preprocessing cache

This repository now includes a distributed preprocessing script for graph classification:

- each MPI rank processes a disjoint shard of graph indices,
- each rank computes full reservoir features locally,
- each graph is saved as a compressed cache file (`.npz`),
- training can be done later (also single GPU) by only reading this cache.

### 1) Build cache with MPI

```bash
mpirun -np 8 python Reservoir_dataset_creation/mpi_reservoir_cache_preprocess.py \
  --dataset-root ~/Dataset/NCI1 \
  --dataset-name NCI1 \
  --output-root ~/Dataset/Reservoir_MPI_Cache \
  --n-units 50 \
  --n-classes 2 \
  --max-k 4 \
  --adjacency-matrix A \
  --runs 0 1 2 3 4
```

Output layout for each run:

- `metadata.json`: cache metadata and completion stats.
- `graphs/graph_XXXXXXXX.npz`: per-graph cache (`reservoir`, `y`, `num_nodes`, `graph_id`).
- `shards/rank_XXXXX.json`: per-rank processing stats.

### 2) Train from cache

```bash
python experiments/train_from_cache.py \
  --cache-dir ~/Dataset/Reservoir_MPI_Cache/run_0_TANH_RES_A_4_n_units_50_NCI1 \
  --n-epochs 500 \
  --n-folds 10 \
  --lr 1e-3 \
  --drop-prob 0.5 \
  --weight-decay 5e-4 \
  --batch-size 32 \
  --output funnel
```

Training loader from cache is implemented in:

- `data_reader/reservoir_cache_dataset.py`

## New: OpenMP CSR + normalization + multihop prep

This branch adds a dedicated preprocessing path focused on CPU parallelism:

- CSR build from edge list (`row_ptr`, `col_idx`)
- GCN normalization on CSR values (`D^{-1/2}(A+I)D^{-1/2}`)
- multi-hop diffusion prep (`K` steps) with OpenMP loops

Implementation files:

- `cpp_omp/graph_preprocess_omp.cpp`: OpenMP kernels
- `utils/omp_graph_preprocess.py`: Python ctypes wrapper + auto-build
- `Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py`: MPI sharding + cache writing
- `benchmarks/benchmark_omp_preprocess_threads.py`: preprocess benchmark vs threads

### 1) Run MPI + OpenMP preprocessing

```bash
mpirun -np 8 python Reservoir_dataset_creation/mpi_omp_csr_multihop_preprocess.py \
  --dataset-root ~/Dataset/PROTEINS \
  --dataset-name PROTEINS \
  --output-root ~/Dataset/Reservoir_OMP_Cache \
  --n-units 50 \
  --n-classes 2 \
  --max-k 4 \
  --runs 0 \
  --omp-threads 8
```

Output (per run):

- `graphs/graph_XXXXXXXX.npz`: `reservoir`, CSR (`row_ptr`, `col_idx`), normalized values
- `metadata.json`: global stats
- `shards/rank_XXXXX.json`: per-rank timing stats

### 2) Benchmark preprocess time vs threads

```bash
python benchmarks/benchmark_omp_preprocess_threads.py \
  --dataset-root ~/Dataset/PROTEINS \
  --dataset-name PROTEINS \
  --max-k 4 \
  --threads 1 2 4 8 16 \
  --num-graphs 256 \
  --repeats 3 \
  --select largest \
  --output-json ./benchmark_results/proteins_omp_threads.json
```
