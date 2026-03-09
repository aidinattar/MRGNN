import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

try:
    from mpi4py import MPI
except ImportError as exc:
    MPI = None
    MPI_IMPORT_ERROR = exc
else:
    MPI_IMPORT_ERROR = None

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian

from model.MRGNN import MRGNN
from utils.omp_graph_preprocess import OMPGraphPreprocessor, compile_openmp_library


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI + OpenMP preprocessing for cached reservoir features."
    )
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path used by TUDataset for native dataset.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="TUDataset name (e.g., PROTEINS, NCI1, COLLAB).")
    parser.add_argument("--output-root", type=str, default="~/Dataset/Reservoir_OMP_Cache",
                        help="Root output folder.")
    parser.add_argument("--n-units", type=int, required=True,
                        help="Reservoir hidden units.")
    parser.add_argument("--n-classes", type=int, default=2,
                        help="Number of classes (for model shape).")
    parser.add_argument("--max-k", type=int, required=True,
                        help="Number of multihop steps including hop-0.")
    parser.add_argument("--adjacency-matrix", type=str, choices=["A", "L", "D", "GCN"],
                        default="GCN",
                        help="Operator mode: A/L/D for reservoir-compatible modes, GCN for legacy OMP_GCN.")
    parser.add_argument("--runs", type=int, nargs="+", default=[0],
                        help="Run IDs used to seed the fixed readout matrix.")
    parser.add_argument("--use-node-attr", action="store_true",
                        help="Enable node attributes in TUDataset.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing graph cache files.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log frequency per rank.")
    parser.add_argument("--omp-lib-path", type=str, default=None,
                        help="Path to libgraph_preprocess_omp.so. Auto-build if missing.")
    parser.add_argument("--force-rebuild-omp-lib", action="store_true",
                        help="Force recompilation of OpenMP library on rank 0.")
    parser.add_argument("--omp-threads", type=int, default=0,
                        help="Set OpenMP threads per rank (0 = keep env/default).")
    parser.add_argument("--store-multi-hop", action="store_true",
                        help="Store multi-hop tensor in graph cache (larger files).")
    parser.add_argument("--no-tanh", action="store_true",
                        help="Disable tanh after each diffusion step.")
    return parser.parse_args()


def ensure_mpi():
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required for this script. Import error: {}".format(MPI_IMPORT_ERROR)
        )
    return MPI.COMM_WORLD


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_run_cache_name(run, adjacency_matrix, max_k, n_units, dataset_name):
    if adjacency_matrix == "GCN":
        return "run_{}_OMP_GCN_{}_n_units_{}_{}".format(
            run, max_k, n_units, dataset_name
        )
    return "run_{}_TANH_RES_{}_{}_n_units_{}_{}".format(
        run, adjacency_matrix, max_k, n_units, dataset_name
    )


def load_native_dataset(args):
    dataset_root = os.path.expanduser(args.dataset_root)
    return TUDataset(
        root=dataset_root,
        name=args.dataset_name,
        use_node_attr=args.use_node_attr,
    )


def read_cached_graph_ids(graph_dir):
    cached_ids = set()
    for file_name in os.listdir(graph_dir):
        if not file_name.endswith(".npz"):
            continue
        if not file_name.startswith("graph_"):
            continue
        try:
            graph_id = int(file_name[len("graph_"):-len(".npz")])
        except ValueError:
            continue
        cached_ids.add(graph_id)
    return cached_ids


def maybe_prepare_x(data, in_channels, dataset_name):
    if data.x is None:
        x = torch.ones(data.num_nodes, 1, dtype=torch.float32)
    else:
        x = data.x.to(torch.float32)

    # Keep compatibility with older PROTEINS variants where a leading column
    # must be dropped to match model input channels.
    if dataset_name == "PROTEINS" and x.shape[1] == in_channels + 1:
        x = x[:, 1:]
    return x


def coo_to_csr(edge_index, edge_weight, num_nodes):
    row = np.asarray(edge_index[0], dtype=np.int64)
    col = np.asarray(edge_index[1], dtype=np.int64)
    val = np.asarray(edge_weight, dtype=np.float32)

    if row.ndim != 1 or col.ndim != 1 or val.ndim != 1:
        raise ValueError("COO arrays must be 1D.")
    if row.shape[0] != col.shape[0] or row.shape[0] != val.shape[0]:
        raise ValueError("COO arrays must have same length.")

    if row.shape[0] == 0:
        return (
            np.zeros(num_nodes + 1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    keys = row * np.int64(num_nodes) + col
    order = np.argsort(keys, kind="mergesort")
    row = row[order]
    col = col[order]
    val = val[order]
    keys = keys[order]

    # Sum duplicate COO entries to match sparse-coalesced behavior.
    unique_mask = np.ones(keys.shape[0], dtype=bool)
    unique_mask[1:] = keys[1:] != keys[:-1]
    unique_idx = np.nonzero(unique_mask)[0]

    row_u = row[unique_idx]
    col_u = col[unique_idx]
    val_u = np.add.reduceat(val, unique_idx).astype(np.float32, copy=False)

    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(row_ptr, row_u + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)

    return (
        np.ascontiguousarray(row_ptr, dtype=np.int64),
        np.ascontiguousarray(col_u, dtype=np.int64),
        np.ascontiguousarray(val_u, dtype=np.float32),
    )


def preprocess_with_operator(omp, data, x_np, args):
    num_nodes = int(data.num_nodes)
    edge_index_np = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)

    if args.adjacency_matrix == "GCN":
        return omp.preprocess_graph(
            edge_index=edge_index_np,
            x0=x_np,
            num_nodes=num_nodes,
            max_k=args.max_k,
            add_self_loops=True,
            apply_tanh=not args.no_tanh,
        )

    if args.adjacency_matrix == "A":
        row_ptr, col_idx = omp.build_csr(
            edge_index=edge_index_np,
            num_nodes=num_nodes,
            add_self_loops=False,
        )
        values = np.ones(col_idx.shape[0], dtype=np.float32)
        hops = omp.multihop_operator(
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values,
            x0=x_np,
            max_k=args.max_k,
            operator_mode="spmm",
            apply_tanh=not args.no_tanh,
        )
        return row_ptr, col_idx, values, hops

    # L / D use normalized Laplacian coefficients from PyG for fidelity with MRGNN.
    L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym")
    L_edge_index_np = L_edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    L_values_np = L_values.detach().cpu().numpy().astype(np.float32, copy=False)
    row_ptr, col_idx, values = coo_to_csr(
        edge_index=L_edge_index_np,
        edge_weight=L_values_np,
        num_nodes=num_nodes,
    )

    if args.adjacency_matrix == "L":
        hops = omp.multihop_operator(
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values,
            x0=x_np,
            max_k=args.max_k,
            operator_mode="spmm",
            apply_tanh=not args.no_tanh,
        )
    else:
        # Fairing: alternating (I - 0.5L) / (I + 2/3 L), no tanh by default.
        hops = omp.multihop_operator(
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values,
            x0=x_np,
            max_k=args.max_k,
            operator_mode="fairing",
            apply_tanh=False,
        )
    return row_ptr, col_idx, values, hops


def save_graph_cache(
    out_path,
    graph_id,
    y,
    num_nodes,
    reservoir,
    row_ptr,
    col_idx,
    norm_values,
    multi_hop=None,
):
    tmp_path = out_path + ".tmp.npz"
    payload = {
        "graph_id": np.asarray([graph_id], dtype=np.int64),
        "y": np.asarray(y, dtype=np.int64).reshape(-1),
        "num_nodes": np.asarray([num_nodes], dtype=np.int64),
        "reservoir": np.asarray(reservoir, dtype=np.float32),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "col_idx": np.asarray(col_idx, dtype=np.int64),
        "norm_values": np.asarray(norm_values, dtype=np.float32),
    }
    if multi_hop is not None:
        payload["multi_hop"] = np.asarray(multi_hop, dtype=np.float32)

    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, out_path)


def main():
    args = parse_args()
    comm = ensure_mpi()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    torch.set_num_threads(1)
    output_root = os.path.expanduser(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    omp_lib_path = args.omp_lib_path
    if omp_lib_path is not None:
        omp_lib_path = os.path.abspath(os.path.expanduser(omp_lib_path))

    if rank == 0:
        compile_openmp_library(
            output_path=omp_lib_path or None,
            force=args.force_rebuild_omp_lib,
        )
    comm.Barrier()

    omp = OMPGraphPreprocessor(
        lib_path=omp_lib_path,
        auto_build=False,
    )
    if args.omp_threads > 0:
        omp.set_threads(args.omp_threads)

    if rank == 0:
        print(
            "[rank 0] OpenMP library loaded. max_threads={} requested_threads={} operator={}".format(
                omp.max_threads(), args.omp_threads, args.adjacency_matrix
            )
        )
        print("[rank 0] Loading native dataset...")
        dataset = load_native_dataset(args)
        print("[rank 0] Dataset ready: {} graphs".format(len(dataset)))
    else:
        dataset = None

    comm.Barrier()
    if rank != 0:
        dataset = load_native_dataset(args)

    n_graphs = len(dataset)
    source_num_features = dataset.num_features if dataset.num_features > 0 else 1
    local_graph_ids = list(range(rank, n_graphs, world_size))

    for run in args.runs:
        run_name = build_run_cache_name(
            run=run,
            adjacency_matrix=args.adjacency_matrix,
            max_k=args.max_k,
            n_units=args.n_units,
            dataset_name=args.dataset_name,
        )
        run_dir = os.path.join(output_root, run_name)
        graph_dir = os.path.join(run_dir, "graphs")
        shard_dir = os.path.join(run_dir, "shards")

        if rank == 0:
            os.makedirs(graph_dir, exist_ok=True)
            os.makedirs(shard_dir, exist_ok=True)
        comm.Barrier()

        seed_everything(run)
        model = MRGNN(
            in_channels=source_num_features,
            out_channels=args.n_units,
            n_class=args.n_classes,
            drop_prob=0.0,
            max_k=args.max_k,
            device="cpu",
        ).to("cpu")
        model.eval()

        processed = 0
        skipped = 0
        total_prep_time = 0.0
        total_reservoir_time = 0.0
        start = time.time()

        with torch.no_grad():
            for i, graph_id in enumerate(local_graph_ids):
                out_path = os.path.join(graph_dir, "graph_{:08d}.npz".format(graph_id))
                if os.path.exists(out_path) and not args.overwrite:
                    skipped += 1
                    continue

                data = dataset[graph_id]
                x = maybe_prepare_x(data, source_num_features, args.dataset_name)
                x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)

                prep_start = time.perf_counter()
                row_ptr, col_idx, norm_values, hops = preprocess_with_operator(
                    omp=omp,
                    data=data,
                    x_np=x_np,
                    args=args,
                )
                total_prep_time += time.perf_counter() - prep_start

                reservoir_start = time.perf_counter()
                hop_concat = np.concatenate(
                    [hops[k] for k in range(args.max_k)],
                    axis=1,
                )
                hop_concat_t = torch.from_numpy(hop_concat).float()
                reservoir = model.lin(hop_concat_t, model.xhi_layer_mask)
                reservoir_np = reservoir.detach().cpu().numpy().astype(np.float32, copy=False)
                total_reservoir_time += time.perf_counter() - reservoir_start

                multi_hop = hops if args.store_multi_hop else None
                save_graph_cache(
                    out_path=out_path,
                    graph_id=int(graph_id),
                    y=data.y.detach().cpu().numpy(),
                    num_nodes=int(data.num_nodes),
                    reservoir=reservoir_np,
                    row_ptr=row_ptr,
                    col_idx=col_idx,
                    norm_values=norm_values,
                    multi_hop=multi_hop,
                )
                processed += 1

                if args.log_every > 0 and (i + 1) % args.log_every == 0:
                    elapsed = time.time() - start
                    print(
                        "[rank {}] run {} | {}/{} local graphs | elapsed {:.2f}s".format(
                            rank, run, i + 1, len(local_graph_ids), elapsed
                        )
                    )

        shard_metadata = {
            "rank": rank,
            "world_size": world_size,
            "run": run,
            "processed_graphs": processed,
            "skipped_graphs": skipped,
            "local_graphs": len(local_graph_ids),
            "elapsed_seconds": time.time() - start,
            "prep_seconds": total_prep_time,
            "reservoir_projection_seconds": total_reservoir_time,
            "omp_threads_setting": args.omp_threads,
        }
        shard_file = os.path.join(shard_dir, "rank_{:05d}.json".format(rank))
        with open(shard_file, "w") as f:
            json.dump(shard_metadata, f, indent=2)

        total_processed = comm.reduce(processed, op=MPI.SUM, root=0)
        total_skipped = comm.reduce(skipped, op=MPI.SUM, root=0)
        total_prep = comm.reduce(total_prep_time, op=MPI.SUM, root=0)
        total_projection = comm.reduce(total_reservoir_time, op=MPI.SUM, root=0)
        comm.Barrier()

        if rank == 0:
            cached_ids = read_cached_graph_ids(graph_dir)
            missing_graph_ids = sorted(set(range(n_graphs)) - cached_ids)
            metadata = {
                "dataset_name": args.dataset_name,
                "dataset_root": os.path.expanduser(args.dataset_root),
                "run": run,
                "operator": args.adjacency_matrix,
                "adjacency_matrix": args.adjacency_matrix,
                "max_k": args.max_k,
                "n_units": args.n_units,
                "n_classes": args.n_classes,
                "use_node_attr": args.use_node_attr,
                "source_num_features": int(source_num_features),
                "expected_graphs": int(n_graphs),
                "cached_graphs": int(len(cached_ids)),
                "missing_graphs": int(len(missing_graph_ids)),
                "processed_this_launch": int(total_processed),
                "skipped_this_launch": int(total_skipped),
                "world_size": int(world_size),
                "omp_threads_setting": int(args.omp_threads),
                "store_multi_hop": bool(args.store_multi_hop),
                "apply_tanh": bool(not args.no_tanh),
                "prep_cpu_seconds_sum": float(total_prep),
                "reservoir_projection_seconds_sum": float(total_projection),
            }
            metadata_path = os.path.join(run_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            if missing_graph_ids:
                print(
                    "[rank 0] WARNING run {}: {} missing graphs. Example ids: {}".format(
                        run, len(missing_graph_ids), missing_graph_ids[:10]
                    )
                )
            else:
                print(
                    "[rank 0] run {} completed. Cached {}/{} graphs | processed {} | skipped {} | prep_sum {:.2f}s".format(
                        run,
                        len(cached_ids),
                        n_graphs,
                        total_processed,
                        total_skipped,
                        total_prep,
                    )
                )
        comm.Barrier()


if __name__ == "__main__":
    main()
