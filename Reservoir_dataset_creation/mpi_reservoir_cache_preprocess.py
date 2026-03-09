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

from model.MRGNN import MRGNN
from utils.utils_method import get_graph_diameter


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI preprocessing for MRGNN reservoir cache (graph classification)."
    )
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path used by TUDataset for native dataset.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="TUDataset name (e.g., NCI1, PROTEINS).")
    parser.add_argument("--output-root", type=str, default="~/Dataset/Reservoir_MPI_Cache",
                        help="Root folder for cache output.")
    parser.add_argument("--n-units", type=int, required=True,
                        help="Reservoir hidden units.")
    parser.add_argument("--n-classes", type=int, default=2,
                        help="Number of classes (only used for model shape compatibility).")
    parser.add_argument("--max-k", type=int, required=True,
                        help="Maximum multiresolution hop count.")
    parser.add_argument("--adjacency-matrix", type=str, choices=["A", "L", "D"], required=True,
                        help="Reservoir operator: A (adjacency), L (Laplacian), D (fairing).")
    parser.add_argument("--runs", type=int, nargs="+", default=[0],
                        help="Run IDs used to seed reservoir initialization.")
    parser.add_argument("--use-node-attr", action="store_true",
                        help="Enable node attributes in TUDataset.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing graph cache files.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log frequency (per rank, number of local graphs).")
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
    return "run_{}_TANH_RES_{}_{}_n_units_{}_{}".format(
        run, adjacency_matrix, max_k, n_units, dataset_name
    )


def select_transform(model, dataset_name, adjacency_matrix):
    if dataset_name == "PROTEINS":
        if adjacency_matrix == "A":
            return model.get_TANH_resevoir_A_PROTEINS
        if adjacency_matrix == "L":
            return model.get_TANH_resevoir_L_PROTEINS
        if adjacency_matrix == "D":
            return model.get_TANH_resevoir_D_PROTEINS
        raise ValueError("Unsupported adjacency matrix '{}'".format(adjacency_matrix))

    if adjacency_matrix == "A":
        return model.get_TANH_resevoir_A
    if adjacency_matrix == "L":
        return model.get_TANH_resevoir_L
    if adjacency_matrix == "D":
        return model.get_TANH_resevoir_D
    raise ValueError("Unsupported adjacency matrix '{}'".format(adjacency_matrix))


def load_native_dataset(args):
    dataset_root = os.path.expanduser(args.dataset_root)
    return TUDataset(
        root=dataset_root,
        name=args.dataset_name,
        pre_transform=get_graph_diameter,
        use_node_attr=args.use_node_attr,
    )


def save_graph_cache(data, graph_id, out_path):
    y = data.y.detach().cpu().view(-1).numpy().astype(np.int64)
    reservoir = data.reservoir.detach().cpu().numpy().astype(np.float32)
    num_nodes = np.asarray([int(data.num_nodes)], dtype=np.int64)
    graph_idx = np.asarray([int(graph_id)], dtype=np.int64)

    tmp_path = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        graph_id=graph_idx,
        num_nodes=num_nodes,
        y=y,
        reservoir=reservoir,
    )
    os.replace(tmp_path, out_path)


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


def main():
    args = parse_args()
    comm = ensure_mpi()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    torch.set_num_threads(1)
    output_root = os.path.expanduser(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    if rank == 0:
        print("[rank 0] Loading native dataset and running pre_transform if needed...")
        dataset = load_native_dataset(args)
        print("[rank 0] Dataset ready: {} graphs".format(len(dataset)))
    else:
        dataset = None

    comm.Barrier()

    if rank != 0:
        dataset = load_native_dataset(args)

    n_graphs = len(dataset)
    source_num_features = dataset.num_features

    local_graph_ids = list(range(rank, n_graphs, world_size))
    comm.Barrier()

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
        transform = select_transform(model, args.dataset_name, args.adjacency_matrix)

        processed = 0
        skipped = 0
        start = time.time()

        with torch.no_grad():
            for i, graph_id in enumerate(local_graph_ids):
                out_path = os.path.join(graph_dir, "graph_{:08d}.npz".format(graph_id))
                if os.path.exists(out_path) and not args.overwrite:
                    skipped += 1
                    continue

                data = dataset[graph_id]
                data = transform(data)
                save_graph_cache(data, graph_id, out_path)
                processed += 1

                if args.log_every > 0 and (i + 1) % args.log_every == 0:
                    elapsed = time.time() - start
                    print(
                        "[rank {}] run {} | processed {}/{} local graphs in {:.2f}s".format(
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
        }
        shard_file = os.path.join(shard_dir, "rank_{:05d}.json".format(rank))
        with open(shard_file, "w") as f:
            json.dump(shard_metadata, f, indent=2)

        total_processed = comm.reduce(processed, op=MPI.SUM, root=0)
        total_skipped = comm.reduce(skipped, op=MPI.SUM, root=0)
        comm.Barrier()

        if rank == 0:
            cached_ids = read_cached_graph_ids(graph_dir)
            missing_graph_ids = sorted(set(range(n_graphs)) - cached_ids)
            metadata = {
                "dataset_name": args.dataset_name,
                "dataset_root": os.path.expanduser(args.dataset_root),
                "run": run,
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
            }
            metadata_path = os.path.join(run_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            if missing_graph_ids:
                preview = missing_graph_ids[:10]
                print(
                    "[rank 0] WARNING run {}: {} missing graphs. Example ids: {}".format(
                        run, len(missing_graph_ids), preview
                    )
                )
            else:
                print(
                    "[rank 0] run {} completed. Cached {}/{} graphs (processed {}, skipped {}).".format(
                        run, len(cached_ids), n_graphs, total_processed, total_skipped
                    )
                )

        comm.Barrier()


if __name__ == "__main__":
    main()
