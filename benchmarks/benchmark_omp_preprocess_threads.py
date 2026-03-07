import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from utils.omp_graph_preprocess import OMPGraphPreprocessor, compile_openmp_library


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenMP preprocess time vs threads."
    )
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path used by TUDataset for native dataset.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="TUDataset name.")
    parser.add_argument("--max-k", type=int, required=True,
                        help="Number of multi-hop steps (including hop-0).")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Thread counts to benchmark.")
    parser.add_argument("--num-graphs", type=int, default=128,
                        help="How many graphs to benchmark.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="How many repeats per thread count.")
    parser.add_argument("--use-node-attr", action="store_true",
                        help="Enable node attributes.")
    parser.add_argument("--select", type=str, choices=["largest", "random"], default="largest",
                        help="Graph selection policy.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for random graph selection.")
    parser.add_argument("--omp-lib-path", type=str, default=None,
                        help="Path to prebuilt OpenMP library.")
    parser.add_argument("--force-rebuild-omp-lib", action="store_true",
                        help="Force recompilation of OpenMP library.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional output JSON path.")
    return parser.parse_args()


def maybe_prepare_x(data):
    if data.x is None:
        return torch.ones(data.num_nodes, 1, dtype=torch.float32)
    return data.x.to(torch.float32)


def pick_graph_ids(dataset, num_graphs, mode, seed):
    num_graphs = min(num_graphs, len(dataset))
    if mode == "random":
        rng = np.random.RandomState(seed)
        return rng.choice(len(dataset), size=num_graphs, replace=False).tolist()

    indexed_sizes = []
    for idx in range(len(dataset)):
        indexed_sizes.append((idx, int(dataset[idx].num_nodes)))
    indexed_sizes.sort(key=lambda p: p[1], reverse=True)
    return [idx for idx, _ in indexed_sizes[:num_graphs]]


def run_once(omp, dataset, graph_ids, max_k):
    start = time.perf_counter()
    total_nodes = 0
    total_edges = 0

    for graph_id in graph_ids:
        data = dataset[graph_id]
        x = maybe_prepare_x(data).detach().cpu().numpy().astype(np.float32, copy=False)
        edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        total_nodes += int(data.num_nodes)
        total_edges += int(edge_index.shape[1])
        omp.preprocess_graph(
            edge_index=edge_index,
            x0=x,
            num_nodes=int(data.num_nodes),
            max_k=max_k,
            add_self_loops=True,
            apply_tanh=True,
        )

    elapsed = time.perf_counter() - start
    return elapsed, total_nodes, total_edges


def main():
    args = parse_args()

    if args.omp_lib_path is not None:
        omp_lib_path = os.path.abspath(os.path.expanduser(args.omp_lib_path))
    else:
        omp_lib_path = None

    compile_openmp_library(
        output_path=omp_lib_path or None,
        force=args.force_rebuild_omp_lib,
    )
    omp = OMPGraphPreprocessor(lib_path=omp_lib_path, auto_build=False)

    dataset = TUDataset(
        root=os.path.expanduser(args.dataset_root),
        name=args.dataset_name,
        use_node_attr=args.use_node_attr,
    )
    graph_ids = pick_graph_ids(
        dataset=dataset,
        num_graphs=args.num_graphs,
        mode=args.select,
        seed=args.seed,
    )

    print("Dataset: {} | graphs: {} | benchmark subset: {}".format(
        args.dataset_name, len(dataset), len(graph_ids)
    ))
    print("max_k={} | selection={} | repeats={}".format(
        args.max_k, args.select, args.repeats
    ))
    print("-" * 64)

    results = []
    for n_threads in args.threads:
        omp.set_threads(n_threads)
        timings = []
        nodes = 0
        edges = 0
        for _ in range(args.repeats):
            elapsed, nodes, edges = run_once(
                omp=omp,
                dataset=dataset,
                graph_ids=graph_ids,
                max_k=args.max_k,
            )
            timings.append(elapsed)

        t_mean = float(np.mean(timings))
        t_std = float(np.std(timings))
        graphs_per_sec = float(len(graph_ids) / t_mean) if t_mean > 0 else 0.0
        result = {
            "threads": int(n_threads),
            "mean_seconds": t_mean,
            "std_seconds": t_std,
            "graphs": int(len(graph_ids)),
            "graphs_per_second": graphs_per_sec,
            "nodes_total": int(nodes),
            "edges_total": int(edges),
        }
        results.append(result)
        print(
            "threads={:>2d} | mean={:8.4f}s | std={:7.4f}s | graphs/s={:8.2f}".format(
                n_threads, t_mean, t_std, graphs_per_sec
            )
        )

    payload = {
        "dataset_name": args.dataset_name,
        "dataset_root": os.path.expanduser(args.dataset_root),
        "max_k": args.max_k,
        "num_graphs": len(graph_ids),
        "repeats": args.repeats,
        "selection": args.select,
        "threads": args.threads,
        "results": results,
    }

    if args.output_json:
        output_json = os.path.abspath(os.path.expanduser(args.output_json))
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("Saved benchmark JSON to {}".format(output_json))


if __name__ == "__main__":
    main()
