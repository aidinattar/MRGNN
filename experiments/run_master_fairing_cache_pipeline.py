#!/usr/bin/env python3
import argparse
import itertools
import os
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Master+fairing grid pipeline by phase: preprocess OR train."
        )
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["preprocess", "train"],
        required=True,
        help="Execution phase. Use separate jobs on HPC.",
    )

    # Shared dataset/cache options.
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="~/Dataset/Reservoir_MPI_Cache")
    parser.add_argument("--n-classes", type=int, required=True)
    parser.add_argument("--use-node-attr", action="store_true")
    parser.add_argument("--runs", type=int, nargs="+", default=[0])

    # Reservoir grid.
    parser.add_argument("--adjacency-matrices", type=str, nargs="+",
                        choices=["A", "L", "D"], default=["A", "L", "D"])
    parser.add_argument("--max-k-list", type=int, nargs="+", required=True)
    parser.add_argument("--n-units-list", type=int, nargs="+", required=True)

    # Training grid.
    parser.add_argument("--lr-list", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--drop-prob-list", type=float, nargs="+", default=[0.5])
    parser.add_argument("--weight-decay-list", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--batch-size-list", type=int, nargs="+", default=[32])
    parser.add_argument("--readout-list", type=str, nargs="+",
                        choices=["funnel", "restricted_funnel", "one_layer"],
                        default=["funnel"])
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--test-epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./test_log/cache_grid")

    # Runtime/execution controls.
    parser.add_argument("--mpi-launcher", type=str, default="mpirun")
    parser.add_argument("--mpi-size-flag", type=str, default="-np")
    parser.add_argument("--mpi-np", type=int, default=8)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--preprocess-script", type=str,
                        default="Reservoir_dataset_creation/mpi_reservoir_cache_preprocess.py")
    parser.add_argument("--train-script", type=str, default="experiments/train_from_cache.py")
    parser.add_argument("--parallel-folds", action="store_true")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--mp-start-method", type=str, default="spawn",
                        choices=["spawn", "fork", "forkserver"])
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--allow-missing-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(base_dir, path):
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def build_cache_name(run, adjacency_matrix, max_k, n_units, dataset_name):
    return "run_{}_TANH_RES_{}_{}_n_units_{}_{}".format(
        run, adjacency_matrix, max_k, n_units, dataset_name
    )


def cmd_to_string(cmd):
    return " ".join(shlex.quote(token) for token in cmd)


def run_command(cmd, dry_run=False):
    print("$ {}".format(cmd_to_string(cmd)))
    if dry_run:
        return 0
    completed = subprocess.run(cmd)
    return completed.returncode


def run_checked(cmd, dry_run=False, continue_on_error=False):
    exit_code = run_command(cmd, dry_run=dry_run)
    if exit_code != 0 and not continue_on_error:
        raise RuntimeError("Command failed with exit code {}: {}".format(exit_code, cmd_to_string(cmd)))
    return exit_code


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    output_root = os.path.expanduser(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.expanduser(args.log_dir), exist_ok=True)

    preprocess_script = resolve_path(repo_root, args.preprocess_script)
    train_script = resolve_path(repo_root, args.train_script)

    reservoir_grid = list(
        itertools.product(args.adjacency_matrices, args.max_k_list, args.n_units_list)
    )
    readout_grid = list(
        itertools.product(
            args.lr_list,
            args.drop_prob_list,
            args.weight_decay_list,
            args.batch_size_list,
            args.readout_list,
        )
    )

    expected_configs = []
    for run in args.runs:
        for adjacency_matrix, max_k, n_units in reservoir_grid:
            expected_configs.append({
                "run": run,
                "adjacency_matrix": adjacency_matrix,
                "max_k": max_k,
                "n_units": n_units,
                "cache_dir": os.path.join(
                    output_root,
                    build_cache_name(run, adjacency_matrix, max_k, n_units, args.dataset_name),
                ),
            })

    print("Reservoir configs: {}".format(len(reservoir_grid)))
    print("Readout configs:   {}".format(len(readout_grid)))
    print("Runs:              {}".format(len(args.runs)))
    print("Expected caches:   {}".format(len(expected_configs)))
    print("Training launches: {}".format(len(expected_configs) * len(readout_grid)))

    failed = 0
    mpi_launcher_tokens = shlex.split(args.mpi_launcher)

    if args.phase == "preprocess":
        print("\n=== Preprocess phase (MPI reservoir cache grid) ===")
        for adjacency_matrix, max_k, n_units in reservoir_grid:
            cmd = (
                mpi_launcher_tokens
                + [
                    args.mpi_size_flag,
                    str(args.mpi_np),
                    args.python_bin,
                    preprocess_script,
                    "--dataset-root",
                    args.dataset_root,
                    "--dataset-name",
                    args.dataset_name,
                    "--output-root",
                    args.output_root,
                    "--n-units",
                    str(n_units),
                    "--n-classes",
                    str(args.n_classes),
                    "--max-k",
                    str(max_k),
                    "--adjacency-matrix",
                    adjacency_matrix,
                    "--runs",
                ]
                + [str(run_id) for run_id in args.runs]
                + [
                    "--log-every",
                    str(args.log_every),
                ]
            )
            if args.use_node_attr:
                cmd.append("--use-node-attr")
            if args.overwrite_cache:
                cmd.append("--overwrite")

            exit_code = run_checked(
                cmd,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            if exit_code != 0:
                failed += 1

    if args.phase == "train":
        print("\n=== Train phase (grid over cache + readout) ===")
        for cfg in expected_configs:
            cache_dir = cfg["cache_dir"]
            if not os.path.isdir(cache_dir):
                message = "Missing cache dir: {}".format(cache_dir)
                if args.allow_missing_cache:
                    print("WARNING: {} (skipping)".format(message))
                    continue
                raise RuntimeError(message)

            for lr, drop_prob, weight_decay, batch_size, readout in readout_grid:
                cmd = [
                    args.python_bin,
                    train_script,
                    "--cache-dir",
                    cache_dir,
                    "--dataset-name",
                    args.dataset_name,
                    "--n-units",
                    str(cfg["n_units"]),
                    "--max-k",
                    str(cfg["max_k"]),
                    "--n-classes",
                    str(args.n_classes),
                    "--n-epochs",
                    str(args.n_epochs),
                    "--n-folds",
                    str(args.n_folds),
                    "--test-epoch",
                    str(args.test_epoch),
                    "--batch-size",
                    str(batch_size),
                    "--lr",
                    str(lr),
                    "--drop-prob",
                    str(drop_prob),
                    "--weight-decay",
                    str(weight_decay),
                    "--output",
                    readout,
                    "--seed",
                    str(args.seed),
                    "--log-dir",
                    args.log_dir,
                    "--mp-start-method",
                    args.mp_start_method,
                ]
                if args.parallel_folds:
                    cmd.append("--parallel-folds")
                if args.gpu_ids is not None and len(args.gpu_ids) > 0:
                    cmd.extend(["--gpu-ids"] + [str(gpu_id) for gpu_id in args.gpu_ids])

                exit_code = run_checked(
                    cmd,
                    dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                )
                if exit_code != 0:
                    failed += 1

    print("\nPipeline completed with {} failed command(s).".format(failed))
    if failed > 0 and not args.continue_on_error:
        raise RuntimeError("Pipeline failed with {} command(s).".format(failed))


if __name__ == "__main__":
    main()
