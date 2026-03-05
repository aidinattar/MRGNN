import ctypes
import os
import subprocess

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CPP_SOURCE = os.path.join(REPO_ROOT, "cpp_omp", "graph_preprocess_omp.cpp")
DEFAULT_BUILD_DIR = os.path.join(REPO_ROOT, "build")
DEFAULT_LIB_PATH = os.path.join(DEFAULT_BUILD_DIR, "libgraph_preprocess_omp.so")


def compile_openmp_library(
    cxx="g++",
    output_path=DEFAULT_LIB_PATH,
    force=False,
    extra_flags=None,
):
    if output_path is None:
        output_path = DEFAULT_LIB_PATH
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not force:
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flags = ["-O3", "-std=c++17", "-fopenmp", "-shared", "-fPIC"]
    if extra_flags:
        flags.extend(extra_flags)

    cmd = [cxx, CPP_SOURCE] + flags + ["-o", output_path]
    subprocess.run(cmd, check=True)
    return output_path


def _as_int64_array(arr, name):
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("{} must be 1D int64".format(name))
    return np.ascontiguousarray(arr)


def _as_float32_matrix(arr, name):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("{} must be 2D float32".format(name))
    return np.ascontiguousarray(arr)


class OMPGraphPreprocessor(object):
    def __init__(self, lib_path=None, auto_build=True, force_rebuild=False):
        if lib_path is None:
            lib_path = DEFAULT_LIB_PATH
        lib_path = os.path.abspath(lib_path)

        if auto_build and (force_rebuild or not os.path.exists(lib_path)):
            compile_openmp_library(output_path=lib_path, force=force_rebuild)

        if not os.path.exists(lib_path):
            raise ValueError(
                "OpenMP library not found at {}. Build it with compile_openmp_library().".format(
                    lib_path
                )
            )

        self.lib_path = lib_path
        self.lib = ctypes.CDLL(lib_path)
        self._configure_signatures()

    def _configure_signatures(self):
        c_int64_p = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")

        self.lib.set_omp_threads.argtypes = [ctypes.c_int]
        self.lib.set_omp_threads.restype = None

        self.lib.get_omp_max_threads.argtypes = []
        self.lib.get_omp_max_threads.restype = ctypes.c_int

        self.lib.build_csr_from_edges.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            c_int64_p,
            c_int64_p,
            ctypes.c_int,
            c_int64_p,
            c_int64_p,
            c_int64_p,
        ]
        self.lib.build_csr_from_edges.restype = ctypes.c_int

        self.lib.compute_gcn_normalized_values.argtypes = [
            ctypes.c_int64,
            c_int64_p,
            c_int64_p,
            c_float_p,
        ]
        self.lib.compute_gcn_normalized_values.restype = ctypes.c_int

        self.lib.csr_multihop_diffusion.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            c_int64_p,
            c_int64_p,
            c_float_p,
            c_float_p,
            ctypes.c_int,
            c_float_p,
        ]
        self.lib.csr_multihop_diffusion.restype = ctypes.c_int

    def set_threads(self, n_threads):
        self.lib.set_omp_threads(int(n_threads))

    def max_threads(self):
        return int(self.lib.get_omp_max_threads())

    def build_csr(self, edge_index, num_nodes, add_self_loops=True):
        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        edge_index = np.ascontiguousarray(edge_index)

        src = np.ascontiguousarray(edge_index[0])
        dst = np.ascontiguousarray(edge_index[1])
        num_edges = int(src.shape[0])

        max_nnz = num_edges + (int(num_nodes) if add_self_loops else 0)
        row_ptr = np.empty(int(num_nodes) + 1, dtype=np.int64)
        col_idx = np.empty(max_nnz, dtype=np.int64)
        out_nnz = np.zeros(1, dtype=np.int64)

        ret = self.lib.build_csr_from_edges(
            int(num_nodes),
            num_edges,
            src,
            dst,
            int(bool(add_self_loops)),
            row_ptr,
            col_idx,
            out_nnz,
        )
        if ret != 0:
            raise RuntimeError("build_csr_from_edges failed with error code {}".format(ret))

        nnz = int(out_nnz[0])
        return row_ptr, np.ascontiguousarray(col_idx[:nnz])

    def compute_gcn_values(self, row_ptr, col_idx):
        row_ptr = _as_int64_array(row_ptr, "row_ptr")
        col_idx = _as_int64_array(col_idx, "col_idx")
        num_nodes = int(row_ptr.shape[0] - 1)
        values = np.empty(col_idx.shape[0], dtype=np.float32)

        ret = self.lib.compute_gcn_normalized_values(
            num_nodes,
            row_ptr,
            col_idx,
            values,
        )
        if ret != 0:
            raise RuntimeError(
                "compute_gcn_normalized_values failed with error code {}".format(ret)
            )

        return values

    def multihop_diffusion(self, row_ptr, col_idx, values, x0, max_k, apply_tanh=True):
        row_ptr = _as_int64_array(row_ptr, "row_ptr")
        col_idx = _as_int64_array(col_idx, "col_idx")
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("values must be 1D float32")
        values = np.ascontiguousarray(values)

        x0 = _as_float32_matrix(x0, "x0")
        num_nodes = int(x0.shape[0])
        feat_dim = int(x0.shape[1])
        max_k = int(max_k)
        if max_k <= 0:
            raise ValueError("max_k must be >= 1")

        out = np.empty((max_k, num_nodes, feat_dim), dtype=np.float32)
        out_flat = np.ascontiguousarray(out.reshape(-1))

        ret = self.lib.csr_multihop_diffusion(
            num_nodes,
            feat_dim,
            max_k,
            row_ptr,
            col_idx,
            values,
            x0.reshape(-1),
            int(bool(apply_tanh)),
            out_flat,
        )
        if ret != 0:
            raise RuntimeError("csr_multihop_diffusion failed with error code {}".format(ret))

        return out_flat.reshape(max_k, num_nodes, feat_dim)

    def preprocess_graph(
        self,
        edge_index,
        x0,
        num_nodes,
        max_k,
        add_self_loops=True,
        apply_tanh=True,
    ):
        row_ptr, col_idx = self.build_csr(
            edge_index=edge_index,
            num_nodes=num_nodes,
            add_self_loops=add_self_loops,
        )
        values = self.compute_gcn_values(row_ptr=row_ptr, col_idx=col_idx)
        hops = self.multihop_diffusion(
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values,
            x0=x0,
            max_k=max_k,
            apply_tanh=apply_tanh,
        )
        return row_ptr, col_idx, values, hops
