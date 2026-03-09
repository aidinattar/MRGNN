#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

int csr_multihop_diffusion_mode(
    int64_t num_nodes,
    int64_t feat_dim,
    int64_t max_k,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* values,
    const float* x0,
    int mode,
    int apply_tanh,
    float* out_hops);

void set_omp_threads(int n_threads) {
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
#else
  (void)n_threads;
#endif
}

int get_omp_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int build_csr_from_edges(
    int64_t num_nodes,
    int64_t num_edges,
    const int64_t* src,
    const int64_t* dst,
    int add_self_loops,
    int64_t* row_ptr,
    int64_t* col_idx,
    int64_t* out_nnz) {
  if (num_nodes <= 0 || num_edges < 0 || src == nullptr || dst == nullptr ||
      row_ptr == nullptr || col_idx == nullptr || out_nnz == nullptr) {
    return -1;
  }

  std::vector<int64_t> row_counts(static_cast<size_t>(num_nodes), 0);

#pragma omp parallel for schedule(static)
  for (int64_t e = 0; e < num_edges; ++e) {
    const int64_t u = src[e];
    if (u < 0 || u >= num_nodes) {
      continue;
    }
#pragma omp atomic update
    row_counts[static_cast<size_t>(u)] += 1;
  }

  if (add_self_loops) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; ++i) {
      row_counts[static_cast<size_t>(i)] += 1;
    }
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<int64_t> chunk_sums(static_cast<size_t>(n_threads), 0);
  std::vector<int64_t> chunk_offsets(static_cast<size_t>(n_threads), 0);

#pragma omp parallel
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    const int64_t start = (num_nodes * tid) / n_threads;
    const int64_t end = (num_nodes * (tid + 1)) / n_threads;
    int64_t local_sum = 0;
    for (int64_t i = start; i < end; ++i) {
      local_sum += row_counts[static_cast<size_t>(i)];
    }
    chunk_sums[static_cast<size_t>(tid)] = local_sum;
  }

  for (int t = 1; t < n_threads; ++t) {
    chunk_offsets[static_cast<size_t>(t)] =
        chunk_offsets[static_cast<size_t>(t - 1)] +
        chunk_sums[static_cast<size_t>(t - 1)];
  }

  row_ptr[0] = 0;
#pragma omp parallel
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    const int64_t start = (num_nodes * tid) / n_threads;
    const int64_t end = (num_nodes * (tid + 1)) / n_threads;
    int64_t running = chunk_offsets[static_cast<size_t>(tid)];

    for (int64_t i = start; i < end; ++i) {
      row_ptr[i] = running;
      running += row_counts[static_cast<size_t>(i)];
      row_ptr[i + 1] = running;
    }
  }

  const int64_t total_nnz = row_ptr[num_nodes];
  *out_nnz = total_nnz;

  std::vector<int64_t> write_ptr(static_cast<size_t>(num_nodes), 0);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < num_nodes; ++i) {
    write_ptr[static_cast<size_t>(i)] = row_ptr[i];
  }

#pragma omp parallel for schedule(static)
  for (int64_t e = 0; e < num_edges; ++e) {
    const int64_t u = src[e];
    const int64_t v = dst[e];
    if (u < 0 || u >= num_nodes || v < 0 || v >= num_nodes) {
      continue;
    }
    int64_t pos = 0;
#pragma omp atomic capture
    {
      pos = write_ptr[static_cast<size_t>(u)];
      write_ptr[static_cast<size_t>(u)] += 1;
    }
    col_idx[pos] = v;
  }

  if (add_self_loops) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; ++i) {
      int64_t pos = 0;
#pragma omp atomic capture
      {
        pos = write_ptr[static_cast<size_t>(i)];
        write_ptr[static_cast<size_t>(i)] += 1;
      }
      col_idx[pos] = i;
    }
  }

  return 0;
}

int compute_gcn_normalized_values(
    int64_t num_nodes,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    float* values) {
  if (num_nodes <= 0 || row_ptr == nullptr || col_idx == nullptr || values == nullptr) {
    return -1;
  }

  std::vector<float> inv_sqrt_deg(static_cast<size_t>(num_nodes), 0.0f);

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < num_nodes; ++i) {
    const int64_t deg = row_ptr[i + 1] - row_ptr[i];
    if (deg > 0) {
      inv_sqrt_deg[static_cast<size_t>(i)] =
          1.0f / std::sqrt(static_cast<float>(deg));
    }
  }

#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < num_nodes; ++i) {
    const float left = inv_sqrt_deg[static_cast<size_t>(i)];
    for (int64_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      const int64_t j = col_idx[p];
      float right = 0.0f;
      if (j >= 0 && j < num_nodes) {
        right = inv_sqrt_deg[static_cast<size_t>(j)];
      }
      values[p] = left * right;
    }
  }

  return 0;
}

int csr_multihop_diffusion(
    int64_t num_nodes,
    int64_t feat_dim,
    int64_t max_k,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* values,
    const float* x0,
    int apply_tanh,
    float* out_hops) {
  // Backward-compatible wrapper: standard sparse propagation (mode 0).
  return csr_multihop_diffusion_mode(
      num_nodes,
      feat_dim,
      max_k,
      row_ptr,
      col_idx,
      values,
      x0,
      0,
      apply_tanh,
      out_hops);
}

int csr_multihop_diffusion_mode(
    int64_t num_nodes,
    int64_t feat_dim,
    int64_t max_k,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* values,
    const float* x0,
    int mode,
    int apply_tanh,
    float* out_hops) {
  if (num_nodes <= 0 || feat_dim <= 0 || max_k <= 0 || row_ptr == nullptr ||
      col_idx == nullptr || values == nullptr || x0 == nullptr || out_hops == nullptr) {
    return -1;
  }
  if (mode != 0 && mode != 1) {
    return -2;
  }

  const int64_t plane = num_nodes * feat_dim;
  std::copy(x0, x0 + plane, out_hops);
  if (max_k == 1) {
    return 0;
  }

  std::vector<float> prev(static_cast<size_t>(plane), 0.0f);
  std::vector<float> next(static_cast<size_t>(plane), 0.0f);
  std::copy(x0, x0 + plane, prev.begin());

  for (int64_t hop = 1; hop < max_k; ++hop) {
#pragma omp parallel for schedule(dynamic, 32)
    for (int64_t i = 0; i < num_nodes; ++i) {
      const float* prev_row = &prev[static_cast<size_t>(i * feat_dim)];
      float* out_row = &next[static_cast<size_t>(i * feat_dim)];

      if (mode == 0) {
        for (int64_t f = 0; f < feat_dim; ++f) {
          out_row[f] = 0.0f;
        }

        for (int64_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
          const int64_t j = col_idx[p];
          if (j < 0 || j >= num_nodes) {
            continue;
          }
          const float w = values[p];
          const float* in_row = &prev[static_cast<size_t>(j * feat_dim)];
          for (int64_t f = 0; f < feat_dim; ++f) {
            out_row[f] += w * in_row[f];
          }
        }

        if (apply_tanh) {
          for (int64_t f = 0; f < feat_dim; ++f) {
            out_row[f] = std::tanh(out_row[f]);
          }
        }
      } else {
        // Fairing mode (mode=1): out = prev + alpha * (Op * prev), with
        // alternating alpha values across hops.
        const float alpha = ((hop - 1) % 2 == 0) ? -0.5f : (2.0f / 3.0f);
        for (int64_t f = 0; f < feat_dim; ++f) {
          out_row[f] = prev_row[f];
        }

        for (int64_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
          const int64_t j = col_idx[p];
          if (j < 0 || j >= num_nodes) {
            continue;
          }
          const float w = alpha * values[p];
          const float* in_row = &prev[static_cast<size_t>(j * feat_dim)];
          for (int64_t f = 0; f < feat_dim; ++f) {
            out_row[f] += w * in_row[f];
          }
        }

        if (apply_tanh) {
          for (int64_t f = 0; f < feat_dim; ++f) {
            out_row[f] = std::tanh(out_row[f]);
          }
        }
      }
    }

    std::copy(next.begin(), next.end(), out_hops + hop * plane);
    prev.swap(next);
  }

  return 0;
}
}
