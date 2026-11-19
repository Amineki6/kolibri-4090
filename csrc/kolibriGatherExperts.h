#pragma once

#include <cstdint>
#include <vector>

#include <torch/extension.h>

namespace kolibri::host {

/// Return the number of detected performance (P) cores on Linux (or total allowed CPUs).
int64_t detected_pcore_count();

/// Gather `expert_ids` rows from each contiguous CPU tensor in `src_tensors`
/// into the corresponding contiguous CPU tensor in `dst_tensors` using a scoped
/// OpenMP team (`num_threads` clause, without mutating PyTorch's global thread count)
/// pinned to P-cores on hybrid CPUs.
void gather_experts_omp(
    std::vector<torch::Tensor> const& src_tensors,
    std::vector<torch::Tensor> const& dst_tensors,
    torch::Tensor const& expert_ids,
    int64_t num_threads
);

}  // namespace kolibri::host
