#pragma once

#include <cstdint>
#include <vector>

#include <torch/extension.h>

namespace kolibri::cache {

/// Issue async H2D `cudaMemcpyAsync` for `count` expert slots across all slabs on `stream_ptr`,
/// then record `event_ptr` on that stream.
void dma_slabs_async(
    std::vector<torch::Tensor> const& src_slabs,
    std::vector<torch::Tensor> const& dst_slabs,
    int64_t count,
    int64_t dst_offset,
    int64_t stream_ptr,
    int64_t event_ptr
);

/// Copy `m` promoted experts from `src_slots` to `dst_slots` across all GPU cache slabs
/// and update `gpu_expert_to_slot` in a single fused CUDA kernel launch.
void promote_experts_cuda(
    std::vector<torch::Tensor> const& cache_slabs,
    torch::Tensor const& gpu_expert_to_slot,
    std::vector<int64_t> const& src_slots,
    std::vector<int64_t> const& dst_slots,
    std::vector<int64_t> const& old_eids,
    std::vector<int64_t> const& new_eids
);

}  // namespace kolibri::cache
