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

/// Append single-token `[1, NKV, HD]` K/V vectors into NHD `k_cache`/`v_cache` at `pos_buf`
/// (modulo `ring_mod` when `ring_mod > 0`), suitable for CUDA Graph capture.
void kv_append_decode_cuda(
    torch::Tensor const& k_in,
    torch::Tensor const& v_in,
    torch::Tensor const& k_cache,
    torch::Tensor const& v_cache,
    torch::Tensor const& pos_buf,
    int64_t ring_mod
);

}  // namespace kolibri::cache
