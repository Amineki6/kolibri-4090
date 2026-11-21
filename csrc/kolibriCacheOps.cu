#include "kolibriCacheOps.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>

namespace kolibri::cache {
namespace {

constexpr int kMaxSlabs = 8;
constexpr int kMaxPromote = 16;

struct PromoteParams {
    int num_large = 0;
    int num_small = 0;
    int m = 0;

    uint8_t* large_ptrs[kMaxSlabs]{};
    size_t large_row_bytes[kMaxSlabs]{};
    size_t large_vec4_per_row[kMaxSlabs]{};

    uint8_t* small_ptrs[kMaxSlabs]{};
    size_t small_row_bytes[kMaxSlabs]{};

    int src_slots[kMaxPromote]{};
    int dst_slots[kMaxPromote]{};
    int32_t* gpu_expert_to_slot = nullptr;
    int32_t num_experts = 0;
    int32_t old_eids[kMaxPromote]{};
    int32_t new_eids[kMaxPromote]{};
};

__global__ void promote_experts_kernel(PromoteParams p) {
    size_t const tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    // Copy 16-byte-aligned slabs with vectorized int4 loads/stores across all blocks.
    for (int k = 0; k < p.num_large; ++k) {
        size_t const n_vec4 = p.large_vec4_per_row[k];
        uint8_t* const base = p.large_ptrs[k];
        size_t const rb = p.large_row_bytes[k];
        for (int i = 0; i < p.m; ++i) {
            int4 const* __restrict__ src =
                reinterpret_cast<int4 const*>(base + static_cast<size_t>(p.src_slots[i]) * rb);
            int4* __restrict__ dst =
                reinterpret_cast<int4*>(base + static_cast<size_t>(p.dst_slots[i]) * rb);
            for (size_t v = tid; v < n_vec4; v += stride) {
                dst[v] = src[v];
            }
        }
    }

    // Copy small unaligned slabs (e.g. scalar global scales) and update expert->slot LUT.
    if (tid == 0) {
        for (int k = 0; k < p.num_small; ++k) {
            uint8_t* const base = p.small_ptrs[k];
            size_t const rb = p.small_row_bytes[k];
            for (int i = 0; i < p.m; ++i) {
                uint8_t const* src = base + static_cast<size_t>(p.src_slots[i]) * rb;
                uint8_t* dst = base + static_cast<size_t>(p.dst_slots[i]) * rb;
                for (size_t b = 0; b < rb; ++b) {
                    dst[b] = src[b];
                }
            }
        }

        if (p.gpu_expert_to_slot != nullptr) {
            for (int i = 0; i < p.m; ++i) {
                int32_t const old_eid = p.old_eids[i];
                if (old_eid >= 0 && old_eid < p.num_experts) {
                    p.gpu_expert_to_slot[old_eid] = -1;
                }
            }
            for (int i = 0; i < p.m; ++i) {
                int32_t const new_eid = p.new_eids[i];
                if (new_eid >= 0 && new_eid < p.num_experts) {
                    p.gpu_expert_to_slot[new_eid] = p.dst_slots[i];
                }
            }
        }
    }
}

}  // namespace

void dma_slabs_async(
    std::vector<torch::Tensor> const& src_slabs,
    std::vector<torch::Tensor> const& dst_slabs,
    int64_t count,
    int64_t dst_offset,
    int64_t stream_ptr,
    int64_t event_ptr
) {
    if (count <= 0 || src_slabs.empty()) {
        return;
    }
    TORCH_CHECK(src_slabs.size() == dst_slabs.size(), "src_slabs and dst_slabs must match in length");
    TORCH_CHECK(src_slabs.size() <= kMaxSlabs, "Too many slabs in dma_slabs_async");
    TORCH_CHECK(dst_offset >= 0, "dst_offset must be non-negative");
    TORCH_CHECK(stream_ptr != 0 && event_ptr != 0, "stream_ptr and event_ptr must be non-null");

    size_t const num_slabs = src_slabs.size();
    void const* src_ptrs[kMaxSlabs]{};
    void* dst_ptrs[kMaxSlabs]{};
    size_t copy_bytes[kMaxSlabs]{};

    auto const dev = dst_slabs[0].device();
    TORCH_CHECK(dev.is_cuda(), "dst_slabs must be CUDA tensors");

    for (size_t k = 0; k < num_slabs; ++k) {
        auto const& src = src_slabs[k];
        auto const& dst = dst_slabs[k];
        TORCH_CHECK(src.device().is_cpu(), "src_slabs must be on CPU");
        TORCH_CHECK(dst.is_cuda() && dst.device() == dev, "dst_slabs must be on the same CUDA device");
        TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst slabs must be contiguous");
        TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src/dst dtypes must match");
        TORCH_CHECK(src.dim() >= 1 && dst.dim() >= 1, "src/dst slabs must have >= 1 dimension");
        TORCH_CHECK(src.size(0) >= count, "src slab has fewer rows than count");
        TORCH_CHECK(dst.size(0) >= dst_offset + count, "dst slab out of bounds");

        size_t const src_rb = static_cast<size_t>(src.stride(0)) * src.element_size();
        size_t const dst_rb = static_cast<size_t>(dst.stride(0)) * dst.element_size();
        TORCH_CHECK(src_rb == dst_rb, "src/dst row byte sizes must match");

        src_ptrs[k] = src.data_ptr();
        dst_ptrs[k] = static_cast<uint8_t*>(dst.data_ptr()) + static_cast<size_t>(dst_offset) * dst_rb;
        copy_bytes[k] = static_cast<size_t>(count) * dst_rb;
    }

    c10::cuda::CUDAGuard device_guard(dev);
    cudaStream_t const stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaEvent_t const event = reinterpret_cast<cudaEvent_t>(event_ptr);

    pybind11::gil_scoped_release release_gil;
    for (size_t k = 0; k < num_slabs; ++k) {
        cudaError_t err = cudaMemcpyAsync(
            dst_ptrs[k], src_ptrs[k], copy_bytes[k], cudaMemcpyHostToDevice, stream
        );
        if (err != cudaSuccess) {
            pybind11::gil_scoped_acquire acquire_gil;
            TORCH_CHECK(false, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        }
    }
    cudaError_t ev_err = cudaEventRecord(event, stream);
    if (ev_err != cudaSuccess) {
        pybind11::gil_scoped_acquire acquire_gil;
        TORCH_CHECK(false, "cudaEventRecord failed: ", cudaGetErrorString(ev_err));
    }
}

void promote_experts_cuda(
    std::vector<torch::Tensor> const& cache_slabs,
    torch::Tensor const& gpu_expert_to_slot,
    std::vector<int64_t> const& src_slots,
    std::vector<int64_t> const& dst_slots,
    std::vector<int64_t> const& old_eids,
    std::vector<int64_t> const& new_eids
) {
    size_t const m = src_slots.size();
    if (m == 0 || cache_slabs.empty()) {
        return;
    }
    TORCH_CHECK(m <= kMaxPromote, "Too many promoted experts in single kernel launch");
    TORCH_CHECK(
        dst_slots.size() == m && old_eids.size() == m && new_eids.size() == m,
        "src_slots, dst_slots, old_eids, new_eids must share length"
    );
    TORCH_CHECK(cache_slabs.size() <= kMaxSlabs, "Too many slabs in promote_experts_cuda");
    TORCH_CHECK(gpu_expert_to_slot.is_cuda() && gpu_expert_to_slot.is_contiguous(), "gpu_expert_to_slot must be contiguous CUDA");
    TORCH_CHECK(gpu_expert_to_slot.scalar_type() == torch::kInt32 && gpu_expert_to_slot.dim() == 1, "gpu_expert_to_slot must be 1D int32");

    auto const dev = cache_slabs[0].device();
    TORCH_CHECK(dev.is_cuda() && gpu_expert_to_slot.device() == dev, "All tensors must share the same CUDA device");

    PromoteParams params{};
    params.m = static_cast<int>(m);
    params.gpu_expert_to_slot = gpu_expert_to_slot.data_ptr<int32_t>();
    params.num_experts = static_cast<int32_t>(gpu_expert_to_slot.numel());

    int64_t num_slots = -1;
    size_t max_vec4 = 0;

    for (size_t k = 0; k < cache_slabs.size(); ++k) {
        auto const& t = cache_slabs[k];
        TORCH_CHECK(t.is_cuda() && t.device() == dev && t.is_contiguous(), "cache_slabs must be contiguous on same CUDA device");
        TORCH_CHECK(t.dim() >= 1, "cache_slab must have >= 1 dim");
        if (num_slots < 0) {
            num_slots = t.size(0);
        } else {
            TORCH_CHECK(t.size(0) == num_slots, "All cache_slabs must share dim 0 size");
        }

        size_t const rb = static_cast<size_t>(t.stride(0)) * t.element_size();
        uint8_t* ptr = static_cast<uint8_t*>(t.data_ptr());
        if ((rb % sizeof(int4) == 0) && (reinterpret_cast<uintptr_t>(ptr) % alignof(int4) == 0)) {
            int const idx = params.num_large++;
            params.large_ptrs[idx] = ptr;
            params.large_row_bytes[idx] = rb;
            params.large_vec4_per_row[idx] = rb / sizeof(int4);
            max_vec4 = std::max(max_vec4, params.large_vec4_per_row[idx]);
        } else {
            int const idx = params.num_small++;
            params.small_ptrs[idx] = ptr;
            params.small_row_bytes[idx] = rb;
        }
    }

    for (size_t i = 0; i < m; ++i) {
        TORCH_CHECK(src_slots[i] >= 0 && src_slots[i] < num_slots, "src_slot out of range");
        TORCH_CHECK(dst_slots[i] >= 0 && dst_slots[i] < num_slots, "dst_slot out of range");
        for (size_t j = 0; j < m; ++j) {
            TORCH_CHECK(src_slots[i] != dst_slots[j], "src_slots and dst_slots must be disjoint");
        }
        params.src_slots[i] = static_cast<int>(src_slots[i]);
        params.dst_slots[i] = static_cast<int>(dst_slots[i]);
        params.old_eids[i] = static_cast<int32_t>(old_eids[i]);
        params.new_eids[i] = static_cast<int32_t>(new_eids[i]);
    }

    c10::cuda::CUDAGuard device_guard(dev);
    cudaStream_t const stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();

    constexpr int kBlockSize = 256;
    int const grid = max_vec4 > 0
        ? static_cast<int>(std::clamp<size_t>((max_vec4 + kBlockSize - 1) / kBlockSize, 1, 256))
        : 1;
    promote_experts_kernel<<<grid, kBlockSize, 0, stream>>>(params);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "promote_experts_kernel launch failed: ", cudaGetErrorString(err));
}

__global__ void kv_append_decode_kernel(
    int4 const* __restrict__ k_in,
    int4 const* __restrict__ v_in,
    int4* __restrict__ k_cache,
    int4* __restrict__ v_cache,
    int64_t const* __restrict__ pos_ptr,
    int64_t ring_mod,
    int n_vec4
) {
    int const tid = threadIdx.x;
    if (tid >= n_vec4) {
        return;
    }
    int64_t const pos = *pos_ptr;
    int64_t const slot = (ring_mod > 0) ? (pos % ring_mod) : pos;
    size_t const off = static_cast<size_t>(slot) * static_cast<size_t>(n_vec4) + static_cast<size_t>(tid);
    k_cache[off] = k_in[tid];
    v_cache[off] = v_in[tid];
}

void kv_append_decode_cuda(
    torch::Tensor const& k_in,
    torch::Tensor const& v_in,
    torch::Tensor const& k_cache,
    torch::Tensor const& v_cache,
    torch::Tensor const& pos_buf,
    int64_t ring_mod
) {
    TORCH_CHECK(
        k_in.is_cuda() && v_in.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() && pos_buf.is_cuda(),
        "kv_append_decode_cuda expects CUDA tensors"
    );
    TORCH_CHECK(
        k_in.is_contiguous() && v_in.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() && pos_buf.is_contiguous(),
        "kv_append_decode_cuda expects contiguous tensors"
    );
    auto const dev = k_cache.device();
    TORCH_CHECK(
        k_in.device() == dev && v_in.device() == dev && v_cache.device() == dev && pos_buf.device() == dev,
        "kv_append_decode_cuda tensors must share CUDA device"
    );
    TORCH_CHECK(pos_buf.scalar_type() == torch::kLong && pos_buf.numel() == 1, "pos_buf must be int64[1]");
    TORCH_CHECK(k_in.scalar_type() == k_cache.scalar_type() && v_in.scalar_type() == v_cache.scalar_type(), "KV dtype mismatch");
    TORCH_CHECK(k_cache.dim() >= 2 && v_cache.dim() >= 2 && k_cache.size(0) == v_cache.size(0), "KV cache shape mismatch");

    size_t const k_in_bytes = static_cast<size_t>(k_in.numel()) * k_in.element_size();
    size_t const v_in_bytes = static_cast<size_t>(v_in.numel()) * v_in.element_size();
    size_t const k_row_bytes = static_cast<size_t>(k_cache.stride(0)) * k_cache.element_size();
    size_t const v_row_bytes = static_cast<size_t>(v_cache.stride(0)) * v_cache.element_size();
    TORCH_CHECK(
        k_in_bytes == k_row_bytes && v_in_bytes == v_row_bytes && k_in_bytes == v_in_bytes,
        "KV token and cache slot byte sizes must match"
    );
    TORCH_CHECK(k_in_bytes % sizeof(int4) == 0, "KV slot bytes must be 16-byte aligned");
    TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(k_in.data_ptr()) % alignof(int4) == 0) &&
            (reinterpret_cast<uintptr_t>(v_in.data_ptr()) % alignof(int4) == 0) &&
            (reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % alignof(int4) == 0) &&
            (reinterpret_cast<uintptr_t>(v_cache.data_ptr()) % alignof(int4) == 0),
        "KV pointers must be 16-byte aligned"
    );

    int const n_vec4 = static_cast<int>(k_in_bytes / sizeof(int4));
    TORCH_CHECK(n_vec4 > 0 && n_vec4 <= 256, "Unsupported KV vector count per token");
    int const block_threads = ((n_vec4 + 31) / 32) * 32;

    c10::cuda::CUDAGuard device_guard(dev);
    cudaStream_t const stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();
    kv_append_decode_kernel<<<1, block_threads, 0, stream>>>(
        reinterpret_cast<int4 const*>(k_in.data_ptr()),
        reinterpret_cast<int4 const*>(v_in.data_ptr()),
        reinterpret_cast<int4*>(k_cache.data_ptr()),
        reinterpret_cast<int4*>(v_cache.data_ptr()),
        pos_buf.data_ptr<int64_t>(),
        ring_mod,
        n_vec4
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kv_append_decode_kernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace kolibri::cache
