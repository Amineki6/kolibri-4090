#include "kolibriRouterKernel.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

namespace cg = cooperative_groups;

namespace kolibri {
namespace kernels {

static constexpr int WARP_SIZE = 32;
static constexpr int NUM_EXPERTS = 384;
static constexpr int TOP_K = 6;
static constexpr int EXPERTS_PER_THREAD = NUM_EXPERTS / WARP_SIZE; // 12
static constexpr int PADDED_EXPERTS_PER_THREAD = 16;                // power of 2 for bitonic sort
static constexpr int WARPS_PER_BLOCK = 8;

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <typename T>
__global__ void kolibriRouterKernel(
    T const* __restrict__ scores_in,
    float const* __restrict__ bias,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_weights,
    int const num_tokens
) {
    __shared__ float smem_raw_scores[WARPS_PER_BLOCK][NUM_EXPERTS];

    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id = (threadIdx.x / WARP_SIZE) % WARPS_PER_BLOCK;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= num_tokens) {
        return;
    }

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_raw_scores[local_warp_id];
    T const* scores_row = scores_in + static_cast<size_t>(global_warp_id) * NUM_EXPERTS;

    float biased_scores[PADDED_EXPERTS_PER_THREAD];
    int32_t indices[PADDED_EXPERTS_PER_THREAD];

    #pragma unroll
    for (int e = 0; e < EXPERTS_PER_THREAD; ++e) {
        int const expert_id = lane_id + e * WARP_SIZE;
        float const raw_score = to_float<T>(scores_row[expert_id]);
        float const b = bias ? bias[expert_id] : 0.0f;
        biased_scores[e] = raw_score + b;
        indices[e] = expert_id;
        my_smem[expert_id] = raw_score;
    }

    #pragma unroll
    for (int e = EXPERTS_PER_THREAD; e < PADDED_EXPERTS_PER_THREAD; ++e) {
        biased_scores[e] = -1e30f;
        indices[e] = 65535;
    }
    __syncwarp();

    float topk_values[TOP_K];
    int32_t topk_indices[TOP_K];
    constexpr float minValue = -1e30f;
    tensorrt_llm::kernels::reduce_topk::reduceTopK<TOP_K, float, PADDED_EXPERTS_PER_THREAD>(
        warp, topk_values, topk_indices, biased_scores, indices, minValue, TOP_K
    );

    if (lane_id < TOP_K) {
        int32_t const expert_id = topk_indices[lane_id];
        float const raw_val = my_smem[expert_id];
        float const s = 1.0f / (1.0f + __expf(-raw_val));
        size_t const out_offset = static_cast<size_t>(global_warp_id) * TOP_K + lane_id;
        out_indices[out_offset] = expert_id;
        out_weights[out_offset] = s;
    }
}

template <typename T>
void invokeKolibriRouter(
    T const* scores_in,
    float const* bias,
    int32_t* out_indices,
    float* out_weights,
    int const num_tokens,
    cudaStream_t stream
) {
    constexpr int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int const blocks = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    kolibriRouterKernel<T><<<blocks, threads_per_block, 0, stream>>>(
        scores_in, bias, out_indices, out_weights, num_tokens
    );
}

template void invokeKolibriRouter<__nv_bfloat16>(
    __nv_bfloat16 const*, float const*, int32_t*, float*, int const, cudaStream_t
);
template void invokeKolibriRouter<half>(
    half const*, float const*, int32_t*, float*, int const, cudaStream_t
);
template void invokeKolibriRouter<float>(
    float const*, float const*, int32_t*, float*, int const, cudaStream_t
);

} // namespace kernels
} // namespace kolibri
