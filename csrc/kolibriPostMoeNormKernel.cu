/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/post_norm/kolibriPostMoeNormKernel.h"

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

// Four-byte packed float container for 32-bit coalesced memory transactions (LDG.E.32 / STG.E.32)
// For BF16/FP16: 2 elements = 4 bytes per thread (coalesces to native 128-byte cache-line per warp)
// For FP32: 1 element = 4 bytes per thread
template <typename T>
struct alignas(4) PackedFloat2;

// Vector specialization for BF16 (2 elements = 4 bytes)
template <>
struct alignas(4) PackedFloat2<__nv_bfloat16>
{
    static constexpr int size = 2;
    __nv_bfloat16 data[2];

    __device__ inline float sum_of_squares() const
    {
        float const f0 = __bfloat162float(data[0]);
        float const f1 = __bfloat162float(data[1]);
        return f0 * f0 + f1 * f1;
    }

    __device__ inline PackedFloat2<__nv_bfloat16> rmsnorm(float s_var, PackedFloat2<__nv_bfloat16> const& gamma) const
    {
        PackedFloat2<__nv_bfloat16> out;
        float const val0 = __bfloat162float(data[0]);
        float const val1 = __bfloat162float(data[1]);
        float const g0 = __bfloat162float(gamma.data[0]);
        float const g1 = __bfloat162float(gamma.data[1]);
        out.data[0] = __float2bfloat16(val0 * s_var * g0);
        out.data[1] = __float2bfloat16(val1 * s_var * g1);
        return out;
    }

    __device__ inline PackedFloat2<__nv_bfloat16> add(PackedFloat2<__nv_bfloat16> const& b) const
    {
        PackedFloat2<__nv_bfloat16> out;
        out.data[0] = __float2bfloat16(__bfloat162float(data[0]) + __bfloat162float(b.data[0]));
        out.data[1] = __float2bfloat16(__bfloat162float(data[1]) + __bfloat162float(b.data[1]));
        return out;
    }
};

// Vector specialization for FP16 (2 elements = 4 bytes)
template <>
struct alignas(4) PackedFloat2<half>
{
    static constexpr int size = 2;
    half data[2];

    __device__ inline float sum_of_squares() const
    {
        float const f0 = __half2float(data[0]);
        float const f1 = __half2float(data[1]);
        return f0 * f0 + f1 * f1;
    }

    __device__ inline PackedFloat2<half> rmsnorm(float s_var, PackedFloat2<half> const& gamma) const
    {
        PackedFloat2<half> out;
        float const val0 = __half2float(data[0]);
        float const val1 = __half2float(data[1]);
        float const g0 = __half2float(gamma.data[0]);
        float const g1 = __half2float(gamma.data[1]);
        out.data[0] = __float2half(val0 * s_var * g0);
        out.data[1] = __float2half(val1 * s_var * g1);
        return out;
    }

    __device__ inline PackedFloat2<half> add(PackedFloat2<half> const& b) const
    {
        PackedFloat2<half> out;
        out.data[0] = __float2half(__half2float(data[0]) + __half2float(b.data[0]));
        out.data[1] = __float2half(__half2float(data[1]) + __half2float(b.data[1]));
        return out;
    }
};

// Vector specialization for FP32 (1 element = 4 bytes)
template <>
struct alignas(4) PackedFloat2<float>
{
    static constexpr int size = 1;
    float data[1];

    __device__ inline float sum_of_squares() const
    {
        return data[0] * data[0];
    }

    __device__ inline PackedFloat2<float> rmsnorm(float s_var, PackedFloat2<float> const& gamma) const
    {
        PackedFloat2<float> out;
        out.data[0] = data[0] * s_var * gamma.data[0];
        return out;
    }

    __device__ inline PackedFloat2<float> add(PackedFloat2<float> const& b) const
    {
        PackedFloat2<float> out;
        out.data[0] = data[0] + b.data[0];
        return out;
    }
};

// Single-barrier block reduction: warp shuffle followed by a NUM_WARPS-warp broadcast read
template <int NUM_WARPS>
__device__ __forceinline__ float blockSumAll(float v, float* red)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        t += red[w];
    }
    return t;
}

// =================================================================================================
// Fused Post-MoE Norm Kernel (4-byte packed float vectorization for 128-byte coalesced warp transfers)
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedAddPostMoeNormRegKernel(
    PackedFloat2<T> const* __restrict__ routed_out,
    PackedFloat2<T> const* __restrict__ shared_out,
    PackedFloat2<T> const* __restrict__ gamma,
    PackedFloat2<T>* __restrict__ moe_out,
    float const eps)
{
    using V = PackedFloat2<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    __shared__ float red[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V c[ITERS];
    V g[ITERS];

    // Standard 4-byte loads: fully coalesced 128-byte warp memory requests
#pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        V routed = routed_out[row + i];
        V shared = shared_out[row + i];
        c[k] = routed.add(shared);
        g[k] = gamma[i];
    }

    // Accumulate sum of squares in registers
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += c[k].sum_of_squares();
    }

    // Intra-block reduction
    float const sum = blockSumAll<NUM_WARPS>(ss, red);
    float const inv = rsqrtf(sum * INV_HIDDEN_DIM + eps);

    // Apply RMS scale and gamma weight, then store directly to global DRAM
#pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_out[row + tid + k * THREADS] = c[k].rmsnorm(inv, g[k]);
    }
}

template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    constexpr int HIDDEN_DIM = 2560;
    TLLM_CHECK_WITH_INFO(hidden_dim == HIDDEN_DIM, "invokeFusedAddPostMoeNorm requires hidden_dim=2560");

    using V = PackedFloat2<T>;
    constexpr int ELEMS_PER_VEC = V::size;
    constexpr int VECS_PER_ROW = HIDDEN_DIM / ELEMS_PER_VEC;

    V const* routed_vec = reinterpret_cast<V const*>(routed_out);
    V const* shared_vec = reinterpret_cast<V const*>(shared_out);
    V const* gamma_vec = reinterpret_cast<V const*>(gamma);
    V* moe_vec = reinterpret_cast<V*>(moe_out);

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>)
    {
        constexpr int NUM_WARPS = 8;
        constexpr int THREADS = NUM_WARPS * 32;
        constexpr int ITERS = VECS_PER_ROW / THREADS; // 1280 / 256 = 5

        dim3 const grid(tokens);
        dim3 const block(THREADS);
        fusedAddPostMoeNormRegKernel<T, NUM_WARPS, ITERS, HIDDEN_DIM><<<grid, block, 0, stream>>>(
            routed_vec, shared_vec, gamma_vec, moe_vec, eps);
    }
    else
    {
        constexpr int NUM_WARPS = 8;
        constexpr int THREADS = NUM_WARPS * 32;
        constexpr int ITERS = VECS_PER_ROW / THREADS; // 2560 / 256 = 10

        dim3 const grid(tokens);
        dim3 const block(THREADS);
        fusedAddPostMoeNormRegKernel<T, NUM_WARPS, ITERS, HIDDEN_DIM><<<grid, block, 0, stream>>>(
            routed_vec, shared_vec, gamma_vec, moe_vec, eps);
    }
    sync_check_cuda_error(stream);
}

template void invokeFusedAddPostMoeNorm<half>(
    half const* routed_out, half const* shared_out, half const* gamma, half* moe_out,
    int const tokens, int const hidden_dim, float const eps, cudaStream_t stream);

#if ENABLE_BF16
template void invokeFusedAddPostMoeNorm<__nv_bfloat16>(
    __nv_bfloat16 const* routed_out, __nv_bfloat16 const* shared_out, __nv_bfloat16 const* gamma, __nv_bfloat16* moe_out,
    int const tokens, int const hidden_dim, float const eps, cudaStream_t stream);
#endif

template void invokeFusedAddPostMoeNorm<float>(
    float const* routed_out, float const* shared_out, float const* gamma, float* moe_out,
    int const tokens, int const hidden_dim, float const eps, cudaStream_t stream);

} // namespace kernels
TRTLLM_NAMESPACE_END
