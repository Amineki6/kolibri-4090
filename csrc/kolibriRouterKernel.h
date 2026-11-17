#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kolibri {
namespace kernels {

template <typename T>
void invokeKolibriRouter(
    T const* scores_in,
    float const* bias,
    int32_t* out_indices,
    float* out_weights,
    int const num_tokens,
    cudaStream_t stream
);

} // namespace kernels
} // namespace kolibri
