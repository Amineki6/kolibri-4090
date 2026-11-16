#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#define sync_check_cuda_error(stream)                                      \
    do {                                                                   \
        (void)(stream);                                                    \
        cudaError_t _err = cudaGetLastError();                             \
        if (_err != cudaSuccess) {                                         \
            throw std::runtime_error(cudaGetErrorString(_err));            \
        }                                                                  \
    } while (0)
