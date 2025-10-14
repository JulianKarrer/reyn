#ifndef COMMON_H_
#define COMMON_H_

#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define BLOCK_SIZE 256
#define BLOCKS(N) \
    {             \
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE}

/// CUDA error checking macro
#define CUDA_CHECK(code)                      \
    {                                         \
        if (code != cudaSuccess)              \
            throw std::runtime_error(         \
                std::format(                  \
                    "CUDA ERROR: %s %s %d\n", \
                    cudaGetErrorString(code), \
                    __FILE__,                 \
                    __LINE__));               \
    }

#endif // COMMON_H_