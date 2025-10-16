#ifndef COMMON_H_
#define COMMON_H_

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <format>    // for std::format
#include <stdexcept> // for std::runtime_error

#define BLOCK_SIZE 256
#define BLOCKS(N) \
    {             \
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE}

/// CUDA error checking macro
/// https://stackoverflow.com/questions/14038589/
#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        // std::string error{std::format("CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__)};
        // throw std::runtime_error(error);
        fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif // COMMON_H_