#ifndef COMMON_H_
#define COMMON_H_

#include <cstdio>

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
        fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif // COMMON_H_