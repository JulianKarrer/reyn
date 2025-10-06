#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "common.h"

// Simple kernel that creates a small grid of particles with some animation
__global__ void generate_points_kernel(float3 *x, double time, int total_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_points)
        return;

    // 10x10 grid of particles with simple animation
    int grid_size = 10;
    int row = idx / grid_size;
    int col = idx % grid_size;
    float radius = 0.1f;
    float base_x = (2. * radius) * (float)col - 1.f;
    float base_y = (2. * radius) * (float)row - 1.f;
    // particles move in small circles
    float angle = time + idx * 0.1f;

    x[idx].x = base_x + radius * cosf(angle);
    x[idx].y = base_y; //+ radius * sinf(angle);
    x[idx].z = 0.;     // radius * sinf(angle);
}

extern "C" void launch_kernel(float3 *x, double t, int total_points)
{
    int blockSize = 256;
    int numBlocks = (total_points + blockSize - 1) / blockSize;

    // launch kernel and check for errors
    generate_points_kernel<<<numBlocks, blockSize>>>(x, t, total_points);
    CUDA_CHECK(cudaGetLastError());

    // block and wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}