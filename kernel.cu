#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "common.h"

// Simple kernel that creates a small grid of particles with some animation
__global__ void generate_points_kernel(float4 *positions, double time, int total_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_points)
        return;

    // 10x10 grid of particles with simple animation
    int grid_size = 10;
    int row = idx / grid_size;
    int col = idx % grid_size;
    float base_x = (col / (float)(grid_size - 1)) * 2.0f - 1.0f;
    float base_y = (row / (float)(grid_size - 1)) * 2.0f - 1.0f;
    // particles move in small circles
    float radius = 0.01f;
    float angle = time + idx * 0.1f;

    positions[idx].x = base_x + radius * cosf(angle);
    positions[idx].y = base_y + radius * sinf(angle);
    positions[idx].z = 0.0f;
    positions[idx].w = 1.0f;
}

extern "C" void launch_kernel(float4 *pos, double t, int total_points)
{
    int blockSize = 256;
    int numBlocks = (total_points + blockSize - 1) / blockSize;

    // launch kernel and check for errors
    generate_points_kernel<<<numBlocks, blockSize>>>(pos, t, total_points);
    CUDA_CHECK(cudaGetLastError());

    // block and wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}