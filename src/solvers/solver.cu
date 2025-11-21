#include "solvers/solver.cuh"

__global__ void integrate_a_v_only(float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ ax,
    float* __restrict__ ay, float* __restrict__ az, const uint N,
    const float dt)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 v_i { v3(i, vx, vy, vz) };
    store_v3(v_i + dt * v3(i, ax, ay, az), i, vx, vy, vz);
}

__global__ void integrate(float* __restrict__ xx, float* __restrict__ xy,
    float* __restrict__ xz, float* __restrict__ vx, float* __restrict__ vy,
    float* __restrict__ vz, float* __restrict__ ax, float* __restrict__ ay,
    float* __restrict__ az, const uint N, const float dt)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // load relevant values to memory
    const float3 a_i { v3(i, ax, ay, az) };
    const float3 v_i { v3(i, vx, vy, vz) };
    const float3 x_i { v3(i, xx, xy, xz) };
    // use semi-implicit Euler integration to update velocities and
    // positions
    const float3 v_i_new { v_i + dt * a_i };
    store_v3(v_i_new, i, vx, vy, vz);
    store_v3(x_i + dt * v_i_new, i, xx, xy, xz);
}
