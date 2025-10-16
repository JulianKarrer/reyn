#include "kernels.cuh"

__host__ __device__ __forceinline__ float C3::w(float q) const
{
    const float t1{sat(1.f - q)};
    const float t2{sat(0.5f - q)};
    return t1 * t1 * t1 - 4.f * t2 * t2 * t2;
}

__host__ __device__ __forceinline__ float C3::dw(float q) const
{
    const float t1{sat(1.f - q)};
    const float t2{sat(0.5f - q)};
    return -3.f * t1 * t1 + 12.f * t2 * t2;
}
