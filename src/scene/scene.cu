#include "scene/scene.cuh"
#include "scene/sample_boundary.cuh"
#include <filesystem>
#include <concepts>
#include "vector_helper.cuh"
#include "scene/loader.h"

/// @brief Kernel used by one of the Scene constructors to initialize a set of
/// dynamic particles in a box using CUDA directly to set each position.
__global__ void init_box_kernel(float3 min, float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ m,
    int3 nxyz, uint N, float h, float rho_0)
{
    // calculate 3d index from 1d index of invocation and nx, ny limits
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    auto nx { nxyz.x };
    auto ny { nxyz.y };
    auto ix { i };
    auto iz { ix / (nx * ny) };
    ix -= iz * nx * ny;
    auto iy { ix / nx };
    ix -= iy * nx;

    // then initialize positions with spacing h
    store_v3(min + v3(ix * h, iy * h, iz * h), i, xx, xy, xz);
    // initial velocities are zero
    store_v3(v3(0.f), i, vx, vy, vz);
    // assume ideal rest mass for now
    m[i] = rho_0 * h * h * h;
}

Scene::Scene(const std::filesystem::path& path, const uint N_desired,
    const float3 min, const float3 max, const float rho_0, Particles& state,
    const float bdy_oversampling_factor)
    : h([&]() {
        // from volume and desired particle count estimate the desired spacing
        // h
        const float3 dxyz { max - min };
        const float h { cbrtf((dxyz.x * dxyz.y * dxyz.z) / (float)N_desired) };
        return cbrtf((dxyz.x * dxyz.y * dxyz.z) / (float)N_desired);
    }())
    , bdy(sample_mesh(
          load_mesh_from_obj(path), h, rho_0, bdy_oversampling_factor))
{
    this->rho_0 = rho_0;

    // compute actual particle count and save it
    const float3 dxyz { max - min };
    const int3 nxyz { floor_div(dxyz, h) };
    N = { (uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z) };
    // exit early if the fluid domain is empty
    if (N == 0)
        throw std::domain_error(
            "Initialization of box failed, zero particles placed");

    // ensure the scene bounds are those of the boundary samples
    this->bound_min = bdy.bound_min;
    this->bound_max = bdy.bound_max;

    state.resize_uninit(N);

    // place particles using cuda
    init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(min, state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), state.m.ptr(), nxyz, N, h, rho_0);
    CUDA_CHECK(cudaGetLastError());

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void _hard_enforce_bounds(const float3 bound_min,
    const float3 bound_max, uint N, float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz)
{
    // calculate index and ensure safety at bounds
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    // if at the boundary, mirror velocity and damp it
    if (x_i <= bound_min || bound_max <= x_i) {
        const float3 v_i { v3(i, vx, vy, vz) };
        store_v3(-0.1 * v_i, i, vx, vy, vz);
    }
    // always clamp positions to the bounds
    store_v3(max(min(x_i, bound_max), bound_min), i, xx, xy, xz);
}

void Scene::hard_enforce_bounds(Particles& state) const
{
    _hard_enforce_bounds<<<BLOCKS(N), BLOCK_SIZE>>>(bound_min, bound_max, N,
        state.xx.ptr(), state.xy.ptr(), state.xz.ptr(), state.vx.ptr(),
        state.vy.ptr(), state.vz.ptr());
    CUDA_CHECK(cudaGetLastError());
};
