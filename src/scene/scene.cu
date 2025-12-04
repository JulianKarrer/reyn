#include "scene/scene.cuh"
#include "scene/sample_boundary.cuh"
#include "utils/vector.cuh"
#include "scene/loader.h"
#include <filesystem>
#include <concepts>
#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/transform.h>

/// @brief Kernel used by one of the Scene constructors to initialize a set of
/// dynamic particles in a box using CUDA directly to set each position.
__global__ void _init_box_kernel(float3 min, float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ m,
    int3 nxyz, uint N, float h, float ρ₀)
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
    m[i] = ρ₀ * h * h * h;
}

__global__ void _cull_stencil(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ bx, const float* __restrict__ by,
    const float* __restrict__ bz, const float cull_sq,
    unsigned char* __restrict__ stencil, UniformGrid<Resort::yes> bdy_grid,
    uint N)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    // count the number of boundary particles within the cull radius
    const int res { bdy_grid.ff_nbrs(
        x_i, bx, by, bz, [&] __device__(uint j, float3 x_ij, float x_ij_l2) {
            return x_ij_l2 <= cull_sq ? 1 : 0;
        }) };
    // if any such neighbour was counted, indicate with a 1
    stencil[i] = res == 0 ? 0 : 1;
}

/// @brief Binary operator functor used by thrust to add pseudorandom zero-mean
/// values of given standard deviation to existing values using
/// `thrust::transform`, where an `N_offset` of the total number of threads that
/// used this functor previously can be used to ensure fresh values are chosen
/// in each invocation
struct GenJitter {
    float jitter_stddev;
    uint N_offset;
    __device__ float operator()(uint idx, float current)
    {
        thrust::default_random_engine rand;
        thrust::normal_distribution<float> uni(0.f, jitter_stddev);
        rand.discard(idx + N_offset);
        return current + uni(rand);
    }
};

Scene::Scene(const std::filesystem::path& path, const uint N_desired,
    const float3 min, const float3 max, const float _rho_0, Particles& state,
    DeviceBuffer<float>& tmp, const float bdy_oversampling_factor,
    const float cull_bdy_radius, const float jitter_stddev)
    : ρ₀(_rho_0)
    , h(cbrtf(prod(max - min) / (float)N_desired))
    , bdy([&]() {
// ignore argument count error for default arguments in cuh header for function
// `sample_mesh`
#ifdef __INTELLISENSE__
#pragma diag_suppress 165
#endif
        // load mesh and sample it
        return sample_mesh(
            load_mesh_from_obj(path), h, _rho_0, bdy_oversampling_factor);
#ifdef __INTELLISENSE__
#pragma diag_default 165
#endif
    }())
    // ensure the scene bounds are those of the boundary samples
    , bound_min(bdy.bound_min)
    , bound_max(bdy.bound_max)
    , grid_builder(UniformGridBuilder(bound_min, bound_max, 2.f * h))
{
    // compute preliminary particle count and save it
    const float3 dxyz { max - min };
    const int3 nxyz { floor_div(dxyz, h) };
    N = { (uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z) };
    // exit early if the fluid domain is empty
    if (N == 0)
        throw std::domain_error(
            "Initialization of box failed, zero particles placed");

    state.resize_uninit(N);

    // place particles using cuda
    _init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(min, state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), state.m.ptr(), nxyz, N, h, ρ₀);
    CUDA_CHECK(cudaGetLastError());

    // remove particles that intersect with the boundary
    // for this purpose, generate a grid
    const auto grid { grid_builder.construct_and_reorder(2. * h, tmp, state) };
    // get relevant pointers
    const auto xs { state.xx.ptr() };
    const auto ys { state.xy.ptr() };
    const auto zs { state.xz.ptr() };
    const auto bx { bdy.xs.ptr() };
    const auto by { bdy.ys.ptr() };
    const auto bz { bdy.zs.ptr() };
    const float cull_sq { cull_bdy_radius * cull_bdy_radius * h * h };
    DeviceBuffer<unsigned char> stencil(N);

    _cull_stencil<<<BLOCKS(N), BLOCK_SIZE>>>(
        xs, ys, zs, bx, by, bz, cull_sq, stencil.ptr(), bdy.grid, N);

    auto dx = thrust::device_pointer_cast(xs);
    auto dy = thrust::device_pointer_cast(ys);
    auto dz = thrust::device_pointer_cast(zs);
    auto first = thrust::make_zip_iterator(thrust::make_tuple(dx, dy, dz,
        state.vx.get().begin(), state.vy.get().begin(), state.vz.get().begin(),
        state.m.get().begin()));
    auto last = first + N;
    auto new_end { thrust::remove_if(thrust::device, first, last,
        stencil.get().begin(), ::cuda::std::identity {}) };
    uint new_N = thrust::distance(first, new_end);

    if (new_N != N) {
        Log::Warn("Culled particles closer than {}h to boundary: {} -> {}",
            cull_bdy_radius, N, new_N);
    }

    // resize tmp buffer to new size to save each component of externally
    // managed memory before resizing it
    state.resize_truncate(new_N, tmp);

    N = new_N;

    // add a pseudorandom jitter to the coordinates
    auto ndx { thrust::device_pointer_cast(state.xx.ptr()) };
    auto ndy { thrust::device_pointer_cast(state.xy.ptr()) };
    auto ndz { thrust::device_pointer_cast(state.xz.ptr()) };
    const float stddev { jitter_stddev * h };
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(new_N), ndx, ndx,
        GenJitter { stddev, 0 });
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(new_N), ndy, ndy,
        GenJitter { stddev, new_N });
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(new_N), ndz, ndz,
        GenJitter { stddev, 2 * new_N });

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    Log::Success(
        "Scene initialized:\tN={}, Nbdy={}, h={}", N, bdy.xs.size(), h);
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
