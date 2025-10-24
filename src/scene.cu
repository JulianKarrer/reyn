#include "scene.cuh"

/// @brief Kernel used by one of the Scene constructors to initialize a set of
/// dynamic particles in a box using CUDA directly to set each position.
__global__ void init_box_kernel(float3 min, float3* x, float3* v, float* m,
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
    x[i] = min + v3(ix * h, iy * h, iz * h);
    // initial velocities are zero
    v[i] = v3(0.f);
    // assume ideal rest mass for now
    m[i] = rho_0 * h * h * h;
}

Scene::Scene(const uint N_desired, const float3 min, const float3 max,
    const float3 bound_min, const float3 bound_max, const float rho_0,
    Particles& state)
{
    // safe info needed for apply step
    this->bound_min = bound_min;
    this->bound_max = bound_max;

    if (((bound_max - bound_min) <= 0.) || ((max - min) <= 0.))
        throw std::invalid_argument(
            "Invalid bounds, minimum is greater than maximum");

    // from volume and desired particle count estimate the desired spacing h
    const float3 dxyz { max - min };
    float h { cbrtf((dxyz.x * dxyz.y * dxyz.z) / (float)N_desired) };
    assert(h >= 0.);
    state.h = h;
    this->h = h;

    const int3 nxyz { floor_div(dxyz, h) };
    // compute actual particle count and save it
    N = { (uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z) };

    if (N == 0)
        throw std::domain_error(
            "Initialization of box failed, zero particles placed");

    state.resize_uninit(N);

    // place particles using cuda
    init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(
        min, state.x.ptr(), state.v.ptr(), state.m.ptr(), nxyz, N, h, rho_0);
    CUDA_CHECK(cudaGetLastError());

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void _hard_enforce_bounds(const float3 bound_min,
    const float3 bound_max, uint N, float3* x, float3* v)
{
    // calculate index and ensure safety at bounds
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // if the point is within the bounding volume, exit early.
    const float3 x_i { x[i] };
    if (bound_min.x <= x_i.x && x_i.x <= bound_max.x && bound_min.y <= x_i.y
        && x_i.y <= bound_max.y && bound_min.z <= x_i.z && x_i.z <= bound_max.z)
        return;
    // if execution reaches this point, find out what violation occured and fix
    // it
    if (x_i.x < bound_min.x) {
        x[i].x = bound_min.x;
        v[i].x *= -1e-1;
    }
    if (x_i.x > bound_max.x) {
        x[i].x = bound_max.x;
        v[i].x *= -1e-1;
    }
    // same for y
    if (x_i.y < bound_min.y) {
        x[i].y = bound_min.y;
        v[i].y *= -1e-1;
    }
    if (x_i.y > bound_max.y) {
        x[i].y = bound_max.y;
        v[i].y *= -1e-1;
    }
    // same for z
    if (x_i.z < bound_min.z) {
        x[i].z = bound_min.z;
        v[i].z *= -1e-1;
    }
    if (x_i.z > bound_max.z) {
        x[i].z = bound_max.z;
        v[i].z *= -1e-1;
    }
}
void Scene::hard_enforce_bounds(Particles& state) const
{
    _hard_enforce_bounds<<<BLOCKS(N), BLOCK_SIZE>>>(
        bound_min, bound_max, N, state.x.ptr(), state.v.ptr());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
};