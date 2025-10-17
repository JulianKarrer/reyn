#include "scene.cuh"

Scene::Scene(uint N_desired, float3 min, float3 max, float3 bound_min, float3 bound_max, float rho_0, Particles &p)
{
    // safe info needed for apply step
    this->bound_min = bound_min;
    this->bound_max = bound_max;

    if (!((bound_max - bound_min) >= 0.) || !((max - min) >= 0.))
        throw std::invalid_argument("Invalid bounds, minimum is greater than maximum");

    // from volume and desired particle count estimate the desired spacing h
    const float3 dxyz{max - min};
    float h{cbrtf((dxyz.x * dxyz.y * dxyz.z) / (float)N_desired)};
    assert(h >= 0.);

    const int3 nxyz{floor_div(dxyz, h)};
    // compute actual particle count and save it
    N = {(uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z)};

    if (N == 0)
        throw std::domain_error("Initialization of box failed, zero particles placed");

    p.resize_uninit(N);

    // place particles using cuda
    init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(min, p.x, p.v, p.m, nxyz, N, h, rho_0);
    CUDA_CHECK(cudaGetLastError());

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void init_box_kernel(float3 min, float3 *x, float3 *v, float *m, int3 nxyz, uint N, float h, float rho_0)
{
    // calculate 3d index from 1d index of invocation and nx, ny limits
    auto i{blockIdx.x * blockDim.x + threadIdx.x};
    if (i >= N)
        return;
    auto nx{nxyz.x};
    auto ny{nxyz.y};
    auto ix{i};
    auto iz{ix / (nx * ny)};
    ix -= iz * nx * ny;
    auto iy{ix / nx};
    ix -= iy * nx;

    // then initialize positions with spacing h
    x[i] = min + v3(ix * h, iy * h, iz * h);
    // initial velocities are zero
    v[i] = v3(0.f);
    // assume ideal rest mass for now
    m[i] = rho_0 * h * h * h;
}