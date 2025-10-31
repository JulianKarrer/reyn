#include "SESPH.cuh"
#include "doctest/doctest.h"
#include <nanobench.h>

template <IsKernel K, Resort R>
__global__ void _compute_densities(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ m, float* rho, const uint N, const K W,
    const UniformGrid<R> grid)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own position only once
    const float3 x_i { v3(i, xx, xy, xz) };
    // mass weighted kernel sum gives densities
    rho[i] = grid.ff_nbrs(
        xx, xy, xz, i, [=] __device__(auto i, auto j, auto x_ij, auto x_ij_l2) {
            return m[j] * W(x_ij);
        });
}

template <IsKernel K, Resort R>
__global__ void _compute_accelerations_and_integrate(float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, const float* __restrict__ m,
    const float* rho, const uint N, const K W, const float k, const float rho_0,
    const float dt, const float nu, const float h, const UniformGrid<R> grid)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own buffered values only once, in coalescing fashion
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 v_i { v3(i, vx, vy, vz) };
    const float rho_i { rho[i] };
    // compute own pressure once
    const float p_i { fmaxf(0., k * (rho_i / rho_0 - 1.f)) };
    // compute acceleration
    const float3 a_i { grid.ff_nbrs(
        xx, xy, xz, i,
        [=] __device__(auto i, auto j, auto x_ij, auto x_ij_l2) {
            const float rho_j { rho[j] };
            const float m_j { m[j] };
            const float3 v_ij { v_i - v3(j, vx, vy, vz) };
            const float3 dW { W.nabla(x_ij) };
            // compute pressure at j repeatedly instead of accessing it,
            // sincethe kernel is memory-bound
            const float p_j { fmaxf(0., k * (rho[j] / rho_0 - 1.f)) };
            // compute and return the contribution of pair ij to the
            // acceleration at particle i
            return (
                // compute viscous acceleration
                (10.f * nu * m_j / rho_j * dot(v_ij, x_ij)
                    / (dot(x_ij, x_ij) + 0.01 * h * h) * dW)
                // compute contribution to pressure acceleration by particle j
                // at particle i note the minus sign!
                - (m_j * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * dW));
        },
        v3(0., -9.81, 0.)) };

    // use semi-implicit Euler integration to update velocities and positions
    const float3 v_i_new { v_i + dt * a_i };
    store_v3(v_i_new, i, vx, vy, vz);
    store_v3(x_i + dt * v_i_new, i, xx, xy, xz);
}

template <IsKernel K, Resort R>
void SESPH<K, R>::step(
    Particles& state, const UniformGrid<R> grid, const float dt)
{
    // first, compute densities
    _compute_densities<K><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.m.ptr(), rho.ptr(), N, W, grid);
    CUDA_CHECK(cudaGetLastError());
    // and lastly, compute accelerations using these density values
    // note that pressure need not be pre-computed and stored since the kernel
    // is memory-bound, rather compute them on the fly from density at neighbour
    // j also note that viscosity and gravity can be computed on the fly in the
    // inner loop since they require only known velocities and the constant g,
    // so that ∇W_{ij} need only be evaluated once and can be reused for
    // pressure- and non-pressure accelerations
    _compute_accelerations_and_integrate<K>
        <<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(), state.xy.ptr(),
            state.xz.ptr(), state.vx.ptr(), state.vy.ptr(), state.vz.ptr(),
            state.m.ptr(), rho.ptr(), N, W, k, rho_0, dt, nu, h, grid);
    CUDA_CHECK(cudaGetLastError());
};

// explicit instantiation in every relevant translation unit

#define X(K) template class SESPH<K, Resort::no>;
FOREACH_KERNEL(X)
#undef X
#define X(K) template class SESPH<K, Resort::yes>;
FOREACH_KERNEL(X)
#undef X
