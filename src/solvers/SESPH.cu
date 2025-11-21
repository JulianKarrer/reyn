#include "SESPH.cuh"
#include "doctest/doctest.h"
#include <nanobench.h>
#include "solvers/solver.cuh"

__device__ inline static float eos(
    const float rho_i, const float k, const float rho_0)
{
    return fmaxf(0., k * (rho_i / rho_0 - 1.f)); // normalize for constant k
    // return fmaxf(0., k * (rho_i - rho_0));
    // return k * (rho_i - rho_0);
}

template <IsKernel K, Resort R>
__global__ void _compute_accelerations(float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ ax,
    float* __restrict__ ay, float* __restrict__ az, const float* __restrict__ m,
    const float* ρ, const uint N, const K W, const float k, const float rho_0,
    const float nu, const float h, const float3 g, const UniformGrid<R> grid,
    const Boundary bdy)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own buffered values only once, in coalescing fashion
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 v_i { v3(i, vx, vy, vz) };
    const float rho_i { ρ[i] };
    // compute own pressure once
    const float p_i { eos(rho_i, k, rho_0) };
    const float p_i_over_rho_i_sq { p_i / (rho_i * rho_i) };
    // compute acceleration
    const float3 a_i {
        grid.ff_nbrs(
            x_i, xx, xy, xz,
            [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                const float rho_j { ρ[j] };
                const float m_j { m[j] };
                const float3 v_ij { v_i - v3(j, vx, vy, vz) };
                const float3 dW { W.nabla(x_ij) };
                // compute pressure at j repeatedly instead of accessing it,
                // sincethe kernel is memory-bound
                const float p_j { eos(rho_j, k, rho_0) };
                // compute and return the contribution of pair ij to the
                // acceleration at particle i
                return (
                    // compute viscous acceleration
                    (10.f * nu * m_j / rho_j * dot(v_ij, x_ij)
                        / (dot(x_ij, x_ij) + 0.01 * h * h) * dW)
                    // compute contribution to pressure acceleration by particle
                    // j at particle i note the minus sign!
                    - (m_j * (p_i_over_rho_i_sq + p_j / (rho_j * rho_j)) * dW));
            },
            g)
        - p_i_over_rho_i_sq
            * bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
                [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                    return bdy.m[j] * W.nabla(x_ij);
                })
    };
    // store the final acceleration
    store_v3(a_i, i, ax, ay, az);
}

template <IsKernel K, Resort R>
void SESPH<K, R>::step(Particles& state, const UniformGrid<R> grid,
    const BoundarySamples& bdy, const float dt)
{
    // first, compute densities
    compute_densities<K><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.m.ptr(), ρ.ptr(), N, W, grid,
        bdy.get());
    CUDA_CHECK(cudaGetLastError());
    // compute accelerations using these density values
    // note that pressure need not be pre-computed and stored since the
    // kernel is memory-bound, rather compute them on the fly from density
    // at neighbour j also note that viscosity and gravity can be computed
    // on the fly in the inner loop since they require only known velocities
    // and the constant g, so that ∇W_{ij} need only be evaluated once and
    // can be reused for pressure- and non-pressure accelerations
    _compute_accelerations<K><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), ax.ptr(), ay.ptr(), az.ptr(), state.m.ptr(), ρ.ptr(), N,
        W, k, rho_0, nu, h, g, grid, bdy.get());
    CUDA_CHECK(cudaGetLastError());
    // integration needs to happen in a seperate step, since reading and
    // writing positions and velocites without global synchronization would
    // elad to race conditions
    integrate<<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(), state.xy.ptr(),
        state.xz.ptr(), state.vx.ptr(), state.vy.ptr(), state.vz.ptr(),
        ax.ptr(), ay.ptr(), az.ptr(), N, dt);
    CUDA_CHECK(cudaGetLastError());
};

// explicit instantiation in every relevant translation unit

#define X(K) template class SESPH<K, Resort::no>;
FOREACH_KERNEL(X)
#undef X
#define X(K) template class SESPH<K, Resort::yes>;
FOREACH_KERNEL(X)
#undef X
