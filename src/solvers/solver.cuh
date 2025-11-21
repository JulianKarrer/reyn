#ifndef SOLVERS_SOLVER_CUH_
#define SOLVERS_SOLVER_CUH_

#include <particles.cuh>
#include "kernels.cuh"
#include "scene/scene.cuh"

__global__ void integrate(float* __restrict__ xx, float* __restrict__ xy,
    float* __restrict__ xz, float* __restrict__ vx, float* __restrict__ vy,
    float* __restrict__ vz, float* __restrict__ ax, float* __restrict__ ay,
    float* __restrict__ az, const uint N, const float dt);

__global__ void integrate_a_v_only(float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ ax,
    float* __restrict__ ay, float* __restrict__ az, const uint N,
    const float dt);

template <IsKernel K, Resort R>
__global__ void compute_densities(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ m, float* ρ, const uint N, const K W,
    const UniformGrid<R> grid, const Boundary bdy)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own position only once
    const float3 x_i { v3(i, xx, xy, xz) };
    // mass weighted kernel sum gives densities
    ρ[i] = grid.ff_nbrs(x_i, xx, xy, xz,
               [=] __device__(
                   auto j, auto x_ij, auto x_ij_l2) { return m[j] * W(x_ij); })
        + bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
            [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                return bdy.m[j] * W(x_ij);
            });
}

template <IsKernel K, Resort R>
__global__ void write_non_prs_acc_ff(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ vx, const float* __restrict__ vy,
    const float* __restrict__ vz, float* __restrict__ ax,
    float* __restrict__ ay, float* __restrict__ az, const float h,
    const float nu, const float3 g, const float* __restrict__ m, float* ρ,
    const uint N, const K W, const UniformGrid<R> grid)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own position and velocity only once
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 v_i { v3(i, vx, vy, vz) };
    // initialize with gravitation acceleration, sum viscosity contributions
    const float3 a_vis_g_i { //
        grid.ff_nbrs(
            x_i, xx, xy, xz,
            [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                const float3 v_ij { v_i - v3(j, vx, vy, vz) };
                return (10.f * nu * m[j] / ρ[j] * dot(v_ij, x_ij)
                    / (dot(x_ij, x_ij) + 0.01 * h * h) * W.nabla(x_ij));
            },
            g)
    };
    // store the computed acceleration, overwriting the current entries
    store_v3(a_vis_g_i, i, ax, ay, az);
}

template <IsKernel K, Resort R>
__global__ void predict_rho(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ vx, const float* __restrict__ vy,
    const float* __restrict__ vz, const float dt, const float rho_0,
    const float* __restrict__ m, float* ρ, const uint N, const K W,
    const UniformGrid<R> grid, const Boundary bdy)
{
    // compute index and ensure safety at bounds
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // read own position and velocity only once
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 v_i { v3(i, vx, vy, vz) };

    // write predicted density
    ρ[i] = grid.ff_nbrs(x_i, xx, xy, xz,
               [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                   const float3 v_ij { v_i - v3(j, vx, vy, vz) };
                   const float m_j { m[j] };
                   // compute current density
                   return m_j // factor out neighbour's mass
                       * (W(x_ij)
                           // add change in density due to velocity divergence
                           + dt * dot(W.nabla(x_ij), v_ij));
               })
        + bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
            [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                // same thing for boundaries, except v_ij = v_i is assumed if
                // boundary is static
                const float m_j { bdy.m[j] };
                return m_j * (W(x_ij) + dt * dot(W.nabla(x_ij), v_i));
            });
}

#endif // SOLVERS_SOLVER_CUH_