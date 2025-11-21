///
///@file PCISPH.cu
///@author Julian Karrer (github.com/JulianKarrer)
///@brief
///@version 0.1
///@date 2025-11-21
///
///@copyright Copyright (c) 2025
///
///

#include "PCISPH.cuh"
#include "solvers/solver.cuh"

__device__ inline static float _eos_pcisph(
    const float rho_i, const float k, const float rho_0)
{
    return fmaxf(0., k * (rho_i - rho_0));
}

template <IsKernel K, Resort R>
__global__ void _write_pcisph_prs_acc(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    const float* __restrict__ m, const float* ρ, const uint N, const K W,
    const float k, const float rho_0, const float rho_0_sq_inv,
    const UniformGrid<R> grid, const Boundary bdy)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    const float rho_i { ρ[i] };
    const float p_i { _eos_pcisph(rho_i, k, rho_0) };
    const float p_i_over_rho_i_sq { p_i / (rho_i * rho_i) };
    const float p_i_over_rho_0_sq { p_i * rho_0_sq_inv };
    const float3 a_i {
        -grid.ff_nbrs(x_i, xx, xy, xz,
            [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                const float rho_j { ρ[j] };
                const float m_j { m[j] };
                const float p_j { _eos_pcisph(rho_j, k, rho_0) };
                return (
                    // compute contribution to pressure acceleration by particle
                    // j at particle i
                    (m_j * (p_i_over_rho_i_sq + p_j / (rho_j * rho_j))
                        * W.nabla(x_ij)));
            })
        - 0.5 * (p_i_over_rho_i_sq + p_i_over_rho_0_sq)
            * bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
                [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                    return bdy.m[j] * W.nabla(x_ij);
                })
    };
    store_v3(a_i, i, ax, ay, az);
}

template <IsKernel K, Resort R>
void PCISPH<K, R>::step(Particles& state, const UniformGrid<R> grid,
    const BoundarySamples& bdy, const float dt)
{
    const Boundary bdy_d { bdy.get() };
    // compute PCISPH stiffness coefficient now that Δt is known
    float k = (float)((δΔt² / dt) / dt);
    // float k = 25.f;
    // compute densities, since they are required for computation of viscosity
    compute_densities<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.m.ptr(), ρ.ptr(), N, W, grid,
        bdy_d);
    CUDA_CHECK(cudaGetLastError());
    // compute non-pressure accelerations (gravity, fluid-fluid viscosity)
    write_non_prs_acc_ff<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), ax.ptr(), ay.ptr(), az.ptr(), h, nu, g, state.m.ptr(),
        ρ.ptr(), N, W, grid);
    CUDA_CHECK(cudaGetLastError());
    uint l { 0 };
    do {
        // accumulate those accelerations into the current velocity
        integrate_a_v_only<<<BLOCKS(N), BLOCK_SIZE>>>(state.vx.ptr(),
            state.vy.ptr(), state.vz.ptr(), ax.ptr(), ay.ptr(), az.ptr(), N,
            dt);
        CUDA_CHECK(cudaGetLastError());
        // compute predicted densities (current + velocity divergence)
        predict_rho<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
            state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
            state.vz.ptr(), dt, ρ₀, state.m.ptr(), ρ.ptr(), N, W, grid, bdy_d);
        CUDA_CHECK(cudaGetLastError());
        // compute pressure acceleration
        _write_pcisph_prs_acc<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
            state.xy.ptr(), state.xz.ptr(), ax.ptr(), ay.ptr(), az.ptr(),
            state.m.ptr(), ρ.ptr(), N, W, k, ρ₀, 1.f / (ρ₀ * ρ₀), grid, bdy_d);
        // repeat while maximum density is above target threshold
        l += 1;
    } while (l < min_iter || ρ.avg() > eta_rho_max * ρ₀);
    // incorporate pressure accelerations and update positions from accumulated
    // velocity
    integrate<<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(), state.xy.ptr(),
        state.xz.ptr(), state.vx.ptr(), state.vy.ptr(), state.vz.ptr(),
        ax.ptr(), ay.ptr(), az.ptr(), N, dt);
    CUDA_CHECK(cudaGetLastError());
};

#define X(K) template class PCISPH<K, Resort::no>;
FOREACH_KERNEL(X)
#undef X
#define X(K) template class PCISPH<K, Resort::yes>;
FOREACH_KERNEL(X)
#undef X
