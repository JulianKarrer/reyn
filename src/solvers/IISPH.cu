#include "IISPH.cuh"
#include "solver.cuh"

///@brief Compute the diagonal elements 𝔸ᵢᵢ and source terms sᵢ of the linear
/// system 𝔸p=s
///
///@tparam K kernel function type
///@tparam R whether the `UniformGrid` provided left the particle data
/// resorted or not
template <IsKernel K, Resort R>
__global__ void _compute_diagonal_elem_and_source_term(
    const float* __restrict__ xx, const float* __restrict__ xy,
    const float* __restrict__ xz, const float* __restrict__ vx,
    const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ m, const float* __restrict__ ρ,
    float* __restrict__ a_ii, float* __restrict__ s_i, const float rho_0,
    const float dt, const uint N, const K W, const UniformGrid<R> grid,
    const Boundary bdy)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 v_i { v3(i, vx, vy, vz) };

    // compute    Σⱼ mⱼ ∇Wᵢⱼ
    // as well as Σⱼ mⱼ ||∇Wᵢⱼ||² = Σⱼ mⱼ ∇Wᵢⱼ · ∇Wᵢⱼ
    // in the same loop over neighbours, compute the velocity divergence
    // Σⱼ mⱼ vᵢⱼ · ∇Wᵢⱼ
    float vel_div_ff { 0. };
    float sum_grad_sq_no_mi { 0. };
    const float3 grad_ff { grid.ff_nbrs(
        x_i, xx, xy, xz, [&] __device__(auto j, auto x_ij, auto x_ij_l2) {
            const float3 nablaWij { W.nabla(x_ij) };
            const float m_j { m[j] };
            const float3 mj_nablaWij { m_j * nablaWij };
            // accumulate velocity divergence
            vel_div_ff += dot(mj_nablaWij, (v_i - v3(j, vx, vy, vz)));
            // accumulate sum of squared norms (without factor mᵢ)
            sum_grad_sq_no_mi += dot(mj_nablaWij, nablaWij);
            // return scaled gradient
            return mj_nablaWij;
        }) };

    // add  Σₖ mₖ ∇Wᵢₖ to sum of gradients for boundary contribution
    // also compute velocity divergence at boundary:
    // Σₖ mₖ vᵢₖ · ∇Wᵢₖ = Σₖ mₖ vᵢ · ∇Wᵢₖ for vₖ = (0,0,0)^T
    const float3 sum_mk_nablaWik { bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
        [=] __device__(auto k, auto x_ik, auto x_ik_l2) {
            return bdy.m[k] * W.nabla(x_ik);
        }) };
    const float3 grad { grad_ff + sum_mk_nablaWik };
    // calculate the square norm of the gradient sum, weighted with masses
    const float grad_sum_sq { dot(grad, grad) };
    // as well as the sum of gradient square norms, weighted with masses
    // (only for fluid neighbours, "Consistent Boundary Handling" with pₖ=0)
    const float sum_grad_sq { m[i] * sum_grad_sq_no_mi };

    // write the diagonal element 𝔸ᵢᵢ, except for a factor of Δt
    // (numerical reasons, can be a very small value)
    a_ii[i] = -dt * (grad_sum_sq + sum_grad_sq) / (rho_0 * rho_0);

    // now also compute the density invariance source term sᵢ
    const float velocity_divergence { vel_div_ff + dot(v_i, sum_mk_nablaWik) };
    s_i[i] = (rho_0 - ρ[i]) / dt - velocity_divergence;
}

template <IsKernel K, Resort R>
__global__ void _set_pressure_accelerations(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    const float* __restrict__ m, const float* __restrict__ p,
    const float rho_0_sq_inv, const uint N, const K W,
    const UniformGrid<R> grid, const Boundary bdy)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    const float p_i { p[i] };
    // compute pressure acceleration for each particle
    float3 a_i { // by assuming rest density 1/ρ₀² can be pulled all the way out
                 // of the sums
        -rho_0_sq_inv
        * (grid.ff_nbrs(x_i, xx, xy, xz,
               [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                   return m[j] * (p_i + p[j]) * W.nabla(x_ij);
               })
            + p_i // p_i is independent of k and can also be pulled out of the
                  // summation
                * bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
                    [=] __device__(auto k, auto x_ik, auto x_ik_l2) {
                        return bdy.m[k] * W.nabla(x_ik);
                    }))
    };
    // store it back
    store_v3(a_i, i, ax, ay, az);
}

/// @brief Computes the Jacobi update ω (sᵢ - (𝔸p)ᵢ)/𝔸ᵢᵢ if 𝔸ᵢᵢ is sufficiently
/// large, otherwise returns zero
/// @param omega Jacobi weighting factor
/// @param a_ii diagonal system matrix element 𝔸ᵢᵢ
/// @param Ap_i i-th row of the system matrix, multiplied with the pressure
/// vector, i.e. (𝔸p)ᵢ
/// @param s_i i-th source term
/// @return an increment to the pressure iterate (to be used in p_i += ...)
__device__ inline static float jacobi_update(
    const float omega, const float a_ii, const float Ap_i, const float s_i)
{
    // note that a_ii is always negative, so this is a "close to zero" check:
    return (a_ii < 1e-6 && a_ii > -1e-6)
        ? 0.f
        : // if diagonal element is too small, do nothing
        max(0.f, // clamp negative pressures to zero
            omega * (s_i - Ap_i) / a_ii);
}

template <IsKernel K, Resort R>
__global__ void _jacobi_update_pressures(const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    const float* __restrict__ ax, const float* __restrict__ ay,
    const float* __restrict__ az, const float* __restrict__ m,
    float* __restrict__ p, const float* __restrict__ a_ii,
    const float* __restrict__ s_i, uint32_t* __restrict__ ρ_err_indicator,
    const float dt, const float omega, const float ρ_err_threshold,
    const uint N, const K W, const UniformGrid<R> grid, const Boundary bdy)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    const float3 a_i { v3(i, ax, ay, az) };

    // compute the left-hand side of the system of equations: (𝔸p)ᵢ
    const float Ap { dt
        * (grid.ff_nbrs(x_i, xx, xy, xz,
               [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
                   const float3 a_ij { a_i - v3(j, ax, ay, az) };
                   return m[j] * dot(a_ij, W.nabla(x_ij));
               })
            + dot(a_i,
                bdy.grid.ff_nbrs(x_i, bdy.xx, bdy.xy, bdy.xz,
                    [=] __device__(auto k, auto x_ik, auto x_ik_l2) {
                        return bdy.m[k] * W.nabla(x_ik);
                    }))) };
    // perform the Jacobi update
    // pᵢ += ω (sᵢ - (𝔸p)ᵢ)/𝔸ᵢᵢ if 𝔸ᵢᵢ is sufficiently large, otherwise add
    // zero.
    const float s_i_i { s_i[i] };
    p[i] = p[i] + jacobi_update(omega, a_ii[i], Ap, s_i_i);

    // check if the predicted density deviation exceed the threshold and pack
    // the corresponding bit
    const float ρ_err_predicted { (Ap - s_i_i) * dt };
    const uint bitpack_adress { i / 32 }; // divisor of i/32
    const uint32_t bit_pos { i - (bitpack_adress * 32) }; // remainder of i/32
    assert(bit_pos < 32);
    const uint32_t bitmask { (ρ_err_predicted > ρ_err_threshold ? 1u : 0u)
        << bit_pos };
    atomicOr(&(ρ_err_indicator[bitpack_adress]), bitmask);
}

template <IsKernel K, Resort R>
uint IISPH<K, R>::step(Particles& state, const UniformGrid<R> grid,
    const BoundarySamples& bdy, const float dt)
{
    const Boundary bdy_d { bdy.get() };

    // compute densities, which are required for non-pressure accelerations
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

    // integrate non-pressure accelerations only now that they are done being
    // read from
    integrate_a_v_only<<<BLOCKS(N), BLOCK_SIZE>>>(state.vx.ptr(),
        state.vy.ptr(), state.vz.ptr(), ax.ptr(), ay.ptr(), az.ptr(), N, dt);
    CUDA_CHECK(cudaGetLastError());

    // now the diagonal elements 𝔸ᵢᵢ and source terms sᵢ can be computed.
    // this part of the 𝔸p = s linear system is sensible to compute once and
    // store store since it does not change during the iterative solving
    // procedure and recomputation is costly
    _compute_diagonal_elem_and_source_term<<<BLOCKS(N), BLOCK_SIZE>>>(
        state.xx.ptr(), state.xy.ptr(), state.xz.ptr(), state.vx.ptr(),
        state.vy.ptr(), state.vz.ptr(), state.m.ptr(), ρ.ptr(), a_ii.ptr(),
        s_i.ptr(), ρ₀, dt, N, W, grid, bdy_d);
    CUDA_CHECK(cudaGetLastError());

    // initialize pressure values using jacobi update with (𝔸p)ᵢ = 0
    // -> exactly the same result as one iteration after cold start with p=0
    const float* a_ii_d { a_ii.ptr() };
    const float* s_i_d { s_i.ptr() };
    const float omega { ω };
    thrust::transform(thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(N), p.get().begin(),
        [a_ii_d, s_i_d, omega] __device__(uint i) -> float {
            return jacobi_update(omega, a_ii_d[i], 0.f, s_i_d[i]);
        });

    // start iterative pressure solve
    uint l { 0 };
    do {
        // reset density error indicators
        thrust::fill(
            ρ_err_threshold.get().begin(), ρ_err_threshold.get().end(), 0);

        // compute pressure accelerations for current iterate of pressure
        _set_pressure_accelerations<K, R>
            <<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(), state.xy.ptr(),
                state.xz.ptr(), ax.ptr(), ay.ptr(), az.ptr(), state.m.ptr(),
                p.ptr(), 1.f / (ρ₀ * ρ₀), N, W, grid, bdy_d);
        CUDA_CHECK(cudaGetLastError());

        // perform jacobi pressure update
        _jacobi_update_pressures<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(
            state.xx.ptr(), state.xy.ptr(), state.xz.ptr(), ax.ptr(), ay.ptr(),
            az.ptr(), state.m.ptr(), p.ptr(), a_ii.ptr(), s_i.ptr(),
            ρ_err_threshold.ptr(), dt, ω, eta_rho_max * ρ₀, N, W, grid, bdy_d);
        CUDA_CHECK(cudaGetLastError());

        // Log::InfoTagged("IISPH",
        //     "iteration {}, error sum {}, threshold {}, avg prs {}", l,
        //     ρ_err_threshold.sum(), eta_rho_max * ρ₀, p.avg());

        // repeat while predicted density deviation is above target threshold.
        // The sum acts as a check if any of the bitpacked values is different
        // from zero, de facto implementing a maximum reduction
        l += 1;
    } while (l < min_iter || ρ_err_threshold.sum() > 0);

    // set pressure accelerations one last time with final updated pressures
    _set_pressure_accelerations<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), ax.ptr(), ay.ptr(), az.ptr(),
        state.m.ptr(), p.ptr(), 1.f / (ρ₀ * ρ₀), N, W, grid, bdy_d);
    CUDA_CHECK(cudaGetLastError());

    // integrate accelerations to velocitites, then velocitites to positions
    integrate<<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(), state.xy.ptr(),
        state.xz.ptr(), state.vx.ptr(), state.vy.ptr(), state.vz.ptr(),
        ax.ptr(), ay.ptr(), az.ptr(), N, dt);
    CUDA_CHECK(cudaGetLastError());

    // return the iteration count
    return l;
};

#define X(K) template class IISPH<K, Resort::no>;
FOREACH_KERNEL(X)
#undef X
#define X(K) template class IISPH<K, Resort::yes>;
FOREACH_KERNEL(X)
#undef X
