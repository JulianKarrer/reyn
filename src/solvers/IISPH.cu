#include "IISPH.cuh"
#include "solver.cuh"

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

    uint l { 0 };
    do {
        // compute predicted densities (current + velocity divergence)
        predict_rho<K, R><<<BLOCKS(N), BLOCK_SIZE>>>(state.xx.ptr(),
            state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
            state.vz.ptr(), dt, ρ₀, state.m.ptr(), ρ.ptr(), N, W, grid, bdy_d);
        CUDA_CHECK(cudaGetLastError());

        // compute pressure acceleration

        // repeat while maximum density is above target threshold
        l += 1;
    } while (l < min_iter || ρ.avg() > eta_rho_max * ρ₀);

    // incorporate pressure accelerations and update positions from
    // accumulated  velocity
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
