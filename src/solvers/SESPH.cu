#include "SESPH.cuh"

template <IsKernel K>
__global__ void _compute_densities(float3 *x, float *m, float* rho, uint N, K W){
    // compute index and ensure safety at bounds
    const auto i{blockIdx.x * blockDim.x + threadIdx.x};
    if (i >= N)
        return;
    // read own position only once
    const float3 x_i{x[i]};
    // mass weighted kernel sum gives densities
    float rho_i{0.};
    for (uint j{0}; j<N; ++j){
        rho_i += m[j] * W(x_i - x[j]);
    }
    rho[i] = rho_i;
}

template <IsKernel K>
__global__ void _compute_accelerations_and_integrate(float3 *x, float3 *v,  float *m, float* rho, uint N, const K W, const float k, const float rho_0, const float dt, const float nu, const float h){
    // compute index and ensure safety at bounds
    const auto i{blockIdx.x * blockDim.x + threadIdx.x};
    if (i >= N)
        return;
    // read own buffered values only once, in coalescing fashion
    const float3 x_i{x[i]};
    const float3 v_i{v[i]};
    const float rho_i{rho[i]};
    // compute own pressure once
    const float p_i{fmaxf(0., k * (rho_i/rho_0 - 1.f))};
    // initialize acceleration, resetting previously held value at i
    float3 a_i{v3(0., -9.81, 0.)};
    for (uint j{0}; j<N; ++j){
        const float rho_j{rho[j]};
        const float m_j{m[j]};
        const float3 v_ij{v_i - v[j]};
        const float3 x_j{x[j]};
        const float3 dW{W.nabla(x_i - x_j)};
        const float3 x_ij{x_i - x_j};
        // compute viscous acceleration
        a_i += (
            10.f * nu 
            * m_j/rho_j 
            * dot(v_ij, x_ij) / (dot(x_ij, x_ij) + 0.01*h*h)
            * dW
        );
        // compute pressure at j repeatedly instead of accessing it, since
        // the kernel is memory-bound
        const float p_j{fmaxf(0., k * (rho[j]/rho_0 - 1.f))};
        // compute contribution to pressure acceleration by particle j at particle i
        // note the minus sign!
        a_i -= m_j * (
            p_i/(rho_i * rho_i) + p_j/(rho_j * rho_j) 
        ) * dW;
    }
    // use semi-implicit Euler integration to update velocities and positions
    const float3 v_i_new{v_i + dt * a_i};
    v[i] = v_i_new;
    x[i] += dt * v_i_new;
}

template <IsKernel K>
void SESPH<K>::compute_accelerations(Particles& state, float dt){
    // first, compute densities
    _compute_densities<K><<<BLOCKS(N), BLOCK_SIZE>>>(state.x, state.m, rho, N, W);
    CUDA_CHECK(cudaGetLastError());
    // then synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    // and lastly, compute accelerations using these density values
    // note that pressure need not be pre-computed and stored since the kernel is memory-bound, rather compute them on the fly from density at neighbour j
    // also note that viscosity and gravity can be computed on the fly in the inner loop since they require only known velocities and the constant g, so that ∇W_{ij} need only be evaluated once and can be reused for pressure- and non-pressure accelerations
    _compute_accelerations_and_integrate<K><<<BLOCKS(N), BLOCK_SIZE>>>(state.x, state.v, state.m, rho, N, W, k, rho_0, dt, nu, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
};

// explicit instantiation in every relevant translation unit
#define X(K) template class SESPH<K>;
FOREACH_KERNEL(X)
#undef X
