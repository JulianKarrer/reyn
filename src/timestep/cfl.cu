#include "cfl.cuh"
#include <thrust/transform_reduce.h>
#include "utils/vector.cuh"

float cfl_time_step(
    const float lambda, const float h, const Particles& state, const float3 g)
{
    const size_t N_fld { state.xx.size() };
    // compute the maximum velocity among particles
    const float* vx { state.vx.ptr() };
    const float* vy { state.vy.ptr() };
    const float* vz { state.vz.ptr() };
    const float v_max { thrust::transform_reduce(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(N_fld),
        [vx, vy, vz] __device__(
            size_t i) -> float { return norm(v3(i, vx, vy, vz)); },
        0.f, ::cuda::maximum<float>()) };
    // compute velocity based CFL time step from the constraint:
    // Δt v_max < λ h       limit 1st order Taylor approx of Δx to λh
    // Δt < λ h / v_max
    const float dt_vel { lambda * h / v_max };

    // Δt can be at least be lower bounded by the influence of gravity in the
    // time step, if no better estimate for acceleration is available:
    // - especially effective if velocity is close to zero so dt_vel grows to
    // infinity since second order is neglected
    // 1/2 g Δt² < λ/2 h
    // Δt < √(λ h / g)
    // use only λ/2 here
    const float dt_g { sqrtf(2. * lambda * h / norm(g)) };

    // for a conservative upper bound on time step, return minimum of all
    // calculated estimated upper bounds. Note that fminf takes care of NaN due
    // to division by zero in dt_vel for example
    return fminf(dt_vel, dt_g);
};

float simple_dt_controller(const uint iter_last, const float dt_last,
    const float lambda, const float h, const Particles& state, const float3 g,
    const uint iter_target, const float increase, const float decrease)
{
    // if iteration count was surpassed, decrease time step, if it was not
    // reached then increase
    const float change_factor { iter_last < iter_target ? increase : decrease };
    const float dt_suggested { dt_last * change_factor };
    // get the current maximum allowed timestep according to CFL condition
    const float Δt_cfl { cfl_time_step(lambda, h, state, g) };
    // clamp the suggested timestep to that value
    return min(Δt_cfl, dt_suggested);
}
