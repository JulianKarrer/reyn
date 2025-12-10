#ifndef TIMSTEP_CFL_CUH_
#define TIMSTEP_CFL_CUH_

#include "particles.cuh"

/// @brief Compute the maximum allowed timestep as a percentage λ∈(0;1] of the
/// time step the fastest particle is estimated to travel a distance of h in
/// @param lambda CFL number
/// @param h particle spacing
/// @param state `Particles` state
/// @param g gravity (used for lower bound on acceleration)
/// @return maximum time step size
float cfl_time_step(
    const float lambda, const float h, const Particles& state, const float3 g);

///@brief A simple time step controller that attempts to keep solver iterations
/// at a desired count by increasing the time step by a factor of `increase`
/// when iterations are too low or multiplying with `decrease` when they are too
/// high, before clamping to the maximum allowed by a `lambda` CFL number
///@param iter_last last iteration count
///@param dt_last last time step size Δt
///@param lambda CFL number
///@param h particle spacing
///@param state `Particles` state
///@param g gravity (used for lower bound on acceleration)
///@param iter_target target solver iteration count
///@param change_factor factor of increase or decrease in Δt per iteration, must
/// be < 1
///@return float
float simple_dt_controller(const uint iter_last, const float dt_last,
    const float lambda, const float h, const Particles& state, const float3 g,
    const uint iter_target = 5, const float change_factor = 0.01);

#endif // TIMSTEP_CFL_CUH_