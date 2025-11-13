#ifndef TIMSTEP_CFL_CUH_
#define TIMSTEP_CFL_CUH_

#include "particles.cuh"

float cfl_time_step(
    const float lambda, const float h, const Particles& state, const float3 g);

#endif // TIMSTEP_CFL_CUH_