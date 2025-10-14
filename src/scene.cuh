#include "particles.h"
#include "gui.h"
#include "common.h"
#include <cuda_gl_interop.h>
#include "vector_helper.cuh"
#include <concepts>

#ifndef SCENE_H_
#define SCENE_H_

class Scene
{
public:
    /// @brief Point marking the lowest extend of the bounding box of the scene along each coordinate axis
    float3 bound_min;
    /// @brief Point marking the highest extend of the bounding box of the scene along each coordinate axis
    float3 bound_max;
    /// @brief Number of dynamic particles in the scene
    uint N;

    /// @brief Construct a scene with a box filled with fluid at as close as possible to the desired number of particles within the bounding box defined by `min` and `max` and at rest density `rho_0`.
    /// @param N_desired desired number of dynamic particles
    /// @param min lower bound of the box of dynamic particles along each axis
    /// @param max upper bound of the box of dynamic particles along each axis
    /// @param min lower bound of the scene bounds along each axis
    /// @param max upper bound of the scene bounds along each axis
    /// @param rho_0 rest density
    Scene(uint N_desired, float3 min, float3 max, float3 bound_min, float3 bound_max, float rho_0, Particles &state);
};

/// @brief Kernel used by one of the Scene constructors to initialize a set of dynamic particles in a box using CUDA directly to set each position.
__global__ void init_box_kernel(float3 min, Particles p, int3 nxyz, uint N, float h, float rho_0);

#endif // SCENE_H_