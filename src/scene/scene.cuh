#include "particles.cuh"
#include "gui.cuh"
#include "common.h"
#include <cuda_gl_interop.h>
#include "vector_helper.cuh"
#include <concepts>

#ifndef SCENE_H_
#define SCENE_H_

class Scene {
public:
    /// @brief Point marking the lowest extend of the bounding box of the scene
    /// along each coordinate axis
    float3 bound_min;
    /// @brief Point marking the highest extend of the bounding box of the scene
    /// along each coordinate axis
    float3 bound_max;
    /// @brief particle spacing h
    float h;
    /// @brief Number of dynamic particles in the scene
    uint N;
    /// @brief Rest density of the fluid
    float rho_0;

    /// @brief Construct a scene with a box filled with fluid at as close as
    /// possible to the desired number of particles within the bounding box
    /// defined by `min` and `max` and at rest density `rho_0`.
    /// @param N_desired desired number of dynamic particles
    /// @param min lower bound of the box of dynamic particles along each axis
    /// @param max upper bound of the box of dynamic particles along each axis
    /// @param bound_min lower bound of the scene bounds along each axis
    /// @param bound_max upper bound of the scene bounds along each axis
    /// @param rho_0 rest density
    /// @param state current state of the particles
    Scene(const uint N_desired, const float3 min, const float3 max,
        const float3 bound_min, const float3 bound_max, const float rho_0,
        Particles& state);

    /// @brief Strictly enforce the simulation bounds set at scene creation,
    /// clamping all particle positions in the argument state to scenes bounding
    /// volume and reflecting any offending velocities while also damping them.
    /// @param state `Particles` to clamp to the bounds
    void hard_enforce_bounds(Particles& state) const;
};

#endif // SCENE_H_