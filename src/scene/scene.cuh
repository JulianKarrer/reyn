#include "particles.cuh"
#include "gui.cuh"
#include "common.h"
#include "datastructure/uniformgrid.cuh"
#include <cuda_gl_interop.h>
#include <filesystem>

#ifndef SCENE_H_
#define SCENE_H_

/// @brief POD collection of pointers and datastructures required to query
/// boundary neighbours on the device
struct Boundary {
    const float* xx;
    const float* xy;
    const float* xz;
    const float* m;
    const UniformGrid<Resort::yes> grid;
};

/// @brief Structure representing a single layer of boundary particles sampled
/// on some surface mesh. Includes respective coordinates, masses due to [Akinci
/// et al. 2012] and a `UniformGrid` for neighbour search, accompanied by the
/// `prefix` sum of particles per cell, since this buffer object must live at
/// least as long as the `UniformGrid` does.
/// Provides a convienience method `get` for collecting raw pointers to relevant
/// buffers as a `__device__`-friendly POD
struct BoundarySamples {
    DeviceBuffer<float> xs;
    DeviceBuffer<float> ys;
    DeviceBuffer<float> zs;
    DeviceBuffer<float> m;
    DeviceBuffer<uint> prefix;
    UniformGrid<Resort::yes> grid;
    float3 bound_min;
    float3 bound_max;
    /// @brief Get a `__device__`-friendly POD structure containing relevant
    /// pointers to positions and masses as well as a datastructure for
    /// neighbour queries
    /// @return a POD to use in CUDA kernels
    Boundary get() const
    {
        return Boundary { xs.ptr(), ys.ptr(), zs.ptr(), m.ptr(), grid };
    }
};

class Scene {
public:
    /// @brief Rest density of the fluid
    float rho_0;
    /// @brief particle spacing h
    const float h;
    /// @brief Number of dynamic particles in the scene
    uint N;
    /// @brief Collection of boundary samples
    const BoundarySamples bdy;
    /// @brief Point marking the lowest extend of the bounding box of the scene
    /// along each coordinate axis
    const float3 bound_min;
    /// @brief Point marking the highest extend of the bounding box of the scene
    /// along each coordinate axis
    const float3 bound_max;

private:
    // list the grid builder now, since its constructor must follow those above

    /// @brief Used to obtain a `UniformGrid` enabling fast fixed radius
    /// neighbourhood queries
    UniformGridBuilder grid_builder;

public:
    /// @brief Construct a scene with a box filled with fluid at as close as
    /// possible to the desired number of particles within the bounding box
    /// defined by `min` and `max` and at rest density `rho_0`.
    /// @param path path to the OBJ file to sample the boundary with
    /// @param N_desired desired number of dynamic particles
    /// @param min lower bound of the box of dynamic particles along each axis
    /// @param max upper bound of the box of dynamic particles along each axis
    /// @param rho_0 rest density
    /// @param state current state of the particles
    /// @param bdy_oversampling_factor ratio of spacing of fluid samples to
    /// spacing of boundary samples
    /// @param cull_bdy_radius radius in units of fluid particle spacing `h`
    /// around any boundary particle in which fluid particles should be culled
    /// to prevent intersections
    Scene(const std::filesystem::path& path, const uint N_desired,
        const float3 min, const float3 max, const float rho_0, Particles& state,
        DeviceBuffer<float>& tmp, const float bdy_oversampling_factor = 2.f,
        const float cull_bdy_radius = 1.f);

    /// @brief Strictly enforce the simulation bounds set at scene creation,
    /// clamping all particle positions in the argument state to scenes bounding
    /// volume and reflecting any offending velocities while also damping them.
    /// @param state `Particles` to clamp to the bounds
    void hard_enforce_bounds(Particles& state) const;

    /// @brief Obtain a `UniformGrid` for neighbourhood search with a fixed
    /// radius specified in units of fluid particle spacing `h`, potentially
    /// reordering particles in `state` for better memory coherency and
    /// coalescing.
    /// @param search_radius maximum radius for the search in units of `h`
    /// @param state `Particles` to query and potentially reorder during grid
    /// construction
    /// @param tmp temporary buffer used for non in-place reordering of buffers
    /// @return a `UniformGrid` to use on the device-side for neigbhbourhood
    /// queries
    UniformGrid<Resort::yes> get_grid(Particles& state,
        DeviceBuffer<float>& tmp, const float search_radius = 2.0)
    {
        return grid_builder.construct_and_reorder(
            search_radius * h, tmp, state);
    };
};

#endif // SCENE_H_