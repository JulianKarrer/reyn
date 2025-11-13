#ifndef SCENE_SAMPLE_CUH_
#define SCENE_SAMPLE_CUH_

#include <thrust/device_vector.h>
#include "loader.h"
#include "../buffer.cuh"
#include "datastructure/uniformgrid.cuh"

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
    /// @brief Get a `__device__`-friendly POD structure containing relevant
    /// pointers to positions and masses as well as a datastructure for
    /// neighbour queries
    /// @return a POD to use in CUDA kernels
    Boundary get() const
    {
        return Boundary { xs.ptr(), ys.ptr(), zs.ptr(), m.ptr(), grid };
    }
};

/// @brief Given boundary particle positions, fluid and boundary spacing,
/// calculate the mass of each boundary particle according to the method
/// described in "Versatile Rigid-Fluid Coupling for Incompressible SPH" [Akinci
/// et al. 2012] for the specified fluid rest density. By specifying
/// `mass_refinement_iterations` greater than one, the masses can be iteratively
/// refined, since they actually depend on each of their neighbours's masses as
/// well.
/// @param xs x-compoenents of boundary particle positions
/// @param ys y-compoenents of boundary particle positions
/// @param zs z-compoenents of boundary particle positions
/// @param h fluid particle spacing
/// @param h_bdy boundary particle spacing
/// @param rho_0 fluid rest density
/// @param mass_refinement_iterations iterations of boundary mass refinement
/// @return `BoundarySamples` struct containing positions, masses and
/// datastructure for neighbour queries
BoundarySamples calculate_boundary_masses(DeviceBuffer<float> xs,
    DeviceBuffer<float> ys, DeviceBuffer<float> zs, const float h,
    const float h_bdy, const float rho_0,
    const uint mass_refinement_iterations = 1);

/// @brief Relax a uniformly random boudnary sampling in the sense that samples
/// should spread out across each face of the mesh evenly to produce a
/// low-discrepancy, blue noise sampling. This method may be called repeatedly
/// to iteratively achieve a more even sampling.
///
/// The function uses SPH to estimate and minimize number density similar to how
/// pressure forces in the fluid minimize density deviations, except negative
/// "pressure" is allowed, stiffness is one, all masses are one and instead of a
/// time step that is integrated over, a scale-dependent particle shift is
/// calculated, before that shift is projected back onto the closest point on
/// the triangle that each sample was created on. This may lead to problems with
/// meshes that consist of triangles much smaller than the boundary sampling,
/// since their ability to spread out is limited by the projection step.
/// @param samples the `BoundarySamples` to relax - this  is modified by the
/// function.
/// @param grid_builder `UniformGridBuilder` for creating a datastructure
/// accelerating the search for neighbouring boundary samples
/// @param vxs x-component of all vertices in the sampled triangle mesh
/// @param vys y-component of all vertices in the sampled triangle mesh
/// @param vzs z-component of all vertices in the sampled triangle mesh
/// @param faces faces of the sampled triangle mesh, where each face is a
/// `uint3` of indices into the vertex buffers
/// @param h_bdy desired spacing of boundary particles
/// boundary particles
/// @param tri_ids the indices of the faces that each boundary sample was
/// created on
/// @param num_den a temporary buffer to store the intermediate number densities
/// computed during relaxation
/// @param relaxation_factor factor akin to the stiffness in the pressure solver
/// that determines the magnitude of the particle shift per iteration. Should be
/// a relatively scale-independent constant, since number densities are
/// normalized by the expected volume of a boundary particle if it were sampled
/// on a regular 3D grid (???: a plane might be a better assumption to make this
/// dimensionless).
void sample_relaxation(BoundarySamples& samples,
    UniformGridBuilder& grid_builder, thrust::device_vector<double> vxs,
    thrust::device_vector<double> vys, thrust::device_vector<double> vzs,
    thrust::device_vector<uint3> faces, float h_bdy,
    thrust::device_vector<int> tri_ids, thrust::device_vector<float>& num_den,
    const float relaxation_factor);

/// @brief Place boundary samples uniformly randomly on the surface of the given
/// `Mesh` with a spacing indicated by `h / oversampling_factor`.
///
/// Internally computes the area of each triangle, transforms it into a discrete
/// CDF using an inclusive prefix sum and uses stratified uniform sampling to
/// select a triangle to place each sample in, then uniformly randomly samples
/// that triangle - this should result in a uniform distribution of samples over
/// the area that matches the expected number of samples required to uphold the
/// desired spacing. The discrapancy of this uniform sampling might be
/// undesirable, so by specifying an iterative relaxation procedure
/// is conducted as described in the function `sample_relaxation`
/// @param mesh input mesh as generated by `load_mesh_from_obj`
/// @param h fluid particle spacing
/// @param rho_0 fluid rest density
/// @param oversampling_factor factor between particle spacing `h` and the
/// desired spacing of boudnary particles, where a higher factor results in
/// more samples
/// @param debug_stream an optional `std::ostream` pointer to write debug
/// information about the convergence of the relaxation procedure in csv format
/// to
/// @param relaxation_iters maximum number of iterations of the relaxation
/// procedure
/// @param relaxation_stiffness stiffness coefficient for the relaxation
/// procedure. Should be roughly in (0; 1.5]
/// @param mass_refinement_iterations MUST be >= 1, boundary mass calculation in
/// [Akinci et al. 2012] Sec. 2.2 can be iterated to get more accurate results.
/// @return collection of coordinates of boundary samples
BoundarySamples sample_mesh(const Mesh mesh, const float h, const float rho_0,
    const double oversampling_factor = 2.0,
    std::ostream* debug_stream = nullptr, const int relaxation_iters = 50,
    const float relaxation_stiffness = 1.,
    const uint mass_refinement_iterations = 1);

#endif // SCENE_SAMPLE_CUH_