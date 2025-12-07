#ifndef SCENE_SAMPLE_CUH_
#define SCENE_SAMPLE_CUH_

#include <thrust/device_vector.h>
#include "loader.h"
#include "../buffer.cuh"
#include "datastructure/uniformgrid.cuh"

/// @brief Device-side representation of a triangular mesh, containing vertex
/// buffers (SoA by component) and a face buffer indexing the vertex buffers
/// with a `uint3`. May be constructed from a `Mesh`, which is stored host-side
/// in RAM, using the `from` method.
struct DeviceMesh {
    DeviceBuffer<double> vxs;
    DeviceBuffer<double> vys;
    DeviceBuffer<double> vzs;
    DeviceBuffer<uint3> faces;
    /// @brief Method for constructing a device-side triangular mesh from a
    /// corresponding host side representation, allocating and copying to device
    /// buffers
    /// @param mesh host-side triangular `Mesh
    /// @return corresponding `DeviceMesh`
    static DeviceMesh from(Mesh mesh)
    {
        return DeviceMesh { //
            DeviceBuffer<double>(mesh.xs.begin(), mesh.xs.end()),
            DeviceBuffer<double>(mesh.ys.begin(), mesh.ys.end()),
            DeviceBuffer<double>(mesh.zs.begin(), mesh.zs.end()),
            DeviceBuffer<uint3>(mesh.faces.begin(), mesh.faces.end())
        };
    }

    ///@brief Compute the volume of the Mesh using signed volumes, which assumes
    /// a watertight mesh with well-defined bounded interior and a consistent
    /// triangle winding.
    ///
    /// Uses the method described in "A Symbolic Method for Calculating the
    /// Integral Properties of Arbitrary Nonconvex Polyhedra" [Sheue-ling Lien
    /// and James T. Kajiya]
    ///@return Volume of the mesh
    double get_volume() const
    {
        const auto vxs_d { vxs.ptr() };
        const auto vys_d { vys.ptr() };
        const auto vzs_d { vzs.ptr() };
        const double determinants { thrust::transform_reduce(
            faces.get().begin(), faces.get().end(),
            [vxs_d, vys_d, vzs_d] __device__(const uint3& face) -> double {
                // load the three vertices of each triangle
                const double3 a { dv3(face.x, vxs_d, vys_d, vzs_d) };
                const double3 b { dv3(face.y, vxs_d, vys_d, vzs_d) };
                const double3 c { dv3(face.z, vxs_d, vys_d, vzs_d) };
                // compute the signed volume of the tetrahedron formed by the
                // triangle and the origin, or the determinant of the matrix T
                // in the paper, formed by the vertex positions as column
                // vectors

                // implemented using a triple product a · (b x c)
                return dot(a, cross(b, c));
            },
            0.,
            [] __device__(
                const double a, const double b) -> double { return a + b; }) };
        return abs(determinants) / 6.;
    };
};

/// @brief Given boundary particle positions, fluid and boundary spacing,
/// calculate the mass of each boundary particle according to the method
/// described in "Versatile Rigid-Fluid Coupling for Incompressible SPH" [Akinci
/// et al. 2012] for the specified fluid rest density. By specifying
/// `mass_refinement_iterations` greater than one, the masses can be iteratively
/// refined, since they actually depend on each of their neighbours's masses as
/// well.
/// @param bdy the boundary samples to calculate the masses of
/// @param h fluid particle spacing
/// @param h_bdy boundary particle spacing
/// @param ρ₀ fluid rest density
/// @param mass_refinement_iterations iterations of boundary mass refinement
/// @return `BoundarySamples` struct containing positions, masses and
/// datastructure for neighbour queries
void calculate_boundary_masses(BoundarySamples& bdy, const float h,
    const float h_bdy, const float ρ₀,
    const uint mass_refinement_iterations = 1);

/// @brief Relax a uniformly random boundary sampling in the sense that samples
/// should spread out across each face of the mesh evenly to produce a
/// low-discrepancy, blue noise sampling.
///
/// The function uses SPH to estimate and minimize number density similar to how
/// pressure forces in the fluid minimize density deviations, except negative
/// "pressure" is allowed, stiffness is one, all masses are one and instead of a
/// time step that is integrated over, a scale-dependent particle shift is
/// calculated, before that shift is projected back onto the closest point on
/// the triangular mesh using an LBVH for efficient distance queries.
/// @param xs x-component of the samples to relax
/// @param ys y-component of the samples to relax
/// @param zs z-component of the samples to relax
/// @param mesh device-side representation of the triangular mesh to sample
/// @param h_bdy desired spacing of boundary particles
/// boundary particles
/// @param stream output stream to print debug information in CSV format to, if
/// any
/// @param relaxation_factor factor akin to the stiffness in the pressure solver
/// that determines the magnitude of the particle shift per iteration, ideally
/// as a fraction of boundary particle spacing (-> typical range from 0 to 1)
/// @param relaxation_iters maximum number of iterations of the relaxation
/// procedure
void relax_sampling(DeviceBuffer<float>& xs, DeviceBuffer<float>& ys,
    DeviceBuffer<float>& zs, DeviceMesh& mesh, const float h_bdy,
    std::ostream* debug_stream, const float relaxation_factor,
    const uint relaxation_iters);

/// @brief Uniformly randomly sample a mesh. Uses stratified sampling of
/// triangles by area using a discrete CDF from the prefix sum of triangle
/// areas, then a uniform sampling within each triangle.
/// @param xs buffer for x-components of the samples, resized by this function
/// @param ys buffer for y-components of the samples, resized by this function
/// @param zs buffer for z-components of the samples, resized by this function
/// @param mesh a `DeviceMesh` of the surface to sample
/// @param h fluid particle spacing
/// @param oversampling_factor ratio of boundary sample spacing to fluid spacing
/// h
void uniform_sample_mesh(DeviceBuffer<float>& xs, DeviceBuffer<float>& ys,
    DeviceBuffer<float>& zs, DeviceMesh& mesh, const float h,
    const float oversampling_factor);

/// @brief Place boundary samples uniformly randomly on the surface of the
/// given `Mesh` with a spacing indicated by `h / oversampling_factor`.
///
/// Internally computes the area of each triangle, transforms it into a
/// discrete CDF using an inclusive prefix sum and uses stratified uniform
/// sampling to select a triangle to place each sample in, then uniformly
/// randomly samples that triangle - this should result in a uniform
/// distribution of samples over the area that matches the expected number
/// of samples required to uphold the desired spacing. The discrapancy of
/// this uniform sampling might be undesirable, so by specifying an
/// iterative relaxation procedure is conducted as described in the function
/// `relax_sampling`
/// @param mesh input mesh as generated by `load_mesh_from_obj`
/// @param h fluid particle spacing
/// @param ρ₀ fluid rest density
/// @param oversampling_factor factor between particle spacing `h` and the
/// desired spacing of boudnary particles, where a higher factor results in
/// more samples
/// @param debug_stream an optional `std::ostream` pointer to write debug
/// information about the convergence of the relaxation procedure in csv
/// format to
/// @param relaxation_iters maximum number of iterations of the relaxation
/// procedure
/// @param relaxation_stiffness stiffness coefficient for the relaxation
/// procedure. Should be roughly in (0; 1.5]
/// @param mass_refinement_iterations MUST be >= 1, boundary mass
/// calculation in [Akinci et al. 2012] Sec. 2.2 can be iterated to get more
/// accurate results.
/// @return collection of coordinates of boundary samples
BoundarySamples sample_mesh(const Mesh mesh, const float h, const float ρ₀,
    const float oversampling_factor = 2.0, std::ostream* debug_stream = nullptr,
    const int relaxation_iters = 50, const float relaxation_factor = 1.,
    const uint mass_refinement_iterations = 5);

#endif // SCENE_SAMPLE_CUH_