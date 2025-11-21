#include "sample_boundary.cuh"

#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>

#include "vector_helper.cuh"
#include "kernels.cuh"
#include "common.h"
#include "scene/ply_io.cuh"
#include "scene.cuh"

#include "doctest/doctest.h"
#include <fstream>

/// @brief Compute the number density of the given positions with a `B3` kernel
/// at \f$2h\f$ support radius
/// @param N number of points
/// @param W kernel function
/// @param grid `UniformGrid`
/// @param xs x-component of positions
/// @param ys y-component of positions
/// @param zs z-component of positions
/// @param num_den output buffer for number density
/// @return nothing
template <Resort R>
__global__ void _compute_number_densities(const uint N, const B3 W,
    const UniformGrid<R> grid, const float* __restrict__ xs,
    const float* __restrict__ ys, const float* __restrict__ zs,
    float* __restrict__ num_den)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xs, ys, zs) };
    num_den[i] = { grid.ff_nbrs(x_i, xs, ys, zs,
        [&W] __device__(
            auto j, const float3 x_ij, auto _x_ij_l2) { return W(x_ij); }) };
};

/// @brief Compute the closest point oin a triangle to a given query point
///
/// Algorithm from "Real-Time Collision Detection" by Christer Ericson
/// @param
/// @return

/// @brief Compute the closest point to `p` on a triangle formed by `a`, `b` and
/// `c`
///
/// Algorithm from "Real-Time Collision Detection" by Christer Ericson
/// https://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
/// @param p query point, the closest point on the triangle to which is sought
/// @param a first vertex of the triangle
/// @param b second vertex of the triangle
/// @param c third vertex of the triangle
/// @return the closest point to `p` in the triangle formed by `a`, `b` and `c`
__device__ inline float3 _closest_point_on_triangle(
    const float3 p, const float3 a, const float3 b, const float3 c)
{
    // this is taken basically verbatim from Christer Ericson, including the
    // helpful comments
    const float3 ab { b - a };
    const float3 ac { c - a };
    const float3 ap { p - a };
    // vertex region outside a
    const float d1 { dot(ab, ap) };
    const float d2 { dot(ac, ap) };
    if (d1 <= 0.f && d2 <= 0.f)
        return a; // barycentric (1,0,0)
    // vertex region outside b
    const float3 bp { p - b };
    const float d3 { dot(ab, bp) };
    const float d4 { dot(ac, bp) };
    if (d3 >= 0.0f && d4 <= d3)
        return b; // barycentric (0,1,0)
    // if in edge region ab, project
    const float vc { d1 * d4 - d3 * d2 };
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return a + v * ab; // barycentric (1-v,v,0)
    }
    // vertex region outside c
    const float3 cp { p - c };
    const float d5 { dot(ab, cp) };
    const float d6 { dot(ac, cp) };
    if (d6 >= 0.0f && d5 <= d6)
        return c; // barycentric (0,0,1)
    // if in edge region ac, project
    const float vb { d5 * d2 - d1 * d6 };
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = { d2 / (d2 - d6) };
        return a + w * ac; // barycentric (1-w,0,w)
    }
    // if in edge region bc, project
    const float va { d3 * d6 - d5 * d4 };
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w { (d4 - d3) / ((d4 - d3) + (d5 - d6)) };
        return b + w * (c - b); // barycentric (0,1-w,w)
    }
    // projected inside face region, use uvw to compute
    const float denom { 1.0f / (va + vb + vc) };
    const float v { vb * denom };
    const float w { vc * denom };
    return a + ab * v + ac * w;
};

__global__ void _relax_sampling(const uint N, const B3 W,
    const UniformGrid<Resort::no> grid, float* __restrict__ xs,
    float* __restrict__ ys, float* __restrict__ zs,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces,
    const int* __restrict__ tri_ids, const float h_bdy,
    float* __restrict__ num_den, const float k, const float rho_0_sq_inv)
{
    // compute index of current boundary sample
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // get the vertices of the face that the boundary particle was sampled on
    const uint3 face { faces[tri_ids[i]] };
    // use a v3 overload that casts the double* vertex attribute buffers to
    // float
    const float3 vert1 { v3(face.x, vxs, vys, vzs) };
    const float3 vert2 { v3(face.y, vxs, vys, vzs) };
    const float3 vert3 { v3(face.z, vxs, vys, vzs) };

    // compute an updated position by slightly moving away from regions of
    // higher number density
    const float term_i { rho_0_sq_inv * num_den[i] };
    const float3 x_i { v3(i, xs, ys, zs) };
    const float3 dx_i { -k * h_bdy * h_bdy
        * grid.ff_nbrs(x_i, xs, ys, zs,
            [&W, &num_den, rho_0_sq_inv, term_i] __device__(
                auto j, const float3 x_ij, auto _x_ij_l2) {
                // symmetric formula
                return (term_i + rho_0_sq_inv * num_den[j]) * W.nabla(x_ij);
            }) };
    const float3 new_x_i { v3(i, xs, ys, zs) + dx_i };

    // find the closest point to this which is inside the triangle and store the
    // result.
    // This writes to the buffer being read from with no synchronization!
    const float3 projected_new_x_i { _closest_point_on_triangle(
        new_x_i, vert1, vert2, vert3) };
    store_v3(projected_new_x_i, i, xs, ys, zs);
};

/// @brief Compute the number density for a hexagonal sampling in a plane with
/// search radius \f$2 h_{bdy}\f$ for the given kernel function
/// @tparam K
/// @param h_bdy particle spacing
/// @param W kernel function
/// @return rest number density for the optimal spacing
template <IsKernel K>
static float _hex_plane_rest_density(
    const float h_bdy, const K W, const int range = 2)
{
    // great resource: https://www.redblobgames.com/grids/hexagons/
    // q axis is horizontal with length h_bdy
    const float3 q_axis { h_bdy * v3(1., 0., 0.) };
    // r axis is 60deg counterclockwise from q axis, same length
    // sin(60deg) = √3 / 2
    // cos(60deg) =  1 / 2
    const float3 r_axis { h_bdy * v3(.5, sqrt(3) * .5, 0.) };
    // these two span a lattice where every combination of q,r in the range
    // gives a unique point
    float ρ₀ { 0. };
    for (int q { -range }; q <= range; ++q)
        for (int r { -range }; r <= range; ++r) {
            ρ₀ += W(q * q_axis + r * r_axis);
        }
    return ρ₀;
};

void relax_sampling(DeviceBuffer<float>& xs, DeviceBuffer<float>& ys,
    DeviceBuffer<float>& zs, const DeviceMesh& mesh, const float h_bdy,
    const DeviceBuffer<int>& tri_ids, std::ostream* debug_stream,
    const float relaxation_factor, const uint relaxation_iters)
{
    // early exit if no iterations are requested
    if (relaxation_iters == 0)
        return;
    // get the number of boundary particles, resize tmp if appropriate
    const uint N { (uint)xs.size() };
    DeviceBuffer<float> num_den(N);

    // get the bounds of the particles
    const float3 min_bound { v3(xs.min(), ys.min(), zs.min()) };
    const float3 max_bound { v3(xs.max(), ys.max(), zs.max()) };
    // in gridbuilder, use larger safety margin of 5h but reuse the
    // UniformGridBuilder across iterations
    UniformGridBuilder grid_builder { UniformGridBuilder(
        min_bound - v3(5.f * (float)h_bdy), max_bound + v3(5.f * (float)h_bdy),
        2.f * (float)h_bdy) };
    // use any kernel function, `B3` is used here
    const B3 W(2.f * h_bdy);
    // compute the resting number density for perfect hexagonal sampling of
    // the plane, given the kernel function and smoothing radius
    const float ρ₀ { _hex_plane_rest_density(h_bdy, W) };
    const float rho_0_sq_inv { 1.f / (ρ₀ * ρ₀) };

    for (uint i { 0 }; i < relaxation_iters; ++i) {
        // build an acceleration datastructure to quickly find neighbours of
        // boundary samples. Don't use resorting, since that would obscur the
        // mapping to faces, which are needed to find the vertices associated
        // with each sample
        const UniformGrid<Resort::no> grid
            = grid_builder.construct(2.f * h_bdy, xs, ys, zs);
        // relax the sampling by moving each boundary particle in the direction
        // of the negative number density gradient, projected onto the plane of
        // the triangle it may move in and clamped to the edges of that triangle

        _compute_number_densities<Resort::no><<<BLOCKS(N), BLOCK_SIZE>>>(
            N, W, grid, xs.ptr(), ys.ptr(), zs.ptr(), num_den.ptr());

        _relax_sampling<<<BLOCKS(N), BLOCK_SIZE>>>(N, W, grid, xs.ptr(),
            ys.ptr(), zs.ptr(), mesh.vxs.ptr(), mesh.vys.ptr(), mesh.vzs.ptr(),
            mesh.faces.ptr(), tri_ids.ptr(), h_bdy, num_den.ptr(),
            relaxation_factor, rho_0_sq_inv);

        // if requested, output debug information
        if (debug_stream) {
            (*debug_stream) << i + 1 << "," << num_den.max() / num_den.min()
                            << "," << num_den.sum() / ((float)N) << std::endl;
        }
    }
};

__global__ void _init_curand(const uint N,
    curandStatePhilox4_32_10_t* __restrict__ states, unsigned long long seed)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // call curand_init with different argument for "subsequence" in each thread
    // but with the same seed
    curand_init(seed, i, 0, &states[i]);
};

template <bool recordTriIds>
__global__ void _uniform_sample_tri_cdf(const uint N, float* __restrict__ xs,
    float* __restrict__ ys, float* __restrict__ zs,
    const double* __restrict__ cdf, const int N_cdf, const double bin_size,
    const double total_area, curandStatePhilox4_32_10_t* __restrict__ states,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces,
    const uint vertex_count, int* __restrict__ tri_ids)
{
    uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    curandStatePhilox4_32_10_t* rng { &states[i] };
    // create a stratified uniformly distributed number within (0; total_area)
    // by dividing the area into N bins, moving i bins over and then sampling
    // uniformly within the size of one bin
    const double area_rand { ((double)i / (double)N) * total_area
        + bin_size * curand_uniform_double(rng) };
    // by inverting the cdf, find the triangle that this random point along the
    // total area belongs to
    // -> binary search for smallest cdf entry greater than or equal to the
    // random number since scan was inclusive
    // -> binary lower bound search
    int low { 0 };
    int high { N_cdf - 1 };
    int tri_id { 0 };
    while (low <= high) {
        const int mid { low + (high - low) / 2 };
        double midval { cdf[mid] };
        if (midval >= area_rand) {
            tri_id = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    assert(0 <= tri_id && tri_id < N_cdf);
    // now tri_id contains the index into the `faces` buffer with the triangle
    // that shall be sampled
    const uint3 sampled_face { faces[tri_id] };
    if constexpr (recordTriIds) {
        // remember the triangle that was sampled for later resampling
        tri_ids[i] = tri_id;
    }

    // from this face (i.e. the three indices into the vertex buffer for the
    // three vertices of the triangle) get the actual triangle vertices
    const uint v1i { sampled_face.x };
    const uint v2i { sampled_face.y };
    const uint v3i { sampled_face.z };
    const double3 v1 { make_double3(vxs[v1i], vys[v1i], vzs[v1i]) };
    const double3 v2 { make_double3(vxs[v2i], vys[v2i], vzs[v2i]) };
    const double3 v3 { make_double3(vxs[v3i], vys[v3i], vzs[v3i]) };
    // get the edged V12 and V13
    const double3 e12 { v2 - v1 };
    const double3 e13 { v3 - v1 };
    // sample a random barycentric coordinate, which is in this case
    // implemented as sampling a random point in (0;1]^2
    const double2 bary_rand { curand_uniform2_double(rng) };
    // uniformly randomly sample the triangle specified by those edges
    // first, interpret the two random numbers as sampling a parallelogram
    // and check if inside the triangle
    const bool inside_triangle { bary_rand.x + bary_rand.y <= 1 };
    // flip s and t if in the wrong half of the parallelogram
    const double s { inside_triangle ? bary_rand.x : 1. - bary_rand.x };
    const double t { inside_triangle ? bary_rand.y : 1. - bary_rand.y };
    // sample the target point
    const double3 sampled { v1 + s * e12 + t * e13 };
    // store the respective cooridnates after finally converting them to
    // floating point precision
    xs[i] = (float)sampled.x;
    ys[i] = (float)sampled.y;
    zs[i] = (float)sampled.z;
};

template <Resort R>
__global__ void _refine_masses(const uint N, const B3 W,
    const UniformGrid<R> grid, const float* __restrict__ xs,
    const float* __restrict__ ys, const float* __restrict__ zs,
    float* __restrict__ m, float ρ₀)
{
    const auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xs, ys, zs) };
    m[i] = ρ₀ * m[i]
        / grid.ff_nbrs(x_i, xs, ys, zs,
            [W, m] __device__(auto j, const float3 x_ij, auto _x_ij_l2) {
                return m[j] * W(x_ij);
            });
};

void calculate_boundary_masses(BoundarySamples& bdy, const float h,
    const float h_bdy, const float ρ₀, const uint mass_refinement_iterations)
{
    // now initialize masses
    const B3 W(2.f * h);
    // "Versatile Rigid-Fluid Coupling for Incompressible SPH" [Akinci et al]
    // The unsimplified first equation in section 2.2 is equivalent to the
    // suggested (4) for m_bi = 1 but can be iterated to more accurately
    // determine the boundary masses (since they each actually depend on all
    // neighbouring boundary masses and them being equal is a simplification).
    // fill the mass buffer with one initially
    const float m_0 { 1.f };
    thrust::fill(bdy.m.get().begin(), bdy.m.get().end(), m_0);

    // BoundarySamples result { std::move(xs), std::move(ys), std::move(zs),
    //     std::move(m), std::move(prefix), grid, bound_min, bound_max };
    // result.grid.prefix = result.prefix.ptr();

    // refine masses iteratively
    const uint N_bdy { (uint)bdy.xs.size() };
    if (mass_refinement_iterations <= 0) {
        throw std::runtime_error(
            "At least one iteration of mass refinements must be run, please "
            "adjust the parameter to `calculate_boundary_masses`.");
    }
    for (uint i { 0 }; i < mass_refinement_iterations; ++i) {
        _refine_masses<<<BLOCKS(N_bdy), BLOCK_SIZE>>>(N_bdy, W, bdy.grid,
            bdy.xs.ptr(), bdy.ys.ptr(), bdy.zs.ptr(), bdy.m.ptr(), ρ₀);
        std::cout << "avg mass " << bdy.m.sum() / (float)bdy.m.size()
                  << std::endl;
    };
};

void uniform_sample_mesh(DeviceBuffer<float>& xs, DeviceBuffer<float>& ys,
    DeviceBuffer<float>& zs, const DeviceMesh& mesh, const float h,
    const float oversampling_factor, DeviceBuffer<int>* tri_ids)
{
    const uint N_face { (uint)mesh.faces.size() };
    const uint N_vert { (uint)mesh.vxs.size() };
    // calculate the surface areas of each face
    // then fuse this with a prefix sum to both compute the total area (last
    // element) and construct a discrete CDF for uniform sampling
    DeviceBuffer<double> area(N_face);
    auto vxs_d { mesh.vxs.ptr() };
    auto vys_d { mesh.vys.ptr() };
    auto vzs_d { mesh.vzs.ptr() };
    const size_t vertex_count { N_vert };
    ::cuda::std::plus<double> plus_double_op;

    thrust::transform_inclusive_scan(
        thrust::device,
        // input iterator begin and end
        mesh.faces.get().begin(), mesh.faces.get().end(),
        // output iterator
        area.get().begin(),
        // unary operator for transform
        [vxs_d, vys_d, vzs_d, vertex_count] __device__(uint3 ids) -> double {
            assert(ids.x < vertex_count && ids.y < vertex_count
                && ids.z < vertex_count);
            // load the three vertices of the face
            const double3 v1 { dv3(ids.x, vxs_d, vys_d, vzs_d) };
            const double3 v2 { dv3(ids.y, vxs_d, vys_d, vzs_d) };
            const double3 v3 { dv3(ids.z, vxs_d, vys_d, vzs_d) };
            // get the edged V12 and V13
            const double3 e12 { v2 - v1 };
            const double3 e13 { v3 - v1 };
            // half the 2-norm of the cross product is the area
            return 0.5 * norm(cross(e12, e13));
        },
        // binary operator for inclusive scan
        plus_double_op //
    );

    if (area.size() <= 0) {
        throw std::runtime_error(
            "Computation of areas of each triangle in mesh unsuccessful: "
            "`area` was empty, suggesting a mesh with no faces");
    }
    // compute the number of boundary samples to place
    const double h_bdy { (double)h / oversampling_factor };
    const double total_area { // load the last element from the device to host
        // memory via thrust overload of operator[]
        area.get()[area.size() - 1]
    };
    // assume hexagonal dense packing of 2-manifold when calculating the desired
    // number of particles in terms of total area and desired spacing
    const float bdy_particle_area { 1.5f * sqrtf(3.f) * (float)h_bdy
        * (float)h_bdy };
    const uint bdy_count { (uint)(ceil(total_area / bdy_particle_area)) };
    const double bin_size { total_area / (double)(bdy_count) };

    std::cout << std::format(
        "total area: {}, h_bdy: {}, bdy_count: {}, face_count: {}", total_area,
        h_bdy, bdy_count, N_face)
              << std::endl;

    // resize buffers for sample coordinates
    xs.resize(bdy_count);
    ys.resize(bdy_count);
    zs.resize(bdy_count);

    // set up one cuRand rng per thread using cudaMalloc to be freed later
    curandStatePhilox4_32_10_t* states;
    cudaMalloc(&states, bdy_count * sizeof(curandStatePhilox4_32_10_t));
    _init_curand<<<BLOCKS(bdy_count), BLOCK_SIZE>>>(
        bdy_count, states, 1614201312);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::cout << "curand initialized" << std::endl;

    // uniformly randomly sample the mesh
    if (tri_ids) {
        // when uniformly sampling, keep track of which sample was placed on
        // which triangle by its id into the `faces` buffer
        tri_ids->resize(bdy_count);
        _uniform_sample_tri_cdf<true>
            <<<BLOCKS(bdy_count), BLOCK_SIZE>>>(bdy_count, xs.ptr(), ys.ptr(),
                zs.ptr(), thrust::raw_pointer_cast(area.ptr()), N_face,
                bin_size, total_area, states, mesh.vxs.ptr(), mesh.vys.ptr(),
                mesh.vzs.ptr(), mesh.faces.ptr(), vertex_count, tri_ids->ptr());
    } else {
        _uniform_sample_tri_cdf<false>
            <<<BLOCKS(bdy_count), BLOCK_SIZE>>>(bdy_count, xs.ptr(), ys.ptr(),
                zs.ptr(), thrust::raw_pointer_cast(area.ptr()), N_face,
                bin_size, total_area, states, mesh.vxs.ptr(), mesh.vys.ptr(),
                mesh.vzs.ptr(), mesh.faces.ptr(), vertex_count, nullptr);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // clean up cuRand states
    cudaFree(states);
    std::cout << "uniformly sampled" << std::endl;

    return;
}

BoundarySamples sample_mesh(const Mesh mesh_host, const float h, const float ρ₀,
    const float oversampling_factor, std::ostream* debug_stream,
    const int relaxation_iters, const float relaxation_factor,
    const uint mass_refinement_iterations)
{
    // move vertex data and faces to the GPU by constructing a `DeviceMesh`
    const DeviceMesh mesh { DeviceMesh::from(mesh_host) };

    const float h_bdy { h / oversampling_factor };
    DeviceBuffer<float> xs(1);
    DeviceBuffer<float> ys(1);
    DeviceBuffer<float> zs(1);

    // 1: uniformly sample mesh
    if (relaxation_iters == 0) {
        // if no blue noise distribution is required, just uniformly sample
        uniform_sample_mesh(xs, ys, zs, mesh, h, oversampling_factor, nullptr);
    } else {
        // if relaxation was requested, remember the faces each boundary
        // particle was sampled on
        DeviceBuffer<int> tri_ids(1);
        uniform_sample_mesh(xs, ys, zs, mesh, h, oversampling_factor, &tri_ids);

        // 2: optionally relax the sampling
        std::cout << "relaxing sampling" << std::endl;
        relax_sampling(xs, ys, zs, mesh, h_bdy, tri_ids, debug_stream,
            relaxation_factor, relaxation_iters);
    };

    // build the final acceleration structure for the boundary samples:
    // now reordering is allowed
    DeviceBuffer<float> m(xs.size());
    DeviceBuffer<uint> prefix(1);
    const float3 bound_min { v3(xs.min(), ys.min(), zs.min()) - v3(h) };
    const float3 bound_max { v3(xs.max(), ys.max(), zs.max()) + v3(h) };

    std::cout << "in sample_mesh bound min x =" << bound_min.x << std::endl;
    std::cout << "in sample_mesh bound min y =" << bound_min.y << std::endl;
    std::cout << "in sample_mesh bound min z =" << bound_min.z << std::endl;
    std::cout << "in sample_mesh bound max x =" << bound_max.x << std::endl;
    std::cout << "in sample_mesh bound max y =" << bound_max.y << std::endl;
    std::cout << "in sample_mesh bound max z =" << bound_max.z << std::endl;

    UniformGridBuilder grid_builder { UniformGridBuilder(
        bound_min, bound_max, 2.f * h) };
    // abuse the mass field for resorting during grid construction
    // use a buffer for cell count prefix sum that outlives the grid_builder
    UniformGrid<Resort::yes> grid { grid_builder.construct_and_reorder(
        2.f * h, m, prefix, xs, ys, zs) };

    BoundarySamples result { std::move(xs), std::move(ys), std::move(zs),
        std::move(m), std::move(prefix), grid, bound_min, bound_max };

    // 3: calculate boundary masses
    std::cout << "calculating bdy masses" << std::endl;
    calculate_boundary_masses(result, h, h_bdy, ρ₀, mass_refinement_iterations);

    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
};

TEST_CASE("Write Boundary Mesh Sampling for docs")
{
#ifdef BENCH
    const float h_bdy { 0.025 };
    const Mesh dragon = load_mesh_from_obj("scenes/dragon.obj");
    const Mesh cube = load_mesh_from_obj("scenes/cube.obj");

    // create and save the uniform sampling with spacing h
    const BoundarySamples uniform_dragon { sample_mesh(
        dragon, h_bdy, 1.0, 1.0, nullptr, 0) };
    save_to_ply("builddocs/_staticc/dragon_uniform.ply", uniform_dragon.xs,
        uniform_dragon.ys, uniform_dragon.zs);

    const BoundarySamples uniform_cube { sample_mesh(
        cube, h_bdy, 1.0, 1.0, nullptr, 0) };
    save_to_ply("builddocs/_staticc/cube_uniform.ply", uniform_cube.xs,
        uniform_cube.ys, uniform_cube.zs);

    // create and save a 100 iterations relaxed sampling with spacing h
    const BoundarySamples relaxed_dragon { sample_mesh(
        dragon, h_bdy, 1.0, 1.0, nullptr, 100) };
    save_to_ply("builddocs/_staticc/dragon_relaxed.ply", relaxed_dragon.xs,
        relaxed_dragon.ys, relaxed_dragon.zs);

    const BoundarySamples relaxed_cube { sample_mesh(
        cube, h_bdy, 1.0, 1.0, nullptr, 100) };
    save_to_ply("builddocs/_staticc/cube_relaxed.ply", relaxed_cube.xs,
        relaxed_cube.ys, relaxed_cube.zs);
#endif
}

TEST_CASE("Observe Influence of Stiffness on Convergence")
{
#ifdef BENCH
    // create a JSON description of the measurements performed
    std::ofstream description("builddocs/_staticc/relaxconv/description.json");
    description << "{\"traces\": [";

    const Mesh cube = load_mesh_from_obj("scenes/cube.obj");

    const float hs[] { 5e-3, 5e-2, 5e-1 };
    const float ks[] { 0.01, 0.1, 0.5, 1., 1.5, 2.0 };
    const int iterations { 20 };
    uint i { 0 };

    for (const float h_bdy : hs)
        for (const float k : ks) {
            i += 1;
            const auto filename { std::format("{}-{}.csv", k, h_bdy) };
            std::ofstream out("builddocs/_staticc/relaxconv/" + filename);
            const BoundarySamples sampling { sample_mesh(
                cube, h_bdy, 1.0, 2.0, &out, iterations, k) };
            description << "{\"k\":" << k << ",\"h_bdy\":" << h_bdy
                        << ",\"filename\":\"" << filename
                        << "\", \"count\":" << sampling.xs.size() << "}";
            // prevent trailing commas in JSON by tracking the number of traces
            // to produce
            if (i * sizeof(float) * sizeof(float) < sizeof(hs) * sizeof(ks)) {
                description << ",";
            };
        }
    description << "]}";
#endif
}
