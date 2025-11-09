#include "sample.cuh"

#include <utility>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <curand_kernel.h>
#include "vector_helper.cuh"
#include <thrust/extrema.h>
#include "kernels.cuh"
#include "common.h"

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
    num_den[i] = { grid.ff_nbrs(xs, ys, zs, i,
        [&W] __device__(auto i, auto j, const float3 x_ij, auto _x_ij_l2) {
            return W(x_ij);
        }) };
};

/// @brief Compute the closest point oin a triangle to a given query point
///
/// Algorithm from "Real-Time Collision Detection" by Christer Ericson
/// https://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
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
    float* __restrict__ ys, float* __restrict__ zs, double* __restrict__ vxs,
    double* __restrict__ vys, double* __restrict__ vzs,
    uint3* __restrict__ faces, int* __restrict__ tri_ids, float h_bdy,
    float* __restrict__ num_den, const float relaxation_factor)
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
    // rho_0 is not exactly the "rest density", especially since the suraface is
    // 2D embedded in a 3D space, but just a rough volume scale so the floating
    // point values are in more scale independent range, closer to one
    const float rho_0_inv
        = (h_bdy * h_bdy * h_bdy); // mass one over expected volume, inverted
    const float rho_i_inv = 1.f / num_den[i];
    const float3 dx_i { -relaxation_factor
        * grid.ff_nbrs(xs, ys, zs, i,
            [&W, &num_den, rho_i_inv, rho_0_inv] __device__(
                auto i, auto j, const float3 x_ij, auto _x_ij_l2) {
                // symmetric formula, like for pressure accelerations but with
                // rho = p
                return (rho_0_inv * rho_i_inv + rho_0_inv * 1. / num_den[j])
                    * W.nabla(x_ij);
            }) };
    const float3 new_x_i { v3(i, xs, ys, zs) + dx_i };

    // find the closest point to this which is inside the triangle and store the
    // result.
    // This writes to the buffer being read from with no synchronization!
    const float3 projected_new_x_i { _closest_point_on_triangle(
        new_x_i, vert1, vert2, vert3) };
    store_v3(projected_new_x_i, i, xs, ys, zs);
};

void sample_relaxation(BoundarySamples& samples,
    UniformGridBuilder& grid_builder, thrust::device_vector<double> vxs,
    thrust::device_vector<double> vys, thrust::device_vector<double> vzs,
    thrust::device_vector<uint3> faces, float h_bdy,
    thrust::device_vector<int> tri_ids, thrust::device_vector<float>& num_den,
    const float relaxation_factor)
{
    std::cout << "relaxing boundary particles" << std::endl;
    // build an acceleration datastructure to quickly find neighbours of
    // boundary samples. Don't use resorting, since that would obscur the
    // mapping to faces, which are needed to find the vertices associated with
    // each sample
    const UniformGrid<Resort::no> grid = grid_builder.construct(
        2.f * h_bdy, samples.xs, samples.ys, samples.zs);
    // use any kernel function, `B3` is used here
    const B3 W(2.f * h_bdy);
    // relax the sampling by moving each boundary particle in the direction of
    // the negative number density gradient, projected onto the plane of the
    // triangle it may move in and clamped to the edges of that triangle
    const uint N { (uint)samples.xs.size() };

    CUDA_CHECK(cudaDeviceSynchronize());

    _compute_number_densities<Resort::no><<<BLOCKS(N), BLOCK_SIZE>>>(N, W, grid,
        samples.xs.ptr(), samples.ys.ptr(), samples.zs.ptr(),
        thrust::raw_pointer_cast(num_den.data()));

    _relax_sampling<<<BLOCKS(N), BLOCK_SIZE>>>(N, W, grid, samples.xs.ptr(),
        samples.ys.ptr(), samples.zs.ptr(),
        thrust::raw_pointer_cast(vxs.data()),
        thrust::raw_pointer_cast(vys.data()),
        thrust::raw_pointer_cast(vzs.data()),
        thrust::raw_pointer_cast(faces.data()),
        thrust::raw_pointer_cast(tri_ids.data()), h_bdy,
        thrust::raw_pointer_cast(num_den.data()), relaxation_factor);
};

std::pair<float3, float3> _get_bounds(const thrust::device_vector<float>& xs,
    const thrust::device_vector<float>& ys,
    const thrust::device_vector<float>& zs)
{
    auto min_x = thrust::min_element(xs.begin(), xs.end());
    auto min_y = thrust::min_element(ys.begin(), ys.end());
    auto min_z = thrust::min_element(zs.begin(), zs.end());
    auto max_x = thrust::max_element(xs.begin(), xs.end());
    auto max_y = thrust::max_element(ys.begin(), ys.end());
    auto max_z = thrust::max_element(zs.begin(), zs.end());
    float3 min { v3(min_x[0], min_y[0], min_z[0]) };
    float3 max { v3(max_x[0], max_y[0], max_z[0]) };
    return { min, max };
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

__global__ void _uniform_sample_tri_cdf(const uint N, float* __restrict__ xs,
    float* __restrict__ ys, float* __restrict__ zs,
    const double* __restrict__ cdf, const int N_cdf, const double bin_size,
    const double total_area, curandStatePhilox4_32_10_t* __restrict__ states,
    double* __restrict__ vxs, double* __restrict__ vys,
    double* __restrict__ vzs, uint3* __restrict__ faces,
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
    // remember the triangle that was sampled for later resampling
    tri_ids[i] = tri_id;

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

BoundarySamples sample_mesh(const Mesh mesh, const float h,
    const double oversampling_factor, const int relaxation_iters)
{
    // move vertex data and faces to the GPU by constructing a
    // `DeviceBuffer` with a single cudaMemcpy underlying each move
    thrust::device_vector<double> vxs(mesh.xs.begin(), mesh.xs.end());
    thrust::device_vector<double> vys(mesh.ys.begin(), mesh.ys.end());
    thrust::device_vector<double> vzs(mesh.zs.begin(), mesh.zs.end());
    thrust::device_vector<uint3> faces(mesh.faces.begin(), mesh.faces.end());

    // calculate the surface areas of each face
    // then fuse this with a prefix sum to both compute the total area (last
    // element) and construct a discrete CDF for uniform sampling
    thrust::device_vector<double> area(mesh.face_count());
    auto vxs_d { thrust::raw_pointer_cast(vxs.data()) };
    auto vys_d { thrust::raw_pointer_cast(vys.data()) };
    auto vzs_d { thrust::raw_pointer_cast(vzs.data()) };
    const size_t vertex_count { mesh.vertex_count() };
    ::cuda::std::plus<double> plus_double_op;

    thrust::transform_inclusive_scan(
        thrust::device,
        // input iterator begin and end
        faces.begin(), faces.end(),
        // output iterator
        area.begin(),
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
    const double total_area {
        area[area.size() - 1] // load the last element from the device to host
                              // memory via thrust overload of operator[]
    };
    const uint bdy_count { (uint)(ceil(total_area / (h_bdy * h_bdy))) };
    const double bin_size { total_area / (double)(bdy_count) };

    std::cout << std::format(
        "total area: {}, h_bdy: {}, bdy_count: {}, face_count: {}", total_area,
        h_bdy, bdy_count, mesh.face_count())
              << std::endl;

    // create `BoundarySamples` struct to return with correct number of
    // entries
    BoundarySamples res {
        DeviceBuffer<float>(bdy_count),
        DeviceBuffer<float>(bdy_count),
        DeviceBuffer<float>(bdy_count),
    };

    // set up one cuRand rng per thread using cudaMalloc to be freed later
    curandStatePhilox4_32_10_t* states;
    cudaMalloc(&states, bdy_count * sizeof(curandStatePhilox4_32_10_t));
    _init_curand<<<BLOCKS(bdy_count), BLOCK_SIZE>>>(
        bdy_count, states, 1614201312);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::cout << "curand initialized" << std::endl;

    // when uniformly sampling, keep track of which sample was placed on which
    // triangle by its id into the `faces` buffer
    thrust::device_vector<int> tri_ids(bdy_count);

    // uniformly randomly sample the mesh
    _uniform_sample_tri_cdf<<<BLOCKS(bdy_count), BLOCK_SIZE>>>(bdy_count,
        res.xs.ptr(), res.ys.ptr(), res.zs.ptr(),
        thrust::raw_pointer_cast(area.data()), mesh.face_count(), bin_size,
        total_area, states, thrust::raw_pointer_cast(vxs.data()),
        thrust::raw_pointer_cast(vys.data()),
        thrust::raw_pointer_cast(vzs.data()),
        thrust::raw_pointer_cast(faces.data()), vertex_count,
        thrust::raw_pointer_cast(tri_ids.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::cout << "boundary sampling complete" << std::endl;

    // clean up cuRand states
    cudaFree(states);

    // if requested, produce a blue noise sampling from the inital, uniformly
    // random sampling
    if (relaxation_iters > 0) {
        // get the bounds of the particles
        auto [min_bound, max_bound]
            = _get_bounds(res.xs.get(), res.ys.get(), res.zs.get());
        std::cout << std::format("bounds: min {} {} {} - max {} {} {}",
            min_bound.x, min_bound.y, min_bound.z, max_bound.x, max_bound.y,
            max_bound.z)
                  << std::endl;
        // relax the sampling to decrease discrepancy
        float h_relax_radius { (float)h_bdy * 2.f };
        // in gridbuilder, use larger safety margin of 5h but reuse the
        // UniformGridBuilder
        UniformGridBuilder grid_builder { UniformGridBuilder(
            min_bound - v3(5.f * (float)h_bdy),
            max_bound + v3(5.f * (float)h_bdy), 2.f * (float)h_bdy) };
        // allocate a buffer to hold number densities
        thrust::device_vector<float> num_den(bdy_count);

        for (uint i { 0 }; i < relaxation_iters; ++i) {
            sample_relaxation(res, grid_builder, vxs, vys, vzs, faces,
                (float)h_bdy, tri_ids, num_den, 50.);
        }
    }
    return res;
};