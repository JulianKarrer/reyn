#include "scene/sample.cuh"
#include "vector_helper.cuh"
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

#include <curand.h>
#include <curand_kernel.h>

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
    const uint vertex_count)
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

BoundarySamples sample_mesh(const Mesh mesh, const float h)
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
            assert(ids.x >= 0 && ids.x < vertex_count && ids.y >= 0
                && ids.y < vertex_count && ids.z >= 0 && ids.z < vertex_count);
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
    const double h_bdy { 0.5 * (double)h };
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

    _uniform_sample_tri_cdf<<<BLOCKS(bdy_count), BLOCK_SIZE>>>(bdy_count,
        res.xs.ptr(), res.ys.ptr(), res.zs.ptr(),
        thrust::raw_pointer_cast(area.data()), mesh.face_count(), bin_size,
        total_area, states, thrust::raw_pointer_cast(vxs.data()),
        thrust::raw_pointer_cast(vys.data()),
        thrust::raw_pointer_cast(vzs.data()),
        thrust::raw_pointer_cast(faces.data()), vertex_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::cout << "boundary sampling complete" << std::endl;

    // clean up cuRand states
    cudaFree(states);

    return res;
};
