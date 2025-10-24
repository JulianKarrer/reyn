#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/scan.h>

#include <random>

#include "datastructure/uniformgrid.cuh"
#include "doctest/doctest.h"
#include <nanobench.h>

/// @brief Called first during uniform grid construction: atomically count the
/// number of particles in each grid cell
__global__ void _count_particles_per_cell(const uint N,
    const float3* __restrict__ x, uint* __restrict__ counts, const uint nx,
    const uint nxny, const float3 bound_min, const float cell_size)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // compute linearized cell index of particle position
    auto index_linear = _index_linear(x[i], bound_min, cell_size, nx, nxny);
    // also, atomically increment the particle count for the cell in the linear
    // list of grid cells
    atomicAdd(&counts[index_linear], 1); // don't use return value
}

/// @brief Called last during uniform grid construction: couting-sort particle
/// indices by obtaining the index into particles of the same cell from the
/// prefix sum and the index within the cell from atomically decrementing the
/// particle counts per cell
__global__ void _counting_sort(const uint N, const float3* __restrict__ x,
    uint* __restrict__ sorted, uint* __restrict__ counts,
    const uint* __restrict__ prefix, const uint nx, const uint nxny,
    const float3 bound_min, const float cell_size)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // recompute linearized particle index
    uint index_linear { _index_linear(x[i], bound_min, cell_size, nx, nxny) };
    // the index to the first particle in the same cell as particle i is given
    // by the number of particles with a lower index, i.e. the prefix sum at i
    uint offset_to_cell { prefix[index_linear] };
    // the offset of i within the cell can be computed by atomically choosing
    // one of the 1..m numbers making up the count of m particles in the same
    // cell, then subtracting one to get an index 0..m-1 for m particles in the
    // same cell
    uint offset_in_cell { atomicSub(&counts[index_linear], 1) - 1 };
    // with both offsets, particle i has a unique spot in the sorted array of
    // particles and can write its index to the corresponding spot
    sorted[offset_to_cell + offset_in_cell] = i;
}

UniformGrid::UniformGrid(const float3 bound_min, const float3 bound_max,
    const float cell_size)
    : // save the cell size of the uniform grid
    _cell_size(cell_size)
    ,
    // the lower bound is offset by a safety margin to make sure all queries
    // in the bounds yield valid indices
    _bound_min(bound_min - v3(cell_size))
    ,
    // after setting bounds and cells size and BEFORE (!) initializing buffers
    // compute their size: compute the number of grid cells along each spatial
    // dimension add one cell size along each axis to account for margins of
    // half a cell
    nxyz(ceil_div(bound_max - _bound_min + v3(2. * cell_size), cell_size))
    ,
    // initialize counts to zero
    counts(nxyz.x * nxyz.y * nxyz.z, 0)
    ,
    // initialization of prefix does not matter, it is overwritten with counts
    prefix(nxyz.x * nxyz.y * nxyz.z)
    ,
    // allocate only minimal memory for now, sorted will be resized to fit the
    // number of particles whenever required
    sorted(1)
{
    // assert that the number of grid cells along each axis is non-negative
    if (nxyz.x <= 0 || nxyz.y <= 0 || nxyz.z <= 0)
        throw std::runtime_error("Negative number of grid cells encountered in "
                                 "construction of uniform grid.");
};

DeviceUniformGrid UniformGrid::update_and_get_pod(const DeviceBuffer<float3>& x)
{
    // get the number of particles
    const uint N { (uint)x.size() };

    // resize the buffer of sorted indices to fit the number of particles, if
    // required initialization does not matter since everything is overwritten
    if (sorted.size() != N)
        sorted.resize(N);

    // - compute the cell index of each particle
    // - linearize it to obtain a pointer into the flat `counts` array
    // - and atomically increment the particle count in the `counts`
    _count_particles_per_cell<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(),
        counts.ptr(), nxyz.x, nxyz.x * nxyz.y, _bound_min, _cell_size);
    CUDA_CHECK(cudaGetLastError());

    // copy counts -> prefix
    // this means one copy of counts can be atomically decremented to sort,
    // while another provides offsets by storing the number of particles with a
    // lower linear index (i.e. the result of the exclusive prefix sum or
    // prescan)
    thrust::copy(
        counts.get().begin(), counts.get().end(), prefix.get().begin());

    // then, take a prefix sum of the device vector
    thrust::exclusive_scan(
        prefix.get().begin(), prefix.get().end(), prefix.get().begin());

    // finally, perform a counting sort:
    // the prefix sum is an offset to particles in the same cell, atomicSub on
    // counts then distributes unique offsets on top of that for each particle
    // in the same cell
    _counting_sort<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(), sorted.ptr(),
        counts.ptr(), prefix.ptr(), nxyz.x, nxyz.x * nxyz.y, _bound_min,
        _cell_size);
    CUDA_CHECK(cudaGetLastError());

    // pack all relevant pointers and information for queries into a POD struct
    // and return it
    return DeviceUniformGrid {
        .bound_min = _bound_min,
        .cell_size = _cell_size,
        .r_c_2 = _cell_size * _cell_size,
        .nx = nxyz.x,
        .nxny = nxyz.x * nxyz.y,
        .prefix = prefix.ptr(),
        .sorted = sorted.ptr(),
    };
}

// DeviceUniformGrid UniformGrid::update_reorder_and_get_pod(Particles& state)
// {
//     // get the number of particles
//     const uint N { (uint)state.x.size() };

//     // resize the buffer of sorted indices to fit the number of particles, if
//     // required initialization does not matter since everything is
//     overwritten if (sorted.size() != N)
//         sorted.resize(N);

//     // - compute the cell index of each particle
//     // - linearize it to obtain a pointer into the flat `counts` array
//     // - and atomically increment the particle count in the `counts`
//     _count_particles_per_cell<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(),
//         counts.ptr(), nxyz.x, nxyz.x * nxyz.y, _bound_min, _cell_size);
//     CUDA_CHECK(cudaGetLastError());

//     // copy counts -> prefix
//     // this means one copy of counts can be atomically decremented to sort,
//     // while another provides offsets by storing the number of particles with
//     a
//     // lower linear index (i.e. the result of the exclusive prefix sum or
//     // prescan)
//     thrust::copy(
//         counts.get().begin(), counts.get().end(), prefix.get().begin());

//     // then, take a prefix sum of the device vector
//     thrust::exclusive_scan(
//         prefix.get().begin(), prefix.get().end(), prefix.get().begin());

//     // finally, perform a counting sort:
//     // the prefix sum is an offset to particles in the same cell, atomicSub
//     on
//     // counts then distributes unique offsets on top of that for each
//     particle
//     // in the same cell
//     _counting_sort<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(), sorted.ptr(),
//         counts.ptr(), prefix.ptr(), nxyz.x, nxyz.x * nxyz.y, _bound_min,
//         _cell_size);
//     CUDA_CHECK(cudaGetLastError());

//     // pack all relevant pointers and information for queries into a POD
//     struct
//     // and return it
//     return DeviceUniformGrid {
//         .bound_min = _bound_min,
//         .cell_size = _cell_size,
//         .r_c_2 = _cell_size * _cell_size,
//         .nx = nxyz.x,
//         .nxny = nxyz.x * nxyz.y,
//         .prefix = prefix.ptr(),
//         .sorted = sorted.ptr(),
//     };
// }

// TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__global__ void _test_kernel_uniform_grid(const uint N,
    const float3* __restrict__ x, uint* __restrict__ count_out,
    float* __restrict__ len2_out, float3* __restrict__ vec_out,
    const DeviceUniformGrid grid, const float r_c_2)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    count_out[i] = grid.ff_nbrs(
        x, i, [r_c_2] __device__(auto i, auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? (i == j ? 0u : 1u) : 0u;
        });
    len2_out[i] = grid.ff_nbrs(
        x, i, [r_c_2] __device__(auto i, auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? dot(x_ij, x_ij) : 0.f;
        });
    vec_out[i] = grid.ff_nbrs(
        x, i, [r_c_2] __device__(auto i, auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? x_ij : v3(0.);
        });
}

__global__ void _test_kernel_uniform_grid_brute_force(const uint N,
    const float3* __restrict__ x, uint* __restrict__ count_out,
    float* __restrict__ len2_out, float3* __restrict__ vec_out,
    const DeviceUniformGrid grid, const float r_c_2)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    const float3 x_i { x[i] };
    for (uint j { 0 }; j < N; ++j) {
        const float3 x_ij { x_i - x[j] };
        const float x_ij_l2 { dot(x_ij, x_ij) };
        if (x_ij_l2 <= r_c_2) {
            count_out[i] += i == j ? 0u : 1u;
            len2_out[i] += x_ij_l2;
            vec_out[i] += x_ij;
        }
    }
}

TEST_CASE("Test Uniform Grid")
{
    const uint N { 50000 };
    const float box_size { 1.f };
    const float cell_size { 0.1f };
    const float r_c_2 { cell_size * cell_size };

    /// create a seeded pseudorandom vector of float3 uniformly randomly
    /// distributed in [0; box_size]^3 on the host side
    thrust::host_vector<float3> x_host(N);
    std::mt19937 rng(161420);
    std::uniform_real_distribution<float> uniform_dist(0.f, box_size);
    for (uint i { 0 }; i < N; ++i) {
        x_host[i] = v3(uniform_dist(rng), uniform_dist(rng), uniform_dist(rng));
    }

    // copy the random host-side buffer to the device
    DeviceBuffer<float3> x(N);
    thrust::copy(x_host.begin(), x_host.end(), x.get().begin());

    // create the uniform grid
    UniformGrid uni_grid { UniformGrid(v3(0.), v3(box_size), cell_size) };
    // build the device-side usable POD
    const DeviceUniformGrid grid { uni_grid.update_and_get_pod(x) };

    // allocate buffers for the results
    DeviceBuffer<uint> d_res_count(N, 0);
    DeviceBuffer<uint> d_res_count_bf(N, 0);
    DeviceBuffer<float> d_res_len2(N, 0.f);
    DeviceBuffer<float> d_res_len2_bf(N, 0.f);
    DeviceBuffer<float3> d_res_vec(N, v3(0.f));
    DeviceBuffer<float3> d_res_vec_bf(N, v3(0.f));

    // execute both kernels
    _test_kernel_uniform_grid_brute_force<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(),
        d_res_count_bf.ptr(), d_res_len2_bf.ptr(), d_res_vec_bf.ptr(), grid,
        r_c_2);
    CUDA_CHECK(cudaGetLastError());

    _test_kernel_uniform_grid<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(),
        d_res_count.ptr(), d_res_len2.ptr(), d_res_vec.ptr(), grid, r_c_2);
    CUDA_CHECK(cudaGetLastError());

    // copy back to host
    thrust::host_vector<uint> h_res_count(
        d_res_count.get().begin(), d_res_count.get().end());
    thrust::host_vector<uint> h_res_count_bf(
        d_res_count_bf.get().begin(), d_res_count_bf.get().end());
    thrust::host_vector<float> h_res_len2(
        d_res_len2.get().begin(), d_res_len2.get().end());
    thrust::host_vector<float> h_res_len2_bf(
        d_res_len2_bf.get().begin(), d_res_len2_bf.get().end());
    thrust::host_vector<float3> h_res_vec(
        d_res_vec.get().begin(), d_res_vec.get().end());
    thrust::host_vector<float3> h_res_vec_bf(
        d_res_vec_bf.get().begin(), d_res_vec_bf.get().end());

    SUBCASE("Uniform Grid Correctness")
    {
        // compare all results
        for (uint i { 0 }; i < N; ++i) {
            // CAPTURE(x.get()[i]);
            CAPTURE(i);
            // make sure there were no out of bounds positions due to potential
            // error in test setup
            CHECK(x_host[i].x >= 0.);
            CHECK(x_host[i].x <= box_size);
            CHECK(x_host[i].y >= 0.);
            CHECK(x_host[i].y <= box_size);
            CHECK(x_host[i].z >= 0.);
            CHECK(x_host[i].z <= box_size);
            // check if the brute-force O(N^2) approach and the uniform grid
            // agree
            CHECK(h_res_count[i] == h_res_count_bf[i]);
            CHECK(h_res_len2[i] == doctest::Approx(h_res_len2_bf[i]));
            CHECK(h_res_vec[i].x == doctest::Approx(h_res_vec_bf[i].x));
            CHECK(h_res_vec[i].y == doctest::Approx(h_res_vec_bf[i].y));
            CHECK(h_res_vec[i].z == doctest::Approx(h_res_vec_bf[i].z));
        }
    }

    // run benchmarks
    ankerl::nanobench::Bench().run("Uniform Grid Construction", [&]() {
        const DeviceUniformGrid grid { uni_grid.update_and_get_pod(x) };
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    ankerl::nanobench::Bench().minEpochIterations(5).run(
        "Uniform Grid Query", [&]() {
            _test_kernel_uniform_grid<<<BLOCKS(N), BLOCK_SIZE>>>(N, x.ptr(),
                d_res_count.ptr(), d_res_len2.ptr(), d_res_vec.ptr(), grid,
                r_c_2);
            CUDA_CHECK(cudaDeviceSynchronize());
        });
}
