#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <iostream>
#include <fstream>

#include <random>

#include "datastructure/uniformgrid.cuh"
#include "doctest/doctest.h"
#include <nanobench.h>
#include "kernels.cuh"

/// @brief Called first during uniform grid construction: atomically count the
/// number of particles in each grid cell
__global__ void _count_particles_per_cell(const uint N,
    const float* __restrict__ xx, const float* __restrict__ xy,
    const float* __restrict__ xz, uint* __restrict__ counts, const uint nx,
    const uint nxny, const float3 bound_min, const float cell_size)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // compute linearized cell index of particle position
    const float3 x_i { v3(i, xx, xy, xz) };
    auto index_linear = _index_linear(x_i, bound_min, cell_size, nx, nxny);
    // also, atomically increment the particle count for the cell in the linear
    // list of grid cells
    atomicAdd(&counts[index_linear], 1); // don't use return value
}

/// @brief Called last during uniform grid construction: couting-sort particle
/// indices by obtaining the index into particles of the same cell from the
/// prefix sum and the index within the cell from atomically decrementing the
/// particle counts per cell
__global__ void _counting_sort(const uint N, const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    uint* __restrict__ sorted, uint* __restrict__ counts,
    const uint* __restrict__ prefix, const uint nx, const uint nxny,
    const float3 bound_min, const float cell_size)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // recompute linearized particle index
    const float3 x_i { v3(i, xx, xy, xz) };
    uint index_linear { _index_linear(x_i, bound_min, cell_size, nx, nxny) };
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

UniformGridBuilder::UniformGridBuilder(const float3 bound_min,
    const float3 bound_max,
    const float search_radius)
    : // save the cell size of the uniform grid
    _cell_size(search_radius / 2.f)
    ,
    // the lower bound is offset by a safety margin to make sure all queries
    // in the bounds yield valid indices
    _bound_min(bound_min - v3(search_radius * 2.))
    ,
    // after setting bounds and cells size and BEFORE (!) initializing buffers
    // compute their size: compute the number of grid cells along each spatial
    // dimension add one cell size along each axis to account for margins of
    // half a cell
    nxyz(ceil_div(
        (bound_max + v3(search_radius * 2.)) - _bound_min, search_radius / 2.f))
    ,
    // initialize counts to zero
    counts(nxyz.x * nxyz.y * nxyz.z, 0)
    ,
    // initialization of prefix does not matter, it is overwritten with counts
    _prefix(1)
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

void UniformGridBuilder::_construct(const DeviceBuffer<float>& xx,
    const DeviceBuffer<float>& xy, const DeviceBuffer<float>& xz,
    DeviceBuffer<uint>& prefix)
{
    // get the number of particles
    const uint N { (uint)xx.size() };
    // the number of entires must be the same across all components
    assert(xx.size() == xy.size() && xx.size() == xz.size());

    // resize the buffer of sorted indices to fit the number of particles,
    // if required.
    //  initialization does not matter since everything is overwritten
    sorted.resize(N);

    // - compute the cell index of each particle
    // - linearize it to obtain a pointer into the flat `counts` array
    // - and atomically increment the particle count in the `counts`
    _count_particles_per_cell<<<BLOCKS(N), BLOCK_SIZE>>>(N, xx.ptr(), xy.ptr(),
        xz.ptr(), counts.ptr(), nxyz.x, nxyz.x * nxyz.y, _bound_min,
        _cell_size);
    CUDA_CHECK(cudaGetLastError());

    // copy counts -> prefix
    if (prefix.size() < nxyz.x * nxyz.y * nxyz.z) {
        prefix.get().resize(nxyz.x * nxyz.y * nxyz.z);
    }
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
    _counting_sort<<<BLOCKS(N), BLOCK_SIZE>>>(N, xx.ptr(), xy.ptr(), xz.ptr(),
        sorted.ptr(), counts.ptr(), prefix.ptr(), nxyz.x, nxyz.x * nxyz.y,
        _bound_min, _cell_size);
    CUDA_CHECK(cudaGetLastError());
}

UniformGrid<Resort::no> UniformGridBuilder::construct(const float search_radius,
    const DeviceBuffer<float>& xx, const DeviceBuffer<float>& xy,
    const DeviceBuffer<float>& xz)
{
    // reconstruct the datastructure from current position data
    _construct(xx, xy, xz, _prefix);
    // pack all relevant pointers and information for queries into a POD struct
    // and return it
    return UniformGrid<Resort::no> { { .sorted = sorted.ptr() }, _bound_min,
        _cell_size, search_radius * search_radius, nxyz.x, nxyz.x * nxyz.y,
        _prefix.ptr() };
}

UniformGrid<Resort::yes> UniformGridBuilder::construct_and_reorder(
    const float search_radius, DeviceBuffer<float>& tmp, Particles& state)
{
    // reconstruct the datastructure from current position data
    _construct(state.xx, state.xy, state.xz, _prefix);
    // then reorder the particle state such that the sorted array becomes the
    // sequence 0...N-1, i.e. redirection through sorted becomes superfluous and
    // a more efficient uniform grid can be returned
    tmp.resize(sorted.size());
    state.reorder(sorted, tmp);

    // pack all relevant pointers and information for queries into a POD
    // struct and return it
    // indicate that corresponding buffers are sorted so no indirection is
    // needed in the query
    return UniformGrid<Resort::yes> {
        .bound_min = _bound_min,
        .cell_size = _cell_size,
        .r_c_2 = search_radius * search_radius,
        .nx = nxyz.x,
        .nxny = nxyz.x * nxyz.y,
        .prefix = _prefix.ptr(),
    };
}

// TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__global__ void _test_kernel_uniform_grid(const uint N,
    const float* __restrict__ xx, const float* __restrict__ xy,
    const float* __restrict__ xz, uint* __restrict__ count_out,
    float* __restrict__ len2_out, float3* __restrict__ vec_out,
    const UniformGrid<Resort::no> grid, const float r_c_2)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    const float3 x_i { v3(i, xx, xy, xz) };
    count_out[i] = grid.ff_nbrs(x_i, xx, xy, xz,
        [r_c_2, i] __device__(auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? (i == j ? 0u : 1u) : 0u;
        });
    len2_out[i] = grid.ff_nbrs(x_i, xx, xy, xz,
        [r_c_2, i] __device__(auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? dot(x_ij, x_ij) : 0.f;
        });
    vec_out[i] = grid.ff_nbrs(x_i, xx, xy, xz,
        [r_c_2, i] __device__(auto j, auto x_ij, auto x_ij_l2) {
            return (x_ij_l2 <= r_c_2) ? x_ij : v3(0.);
        });
}
__global__ void _test_kernel_uniform_grid_brute_force(const uint N,
    const float* __restrict__ xx, const float* __restrict__ xy,
    const float* __restrict__ xz, uint* __restrict__ count_out,
    float* __restrict__ len2_out, float3* __restrict__ vec_out,
    const UniformGrid<Resort::no> grid, const float r_c_2)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    const float3 x_i { v3(i, xx, xy, xz) };
    for (uint j { 0 }; j < N; ++j) {
        const float3 x_ij { x_i - v3(j, xx, xy, xz) };
        const float x_ij_l2 { dot(x_ij, x_ij) };
        if (x_ij_l2 <= r_c_2) {
            count_out[i] += i == j ? 0u : 1u;
            len2_out[i] += x_ij_l2;
            vec_out[i] += x_ij;
        }
    }
}

template <IsKernel K, Resort R>
__global__ void _benchmark_kernel(const uint N, const float* __restrict__ xx,
    const float* __restrict__ xy, const float* __restrict__ xz,
    float* __restrict__ res_out, const UniformGrid<R> grid, const K W)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    const float3 x_i { v3(i, xx, xy, xz) };
    res_out[i] = grid.ff_nbrs(
        x_i, xx, xy, xz, [=] __device__(auto j, auto x_ij, auto x_ij_l2) {
            return xx[i] * xx[j] * W(x_ij);
        });
}

TEST_CASE("Test Uniform Grid")
{

#ifdef BENCH
    const uint nanobench_minepoch_qeuery { 50 };
    const uint side_length { 100 };
#else
    const uint nanobench_minepoch_qeuery { 10 };
    const uint side_length { 50 };
#endif
    const uint N { side_length * side_length * side_length };
    const float h { 1.1 };
    const float box_size { h * (float)side_length };
    const float search_radius { 2. * h };
    const float r_c_2 { search_radius * search_radius };

    /// create a seeded pseudorandom vector of float3 uniformly randomly
    /// distributed in [0; box_size]^3 on the host side
    thrust::host_vector<float> xx_host(N);
    thrust::host_vector<float> xy_host(N);
    thrust::host_vector<float> xz_host(N);
    std::mt19937 rng(161);
    std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);

    for (uint i { 0 }; i < N; ++i) {
        xx_host[i] = box_size * uniform_dist(rng);
        xy_host[i] = box_size * uniform_dist(rng);
        xz_host[i] = box_size * uniform_dist(rng);
    }

    // copy the random host-side buffer to the device
    DeviceBuffer<float> xx(N);
    DeviceBuffer<float> xy(N);
    DeviceBuffer<float> xz(N);
    thrust::copy(xx_host.begin(), xx_host.end(), xx.get().begin());
    thrust::copy(xy_host.begin(), xy_host.end(), xy.get().begin());
    thrust::copy(xz_host.begin(), xz_host.end(), xz.get().begin());

    // create the uniform grid
    UniformGridBuilder uni_grid { UniformGridBuilder(
        v3(0.), v3(box_size), search_radius) };
    // build the device-side usable POD
    const UniformGrid grid { uni_grid.construct(search_radius, xx, xy, xz) };

    // allocate buffers for the results
    DeviceBuffer<uint> d_res_count(N, 0);
    DeviceBuffer<uint> d_res_count_bf(N, 0);
    DeviceBuffer<float> d_res_len2(N, 0.f);
    DeviceBuffer<float> d_res_len2_bf(N, 0.f);
    DeviceBuffer<float3> d_res_vec(N, v3(0.f));
    DeviceBuffer<float3> d_res_vec_bf(N, v3(0.f));

    // execute both kernels
    _test_kernel_uniform_grid_brute_force<<<BLOCKS(N), BLOCK_SIZE>>>(N,
        xx.ptr(), xy.ptr(), xz.ptr(), d_res_count_bf.ptr(), d_res_len2_bf.ptr(),
        d_res_vec_bf.ptr(), grid, r_c_2);
    CUDA_CHECK(cudaGetLastError());

    _test_kernel_uniform_grid<<<BLOCKS(N), BLOCK_SIZE>>>(N, xx.ptr(), xy.ptr(),
        xz.ptr(), d_res_count.ptr(), d_res_len2.ptr(), d_res_vec.ptr(), grid,
        r_c_2);
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
            // make sure there were no out of bounds positions due to
            // potential error in test setup
            CHECK(xx_host[i] >= 0.);
            CHECK(xx_host[i] <= box_size);
            CHECK(xy_host[i] >= 0.);
            CHECK(xy_host[i] <= box_size);
            CHECK(xz_host[i] >= 0.);
            CHECK(xz_host[i] <= box_size);
            // check if the brute-force O(N^2) approach and the uniform grid
            // agree
            CHECK(h_res_count[i] == h_res_count_bf[i]);
            CHECK(h_res_len2[i] == doctest::Approx(h_res_len2_bf[i]));
            CHECK(h_res_vec[i].x == doctest::Approx(h_res_vec_bf[i].x));
            CHECK(h_res_vec[i].y == doctest::Approx(h_res_vec_bf[i].y));
            CHECK(h_res_vec[i].z == doctest::Approx(h_res_vec_bf[i].z));
        }
    }

    // prepare benchmarks
    // use half-jittered uniform grid for more realistic setting for SPH
    uint i_grid { 0 };
    for (uint x { 0 }; x < side_length; ++x)
        for (uint y { 0 }; y < side_length; ++y)
            for (uint z { 0 }; z < side_length; ++z) {
                xx_host[i_grid] = (float)x * h + 0.5 * h * uniform_dist(rng);
                xy_host[i_grid] = (float)y * h + 0.5 * h * uniform_dist(rng);
                xz_host[i_grid] = (float)z * h + 0.5 * h * uniform_dist(rng);
                ++i_grid;
            }
    thrust::copy(xx_host.begin(), xx_host.end(), xx.get().begin());
    thrust::copy(xy_host.begin(), xy_host.end(), xy.get().begin());
    thrust::copy(xz_host.begin(), xz_host.end(), xz.get().begin());

    // create a random permutation of the sequence 0..=N-1 to shuffle all three
    // cooridnates with the same random order
    thrust::device_vector<uint> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());
    thrust::default_random_engine rng_d(1614201312);
    thrust::shuffle(permutation.begin(), permutation.end(), rng_d);

    // reorder coordinates randomly, using d_res_len2_bf as a temporary buffer
    thrust::gather(permutation.begin(), permutation.end(), xx.get().begin(),
        d_res_len2_bf.get().begin());
    d_res_len2_bf.get().swap(xx.get());
    thrust::gather(permutation.begin(), permutation.end(), xy.get().begin(),
        d_res_len2_bf.get().begin());
    d_res_len2_bf.get().swap(xy.get());
    thrust::gather(permutation.begin(), permutation.end(), xz.get().begin(),
        d_res_len2_bf.get().begin());
    d_res_len2_bf.get().swap(xz.get());

    // run benchmarks
    ankerl::nanobench::Bench bench;
    bench.title("Uniform Grid Benchmarks");
    const B3 W(2.f * h);

    bench.minEpochIterations(nanobench_minepoch_qeuery)
        .run("Construction (No Reordering)", [&]() {
            const UniformGrid _ { uni_grid.construct(2. * h, xx, xy, xz) };
            CUDA_CHECK(cudaDeviceSynchronize());
        });

    const UniformGrid unordered_grid { uni_grid.construct(2. * h, xx, xy, xz) };
    bench.minEpochIterations(nanobench_minepoch_qeuery)
        .run("Query (No Reordering)", [&]() {
            _benchmark_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(N, xx.ptr(), xy.ptr(),
                xz.ptr(), d_res_len2.ptr(), unordered_grid, W);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

    // now reorder the position attributes using d_res_len2 as a temprorary
    // buffer

    bench.run("Construction (Reordering)", [&]() {
        // to make the comparison more fair, reorder 4 additional arrays to
        // emulate reordering m,vx,vy,vz - but in this benchmark just reorder
        // the same array 4 times for this
        const UniformGrid _ { uni_grid.construct_and_reorder(2. * h, d_res_len2,
            xx, xy, xz, &d_res_len2_bf, &d_res_len2_bf, &d_res_len2_bf,
            &d_res_len2_bf) };
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    const UniformGrid ordered_grid { uni_grid.construct_and_reorder(
        2. * h, d_res_len2, xx, xy, xz) };
    bench.minEpochIterations(nanobench_minepoch_qeuery)
        .run("Query (Reordering)", [&]() {
            _benchmark_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(N, xx.ptr(), xy.ptr(),
                xz.ptr(), d_res_len2.ptr(), ordered_grid, W);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

#ifdef BENCH
    // output benchmark to json
    std::ofstream jsonout("./builddocs/_staticc/benchmarks/Current.json");
    bench.render(ankerl::nanobench::templates::json(), jsonout);
#endif
}
