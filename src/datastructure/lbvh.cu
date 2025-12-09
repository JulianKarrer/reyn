#include "datastructure/lbvh.cuh"

#include "doctest/doctest.h"
#include <nanobench.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <random>

__host__ __device__ inline static AABB AABB_from_verts(
    double3 vert1, double3 vert2, double3 vert3)
{
    const float3 mini { v3(
        static_cast<float>(fmin(fmin(vert1.x, vert2.x), vert3.x)),
        static_cast<float>(fmin(fmin(vert1.y, vert2.y), vert3.y)),
        static_cast<float>(fmin(fmin(vert1.z, vert2.z), vert3.z))) };
    const float3 maxi { v3(
        static_cast<float>(fmax(fmax(vert1.x, vert2.x), vert3.x)),
        static_cast<float>(fmax(fmax(vert1.y, vert2.y), vert3.y)),
        static_cast<float>(fmax(fmax(vert1.z, vert2.z), vert3.z))) };
    const float3 v { maxi - mini };
    // prevent degenerate AABBs with zero volume:
    if (min(v.x, min(v.y, v.z)) == 0.f) {
        // add a small epsilon offset relative to largest extent to mini and
        // maxi
        const float3 ε { v3(FLT_EPSILON) };
        // const float max_extent { max(v.x, max(v.y, v.z)) };
        // const float3 padding { v3(1e-8f * max_extent) };
        // return AABB { mini - padding, maxi + padding };
        return AABB { mini - ε, maxi + ε };
    }
    return AABB { mini, maxi };
}

__global__ void generate_tree(const uint64_t* __restrict__ codes, const uint N,
    ChildNode* __restrict__ children_l, ChildNode* __restrict__ children_r,
    uint* __restrict__ parent, uint* __restrict__ leaf_parent)
{
    const uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N - 1)
        return;

    // find the range of values corresponding to the current node
    const int2 range = determine_range(codes, N, static_cast<int>(i));
    const int first { range.x };
    const int last { range.y };

    // determine where to split the node
    const int split { find_split(codes, first, last) };

    // record left child
    const uint32_t l_idx { (uint32_t)split };
    const bool l_leaf { l_idx == first };
    children_l[i] = ChildNode::create(l_idx, l_leaf);

    // record right child
    const uint32_t r_idx { (uint32_t)split + 1 };
    const bool r_leaf { r_idx == last };
    children_r[i] = ChildNode::create(r_idx, r_leaf);

    // record parent node of internal nodes
    if (l_leaf) {
        leaf_parent[split] = i;
    } else {
        parent[split] = i;
    }

    if (r_leaf) {
        leaf_parent[split + 1] = i;
    } else {
        parent[split + 1] = i;
    }
}

__global__ void compute_aabbs(const uint* __restrict__ parent,
    const uint* __restrict__ leaf_parent, uint* __restrict__ visited,
    const ChildNode* __restrict__ children_l,
    const ChildNode* __restrict__ children_r, const uint N,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces,
    AABB* __restrict__ aabbs_leaf, AABB* __restrict__ aabbs_internal)
{
    uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;

    // first, compute the initial AABB for the leaf node i
    // leaves correspond directly to faces
    const uint3 face { faces[i] };
    const double3 vert1 { dv3(vxs[face.x], vys[face.x], vzs[face.x]) };
    const double3 vert2 { dv3(vxs[face.y], vys[face.y], vzs[face.y]) };
    const double3 vert3 { dv3(vxs[face.z], vys[face.z], vzs[face.z]) };
    AABB aabb { AABB_from_verts(vert1, vert2, vert3) };
    // store the computed leaf aabb
    aabbs_leaf[i] = aabb;
    bool i_am_leaf { true };

    do {
        // find parent node
        // (i only referrs to leaf in the first iteration, see `i_am_leaf`)
        uint i_parent { i_am_leaf ? leaf_parent[i] : parent[i] };

        // make previous write to aabb buffers visible to all other threads
        // before reading flags and updating them atomically
        __threadfence();

        // contest with other thread, only the second thread may continue
        uint own_thread_order { atomicAdd(&visited[i_parent], 1) };
        // only the second thread may proceed, ensuring that all sibling nodes
        // are done
        if (own_thread_order == 0) {
            return;
        }
        // check if current thread ascended the tree from the left or right,
        // fetching the sibling id from the repective other `children` array
        const ChildNode sibling_l { children_l[i_parent] };
        const uint32_t sibling_l_idx { sibling_l.idx() };
        if (sibling_l_idx == i) {
            // current thread is left child of i_parent:
            // load right child as well
            const ChildNode sibling_r { children_r[i_parent] };
            const uint32_t sibling_idx { sibling_r.idx() };
            // get aabb from leaf or internal array
            const AABB sibling_aabb { sibling_r.is_leaf()
                    ? aabbs_leaf[sibling_idx]
                    : aabbs_internal[sibling_idx] };
            // combine the sibling's AABB with the own AABB
            aabb = aabb | sibling_aabb;
        } else {
            // current thread is right child of i_parent
            // left sibling id is already known, lead its AABB
            const AABB sibling_aabb { sibling_l.is_leaf()
                    ? aabbs_leaf[sibling_l_idx]
                    : aabbs_internal[sibling_l_idx] };
            // combine the sibling's AABB with the own AABB
            aabb = aabb | sibling_aabb;
        }

        // this is the parent internal node's aabb
        aabbs_internal[i_parent] = aabb;
        // make the parent node the current one, get a new parent
        i = i_parent;
        // from now on, look parent node must be found in `parent`, not
        // `leaf_parent`
        i_am_leaf = false;
    } while (i > 0);

    // if we arrive here, i==0 is the root node
    assert(i == 0);
}

__global__ void compute_tree_height(const uint N,
    const uint* __restrict__ leaf_parents, const uint* __restrict__ parent,
    uint* __restrict__ depths)
{
    // compute own index
    uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    // start from the parent of the i-th leaf node
    uint parent_i { leaf_parents[i] };
    uint depth { 1 };
    // iteratively traverse the tree upwards until the root (i=0) is reached
    while (parent_i) {
        parent_i = parent[parent_i];
        depth += 1;
    }
    // arrived at the root node, store accumulated depth
    assert(parent_i == 0);
    depths[i] = depth;
};

/// @brief Check if a BVH leaf node AABB contains the vertices of the
/// corresponding primitive/face of a triangular mesh. Used to check the
/// correctness of the LBVH implementation.
/// @param leaf_aabbs array of leaf-node AABBs
/// @param faces faces of the triangular mesh, indexing into the vertex
/// buffers
/// @param vxs x-component of the vertex buffer of the triangular mesh
/// @param vys y-component of the vertex buffer of the triangular mesh
/// @param vzs z-component of the vertex buffer of the triangular mesh
/// @param i index of the leaf node in `leaf_aabbs` to check
/// @return whether all three vertices are contained in the associated AABB
bool check_lbvh_correctness_leaf(const thrust::host_vector<AABB>& leaf_aabbs,
    const thrust::host_vector<uint3>& faces,
    const thrust::host_vector<double>& vxs,
    const thrust::host_vector<double>& vys,
    const thrust::host_vector<double>& vzs, const uint i)
{
    // get current leaf AABB
    const AABB aabb { leaf_aabbs[i] };
    CHECK(!(aabb.mini >= aabb.maxi));
    // load all vertices of the primitive
    const uint3 face { faces[i] };
    const float3 vert1 { v3(static_cast<float>(vxs[face.x]),
        static_cast<float>(vys[face.x]), static_cast<float>(vzs[face.x])) };
    const float3 vert2 { v3(static_cast<float>(vxs[face.y]),
        static_cast<float>(vys[face.y]), static_cast<float>(vzs[face.y])) };
    const float3 vert3 { v3(static_cast<float>(vxs[face.z]),
        static_cast<float>(vys[face.z]), static_cast<float>(vzs[face.z])) };
    // for each vertex, check if it is contained in the aabb and return the
    // result
    return aabb.contains(vert1) && aabb.contains(vert2) && aabb.contains(vert3);
}

/// @brief Check whether a constructed BVH is correct, in the sense that a
/// recursive descent from the root node yields a series of child AABBs that are
/// entirely contained in the parent AABB, and the leaf nodes entirely contain
/// the respective primitives they represent, as checked by
/// `check_lbvh_correctness_leaf`.
/// @param children_l index of the left child node, and whether it is a leaf
/// @param children_r index of the right child node, and whether it is a leaf
/// @param leaf_aabbs the AABBs of all leaf nodes in order
/// @param internal_aabbs the AABBs of all internal nodes in order
/// @param faces faces of the triangular mesh to check against
/// @param vxs x-component of the vertex buffer of the triangular mesh to check
/// @param vys y-component of the vertex buffer of the triangular mesh to check
/// @param vzs z-component of the vertex buffer of the triangular mesh to check
/// @param i current index of the internal node to check. Initially zero at the
/// root node, then changed to child indices during recursion
/// @return whether or not all child AABBs and primitives are contained in their
/// respective parent nodes for a given triangular mesh and corresponding LBVH
bool check_lbvh_correctness_internal(
    const thrust::host_vector<ChildNode>& children_l,
    const thrust::host_vector<ChildNode>& children_r,
    const thrust::host_vector<AABB>& leaf_aabbs,
    const thrust::host_vector<AABB>& internal_aabbs,
    const thrust::host_vector<uint3>& faces,
    const thrust::host_vector<double>& vxs,
    const thrust::host_vector<double>& vys,
    const thrust::host_vector<double>& vzs, const uint i)
{
    // get left and right child indices and check if they are leaves
    const ChildNode l { children_l[i] };
    const ChildNode r { children_r[i] };

    const uint32_t i_l { l.idx() };
    const uint32_t i_r { r.idx() };

    const bool l_leaf { l.is_leaf() };
    const bool r_leaf { r.is_leaf() };
    // get the respective AABBs
    const AABB l_aabb { (l_leaf ? leaf_aabbs : internal_aabbs)[i_l] };
    const AABB r_aabb { (r_leaf ? leaf_aabbs : internal_aabbs)[i_r] };
    // assert that AABBs do not violate mini <= maxi invariant
    CHECK(!(l_aabb.mini >= l_aabb.maxi));
    CHECK(!(r_aabb.mini >= r_aabb.maxi));
    // assert inclusion in the parent AABB, which must be associated with an
    // internal node
    const AABB own_aabb { internal_aabbs[i] };
    CHECK(!(own_aabb.mini >= own_aabb.maxi));
    if (!(l_aabb <= own_aabb) || !(r_aabb <= own_aabb)) {
        // if l_aabb ⊈ own_aabb or r_aabb ⊈ own_aabb then
        // a child AABB is not included in its parent and there was an error
        return false;
    }
    // otherwise, recursively check children
    const bool left_okay { l_leaf
            ? check_lbvh_correctness_leaf(leaf_aabbs, faces, vxs, vys, vzs, i_l)
            : check_lbvh_correctness_internal(children_l, children_r,
                  leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs, i_l) };
    if (!left_okay) {
        // on violation, early exit indicating failure
        return false;
    }
    const bool right_okay { r_leaf
            ? check_lbvh_correctness_leaf(leaf_aabbs, faces, vxs, vys, vzs, i_r)
            : check_lbvh_correctness_internal(children_l, children_r,
                  leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs, i_r) };
    // since left was checked, overall correctness is now the correctness of the
    // right child node
    return right_okay;
}

/// @brief Kernel to test the BVH Query by brute-force looping over every face
/// in the mesh for every query point and projecting it to the closest point on
/// the mesh. The result can then be compared to the optimized, LBVH-based
/// traversal in a doctest.
__global__ void test_brute_force_project(const uint N_ps, const uint N_faces,
    float* __restrict__ xs, float* __restrict__ ys, float* __restrict__ zs,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces)
{
    uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N_ps)
        return;
    // get query point
    const float3 q { v3(i, xs, ys, zs) };
    // initialize closest  projected point so far and distance to it
    float closest_point_dist_sq { FLT_MAX };
    float3 closest_point { v3(0.f) };
    // simply brute-force iterate over all faces, updating the closest
    // projected point
    for (uint i { 0 }; i < N_faces; ++i) {
        const uint3 face { faces[i] };
        const float3 vert1 { v3(face.x, vxs, vys, vzs) };
        const float3 vert2 { v3(face.y, vxs, vys, vzs) };
        const float3 vert3 { v3(face.z, vxs, vys, vzs) };
        const float3 projected_q { closest_point_on_triangle(
            q, vert1, vert2, vert3) };
        const float3 diff { projected_q - q };
        const float dist_sq { dot(diff, diff) };
        // if the squared distance is lower than or equal to the recorded
        // minimum, update the current minimum
        if (dist_sq <= closest_point_dist_sq) {
            closest_point_dist_sq = dist_sq;
            closest_point = projected_q;
        }
    }
    // store the result
    store_v3(closest_point, i, xs, ys, zs);
    return;
}

TEST_CASE("LBVH Correctness")
{
    Log::stop_logging();
    // load device mesh
    DeviceMesh mesh_d { DeviceMesh::from(
        load_mesh_from_obj("scenes/dragon.obj")) };
    // load_mesh_from_obj("scenes/ico_sphere.obj")) };

    // compute LBVH
    const LBVH lbvh(&mesh_d);

    // copy over relevant buffers to the host-side
    const thrust::host_vector<ChildNode> children_l = lbvh.children_l.get();
    const thrust::host_vector<ChildNode> children_r = lbvh.children_r.get();
    const thrust::host_vector<AABB> leaf_aabbs = lbvh.aabbs_leaf.get();
    const thrust::host_vector<AABB> internal_aabbs = lbvh.aabbs_internal.get();
    const thrust::host_vector<uint3> faces = mesh_d.faces.get();
    const thrust::host_vector<double> vxs = mesh_d.vxs.get();
    const thrust::host_vector<double> vys = mesh_d.vys.get();
    const thrust::host_vector<double> vzs = mesh_d.vzs.get();
    SUBCASE("LBVH BVH Descent Check")
    {
        // recursively check if parent AABB contains all child AABBs
        // and if leaves contain primitives entirely
        // this uses a recursive descent funciton started from the root node i=0
        CHECK(check_lbvh_correctness_internal(children_l, children_r,
            leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs, 0));
    }

    SUBCASE("LBVH Node Indices Check")
    {
        // now check if all leaves are reachable:
        // - first, assert that all leaves are referenced in children_l or
        // children_r
        const uint N { lbvh.N_faces };
        thrust::host_vector<uint32_t> leaf_indices;
        leaf_indices.reserve(N);
        for (const auto& c : children_l) {
            if (c.is_leaf()) {
                leaf_indices.push_back(c.idx());
            }
        }
        for (const auto& c : children_r) {
            if (c.is_leaf()) {
                leaf_indices.push_back(c.idx());
            }
        }
        thrust::sort(leaf_indices.begin(), leaf_indices.end());
        for (uint32_t i { 0 }; i < N; i++) {
            CHECK(leaf_indices[i] == i);
        }
        // - then, assert that all internal node indices apart from the root
        // (1,2,...,N-2) are referenced
        thrust::host_vector<uint32_t> internal_indices;
        internal_indices.reserve(N - 2);
        for (const auto& c : children_l) {
            if (!c.is_leaf()) {
                internal_indices.push_back(c.idx());
            }
        }
        for (const auto& c : children_r) {
            if (!c.is_leaf()) {
                internal_indices.push_back(c.idx());
            }
        }
        thrust::sort(internal_indices.begin(), internal_indices.end());
        CHECK(internal_indices[0] == 1);
        for (uint32_t i { 1 }; i < N - 2; i++) {
            CHECK(internal_indices[i] == internal_indices[i - 1] + 1);
        }
    }

    // now sample a bunch of points and project them onto the mesh
    const uint N_points { 10000 };
    thrust::host_vector<float> xx_host(N_points);
    thrust::host_vector<float> xy_host(N_points);
    thrust::host_vector<float> xz_host(N_points);
    std::mt19937 rng(420);
    std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);

    // compute volume of the mesh
    const float min_x { static_cast<float>(mesh_d.vxs.min()) };
    const float min_y { static_cast<float>(mesh_d.vys.min()) };
    const float min_z { static_cast<float>(mesh_d.vzs.min()) };
    const float max_x { static_cast<float>(mesh_d.vxs.max()) };
    const float max_y { static_cast<float>(mesh_d.vys.max()) };
    const float max_z { static_cast<float>(mesh_d.vzs.max()) };
    // sample random points in that volume
    for (uint i { 0 }; i < N_points; ++i) {
        xx_host[i] = (max_x - min_x) * uniform_dist(rng) + min_x;
        xy_host[i] = (max_y - min_y) * uniform_dist(rng) + min_y;
        xz_host[i] = (max_z - min_z) * uniform_dist(rng) + min_z;
    }

    // allocate buffers for the test data
    DeviceBuffer<float> xs(N_points);
    DeviceBuffer<float> ys(N_points);
    DeviceBuffer<float> zs(N_points);
    thrust::device_vector<float> xs_b(N_points);
    thrust::device_vector<float> ys_b(N_points);
    thrust::device_vector<float> zs_b(N_points);

    // copy the random points to the device
    thrust::copy(xx_host.begin(), xx_host.end(), xs_b.begin());
    thrust::copy(xx_host.begin(), xx_host.end(), xs.get().begin());

    thrust::copy(xy_host.begin(), xy_host.end(), ys_b.begin());
    thrust::copy(xy_host.begin(), xy_host.end(), ys.get().begin());

    thrust::copy(xz_host.begin(), xz_host.end(), zs_b.begin());
    thrust::copy(xz_host.begin(), xz_host.end(), zs.get().begin());

    // now run the project method of the lbvh
    lbvh.project(xs, ys, zs);

    // and the brute force method for comparison
    test_brute_force_project<<<BLOCKS(N_points), BLOCK_SIZE>>>(N_points,
        lbvh.N_faces, thrust::raw_pointer_cast(xs_b.data()),
        thrust::raw_pointer_cast(ys_b.data()),
        thrust::raw_pointer_cast(zs_b.data()), mesh_d.vxs.ptr(),
        mesh_d.vys.ptr(), mesh_d.vzs.ptr(), mesh_d.faces.ptr());

    // now copy both sets of results back to the host
    thrust::host_vector<float> xs_res(N_points);
    thrust::host_vector<float> ys_res(N_points);
    thrust::host_vector<float> zs_res(N_points);
    thrust::host_vector<float> xs_res_b(N_points);
    thrust::host_vector<float> ys_res_b(N_points);
    thrust::host_vector<float> zs_res_b(N_points);

    thrust::copy(xs_b.begin(), xs_b.end(), xs_res_b.begin());
    thrust::copy(ys_b.begin(), ys_b.end(), ys_res_b.begin());
    thrust::copy(zs_b.begin(), zs_b.end(), zs_res_b.begin());
    thrust::copy(xs.get().begin(), xs.get().end(), xs_res.begin());
    thrust::copy(ys.get().begin(), ys.get().end(), ys_res.begin());
    thrust::copy(zs.get().begin(), zs.get().end(), zs_res.begin());

    SUBCASE("LBVH Projection Query Correctness")
    {
        const float ε { 375 * FLT_EPSILON };
        for (uint i { 0 }; i < N_points; ++i) {
            const float3 q_proj { v3(xs_res[i], ys_res[i], zs_res[i]) };
            const float3 q_proj_b { v3(xs_res_b[i], ys_res_b[i], zs_res_b[i]) };
            const float3 diff { q_proj - q_proj_b };
            CHECK(dot(diff, diff) == doctest::Approx(0).epsilon(ε));
        }
    }

    SUBCASE("Write Dump to File")
    {

        // write leaf AABBs to file
        {
            std::ofstream leafs("builddocs/_staticc/lbvh_leafs.bin");
            thrust::host_vector<AABB> leaf_aabbs = lbvh.aabbs_leaf.get();
            for (uint i { 0 }; i < lbvh.N_faces; ++i) {
                const float3 centroid { leaf_aabbs[i].get_centroid() };
                const float3 scale { leaf_aabbs[i].get_size() };
                leafs.write(
                    reinterpret_cast<const char*>(&centroid.x), sizeof(float));
                leafs.write(
                    reinterpret_cast<const char*>(&centroid.y), sizeof(float));
                leafs.write(
                    reinterpret_cast<const char*>(&centroid.z), sizeof(float));
                leafs.write(
                    reinterpret_cast<const char*>(&scale.x), sizeof(float));
                leafs.write(
                    reinterpret_cast<const char*>(&scale.y), sizeof(float));
                leafs.write(
                    reinterpret_cast<const char*>(&scale.z), sizeof(float));
            }
        }
        // write internal AABBs to file
        {
            std::ofstream intern("builddocs/_staticc/lbvh_internals.bin");
            thrust::host_vector<AABB> internal_aabbs
                = lbvh.aabbs_internal.get();
            for (uint i { 0 }; i < lbvh.N_faces - 1; ++i) {
                const float3 centroid { internal_aabbs[i].get_centroid() };
                const float3 scale { internal_aabbs[i].get_size() };
                intern.write(
                    reinterpret_cast<const char*>(&centroid.x), sizeof(float));
                intern.write(
                    reinterpret_cast<const char*>(&centroid.y), sizeof(float));
                intern.write(
                    reinterpret_cast<const char*>(&centroid.z), sizeof(float));
                intern.write(
                    reinterpret_cast<const char*>(&scale.x), sizeof(float));
                intern.write(
                    reinterpret_cast<const char*>(&scale.y), sizeof(float));
                intern.write(
                    reinterpret_cast<const char*>(&scale.z), sizeof(float));
            }
        }
        // write the original points to a file
        {
            std::ofstream points_o("builddocs/_staticc/points_orig.bin");
            for (uint i { 0 }; i < N_points; ++i) {
                points_o.write(
                    reinterpret_cast<const char*>(&xx_host[i]), sizeof(float));
                points_o.write(
                    reinterpret_cast<const char*>(&xy_host[i]), sizeof(float));
                points_o.write(
                    reinterpret_cast<const char*>(&xz_host[i]), sizeof(float));
            }
        }
        {
            std::ofstream points_t("builddocs/_staticc/points_proj.bin");
            for (uint i { 0 }; i < N_points; ++i) {
                points_t.write(
                    reinterpret_cast<const char*>(&xs_res[i]), sizeof(float));
                points_t.write(
                    reinterpret_cast<const char*>(&ys_res[i]), sizeof(float));
                points_t.write(
                    reinterpret_cast<const char*>(&zs_res[i]), sizeof(float));
            }
        }
    }
    Log::start_logging();
}

TEST_CASE("LBVH Benchmarks")
{
    Log::stop_logging();
    ankerl::nanobench::Bench bench;
    bench.title("LBVH Benchmarks");

    DeviceMesh mesh_d { DeviceMesh::from(
        load_mesh_from_obj("scenes/dragon.obj")) };

    // construction benchmarks
    bench.run("Construction", [&]() {
        const LBVH lbvh(&mesh_d);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // now sample a bunch of points and project them onto the mesh
    const uint N_points { 1000000 };
    thrust::host_vector<float> xx_host(N_points);
    thrust::host_vector<float> xy_host(N_points);
    thrust::host_vector<float> xz_host(N_points);
    std::mt19937 rng(161);
    std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);

    // compute volume of the mesh
    const float min_x { static_cast<float>(mesh_d.vxs.min()) };
    const float min_y { static_cast<float>(mesh_d.vys.min()) };
    const float min_z { static_cast<float>(mesh_d.vzs.min()) };
    const float max_x { static_cast<float>(mesh_d.vxs.max()) };
    const float max_y { static_cast<float>(mesh_d.vys.max()) };
    const float max_z { static_cast<float>(mesh_d.vzs.max()) };
    // sample random points in that volume
    for (uint i { 0 }; i < N_points; ++i) {
        xx_host[i] = (max_x - min_x) * uniform_dist(rng) + min_x;
        xy_host[i] = (max_y - min_y) * uniform_dist(rng) + min_y;
        xz_host[i] = (max_z - min_z) * uniform_dist(rng) + min_z;
    }

    DeviceBuffer<float> xs(N_points);
    DeviceBuffer<float> ys(N_points);
    DeviceBuffer<float> zs(N_points);
    thrust::copy(xx_host.begin(), xx_host.end(), xs.get().begin());
    thrust::copy(xy_host.begin(), xy_host.end(), ys.get().begin());
    thrust::copy(xz_host.begin(), xz_host.end(), zs.get().begin());
    const LBVH lbvh(&mesh_d);
    bench.minEpochIterations(5).run("Query (Project to Surface)", [&]() {
        lbvh.project(xs, ys, zs);
        CUDA_CHECK(cudaDeviceSynchronize());
    });
    Log::start_logging();
}