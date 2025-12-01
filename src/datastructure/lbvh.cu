#include "datastructure/lbvh.cuh"

#include "doctest/doctest.h"
#include <fstream>
#include <thrust/host_vector.h>

__host__ __device__ inline static AABB AABB_from_verts(
    double3 vert1, double3 vert2, double3 vert3)
{
    return AABB { (v3((float)fmin(fmin(vert1.x, vert2.x), vert3.x),
                      (float)fmin(fmin(vert1.y, vert2.y), vert3.y),
                      (float)fmin(fmin(vert1.z, vert2.z), vert3.z))),
        (v3((float)fmax(fmax(vert1.x, vert2.x), vert3.x),
            (float)fmax(fmax(vert1.y, vert2.y), vert3.y),
            (float)fmax(fmax(vert1.z, vert2.z), vert3.z))) };
}

__global__ void generate_tree(const uint32_t* __restrict__ codes, const uint N,
    uint* __restrict__ children_l, uint* __restrict__ children_r,
    uint* __restrict__ parent, uint* __restrict__ leaf_parent,
    bool* __restrict__ l_is_leaf, bool* __restrict__ r_is_leaf)
{
    const uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N - 1)
        return;

    // find the range of values corresponding to the current node
    const int2 range = determine_range(codes, N, (int)i);
    const int first { range.x };
    const int last { range.y };

    // determine where to split the node
    const int split { find_split(codes, first, last) };

    // record left child
    const bool l_leaf { split == first };
    l_is_leaf[i] = l_leaf;
    children_l[i] = split;

    // record right child
    const bool r_leaf { split + 1 == last };
    r_is_leaf[i] = r_leaf;
    children_r[i] = split + 1;

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

__global__ void compute_aabbs(uint* __restrict__ parent,
    uint* __restrict__ leaf_parent, uint* __restrict__ visited,
    uint* __restrict__ children_l, uint* __restrict__ children_r, uint N,
    double* __restrict__ vxs, double* __restrict__ vys,
    double* __restrict__ vzs, uint3* __restrict__ faces,
    bool* __restrict__ l_is_leaf, bool* __restrict__ r_is_leaf,
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
        const bool current_is_left { (children_l[i_parent] == i) };
        const uint sibling_i { current_is_left ? children_r[i_parent]
                                               : children_l[i_parent] };
        // get the siblings aabb
        const AABB sibling_aabb {
            ((current_is_left ? r_is_leaf : l_is_leaf)[i_parent])
                ?
                // sibling is leaf
                aabbs_leaf[sibling_i]
                :
                // sibling is not a leaf
                aabbs_internal[sibling_i]
        };
        // combine the sibling's AABB with the own AABB
        aabb = aabb | sibling_aabb;
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
    return;
}

/// @brief Check if a BVH leaf node AABB contains the vertices of the
/// corresponding primitive/face of a triangular mesh. Used to check the
/// correctness of the LBVH implementation.
/// @param leaf_aabbs array of leaf-node AABBs
/// @param faces faces of the triangular mesh, indexing into the vertex buffers
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
    // load all vertices of the primitive
    const uint3 face { faces[i] };
    const float3 vert1 { v3(
        (float)vxs[face.x], (float)vys[face.x], (float)vzs[face.x]) };
    const float3 vert2 { v3(
        (float)vxs[face.y], (float)vys[face.y], (float)vzs[face.y]) };
    const float3 vert3 { v3(
        (float)vxs[face.z], (float)vys[face.z], (float)vzs[face.z]) };
    // for each vertex, check if it is contained in the aabb and return the
    // result
    return aabb.contains(vert1) && aabb.contains(vert2) && aabb.contains(vert3);
}

/// @brief Check whether a constructed BVH is correct, in the sense that a
/// recursive descent from the root node yields a series of child AABBs that are
/// entirely contained in the parent AABB, and the leaf nodes entirely contain
/// the respective primitives they represent, as checked by
/// `check_lbvh_correctness_leaf`.
/// @param children_l index of the left child node of i, indexed by i
/// @param children_r index of the right child node of i, indexed by i
/// @param l_is_leaf whether the left child node is a leaf node
/// @param r_is_leaf whether the right child node is a leaf node
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
    const thrust::host_vector<uint>& children_l,
    const thrust::host_vector<uint>& children_r,
    const thrust::host_vector<bool>& l_is_leaf,
    const thrust::host_vector<bool>& r_is_leaf,
    const thrust::host_vector<AABB>& leaf_aabbs,
    const thrust::host_vector<AABB>& internal_aabbs,
    const thrust::host_vector<uint3>& faces,
    const thrust::host_vector<double>& vxs,
    const thrust::host_vector<double>& vys,
    const thrust::host_vector<double>& vzs, const uint i)
{
    // get left and right child indices
    const uint i_l { children_l[i] };
    const uint i_r { children_r[i] };
    // check if they are leafs
    const bool l_leaf { l_is_leaf[i] };
    const bool r_leaf { r_is_leaf[i] };
    // get the respective AABBs
    const AABB l_aabb { (l_leaf ? leaf_aabbs : internal_aabbs)[i_l] };
    const AABB r_aabb { (r_leaf ? leaf_aabbs : internal_aabbs)[i_r] };
    // assert inclusion in the parent AABB, which must be associated with an
    // internal node
    const AABB own_aabb { internal_aabbs[i] };
    if (!(l_aabb <= own_aabb) || !(r_aabb <= own_aabb)) {
        // if l_aabb ⊈ own_aabb or r_aabb ⊈ own_aabb then
        // a child AABB is not included in its parent and there was an error
        return false;
    }
    // otherwise, recursively check children
    const bool left_okay { l_leaf
            ? check_lbvh_correctness_leaf(leaf_aabbs, faces, vxs, vys, vzs, i_l)
            : check_lbvh_correctness_internal(children_l, children_r, l_is_leaf,
                  r_is_leaf, leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs,
                  i_l) };
    if (!left_okay) {
        // on violation, early exit indicating failure
        return false;
    }
    const bool right_okay { r_leaf
            ? check_lbvh_correctness_leaf(leaf_aabbs, faces, vxs, vys, vzs, i_r)
            : check_lbvh_correctness_internal(children_l, children_r, l_is_leaf,
                  r_is_leaf, leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs,
                  i_r) };
    // since left was checked, overall correctness is now the correctness of the
    // right child node
    return right_okay;
}

TEST_CASE("Check LBVH correctness")
{
    // load device mesh
    DeviceMesh device_mesh { DeviceMesh::from(
        load_mesh_from_obj("scenes/ico_sphere.obj")) };

    // compute LBVH
    const LBVH lbvh(&device_mesh);

    // copy over relevant buffers to the host-side
    const thrust::host_vector<uint> children_l = lbvh.children_l.get();
    const thrust::host_vector<uint> children_r = lbvh.children_r.get();
    const thrust::host_vector<bool> l_is_leaf = lbvh.l_is_leaf.get();
    const thrust::host_vector<bool> r_is_leaf = lbvh.r_is_leaf.get();
    const thrust::host_vector<AABB> leaf_aabbs = lbvh.aabbs_leaf.get();
    const thrust::host_vector<AABB> internal_aabbs = lbvh.aabbs_internal.get();
    const thrust::host_vector<uint3> faces = device_mesh.faces.get();
    const thrust::host_vector<double> vxs = device_mesh.vxs.get();
    const thrust::host_vector<double> vys = device_mesh.vys.get();
    const thrust::host_vector<double> vzs = device_mesh.vzs.get();

    // recursively check if parent AABB contains all child AABBs
    // and if leaves contain primitives entirely
    // this uses a recursive descent funciton started from the root node i=0
    CHECK(check_lbvh_correctness_internal(children_l, children_r, l_is_leaf,
        r_is_leaf, leaf_aabbs, internal_aabbs, faces, vxs, vys, vzs, 0));
}

TEST_CASE("Write LBVH dump to file")
{
    // #ifdef BENCH
    // load device mesh
    DeviceMesh dragon_d { DeviceMesh::from(
        load_mesh_from_obj("scenes/ico_sphere.obj")) };
    // compute LBVH
    const LBVH lbvh(&dragon_d);
    // write leaf AABBs to file
    {
        std::ofstream leafs("builddocs/_staticc/lbvh_leafs.bin");
        thrust::host_vector<AABB> leaf_aabbs = lbvh.aabbs_leaf.get();
        for (uint i { 0 }; i < lbvh.N_faces; ++i) {
            const float3 centroid { leaf_aabbs[i].get_centroid() };
            const float3 scale { leaf_aabbs[i].get_volume() };
            leafs.write(
                reinterpret_cast<const char*>(&centroid.x), sizeof(float));
            leafs.write(
                reinterpret_cast<const char*>(&centroid.y), sizeof(float));
            leafs.write(
                reinterpret_cast<const char*>(&centroid.z), sizeof(float));
            leafs.write(reinterpret_cast<const char*>(&scale.x), sizeof(float));
            leafs.write(reinterpret_cast<const char*>(&scale.y), sizeof(float));
            leafs.write(reinterpret_cast<const char*>(&scale.z), sizeof(float));
        }
    }
    // write internal AABBs to file
    {
        std::ofstream intern("builddocs/_staticc/lbvh_internals.bin");
        thrust::host_vector<AABB> internal_aabbs = lbvh.aabbs_internal.get();
        for (uint i { 0 }; i < lbvh.N_faces - 1; ++i) {
            const float3 centroid { internal_aabbs[i].get_centroid() };
            const float3 scale { internal_aabbs[i].get_volume() };
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
    // #endif
}