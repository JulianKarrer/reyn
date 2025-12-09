#ifndef DATASTRUCTURE_LBVH_CUH_
#define DATASTRUCTURE_LBVH_CUH_

#include "scene/sample_boundary.cuh"
#include "buffer.cuh"
#include "utils/vector.cuh"
#include "utils/geometry.cuh"
#include "utils/dispatch.cuh"
#include <thrust/sort.h>
#include "log.h"

///@brief  Expands a 10-bit integer into 30 bits by inserting 2 zeros after
/// each bit.
///
/// [Tero Karras]
/// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__device__ static inline uint32_t expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

///@brief  Calculates a 30-bit Morton code for the  given 3D point located
/// within the unit cube [0,1].
///
/// [Tero Karras]
/// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
/// @param v a vertex to assign a morton code to, which should be in [0.f ;
/// 1.f] along every spatial dimension
__device__ static inline uint32_t morton_3d(double3 v)
{
    assert(0. <= v.x && v.x <= 1.);
    assert(0. <= v.y && v.y <= 1.);
    assert(0. <= v.z && v.z <= 1.);
    const float x = static_cast<float>(min(max(v.x * 1024., 0.), 1023.));
    const float y = static_cast<float>(min(max(v.y * 1024., 0.), 1023.));
    const float z = static_cast<float>(min(max(v.z * 1024., 0.), 1023.));
    uint32_t xx = expand_bits(static_cast<uint32_t>(x));
    uint32_t yy = expand_bits(static_cast<uint32_t>(y));
    uint32_t zz = expand_bits(static_cast<uint32_t>(z));
    return xx * 4 + yy * 2 + zz;
}

/// @brief Struct representing an Axis-Aligned Bounding Box or AABB
struct AABB {
    /// @brief minimum coordinate of the box along each axis
    float3 mini;
    /// @brief maximum coordinate of the box along each axis
    float3 maxi;

    /// @brief Merge two AABBs, taking their union (minimum of mini and
    /// maximum of maxi)
    /// @param other other AABB to merhe into `this`
    /// @return the union of both AABBs
    __host__ __device__ AABB operator|(const AABB other) const
    {
        const AABB res { min(this->mini, other.mini),
            max(this->maxi, other.maxi) };
        const float3 v { res.maxi - res.mini };
        if (min(v.x, min(v.y, v.z)) == 0.f) {
            // add a small epsilon offset relative to largest extent to mini and
            // maxi
            const float3 ε { v3(FLT_EPSILON) };
            return AABB { mini - ε, maxi + ε };
        }
        return res;
    };

    ///@brief Get the centroid of the AABB
    ///@return ½ (mini + maxi)
    __host__ __device__ float3 get_centroid() const
    {
        return 0.5f * (maxi + mini);
    };

    ///@brief check if the AABB contains a vertex v
    ///@param v vertex to check
    ///@return whether or not the vertex is contained in the AABB
    __host__ __device__ bool contains(const float3& v) const
    {
        return (mini.x <= v.x && v.x <= maxi.x && mini.y <= v.y && v.y <= maxi.y
            && mini.z <= v.z && v.z <= maxi.z);
    };

    /// @brief Get the size of the AABB along every coordinate axis, i.e. `maxi
    /// - mini`
    /// @return size of the AABB along every axis
    __host__ __device__ float3 get_size() const { return maxi - mini; };

    /// @brief Lower bound of the squared euclidean distance from the query
    /// point `p` to any point inside the AABB. Described in "Nearest Neighbor
    /// Queries" [Roussopoulos, Kelley, Vincent]
    /// @param p query point
    /// @return lower bound of squared distance between p and any point in the
    /// AABB
    __host__ __device__ float min_dist(const float3 p) const
    {
        // r_i in the paper is equivalent to clamping between mini and maxi
        // (if the invariant mini <= maxi is not violated)
        assert(mini <= maxi);
        const float3 r { min(max(p, mini), maxi) };
        // return the sum of squared distances from p to r
        const float3 d { r - p };
        return dot(d, d);
    };

    ///@brief Upper bound on the squared euclidean distance from query point p
    /// to any point in the AABB. Described in "Nearest Neighbor
    /// Queries" [Roussopoulos, Kelley, Vincent]
    ///@param p query point
    ///@return upper bound of euclidean distance between p and a primtive in the
    /// AABB
    __host__ __device__ float min_max_dist(const float3 p) const
    {
        // project each component of p to the closest face of the AABB
        const float3 c { get_centroid() };
        const float3 rm { v3( //
            p.x <= c.x ? mini.x : maxi.x, //
            p.y <= c.y ? mini.y : maxi.y, //
            p.z <= c.z ? mini.z : maxi.z) };
        // also do the opposite along each axis
        const float3 rM { v3( //
            p.x >= c.x ? mini.x : maxi.x, //
            p.y >= c.y ? mini.y : maxi.y, //
            p.z >= c.z ? mini.z : maxi.z) };
        // compute the squared distances from p to rm
        const float3 p_rm { p - rm };
        const float3 p_rm_sq { p_rm * p_rm };
        // compute the squared distances from p to rM
        const float3 p_rM { p - rM };
        const float3 p_rM_sq { p_rM * p_rM };
        // compute the minimum as described in the paper:
        // each option is one component of p_rm_sq and the complementary two
        // components of p_rM_sq
        return min(min(
                       // x-component
                       p_rm_sq.x + p_rM_sq.y + p_rM_sq.z,
                       // y-component
                       p_rm_sq.y + p_rM_sq.x + p_rM_sq.z),
            // z-component
            p_rm_sq.z + p_rM_sq.x + p_rM_sq.y);
    }
};

/// @brief Check if the left AABB is contained in the right AABB
/// @param lhs included AABB
/// @param rhs including AABB
/// @return whether or not lhs ⊆ rhs is true
__host__ __device__ inline bool operator<=(const AABB& lhs, const AABB& rhs)
{
    return (rhs.mini.x <= lhs.mini.x && rhs.mini.y <= lhs.mini.y
        && rhs.mini.z <= lhs.mini.z && lhs.maxi.x <= rhs.maxi.x
        && lhs.maxi.y <= rhs.maxi.y && lhs.maxi.z <= rhs.maxi.z);
}

/// @brief A wrapper type around an unsigned integer, used to enforce safe
/// handling of pointers to child nodes in binary trees that have
/// leaf-vs-internal node information bitpacked into the index to the child.
struct ChildNode {
    /// @brief index to a child node, shifted once to the left, so that the
    /// least significant, rightmost bit may indicate whether the child is a
    /// leaf or not:
    ///
    /// - last bit is zero = leaf
    ///
    /// - last bit is one = internal node
    uint32_t __i;

    /// @brief Whether the child node is a leaf node or not
    /// @return `true` if leaf node, `false` if internal node
    __host__ __device__ inline bool is_leaf() const { return (__i & 1) == 0; };

    /// @brief Get the index of the child node without leaf vs. internal node
    /// information
    /// @return index of the child
    __host__ __device__ inline uint32_t idx() const { return (__i >> 1); };

    /// @brief Create an instance of `ChildNode`. Static method used instead of
    /// a constructor to keep the type trivially initializable and maximally
    /// `__device__`-compatible.
    /// @param child_index index of the child node
    /// @param is_leaf whether that child node is a leaf node
    /// @return a pointer to the child node with bitpacked leaf information
    __host__ __device__ static ChildNode create(
        const uint32_t child_index, const bool is_leaf)
    {
        // put 0 in LSB if leaf, 1 otherwise
        return ChildNode { (child_index << 1) | (is_leaf ? 0 : 1) };
    }
};

/// @brief Construct an AABB encompassing the three `double`-valued
/// vertices of a triangle. Uses `FLT_EPSILON` to ensure non-zero volume of the
/// resulting AABB.
/// @param vert1 1st vertex
/// @param vert2 2nd vertex
/// @param vert3 3rd vertex
__host__ __device__ inline static AABB AABB_from_verts(
    double3 vert1, double3 vert2, double3 vert3);

struct MortonCodeGenerator {
    const double3 bounds_min;
    const double3 volume_inv;
    const double* vxs;
    const double* vys;
    const double* vzs;
    MortonCodeGenerator(double3 _bounds_min, double3 _volume, DeviceMesh* mesh)
        : bounds_min(_bounds_min)
        , volume_inv(dv3((1. / _volume.x), (1. / _volume.y), (1. / _volume.z)))
        , vxs(mesh->vxs.ptr())
        , vys(mesh->vys.ptr())
        , vzs(mesh->vzs.ptr()) {};
    __device__ uint64_t operator()(const uint3 face) const
    {
        const double3 v1 { dv3(vxs[face.x], vys[face.x], vzs[face.x]) };
        const double3 v2 { dv3(vxs[face.y], vys[face.y], vzs[face.y]) };
        const double3 v3 { dv3(vxs[face.z], vys[face.z], vzs[face.z]) };
        // calculate the AABB of the triangle
        const double one_third { 1. / 3. };
        const double3 centroid { one_third * (v1 + v2 + v3) - bounds_min };
        // normalize the centroid in a range [0.f ; 1024.f]^3 in the
        // volume of the overall AABB normalize the vertices
        const double3 centroid_normalized { centroid * volume_inv };
        const uint32_t morton_code { morton_3d(centroid_normalized) };
        return (uint64_t)morton_code;
    }
};

///@brief Augment `uint64_t` Morton codes that only occupy the least significant
/// 32 bits with the `uint32_t` index of the code, making it a unique 64 bit
/// code while preserving the sorting of the array of codes.
///
/// The Morton code will take up the most significant 32 bits while the lower 32
/// bits are taken up by the index
struct MortonCodeExtender {
    MortonCodeExtender() {};
    __device__ uint64_t operator()(
        const thrust::tuple<uint32_t, uint64_t>& tuple) const
    {
        const uint32_t i { thrust::get<0>(tuple) };
        const uint64_t code { thrust::get<1>(tuple) };
        // explicitly cast index to 64 bits unsigned
        const uint64_t i_long { (uint64_t)i };
        // shift the Morton code up and move the index into the gap using
        // bitwise or
        return (code << 32) | i_long;
    }
};

/// @brief Find the index in [frist, last] to split morton codes such that the
/// highest differing bit is zero in one partition and one in the other, or use
/// a median split if this is not possible. Function taken from:
/// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
/// @param codes morton codes to split
/// @param first first index into the codes (inclusive)
/// @param last last index into the codes (inclusive)
/// @return index of the code to split on
__device__ static inline int find_split(
    const uint64_t* __restrict__ codes, const int first, const int last)
{
    const uint64_t first_code { codes[first] };
    const uint64_t last_code { codes[last] };

    // median split if codes are identical
    if (first_code == last_code) {
        return (first + last) >> 1; // shift to divide
    }

    // count leading zeros (CLZ intrinsic) of XOR'ed codes to find the highest
    // differing bit
    const int lcp { __clzll(first_code ^ last_code) };

    // binary search for split point, which is the highest id where more than
    // lcp bits are shared with the `first` code
    int split { first };
    int step { last - first };
    do {
        step = (step + 1) >> 1; // exponential decrease
        const int new_split { split + step };
        if (new_split < last) {
            const int new_lcp { __clzll(first_code ^ codes[new_split]) };
            if (new_lcp > lcp)
                split = new_split;
        }
    } while (step > 1);
    return split;
}

/// @brief Determine range of nodes that have the node at index `i` as their
/// ancestor.
/// Sources:
///
/// - "Maximizing Parallelism in the Construction of BVHs, Octrees, and
/// k-d Trees" [Tero Karras]
///
/// - https://github.com/ToruNiina/lbvh/blob/master/lbvh/bvh.cuh
/// @param codes morton codes
/// @param N number of primitives
/// @param i current index
/// @return range [lower, upper] of ndoes associated with node i
__device__ static inline int2 determine_range(
    const uint64_t* __restrict__ codes, const uint N, const int i)
{
    if (i == 0) {
        return make_int2(0, N - 1);
    }
    const uint64_t own_code { codes[i] };
    const int delta_l { __clzll(own_code ^ codes[i - 1]) };
    const int delta_r { __clzll(own_code ^ codes[i + 1]) };
    const int d { delta_r > delta_l ? 1 : -1 }; // = sign(δᵣ - δₗ)

    // find one end of the range
    const int delta_min { min(delta_l, delta_r) };
    int lmax { 2 };

    while (true) {
        const int j { i + d * lmax };
        if (j < 0 || j >= N || __clzll(own_code ^ codes[j]) <= delta_min) {
            break;
        }
        lmax <<= 1; // lmax <- lmax * 2
    }
    // find other end of the range
    int l { 0 };
    int t { lmax >> 1 };
    while (t >= 1) {
        const int j { i + d * (l + t) };
        if (j >= 0 && j < N) {
            const int delta { __clzll(own_code ^ codes[j]) };
            if (delta > delta_min) {
                l += t;
            }
        }
        t >>= 1;
    }

    const int end { i + l * d };
    return make_int2(min(i, end), max(i, end));
}

/// @brief Generate a BVH binary radix tree from a sorted list of mortion codes,
/// recording the resulting links between parent and child nodes, as well as
/// which child nodes are leaves (the index of which is simply an index into the
/// sorted buffer of primitives).
///
/// Algorithm from "Maximizing Parallelism in the Construction of BVHs, Octrees,
/// and k-d Trees" [Tero Karras]
/// @param codes morton codes
/// @param N number of inner nodes of the tree (= #primitives - 1)
/// @param children_l index of the left child node, and whether it is a leaf or
/// not
/// @param children_r index of the right child node, and whether it is a leaf or
/// not
/// @param parent index of the parent of the i-th internal node
/// @param leaf_parent index of the parent of the i-th leaf node
__global__ void generate_tree(const uint64_t* __restrict__ codes, const uint N,
    ChildNode* __restrict__ children_l, ChildNode* __restrict__ children_r,
    uint* __restrict__ parent, uint* __restrict__ leaf_parent);

/// @brief From a binary radix tree representing the structure of a BVH, compute
/// the AABB of each node, whether it is a leaf node (primitive) or internal
/// node. Parent AABBs must encompass all descendants' AABBs.
///
/// Algorithm from "Maximizing Parallelism in the Construction of BVHs, Octrees,
/// and k-d Trees" [Tero Karras]:
///
/// Constructed bottom-up from N threads, where N is the number of primitives or
/// leaf nodes. Synchronization is handled via global atomic increments to flags
/// in the `visited` buffer, such that only the second child arriving bottom-up
/// at the parent node assumes the role of computing the AABB at the parent
/// node, when the other sibling is already done writing its AABB. then proceeds
/// iteratively, such that half the threads exit early on each layer until only
/// one thread remains at the root.
/// @param parent index of the parent node of internal node i
/// @param leaf_parent index of the parent node of leaf node i
/// @param visited flags indicating whether the parent at the given i was
/// already visited by the sibling thread and the current thread may proceed or
/// not. Could be bool but is taken as
/// @param children_l index of the left child node, and whether it is a leaf or
/// not
/// @param children_r index of the right child node, and whether it is a leaf or
/// not
/// @param N number of leaf nodes or primitives
/// @param vxs x-component of the vertex buffer
/// @param vys y-component of the vertex buffer
/// @param vzs z-component of the vertex buffer
/// @param faces sorted faces or primitives, which index into the three vertex
/// componentn buffers
/// @param aabbs_leaf the AABBs of each leaf node
/// @param aabbs_internal the AABBs of each internal node
/// @return
__global__ void compute_aabbs(const uint* __restrict__ parent,
    const uint* __restrict__ leaf_parent, uint* __restrict__ visited,
    const ChildNode* __restrict__ children_l,
    const ChildNode* __restrict__ children_r, const uint N,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces,
    AABB* __restrict__ aabbs_leaf, AABB* __restrict__ aabbs_internal);

///@brief Compute the height of an LBVH binary radix tree by traversing upwards
/// from all leaves and storing the number of edges encountered along each path
/// in `depths`. A reduction over `depths` can then yield the maximum depth of
/// the tree
///@param N number of leaf nodes or primitives
///@param leaf_parents parent indices of leaf nodes
///@param parent parent indices of internal nodes
///@param depths the output: number of edges from root to the respective node
__global__ void compute_tree_height(const uint N,
    const uint* __restrict__ leaf_parents, const uint* __restrict__ parent,
    uint* __restrict__ depths);

template <uint STACK_SIZE, typename TargetType, class F>
__device__ inline static auto bvh_find_closest(float3 q,
    const ChildNode* __restrict__ children_l,
    const ChildNode* __restrict__ children_r,
    const AABB* __restrict__ aabbs_leaf,
    const AABB* __restrict__ aabbs_internal, F f)
{
    // construct stack that can hold sufficient entries for traversing the
    // entire tree
    uint32_t stack_idx[STACK_SIZE];
    float stack_mindist[STACK_SIZE];
    // initialize the first entry in the stack
    stack_idx[0] = 0; // initial idx points to root node
    stack_mindist[0]
        = aabbs_internal[0].min_dist(q); // lower bound of dist to root AABB
    // stack initially points to first empty element (at index 1)
    int stack_ptr { 1 };

    // initialize nearest object index and distance lower bound
    float closest_point_dist_sq { FLT_MAX };
    TargetType closest_target {};

    // lambda function for traversing a child node
    const auto traverse = [&] __device__(const float min_dist,
                              const bool is_leaf, const uint32_t idx) {
        // prune child nodes that at best (min_dist) are still further away than
        // the current closest found point
        if ( //
             // l_min_dist <= r_min_max_dist &&
            min_dist <= closest_point_dist_sq //
        ) {
            // check if the child is a leaf node, which may update the closest
            // point, or an internal node that can be put on the stack
            if (is_leaf) {
                // compute actual squared distance to the primitive
                const auto tup { f(idx) };
                const float dist_sq { thrust::get<0>(tup) };
                const TargetType acc { thrust::get<1>(tup) };
                // if this distance is a new best, update the closest point so
                // far
                if (dist_sq <= closest_point_dist_sq) {
                    closest_point_dist_sq = dist_sq;
                    closest_target = acc;
                }
            } else {
                // in this branch, the node is an internal node, put it on the
                // stack
                stack_idx[stack_ptr] = idx;
                stack_mindist[stack_ptr] = min_dist;
                stack_ptr++;
            }
        }
    };

    do {
        // decrement stack pointer
        stack_ptr--;

        // if the AABB is at best further away than the current minimum
        // point, prune this entry from the stack
        if (stack_mindist[stack_ptr] > closest_point_dist_sq) {
            continue;
        }

        // get index of the current node from the stack
        const uint32_t idx { stack_idx[stack_ptr] };

        // compute lower bound on squared distance to left child
        const ChildNode l { children_l[idx] };
        const bool l_is_leaf { l.is_leaf() };
        const uint32_t l_idx { l.idx() };
        const AABB l_aabb { (l_is_leaf ? aabbs_leaf : aabbs_internal)[l_idx] };
        const float l_min_dist { l_aabb.min_dist(q) };
        // const float l_min_max_dist { l_aabb.min_max_dist(q) };

        // compute lower bound on squared distance to right child
        const ChildNode r { children_r[idx] };
        const bool r_is_leaf { r.is_leaf() };
        const uint32_t r_idx { r.idx() };
        const AABB r_aabb { (r_is_leaf ? aabbs_leaf : aabbs_internal)[r_idx] };
        const float r_min_dist { r_aabb.min_dist(q) };
        // const float r_min_max_dist { r_aabb.min_max_dist(q) };

        // ordering matters! the closer children should be visited first, i.e.
        // put on the stack last. use min_dist as an optimistic heuristic for
        // actual distance to a primitive in the respective subtree
        if (l_min_dist < r_min_dist) {
            traverse(r_min_dist, r_is_leaf, r_idx);
            traverse(l_min_dist, l_is_leaf, l_idx);
        } else {
            traverse(l_min_dist, l_is_leaf, l_idx);
            traverse(r_min_dist, r_is_leaf, r_idx);
        }

        // assert(0 <= stack_ptr && stack_ptr < MAX_TREE_DEPTH);
    } while (stack_ptr > 0); // repeat until stack is empty

    // now, the entire stack is empty and the closest target was found,
    // store it back
    return closest_target;
}

template <uint STACK_SIZE>
///@brief Kernel function to project points to the respectively closest point on
/// a triangular mesh represented by an LBVH. Used in `LBVH::projection` via the
/// `operator()` of the `ProjectionLaunchFunctor` functor
__global__ void project_points(const uint N_ps, float* __restrict__ xs,
    float* __restrict__ ys, float* __restrict__ zs,
    const double* __restrict__ vxs, const double* __restrict__ vys,
    const double* __restrict__ vzs, const uint3* __restrict__ faces,
    const ChildNode* __restrict__ children_l,
    const ChildNode* __restrict__ children_r,
    const AABB* __restrict__ aabbs_leaf,
    const AABB* __restrict__ aabbs_internal)
{
    uint i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N_ps)
        return;
    // get query point
    const float3 q { v3(i, xs, ys, zs) };

    // traversal function must return squared distance to closest point on the
    // primitive, as well as the target property of the closest point that
    // should be kept track of, in this case the clsoest point itself
    const auto traversal_func = [&] __device__(uint idx) {
        // get closest point on idx-th triangle
        const float3 projected_q { closest_point_on_triangle(
            q, faces[idx], vxs, vys, vzs) };
        // compute squared distance
        const float3 diff { projected_q - q };
        const float dist_sq { dot(diff, diff) };
        // return tuple
        return thrust::make_tuple(dist_sq, projected_q);
    };

    // pass `traversal_func` to a generic bvh traversal routine to keep track of
    // closest points found so far, finally yielding the closest point to the
    // mesh overall
    const float3 closest_q { bvh_find_closest<STACK_SIZE, float3>(q, children_l,
        children_r, aabbs_leaf, aabbs_internal, traversal_func) };

    // store the closest point back to the buffer
    store_v3(closest_q, i, xs, ys, zs);
};

/// @brief Wrapper for launching the `project_points` kernel with a `STACK_SIZE`
/// decided at runtime by `RuntimeTemplateSelectList` or similar
/// @tparam STACK_SIZE stack size for BVH traversal
template <unsigned STACK_SIZE> struct ProjectionLaunchFunctor {
    void operator()(const uint N_ps, DeviceBuffer<float>& xs,
        DeviceBuffer<float>& ys, DeviceBuffer<float>& zs,
        const DeviceMesh* mesh, const DeviceBuffer<AABB>& aabbs_leaf,
        const DeviceBuffer<AABB>& aabbs_internal,
        const DeviceBuffer<ChildNode>& children_l,
        const DeviceBuffer<ChildNode>& children_r) const
    {
        // Log::InfoTagged("LBVH", "Traversal Stack Size {}", STACK_SIZE);
        project_points<STACK_SIZE>
            <<<BLOCKS(N_ps), BLOCK_SIZE>>>(static_cast<uint>(N_ps), xs.ptr(),
                ys.ptr(), zs.ptr(), mesh->vxs.ptr(), mesh->vys.ptr(),
                mesh->vzs.ptr(), mesh->faces.ptr(), children_l.ptr(),
                children_r.ptr(), aabbs_leaf.ptr(), aabbs_internal.ptr());
    };
};

///@brief POD struct containing raw pointers to buffers managed by an `LBVH`
/// instance, for use in `__device__` functions and lambdas
struct DeviceLBVH {
    /// @brief underlying triangular mesh
    const DeviceMesh* mesh;
    ///@brief height of the binary tree
    const uint tree_height;
    ///@brief AABBs of leaf nodes of the tree
    const AABB* aabbs_leaf;
    ///@brief AABBs of internal nodes of the tree
    const AABB* aabbs_internal;
    ///@brief index and leaf vs. internal node info for left child of each
    /// internal node
    const ChildNode* children_l;
    ///@brief index and leaf vs. internal node info for right child of each
    /// internal node
    const ChildNode* children_r;
};

///@brief A Linear Bounding Volume Hierarchy (LBVH) constructed bottom-up from
/// Z-order/Morton-ordering of primitives to produce a balanced binary radix
/// tree. Enables fast spatial queries such as finding the closest point on a
/// triangular mesh to a query point (as used in the `project` method).
///
/// Stack-based traversal uses an optimistic heuristic for ordering of child
/// nodes (minimum distance to AABB) and prunes the search tree using the
/// closest squared distance observed so far during traversal.
///
/// Construction follows "Maximizing Parallelism in the Construction of BVHs,
/// Octrees, and k-d Tree" by [Tero Karras] and traversal is inspired by
/// "Nearest Neighbor Queries" by [Roussopoulos, Kelley, Vincent], similar to
/// but not quite the same as the implementation by ToruNiina
/// (https://github.com/ToruNiina/lbvh) which does not resort primitives by
/// Morton code, uses a AoS layout, a different traversal method etc.
class LBVH {
private:
    /// @brief pointer to the device mesh underlying the LBVH
    DeviceMesh* mesh;
    ///@brief height of the binary radix tree built, which is also the stack
    /// size required for traversal
    uint tree_height { 0 };

public:
    ///@brief number of primitives
    uint N_faces;
    ///@brief AABBs of leaf nodes of the BVH
    DeviceBuffer<AABB> aabbs_leaf;
    ///@brief AABBs of internal nodes of the BVH
    DeviceBuffer<AABB> aabbs_internal;
    ///@brief for every internal node i, `children_l[i]` yields a struct that
    /// contains a bitpacked representation of 1.) the index of the left child
    /// of node i and a.) whether the left child is a leaf or internal node
    DeviceBuffer<ChildNode> children_l;
    ///@brief for every internal node i, `children_r[i]` yields a struct that
    /// contains a bitpacked representation of 1.) the index of the right child
    /// of node i and a.) whether the right child is a leaf or internal node
    DeviceBuffer<ChildNode> children_r;
    ///@brief overall lower bound along each axis of the extent of all vertices
    /// in the underlying mesh, together with `bounds_max` forms the
    /// AABB encompassing the LBVH
    double3 bounds_min;
    ///@brief overall upper bound along each axis of the extent of all vertices
    /// in the underlying mesh, together with `bounds_min` forms the
    /// AABB encompassing the LBVH
    double3 bounds_max;

    ///@brief Construct a new LBVH object
    ///@param _mesh Mesh to construct an LBVH for
    LBVH(DeviceMesh* _mesh)
        : mesh(_mesh)
        , N_faces(static_cast<uint>(_mesh->faces.size()))
        , aabbs_leaf(N_faces)
        , aabbs_internal(N_faces - 1)
        , children_l(N_faces - 1, ChildNode { UINT32_MAX })
        , children_r(N_faces - 1, ChildNode { UINT32_MAX })
        , bounds_min(dv3(mesh->vxs.min(), mesh->vys.min(), mesh->vzs.min()))
        , bounds_max(dv3(mesh->vxs.max(), mesh->vys.max(), mesh->vzs.max()))
    {
        // compute morton codes for each face
        Log::InfoTagged(
            "LBVH", "Creating Morton Codes for {} primitives", N_faces);
        const double3 volume { bounds_max - bounds_min };
        MortonCodeGenerator gen(bounds_min, volume, mesh);

        // allocate buffer for morton codes
        DeviceBuffer<uint64_t> codes(N_faces);

        // compute codes
        thrust::transform(mesh->faces.get().begin(), mesh->faces.get().end(),
            codes.get().begin(), gen);

        // sort the faces by morton code
        Log::InfoTagged("LBVH", "Sorting Primitives By Morton Codes");
        thrust::sort_by_key(
            codes.get().begin(), codes.get().end(), mesh->faces.get().begin());

        // append indices to morton codes, now that the faces are sorted
        // note that this preserves the sorting but should make keys unique even
        // if morton codes coincide
        auto codes_start = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::counting_iterator<uint32_t>(0), codes.get().begin()));
        auto codes_end = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::counting_iterator<uint32_t>(codes.get().size()),
            codes.get().end()));
        thrust::transform(
            codes_start, codes_end, codes.get().begin(), MortonCodeExtender {});

        // create the tree internal nodes as a SoA
        {
            DeviceBuffer<uint> parent(N_faces - 1);
            DeviceBuffer<uint> leaf_parents(N_faces);

            // generate the tree
            Log::InfoTagged("LBVH", "Generating LBVH Tree");
            generate_tree<<<BLOCKS(N_faces - 1), BLOCK_SIZE>>>(codes.ptr(),
                N_faces, children_l.ptr(), children_r.ptr(), parent.ptr(),
                leaf_parents.ptr());

            // create bounding boxes
            Log::InfoTagged("LBVH", "Computing Bounding Boxes");
            DeviceBuffer<uint> visited(N_faces, 0);
            compute_aabbs<<<BLOCKS(N_faces), BLOCK_SIZE>>>(parent.ptr(),
                leaf_parents.ptr(), visited.ptr(), children_l.ptr(),
                children_r.ptr(), N_faces, mesh->vxs.ptr(), mesh->vys.ptr(),
                mesh->vzs.ptr(), mesh->faces.ptr(), aabbs_leaf.ptr(),
                aabbs_internal.ptr());

            // compute the tree height, reusing the `visited` buffer
            Log::InfoTagged("LBVH", "Computing Tree height");
            compute_tree_height<<<BLOCKS(N_faces), BLOCK_SIZE>>>(
                N_faces, leaf_parents.ptr(), parent.ptr(), visited.ptr());
            tree_height = visited.max();
            Log::InfoTagged(
                "LBVH", "Height of the LBVH radix tree: {}", tree_height);

            // end the current scope, which calls the destructors of the buffers
            // containing information about the parent node of each leaf and
            // internal node - this info is no longer needed after AABB
            // construction
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        Log::SuccessTagged("LBVH", "Generated LBVH Tree");
    };

    void project(DeviceBuffer<float>& xs, DeviceBuffer<float>& ys,
        DeviceBuffer<float>& zs) const
    {
        // assert that coordinate buffers have the same size
        size_t N_ps { xs.size() };
        if (ys.size() != N_ps || zs.size() != N_ps) {
            throw std::runtime_error("LBVH::project called with coordinate "
                                     "buffers of differing sizes");
        }
        // perform the projection of each point to the closest point on the
        // surface
        RuntimeTemplateSelectList<ProjectionLaunchFunctor, 4, 8, 16, 24, 32, 48,
            64, 128>::dispatch(tree_height, static_cast<uint>(N_ps), xs, ys, zs,
            mesh, aabbs_leaf, aabbs_internal, children_l, children_r);
    };

    ///@brief Get a POD struct with raw pointers to the buffers underlying the
    /// LBVH, for use in `__device__` side functions, lambdas and kernels
    ///@return `DeviceLBVH`
    DeviceLBVH get_pod() const
    {
        return DeviceLBVH {
            mesh,
            tree_height,
            aabbs_leaf.ptr(),
            aabbs_internal.ptr(),
            children_l.ptr(),
            children_r.ptr(),
        };
    };
};

#endif // DATASTRUCTURE_LBVH_CUH_
