#ifndef DATASTRUCTURE_LBVH_CUH_
#define DATASTRUCTURE_LBVH_CUH_

#include "scene/sample_boundary.cuh"
#include "buffer.cuh"
#include "vector_helper.cuh"
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
/// 1024.f] along every spatial dimension
__device__ static inline uint32_t morton_3d(double3 v)
{
    const float x = (float)min(max(v.x * 1024., 0.), 1023.);
    const float y = (float)min(max(v.y * 1024., 0.), 1023.);
    const float z = (float)min(max(v.z * 1024., 0.), 1023.);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

/// @brief Compare two 32 bit Morton codes, returning the length of the longest
/// common prefix. For i!=j, this function emulates unique morton codes by using
/// the respective indices as a tiebreaker when Morton codes are equal, meaning
/// 32 will never be returned.
///
/// This performs `__clz(kᵢ ⊗ kⱼ)` if kᵢ ≠ kⱼ (keys are distinct) and
/// `__clz(kᵢ' ⊗ kⱼ')` with `kᵢ' = (kᵢ << 32 | i)` etc. otherwise
/// @param code_i first Morton code to compare
/// @param i index of first Morton code
/// @param codes second Morton code to compare to the first
/// @param j index of second Morton code
/// @return the length of the longest common prefix of either the codes, or the
/// codes and their concatenated indices for tiebreaks.
__device__ static inline int common_prefix(
    const uint32_t code_i, const int i, const uint32_t code_j, const int j)
{
    if (code_i != code_j) {
        return __clz(code_i ^ code_j);
    } else {
        const uint64_t k_i { code_i };
        const uint64_t k_j { code_j };
        return __clzll(((k_i << 32) | i) ^ ((k_j << 32) | j));
    }
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
        return AABB { min(this->mini, other.mini),
            max(this->maxi, other.maxi) };
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

    __host__ __device__ float3 get_volume() const { return maxi - mini; };
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
/// vertices of a triangle
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
        , volume_inv(dv3(
              (1024. / _volume.x), (1024. / _volume.y), (1024. / _volume.z)))
        , vxs(mesh->vxs.ptr())
        , vys(mesh->vys.ptr())
        , vzs(mesh->vzs.ptr()) {};
    __device__ uint32_t operator()(const uint3 face) const
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
        return morton_code;
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
    const uint32_t* __restrict__ codes, const int first, const int last)
{
    const uint32_t first_code { codes[first] };
    const uint32_t last_code { codes[last] };

    // median split if codes are identical
    if (first_code == last_code) {
        return (first + last) >> 1; // shift to divide
    }

    // count leading zeros (CLZ intrinsic) of XOR'ed codes to find the highest
    // differing bit
    const int lcp { common_prefix(first_code, first, last_code, last) };

    // binary search for split point, which is the highest id where more than
    // lcp bits are shared with the `first` code
    int split { first };
    int step { last - first };
    do {
        step = (step + 1) >> 1; // exponential decrease
        const int new_split { split + step };
        if (new_split < last) {
            const int new_lcp { common_prefix(
                first_code, first, codes[new_split], new_split) };
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
    const uint32_t* __restrict__ codes, const uint N, const int i)
{
    if (i == 0) {
        return make_int2(0, N - 1);
    }
    const uint32_t own_code { codes[i] };
    const int delta_r { common_prefix(own_code, i, codes[i + 1], i + 1) };
    const int delta_l { common_prefix(own_code, i, codes[i - 1], i - 1) };
    const int d { delta_r > delta_l ? 1 : -1 }; // = sign(δᵣ - δₗ)

    // find one end of the range
    const int delta_min { min(delta_l, delta_r) };
    int lmax { 2 };

    while (true) {
        const int j { i + d * lmax };
        if (j < 0 || j >= N
            || common_prefix(own_code, i, codes[j], j) <= delta_min) {
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
            const int delta { common_prefix(own_code, i, codes[j], j) };
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
__global__ void generate_tree(const uint32_t* __restrict__ codes, const uint N,
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
__global__ void compute_aabbs(uint* __restrict__ parent,
    uint* __restrict__ leaf_parent, uint* __restrict__ visited,
    ChildNode* __restrict__ children_l, ChildNode* __restrict__ children_r,
    uint N, double* __restrict__ vxs, double* __restrict__ vys,
    double* __restrict__ vzs, uint3* __restrict__ faces,
    AABB* __restrict__ aabbs_leaf, AABB* __restrict__ aabbs_internal);

class LBVH {
private:
    DeviceMesh* mesh;

public:
    uint N_faces;
    DeviceBuffer<AABB> aabbs_leaf;
    DeviceBuffer<AABB> aabbs_internal;
    DeviceBuffer<ChildNode> children_l;
    DeviceBuffer<ChildNode> children_r;

    LBVH(DeviceMesh* _mesh)
        : mesh(_mesh)
        , N_faces((uint)_mesh->faces.size())
        , aabbs_leaf(N_faces)
        , aabbs_internal(N_faces - 1)
        , children_l(N_faces - 1)
        , children_r(N_faces - 1)
    {
        Log::InfoTagged("LBVH", "Computing Bounds");
        // assemble or compute the minimum and maximum bounds of overall
        // AABB encompassing all vertices
        double3 bounds_min { dv3(
            mesh->vxs.min(), mesh->vys.min(), mesh->vzs.min()) };
        double3 bounds_max { dv3(
            mesh->vxs.max(), mesh->vys.max(), mesh->vzs.max()) };

        // compute morton codes for each face
        Log::InfoTagged("LBVH", "Creating Morton Codes");
        const double3 volume { bounds_max - bounds_min };
        MortonCodeGenerator gen(bounds_min, volume, mesh);
        DeviceBuffer<uint32_t> codes(N_faces);
        thrust::transform(mesh->faces.get().begin(), mesh->faces.get().end(),
            codes.get().begin(), gen);
        // sort the faces by morton code
        Log::InfoTagged("LBVH", "Sorting Primitives By Morton Codes");
        thrust::sort_by_key(
            codes.get().begin(), codes.get().end(), mesh->faces.get().begin());

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
            Log::InfoTagged(
                "LBVH", "Computing Bounding Boxes of internal nodes");
            DeviceBuffer<uint> visited(N_faces - 1, 0);
            compute_aabbs<<<BLOCKS(N_faces), BLOCK_SIZE>>>(parent.ptr(),
                leaf_parents.ptr(), visited.ptr(), children_l.ptr(),
                children_r.ptr(), N_faces, mesh->vxs.ptr(), mesh->vys.ptr(),
                mesh->vzs.ptr(), mesh->faces.ptr(), aabbs_leaf.ptr(),
                aabbs_internal.ptr());

            // end the current scope, which calls the destructors of the buffers
            // containing information about the parent node of each leaf and
            // internal node - this info is no longer needed after AABB
            // construction
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        Log::SuccessTagged("LBVH", "Generated LBVH Tree");
    };
};

#endif // DATASTRUCTURE_LBVH_CUH_
