#ifndef DATASTRUCTURE_UNIFORMGRID_CUH_
#define DATASTRUCTURE_UNIFORMGRID_CUH_

#include "buffer.cuh"
#include "vector_helper.cuh"
#include "particles.cuh"

/// @brief Compute the 3-dimensional index of a position `pos` within a grid of
/// specified lower bound of AABB and cell size
/// @param pos queried position
/// @param bound_min the lower bound of the AABB containing query points,
/// including safety margin
/// @param cell_size the cell size of the uniform grid
/// @return a 3-dimensional cell index, should be non-negative to avoid memory
/// errors
__device__ inline int3 _index3(
    const float3 pos, const float3 bound_min, const float cell_size)
{
    // compute cell index along each dimension
    return floor_div(pos - bound_min, cell_size);
}

/// @brief Linearize a 3-dimensional cell index to a single integer using the
/// natural order or XYZ space filling curve by providing the number of grid
/// cells in x-direction and the number of grid cells in a xy-plane
/// @param x grid cell index in x-direction
/// @param y grid cell index in y-direction
/// @param z grid cell index in z-direction
/// @param nx number of grid cells in one x-aligned strip
/// @param nxny number of grid cells in one xy-plane
/// @return
__device__ inline int _linearize(
    const int x, const int y, const int z, const uint nx, const uint nxny)
{
    // compute the linear index from this, using the natural order or
    // XYZ space-filling curve, where x is the fastest varying index followed
    // by y, then z
    return (z * nxny + (y * nx + x));
}

/// @brief Compute the linear cell index at a query position. Used during grid
/// construction and uses the same `_index3` and  `_linearize` functions used
/// during the querying procedure to ensure the consistency of indices.
/// @param pos queried position
/// @param bound_min the lower bound of the AABB containing query points,
/// including safety margin
/// @param cell_size the cell size of the uniform grid
/// @param nx number of grid cells in one x-aligned strip
/// @param nxny number of grid cells in one xy-plane
/// @return an unsigned, linearized cell id of the cell that the queried point
/// resides in. ATTENTION: only valid if the position is within the bounds of
/// the uniform grid, a negative cell coordinate along any axis may cause an
/// underflow of the unsigned integer and therefore memory errors!
__device__ inline uint _index_linear(const float3 pos, const float3 bound_min,
    const float cell_size, const uint nx, const uint nxny)
{
    // compute cell index along each dimension
    const int3 index_xyz { _index3(pos, bound_min, cell_size) };
    // compute the linear index from this, using the natural order or
    // XYZ space-filling curve, where x is the fastest varying index followed
    // by y, then z
    return _linearize(index_xyz.x, index_xyz.y, index_xyz.z, nx, nxny);
}

/// @brief type alias for bool indicating whether a uniform grid was constructed
/// for use with a resorted particle state, requiring no further indirection, or
/// if a `sorted` buffer must be stored to keep track of the indices of
/// particles sorted along a XYZ curve
enum class Resort : bool { no = false, yes = true };

/// @brief Struct parameterized over whether the particles were resorted or not,
/// which correspondingly contains a pointer to the `sorted` buffer or not
/// @tparam if `Resort::yes` then the struct is empty, otherwise it contains
/// `const uint* sorted`
template <Resort> struct MaybeSorted;
template <> struct MaybeSorted<Resort::no> {
    /// @brief pointer to CUDA buffer particle indices sorted by the
    /// space-filling curve underlying the uniform grid
    const uint* sorted;
};
template <> struct MaybeSorted<Resort::yes> { };

template <Resort R> struct UniformGrid : MaybeSorted<R> {
    /// @brief lower bound
    const float3 bound_min;
    /// @brief cell size of the uniform grid
    const float cell_size;
    /// @brief squared search radius for neighbourhood pruning
    const float r_c_2;
    /// @brief number of grid cells along x-direction
    const int nx;
    /// @brief number of grid cells in a xy-plane
    const int nxny;
    /// @brief pointer to CUDA buffer of prefix sums, indexing into the sorted
    /// array of particle indices
    const uint* prefix;

    __device__ inline uint sorted_lookup(const uint index) const
    {
        if constexpr (R == Resort::yes) {
            return index;
        } else {
            return this->sorted[index];
        }
    }

    template <typename MapOp>
    using AccType = std::invoke_result_t<MapOp, const uint, const uint,
        const float3, const float>;

    template <typename MapOp>
    __device__ inline auto ff_nbrs(const float3* __restrict__ x, const uint i,
        MapOp map,
        // the default initial value of the accumulator is the default
        // zero-initialized value of the type that is being accumulated
        AccType<MapOp> initial_value = AccType<MapOp> {}) const
    {
        AccType<MapOp> acc { initial_value };
        // compute the cell id of the queried position
        const float3 x_i { x[i] };
        const int3 cid3 { _index3(x_i, bound_min, cell_size) };

        // this strip must be shifted in y and z direction to nine different
        // starting positions to cover the full cube of 27 neighbouring cells,
        // while ensuring that no out-of-bounds accesses occur
        for (int iz { -1 }; iz <= 1; ++iz) {
            for (int iy { -1 }; iy <= 1; ++iy) {
                const int cid_y { cid3.y + iy };
                const int cid_z { cid3.z + iz };

                // check if the index provided is in the viable range
                // otherwise, at the edge of the datastructure, skip
                // non-existant cells
                // if (cid_y < 0 || cid_z < 0 || cid_y >=
                // nxyz.y || cid_z >= nxyz.z)
                //     continue;

                // from here on, at the given y and z offset, traverse a strip
                // of grid cells in x-direction, looking for neighbouring
                // particles:

                // start iterating one cell prior (x-1)
                // clamping should not be required due to margins:
                // const int cid_x{max(0, cid3.x - 1)};
                const int cid_x { cid3.x - 1 };
                // now, the linearized cell id of the cell at (x-1,y,z) can be
                // computed
                const int cid { _linearize(cid_x, cid_y, cid_z, nx, nxny) };
                // the prefix sum there is the number of particles with a lower
                // cid, which should be the index to the first particle in the
                // cell (x-1,y,z)
                const uint start_id { prefix[cid] };
                // similarly, look up the index of the first particle past the
                // cell (x+1,y,z): the index of the last particle in the x+1
                // cell is < that of the first particle two cells over, i.e. in
                // (x+2,y,z) or cid_i+3

                // clamping should not be required due to margins:
                // const uint end_id{prefix[min(nxnynz, cid + 3)]};
                const uint end_id { prefix[cid + 3] };

                // iterate from start_id inclusive to end_id exclusive
                for (uint j { start_id }; j < end_id; ++j) {
                    // skip any sample that is not within the search radius
                    const uint s_j { sorted_lookup(j) };
                    const float3 x_j { x[s_j] };
                    const float3 x_ij { x_i - x_j };
                    const float x_ij_l2 { dot(x_ij, x_ij) };
                    // for maximum perfomance, avoid sqrt and compare squared
                    // distance to squared search radius, since squaring is
                    // strictly monotonous on positive distances
                    // TODO: benchmark - this might have to be replaced with a
                    // conditional if the branch is not compiled such that warp
                    // divergence is avoided
                    if (x_ij_l2 <= r_c_2) {
                        // now, j is an actual neighbour of i
                        // call the map operators and reduce operators
                        acc += map(i, s_j, x_ij, x_ij_l2);
                    }
                }
            }
        }
        // finally, return the accumulator
        return acc;
    };
};

/// @brief A uniform grid implemented using a counting sort of particles within
/// the bounds specified at construction with a single prefix sum for efficient
/// access of neighbouring particles from a query position via `__device__`
/// functors.
///
/// Usage: specify the bounds, cell size of the uniform grid
///
/// The implementation largely follows [Hoetzlein 2014]:
/// https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf page
/// 20
///
/// ATTENTION: Since explicit bound checks are disabled in queries for
/// efficiency, querying points outside of the specified bounds is NOT SAFE! and
/// will cause memory errors.
class UniformGridBuilder {
private:
    // here order matters due to the initializer list relying on earlier values
    // to compute later ones:

    /// @brief  Cell size of the uniform grid
    const float _cell_size;
    /// @brief  Lower bound of the AABB containing all query points, including a
    /// safety margin of one cell size to exclude edge cases and simplify query
    /// logic to remove bound checking
    const float3 _bound_min;
    /// @brief  Spatial extend of the grid in terms of number of grid cells per
    /// spatial dimension
    const int3 nxyz;

    // buffers with as many entries as there are grid cells

    /// @brief  Buffer of size (#grid cells) for a prefix sum over the number of
    /// particles per cell. Used to index into the sorted array of particle
    /// indices
    DeviceBuffer<uint> prefix;
    /// @brief  Buffer of size (#grid cells) for the number of particles in each
    /// grid cell, to be atomically incremented and decremented during
    /// construction of the `UniformGrid`
    DeviceBuffer<uint> counts;

    // buffer with as many entries as there are particles
    /// @brief  Buffer of size (#particles) to hold the indices of particles,
    /// sorted by the space filling curve linearizing the 3-dimensional
    /// coordinate of each cell a particle may reside in
    DeviceBuffer<uint> sorted;

    void _construct(const DeviceBuffer<float3>& x);

public:
    /// @brief Construct a uniform grid to efficiently query the neighbourhood
    /// within some search radius of particles that MUST be contained in the
    /// range between `bound_min` and `bound_max` along each axis, using a grid
    /// of some given cell size.
    /// @param bound_min the lower bound of the AABB containing query points
    /// @param bound_max the upper bound of the AABB containing query points
    /// @param cell_size the cell size of the uniform grid
    UniformGridBuilder(
        const float3 bound_min, const float3 bound_max, const float cell_size);

    /// @brief Construct the uniform grid for the given buffer of query points,
    /// returning a POD structure that may be used on the device for querying
    /// neighbouring particles at positions within the AABB defined at
    /// construction of this `UniformGridBuilder`.
    /// @param x the buffer of positions to query
    /// @return a POD usable in a `__device__` context to providee functors that
    /// map and reduce over neighbouring particles around some query position
    UniformGrid<Resort::no> construct(const DeviceBuffer<float3>& x);

    UniformGrid<Resort::yes> construct_and_reorder(Particles& state);

    // no copying
    UniformGridBuilder(const UniformGridBuilder&) = delete;
    UniformGridBuilder& operator=(const UniformGridBuilder&) = delete;
};

#endif // DATASTRUCTURE_UNIFORMGRID_CUH_