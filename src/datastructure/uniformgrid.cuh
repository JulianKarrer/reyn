#ifndef DATASTRUCTURE_UNIFORMGRID_CUH_
#define DATASTRUCTURE_UNIFORMGRID_CUH_

#include "buffer.cuh"
#include "utils/vector.cuh"
#include "particles.cuh"
#include <thrust/device_vector.h>

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
    using AccType
        = std::invoke_result_t<MapOp, const uint, const float3, const float>;

    template <typename MapOp>
    __device__ inline auto ff_nbrs(const float3 x_i,
        const float* __restrict__ xx, const float* __restrict__ xy,
        const float* __restrict__ xz, MapOp map,
        // the default initial value of the accumulator is the default
        // zero-initialized value of the type that is being accumulated
        AccType<MapOp> initial_value = AccType<MapOp> {}) const
    {
        AccType<MapOp> acc { initial_value };
        const AccType<MapOp> neutral_element {};
        // compute the cell id of the queried position
        const int3 cid3 { _index3(x_i, bound_min, cell_size) };

        // the range is the ceil of the ratio of search radius over cell size,
        // i.e. how many cells must be searched in each direction to guarantee
        // that the search radius is covered.
        // range = 1 if search radius is at most the cell size
        // range = 2 if search radius is at most twice the cell size
        // and so on
        constexpr int range { 2 };

        // this strip must be shifted in y and z direction to all adjacent
        // starting positions to cover the full cube of (2*range+1)^d
        // neighbouring cells, while ensuring that no out-of-bounds accesses
        // occur
        for (int z_id { cid3.z - range }; z_id <= (cid3.z + range); ++z_id)
            for (int y_id { cid3.y - range }; y_id <= (cid3.y + range);
                 ++y_id) {
                // traverse a strip of grid cells in x-direction, looking for
                // neighbouring particles:

                // start iterating at lower tip of the strip (x - range)
                const int x_id { cid3.x - range };
                // now, the linearized cell id of the cell at (x-range,y,z) can
                // be computed
                const int cid { _linearize(x_id, y_id, z_id, nx, nxny) };
                // the prefix sum there is the number of particles with a lower
                // cid, which should be the index to the first particle in the
                // cell (x-range,y,z)
                const uint start_id { prefix[cid] };
                // similarly, look up the index of the first particle past the
                // cell (x+range,y,z): the index of the last particle in the
                // x+range cell is < that of the first particle in the next cell
                // over
                const uint end_id { prefix[cid + (2 * range + 1)] };

                // iterate from start_id inclusive to end_id exclusive
                for (uint j { start_id }; j < end_id; ++j) {
                    // skip any sample that is not within the search radius
                    const uint s_j { sorted_lookup(j) };
                    const float3 x_j { v3(s_j, xx, xy, xz) };
                    const float3 x_ij { x_i - x_j };
                    const float x_ij_l2 { dot(x_ij, x_ij) };
                    // for better perfomance, avoid slow `sqrt` computation and
                    // instead compare squared distance to squared search
                    // radius, since squaring is strictly monotonous on positive
                    // distances
                    if (x_ij_l2 <= r_c_2) {
                        // now, j is an actual neighbour of i
                        // call the map operators and reduce in the accumulator
                        acc += map(s_j, x_ij, x_ij_l2);
                    }
                }
            }
        // finally, return the accumulator
        return acc;
    };
};

/// Concept describing a DeviceBuffer<float>& for variadic overload of
/// `construct_and_reorder`
template <typename T>
concept IsFltDevBufPtr = std::is_same_v<T, DeviceBuffer<float>*>;

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
    DeviceBuffer<uint> _prefix;
    /// @brief  Buffer of size (#grid cells) for the number of particles in each
    /// grid cell, to be atomically incremented and decremented during
    /// construction of the `UniformGrid`
    DeviceBuffer<uint> counts;

    // buffer with as many entries as there are particles
    /// @brief  Buffer of size (#particles) to hold the indices of particles,
    /// sorted by the space filling curve linearizing the 3-dimensional
    /// coordinate of each cell a particle may reside in
    DeviceBuffer<uint> sorted;

    void _construct(const DeviceBuffer<float>& xx,
        const DeviceBuffer<float>& xy, const DeviceBuffer<float>& xz,
        DeviceBuffer<uint>& prefix);

    // common implementation of uniform grid construction with variadic buffers
    // to resort along with the particle positions and a specified buffer for
    // storing the prefix sum, such that either the member variable can be used
    // to allocate only once when the grid is frequently built, or a buffer
    // external to this object can be used to provide persistance of the
    // datastructure beyond this builder's lifetime when e.g. a constant uniform
    // grid is built once for the remainder of the simulation
    template <IsFltDevBufPtr... MoreBufs>
        requires(sizeof...(MoreBufs) >= 0) // at least one buffer to reorder
    UniformGrid<Resort::yes> _construct_reorder_variadic(
        const float search_radius, DeviceBuffer<float>& tmp,
        DeviceBuffer<uint>& prefix, DeviceBuffer<float>& xx,
        DeviceBuffer<float>& xy, DeviceBuffer<float>& xz, MoreBufs... reorder)
    {
        // implementation needs to be here to be visible to other translation
        // units due to the template arguments, unfortunately

        // construct
        _construct(xx, xy, xz, prefix);

        // reorder (variadic)
        // reuse the tmp buffer to save memory at the cost of not being able to
        // use multiple cudaStreams reorder all position vectors to ensure the
        // query is correct, but also any of the variable number of additional
        // buffers that were passed in
        uint N { static_cast<uint>(sorted.size()) };
        tmp.resize(sorted.size());
        DeviceBuffer<float>* buffers[] = { reorder... };
        for (DeviceBuffer<float>* buf : buffers) {
            buf->reorder(sorted, tmp);
            // ensure that all resorted buffers fit to reduce the risk of
            // misusage and runtime errors
            if (buf->size() != N) {
                throw std::runtime_error(
                    "Attempted to resort an array of grid construction the "
                    "dimensions of which don't match the number of particles "
                    "in "
                    "the sorting.");
            }
        }
        xx.reorder(sorted, tmp);
        xy.reorder(sorted, tmp);
        xz.reorder(sorted, tmp);

        // return datastructure indicating that corresponding buffers are sorted
        // so no indirection is needed in the query
        return UniformGrid<Resort::yes> {
            .bound_min = _bound_min,
            .cell_size = _cell_size,
            .r_c_2 = search_radius * search_radius,
            .nx = nxyz.x,
            .nxny = nxyz.x * nxyz.y,
            .prefix = prefix.ptr(),
        };
    }

public:
    /// @brief Construct a uniform grid to efficiently query the neighbourhood
    /// within some search radius of particles that MUST be contained in the
    /// range between `bound_min` and `bound_max` along each axis, using a grid
    /// of some given cell size.
    /// @param bound_min the lower bound of the AABB containing query points
    /// @param bound_max the upper bound of the AABB containing query points
    /// @param search_radius the search radius to expect
    UniformGridBuilder(const float3 bound_min, const float3 bound_max,
        const float search_radius);

    /// @brief Construct the uniform grid for the given buffer of query points,
    /// returning a POD structure that may be used on the device for querying
    /// neighbouring particles at positions within the AABB defined at
    /// construction of this `UniformGridBuilder`.
    /// @param search_radius radius outside of which candidate neighbours around
    /// the query point are pruned
    /// @param xx x-components of positions to query
    /// @param xy y-components of positions to query
    /// @param xz z-components of positions to query
    /// @return a POD usable in a `__device__` context to providee functors that
    /// map and reduce over neighbouring particles around some query position
    UniformGrid<Resort::no> construct(const float search_radius,
        const DeviceBuffer<float>& xx, const DeviceBuffer<float>& xy,
        const DeviceBuffer<float>& xz);

    /// @brief Construct the uniform grid for the given buffer of query points,
    /// returning a POD structure that may be used on the device for querying
    /// neighbouring particles at positions within the AABB defined at
    /// construction of this `UniformGridBuilder`.
    ///
    /// In contrast to the `construct` method, this calls on the `Particles` to
    /// reorder according to the sorting used by the grid to improve memory
    /// coherency.
    /// @param search_radius radius outside of which candidate neighbours around
    /// the query point are pruned
    /// @param state the `Particles` state containing the positions to query and
    /// the buffers to reorder
    /// @param tmp a temporary buffer used for the gathering operation that
    /// reorders particle quantities
    /// @return a POD usable in a `__device__` context to providee functors that
    /// map and reduce over neighbouring particles around some query position
    UniformGrid<Resort::yes> construct_and_reorder(
        const float search_radius, DeviceBuffer<float>& tmp, Particles& state);

    /// @brief Construct the uniform grid for the given buffer of query points,
    /// returning a POD structure that may be used on the device for querying
    /// neighbouring particles at positions within the AABB defined at
    /// construction of this `UniformGridBuilder`.
    ///
    /// In contrast to the `construct` method, this calls on the `Particles` to
    /// reorder according to the sorting used by the grid to improve memory
    /// coherency.
    /// @tparam ...MoreBufs Variable number of zero or more
    /// `DeviceBuffer<float>*` to resort along the space-filling curve, must all
    /// have the same number of entries
    /// @param search_radius radius outside of which candidate neighbours around
    /// the query point are pruned
    /// @param tmp a temporary buffer used for the gathering operation that
    /// reorders particle quantities
    /// @param xx x-components of all points to index
    /// @param xy y-components of all points to index
    /// @param xz z-components of all points to index
    /// @param ...reorder pack of zero or more `DeviceBuffer<float>` to resort
    /// along the space-filling curve
    /// @return a POD usable in a `__device__` context to providee functors that
    /// map and reduce over neighbouring particles around some query position
    template <IsFltDevBufPtr... MoreBufs>
        requires(sizeof...(MoreBufs) >= 0) // at least one buffer to reorder
    UniformGrid<Resort::yes> construct_and_reorder(const float search_radius,
        DeviceBuffer<float>& tmp, DeviceBuffer<float>& xx,
        DeviceBuffer<float>& xy, DeviceBuffer<float>& xz, MoreBufs... reorder)
    {
        // in this overload, use the private prefix member
        return _construct_reorder_variadic(
            search_radius, tmp, _prefix, xx, xy, xz, reorder...);
    };

    /// @brief Construct the uniform grid for the given buffer of query points,
    /// returning a POD structure that may be used on the device for querying
    /// neighbouring particles at positions within the AABB defined at
    /// construction of this `UniformGridBuilder`.
    ///
    /// In contrast to the `construct` method, this calls on the `Particles` to
    /// reorder according to the sorting used by the grid to improve memory
    /// coherency.
    /// @tparam ...MoreBufs Variable number of zero or more
    /// `DeviceBuffer<float>*` to resort along the space-filling curve, must all
    /// have the same number of entries
    /// @param search_radius radius outside of which candidate neighbours around
    /// the query point are pruned
    /// @param tmp a temporary buffer used for the gathering operation that
    /// reorders particle quantities
    /// @param prefix output buffer for storing the prefix sum, in case it must
    /// outlive this `UniformGridBuilder`
    /// @param xx x-components of all points to index
    /// @param xy y-components of all points to index
    /// @param xz z-components of all points to index
    /// @param ...reorder pack of zero or more `DeviceBuffer<float>` to resort
    /// along the space-filling curve
    /// @return a POD usable in a `__device__` context to providee functors that
    /// map and reduce over neighbouring particles around some query position
    template <IsFltDevBufPtr... MoreBufs>
        requires(sizeof...(MoreBufs) >= 0) // at least one buffer to reorder
    UniformGrid<Resort::yes> construct_and_reorder(const float search_radius,
        DeviceBuffer<float>& tmp, DeviceBuffer<uint>& prefix,
        DeviceBuffer<float>& xx, DeviceBuffer<float>& xy,
        DeviceBuffer<float>& xz, MoreBufs... reorder)
    {
        // in this overload, use the prefix provided via argument
        return _construct_reorder_variadic(
            search_radius, tmp, prefix, xx, xy, xz, reorder...);
    };

    // no copying allowed, protect owned data in DeviceBuffers
    UniformGridBuilder(const UniformGridBuilder&) = delete;
    // no copying allowed, protect owned data in DeviceBuffers
    UniformGridBuilder& operator=(const UniformGridBuilder&) = delete;
};

#endif // DATASTRUCTURE_UNIFORMGRID_CUH_