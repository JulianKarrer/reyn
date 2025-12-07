#include "scene/scene.cuh"
#include "scene/sample_boundary.cuh"
#include "datastructure/lbvh.cuh"
#include "utils/vector.cuh"
#include "scene/loader.h"
#include <filesystem>
#include <concepts>
#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/transform.h>

/// @brief Kernel used by one of the Scene constructors to initialize a set of
/// dynamic particles in a box using CUDA directly to set each position.
__global__ void _init_box_kernel(float3 min, float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ m,
    int3 nxyz, uint N, float h, float ρ₀)
{
    // calculate 3d index from 1d index of invocation and nx, ny limits
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    auto nx { nxyz.x };
    auto ny { nxyz.y };
    auto ix { i };
    auto iz { ix / (nx * ny) };
    ix -= iz * nx * ny;
    auto iy { ix / nx };
    ix -= iy * nx;

    // then initialize positions with spacing h
    store_v3(min + v3(ix * h, iy * h, iz * h), i, xx, xy, xz);
    // initial velocities are zero
    store_v3(v3(0.f), i, vx, vy, vz);
    // assume ideal rest mass for now
    m[i] = ρ₀ * h * h * h;
}

/// @brief Binary operator functor used by thrust to add pseudorandom zero-mean
/// values of given standard deviation to existing values using
/// `thrust::transform`, where an `N_offset` of the total number of threads that
/// used this functor previously can be used to ensure fresh random values are
/// chosen in each invocation
struct GenJitter {
    float jitter_stddev;
    uint N_offset;
    __device__ float operator()(uint idx, float current)
    {
        thrust::default_random_engine rand;
        thrust::normal_distribution<float> uni(0.f, jitter_stddev);
        rand.discard(idx + N_offset);
        return current + uni(rand);
    }
};

/// @brief Unary operator functor used by thrust to cull fluid particles that
/// are too close to the boundary, as indicated by the `cull_bdy_radius` in
/// units of particle spacing `h`
struct CullPredicate {
    UniformGrid<Resort::yes> bdy_grid;
    const float* bx;
    const float* by;
    const float* bz;
    float cull_sq;

    template <typename Tuple> __device__ bool operator()(Tuple const& t)
    {
        // first three components of the inbound zip iterator are the x,y and z
        // coordinates of the fluid particle
        const float x = thrust::get<0>(t);
        const float y = thrust::get<1>(t);
        const float z = thrust::get<2>(t);
        const float3 x_i { v3(x, y, z) };
        // count the number of boundary particles within the cull radius
        const int res { bdy_grid.ff_nbrs(x_i, bx, by, bz,
            [&] __device__(uint j, float3 x_ij, float x_ij_l2) {
                return x_ij_l2 <= cull_sq ? 1 : 0;
            }) };
        // if any such neighbour was counted, indicate with a 1
        return res == 0 ? 0 : 1;
    };
};

/// @brief Unary operator functor used by thrust to cull fluid particles that
/// are outside of a mesh corresponding to the boundary of the fluid, using the
/// calculated normal of the triangle hosting the closest point on the mesh to
/// each query point. Uses an LBVH  for efficient distance queries.
template <uint STACK_SIZE> struct InsidePredicate {
    const DeviceLBVH lbvh;
    const double* vxs;
    const double* vys;
    const double* vzs;
    const uint3* faces;

    template <typename Tuple> __device__ bool operator()(Tuple const& t)
    {
        // first three components of the inbound zip iterator are the x,y and z
        // coordinates of the fluid particle
        const float x = thrust::get<0>(t);
        const float y = thrust::get<1>(t);
        const float z = thrust::get<2>(t);
        const float3 q { v3(x, y, z) };

        // create the traversal function that, given the index of a primitive
        // (fluid boundary mesh triangle id) returns the squared distance to the
        // closest point on the primitive and the quantity of interest, in this
        // case a boolean indicating if the triangle's normal points towards or
        // away from the query point
        const auto traversal_func
            = [&] __device__(uint idx) -> thrust::tuple<float, bool> {
            // load triangle vertices
            const uint3 face { faces[idx] };
            const float3 a { v3(face.x, vxs, vys, vzs) };
            const float3 b { v3(face.y, vxs, vys, vzs) };
            const float3 c { v3(face.z, vxs, vys, vzs) };
            // get closest point on idx-th triangle
            const float3 projected_q { closest_point_on_triangle(q, a, b, c) };
            // compute squared distance
            const float3 diff { projected_q - q }; // Q to projected Q'
            const float dist_sq { dot(diff, diff) };
            // compute triangle normal
            const float3 e1 { b - a }; // edge from A to B
            const float3 e2 { c - a }; // edge from A to C
            const float3 tri_normal { cross(e1, e2) };
            // check on which side of the triangle we are:
            // nthe triangle normal should point away from q to be inside a
            // mesh, so Q to Q' and normal should coincide, yielding a
            // non-negative projection via dot product
            const bool outside { dot(tri_normal, diff) >= 0. };
            // return tuple
            return thrust::make_tuple(dist_sq, !outside);
        };

        // now apply the higher order function for BVH traversal using this
        // lambda and return the result
        return bvh_find_closest<STACK_SIZE, bool>(q, lbvh.children_l,
            lbvh.children_r, lbvh.aabbs_leaf, lbvh.aabbs_internal,
            traversal_func);
    };
};

/// @brief Launcher functor for culling fluid particles according to the
/// `InsidePredicate` that decides if a position is inside or outside a given
/// mesh. This is an adapter used to facilitate using
/// `RuntimeTemplateSelectList` to select a sufficient but small stack size for
/// BVH traversal at runtime using compile-time recursion to search an ordered
/// list of possible template parameters and picking the earliest (smallest)
/// that fits the tree height.
/// @tparam STACK_SIZE
template <uint STACK_SIZE> struct LaunchInsidePredicate {
    void operator()(const DeviceLBVH& lbvh_pod, const DeviceMesh& fluid_mesh,
        const uint N, float h, Particles& state, DeviceBuffer<float>& tmp,
        uint& new_N) const
    {
        InsidePredicate<STACK_SIZE> inside_predicate { lbvh_pod,
            fluid_mesh.vxs.ptr(), fluid_mesh.vys.ptr(), fluid_mesh.vzs.ptr(),
            fluid_mesh.faces.ptr() };
        new_N = cull_particles_by_predicate(N, h, state, tmp, inside_predicate);
    };
};

///@brief Struct used for static assertion that predicates passed into
///`cull_particles_by_predicate` have a fitting `operator()` implementation.
template <typename T, typename Tuple, typename = void>
struct has_templated_call_operator : std::false_type { };

///@brief Struct used for static assertion that predicates passed into
///`cull_particles_by_predicate` have a fitting `operator()` implementation.
template <typename T, typename Tuple>
struct has_templated_call_operator<T, Tuple,
    std::void_t<decltype(std::declval<T>().template operator()<Tuple>(
        std::declval<Tuple const&>()))>> : std::true_type { };

template <typename Pred>
/// @brief From the `Particles`, remove all fluid particles that fulfill the
/// `Predicate` functor `predicate`. This functor must have a `bool
/// operator(Tuple t) const`, where `Tuple` is a thrust tuple of
/// `x,y,z,vx,vy,vz,m` for each particle.
/// @param h particle spacing
/// @param state `Particles`
/// @param tmp temporary buffer for resorting
/// @return the remaining number of fluid particles
uint cull_particles_by_predicate(const uint N, const float h, Particles& state,
    DeviceBuffer<float>& tmp, Pred predicate)
{
    // build a zipped iterator to apply remove_if to positions, velocities and
    // masses in one go using the same predicate functor
    auto dx = thrust::device_pointer_cast(state.xx.ptr());
    auto dy = thrust::device_pointer_cast(state.xy.ptr());
    auto dz = thrust::device_pointer_cast(state.xz.ptr());
    auto first = thrust::make_zip_iterator(thrust::make_tuple(dx, dy, dz,
        state.vx.get().begin(), state.vy.get().begin(), state.vz.get().begin(),
        state.m.get().begin()));
    auto last = first + N;

    using Tuple = thrust::tuple<decltype(dx[0]), decltype(dy[0]),
        decltype(dz[0]), decltype(state.vx.get().begin()[0]),
        decltype(state.vy.get().begin()[0]),
        decltype(state.vz.get().begin()[0]),
        decltype(state.m.get().begin()[0])>;
    static_assert(has_templated_call_operator<Pred, Tuple>::value,
        "Pred must provide: template <typename Tuple> bool operator()(Tuple "
        "const&) (callable with the particle thrust::tuple).");

    // apply remove_if using the culling predicate for particles too close to
    // the boundary
    auto new_end { thrust::remove_if(thrust::device, first, last, predicate) };

    // get the remaining number of particles
    uint new_N = thrust::distance(first, new_end);

    // resize tmp buffer to new size to save each component of externally
    // managed memory before resizing it
    state.resize_truncate(new_N, tmp);

    return new_N;
}

/// @brief Add a small, normally distributed jitter with (σ = `jitter_stddev *
/// h`) to the particle positions in `state`
/// @param N number of particles
/// @param h particle spacing
/// @param state `Particles` that contain positions to randomly perturb
/// @param jitter_stddev standard deviation of normally distributed jitter in
/// units of particle spacing h
void jitter_positions(
    const uint N, const float h, Particles& state, const float jitter_stddev)
{
    auto ndx { thrust::device_pointer_cast(state.xx.ptr()) };
    auto ndy { thrust::device_pointer_cast(state.xy.ptr()) };
    auto ndz { thrust::device_pointer_cast(state.xz.ptr()) };
    const float stddev { jitter_stddev * h };
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N), ndx, ndx, GenJitter { stddev, 0 });
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N), ndy, ndy, GenJitter { stddev, N });
    thrust::transform(thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N), ndz, ndz,
        GenJitter { stddev, 2 * N });
}

Scene::Scene(const std::filesystem::path& path, const uint N_desired,
    const float3 min, const float3 max, const float _ρ₀, Particles& state,
    DeviceBuffer<float>& tmp, const float bdy_oversampling_factor,
    const float cull_bdy_radius, const float jitter_stddev)
    : ρ₀(_ρ₀)
    , h(cbrtf(prod(max - min) / (float)N_desired))
    , bdy(sample_mesh(load_mesh_from_obj(path, { "fluid" }), h, _ρ₀,
          bdy_oversampling_factor))
    // ensure the scene bounds are those of the boundary samples
    , bound_min(bdy.bound_min)
    , bound_max(bdy.bound_max)
    , grid_builder(UniformGridBuilder(bound_min, bound_max, 2.f * h))
{
    // compute preliminary particle count and save it
    const float3 dxyz { max - min };
    const int3 nxyz { floor_div(dxyz, h) };
    N = { (uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z) };
    // exit early if the fluid domain is empty
    if (N == 0)
        throw std::domain_error(
            "Initialization of box failed, zero particles placed");

    // resize without regard for existing data in `state`, just allocate
    // sufficient amount of uninitialized memory
    state.resize_uninit(N);

    // place fluid particles
    _init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(min, state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), state.m.ptr(), nxyz, N, h, ρ₀);
    CUDA_CHECK(cudaGetLastError());

    // build a predicate for culling particles too close to the boundary
    const auto bx { bdy.xs.ptr() };
    const auto by { bdy.ys.ptr() };
    const auto bz { bdy.zs.ptr() };
    const float cull_sq { cull_bdy_radius * cull_bdy_radius * h * h };
    const auto bdy_grid { bdy.grid }; // get POD for device-lambda
    CullPredicate cull_predicate { bdy_grid, bx, by, bz, cull_sq };

    // remove particles too close to the boundary
    const uint new_N { cull_particles_by_predicate(
        N, h, state, tmp, cull_predicate) };

    if (new_N != N) {
        Log::Warn("Culled particles closer than {}h to boundary: {} -> {}",
            cull_bdy_radius, N, new_N);
    }
    N = new_N;

    // add a pseudorandom jitter to the coordinates
    jitter_positions(new_N, h, state, jitter_stddev);

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    Log::Success(
        "Scene initialized:\tN={}, Nbdy={}, h={}", N, bdy.xs.size(), h);
}

Scene Scene::from_obj(const std::filesystem::path& path, const uint N_desired,
    const float ρ₀, Particles& state, DeviceBuffer<float>& tmp,
    const float bdy_oversampling_factor, const float cull_bdy_radius,
    const float jitter_stddev)
{
    // load fluid boundary mesh
    DeviceMesh fluid_mesh { DeviceMesh::from(
        load_mesh_from_obj(path, {}, "fluid")) };

    // calculate particle spacing from fluid volume to achieve desired particle
    // count
    const float h { (float)cbrt(fluid_mesh.get_volume() / (double)N_desired) };

    // build an lbvh to accelerate closest point queries to the fluid
    // boundary mesh
    const LBVH lbvh(&fluid_mesh);

    // sample boundary
    BoundarySamples bdy { sample_mesh(load_mesh_from_obj(path, { "fluid" }), h,
        ρ₀, bdy_oversampling_factor) };

    // extend the scene boundaries to fit the fluid if required
    const float3 fluid_min { v3(lbvh.bounds_min) };
    const float3 fluid_max { v3(lbvh.bounds_max) };
    const float3 ε { v3(0.5 * h) }; // safety margin for bound adjustment
    float3 bound_min { min(bdy.bound_min, fluid_min - ε) };
    float3 bound_max { max(bdy.bound_max, fluid_max + ε) };

    // compute an upper bound of the fluid particle count from LBVH bounds
    const float3 dxyz { fluid_max - fluid_min };
    const int3 nxyz { floor_div(dxyz, h) };
    uint N { (uint)abs(nxyz.x) * (uint)abs(nxyz.y) * (uint)abs(nxyz.z) };
    // exit early if the fluid domain is empty
    if (N == 0)
        throw std::domain_error(
            "Initialization of box failed, zero particles placed");

    // resize without regard for existing data in `state`, just allocate
    // sufficient amount of uninitialized memory
    state.resize_uninit(N);

    // place fluid particles
    _init_box_kernel<<<BLOCKS(N), BLOCK_SIZE>>>(fluid_min, state.xx.ptr(),
        state.xy.ptr(), state.xz.ptr(), state.vx.ptr(), state.vy.ptr(),
        state.vz.ptr(), state.m.ptr(), nxyz, N, h, ρ₀);
    CUDA_CHECK(cudaGetLastError());

    // cull fluid particles that are outside of the specified volume:
    // if the triangle normal at the closest point on the mesh points towards
    // the sample, cull it
    const DeviceLBVH lbvh_pod { lbvh.get_pod() };
    uint new_N_inside { N };

    RuntimeTemplateSelectList<LaunchInsidePredicate, 2, 4, 8, 16, 32, 64,
        128>::dispatch(lbvh_pod.tree_height, lbvh_pod, fluid_mesh, N, h, state,
        tmp, new_N_inside);

    // cull fluid particles that are too close to the boundary
    const auto bx { bdy.xs.ptr() };
    const auto by { bdy.ys.ptr() };
    const auto bz { bdy.zs.ptr() };
    const float cull_sq { cull_bdy_radius * cull_bdy_radius * h * h };
    const auto bdy_grid { bdy.grid }; // get POD for device-lambda
    CullPredicate cull_predicate { bdy_grid, bx, by, bz, cull_sq };
    const uint new_N { cull_particles_by_predicate(
        new_N_inside, h, state, tmp, cull_predicate) };

    if (new_N != N) {
        Log::Warn("Culled particles closer than {}h to boundary: {} -> {}",
            cull_bdy_radius, N, new_N);
    }
    N = new_N;

    // add a pseudorandom jitter to the coordinates
    jitter_positions(new_N, h, state, jitter_stddev);

    // block and wait for operation to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    Log::Success(
        "Scene initialized:\tN={}, Nbdy={}, h={}", N, bdy.xs.size(), h);
    return Scene(ρ₀, h, N, std::move(bdy), bound_min, bound_max);
}

__global__ void _hard_enforce_bounds(const float3 bound_min,
    const float3 bound_max, uint N, float* __restrict__ xx,
    float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ vx,
    float* __restrict__ vy, float* __restrict__ vz)
{
    // calculate index and ensure safety at bounds
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    const float3 x_i { v3(i, xx, xy, xz) };
    // if at the boundary, mirror velocity and damp it
    if (x_i <= bound_min || bound_max <= x_i) {
        const float3 v_i { v3(i, vx, vy, vz) };
        store_v3(-0.1 * v_i, i, vx, vy, vz);
    }
    // always clamp positions to the bounds
    store_v3(max(min(x_i, bound_max), bound_min), i, xx, xy, xz);
}

void Scene::hard_enforce_bounds(Particles& state) const
{
    _hard_enforce_bounds<<<BLOCKS(N), BLOCK_SIZE>>>(bound_min, bound_max, N,
        state.xx.ptr(), state.xy.ptr(), state.xz.ptr(), state.vx.ptr(),
        state.vy.ptr(), state.vz.ptr());
    CUDA_CHECK(cudaGetLastError());
};
