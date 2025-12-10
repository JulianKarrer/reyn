#ifndef PARTICLES_H_
#define PARTICLES_H_

class GUI;

#include "common.h"
#include "buffer.cuh"
#include <iostream>

/// @brief An object holding the minimum amount of information required to
/// describe the state of the particle system at any point in time, i.e.
///
/// - positions
///
/// - velocities
///
/// - masses
///
/// - some useful constants such as rest density `ρ₀` and particle spacing
/// `h`
///
/// Other properties such as densities, accelerations etc. should be owned,
/// computed and handled by the respective pressure solver as required in each
/// time step.
class Particles {
public:
    DeviceBuffer<float> xx;
    DeviceBuffer<float> xy;
    DeviceBuffer<float> xz;
    DeviceBuffer<float> vx;
    DeviceBuffer<float> vy;
    DeviceBuffer<float> vz;
    DeviceBuffer<float> m;
    float ρ₀;

    Particles(const int N, float ρ₀);
    Particles(GUI* _gui, float _rho_0);

    /// @brief Resize all buffers. This leaves the positions buffer
    /// uninitialized if externally handled by the GUI for OpenGL interop!
    /// @param N new desired number of particles
    /// @param positions_only only resize and initialize positions, leaving all
    /// other fields untouched. ATTENTION! this leaves the buffer sizes
    /// inconsistent and should only be used in conjunction with e.g. a later
    /// `resize_truncate`, such as when batching candidate initial particle
    /// positions in `Scene` construction to reduce peak memory usage
    void resize_uninit(uint N, bool positions_only = false);

    /// @brief Resize all buffers, keeping the data but potentially truncating
    /// it if the size decreases
    /// @param N new desired number of particles
    /// @param tmp temporary buffer for resizing externally managed buffers
    /// without losing data
    void resize_truncate(uint N, DeviceBuffer<float>& tmp);

    /// @brief Pointer to the GUI instance managing the position buffer, if any,
    /// and `nullptr` otherwise
    GUI* const gui { nullptr };

    /// @brief Reorder all particle attributes (mass, velocities, positions) in
    /// the order provided by the map `sorted`, which must be a permutation of
    /// the numbers \f$[0; N-1]\f$. This is essentially a gather operation,
    /// lifted on all particle attributes.
    /// @param sorted permutation of \f$[0; N-1]\f$ to resort
    /// particle attributes with. Must have length `N` for `N` particles.
    /// @param tmp a temporary buffer used to resort efficiently (and not
    /// in-place). Must not be an externally managed `DeviceBuffer` since
    /// `thrust` functionality is used, so this would throw an error.
    void reorder(const DeviceBuffer<uint>& sorted, DeviceBuffer<float>& tmp);

    // Explicitly forbid a copy constructor, since destructor must only be
    // called once to ensure cudaFree does not free twice
    Particles(const Particles&) = delete;
    Particles& operator=(const Particles&) = delete;
};

#endif // PARTICLES_H_