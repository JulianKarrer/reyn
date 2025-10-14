#ifndef PARTICLES_H_
#define PARTICLES_H_

class GUI;

#include "common.h"
#include <iostream>

/// @brief An object holding the minimum amount of information required to describe the state of the particle system at any point in time, i.e.
///
/// - positions
///
/// - velocities
///
/// - masses
///
/// - some useful constants such as rest density `rho_0` and particle spacing `h`
///
/// Other properties such as densities, accelerations etc. should be owned, computed and handled by the respective pressure solver as required in each time step.
class Particles
{
public:
    float3 *x{nullptr};
    float3 *v{nullptr};
    float *m{nullptr};
    const float h;
    const float rho_0;

    Particles(const int N, float h, float rho_0);
    Particles(GUI *_gui, float _h, float _rho_0);
    ~Particles();

    /// @brief Set the pointer to the position buffer. Used by GUI to ensure externally managed position buffers that are shared with OpenGL VBOs and mapped for use by CUDA are consistent.
    /// @param x new pointer to positions
    void set_x(float3 *x);

    /// Resize all buffers. This leaves the new memory uninitialized!
    /// Internally uses `cudaFree`, then `cudaMalloc` but handles externally managed position buffers from the GUI
    void resize_uninit(uint N);

    /// @brief Pointer to the GUI instance managing the position buffer, if any, and `nullptr` otherwise
    GUI *const gui{nullptr};
};

#endif // PARTICLES_H_