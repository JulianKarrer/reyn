
#ifndef PARTICLES_H_
#define PARTICLES_H_
#include "gui.h"
#include "common.h"
#include <cuda_gl_interop.h>
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
    Particles(GUI &gui, const int N, float _h, float _rho_0);
    ~Particles();

    void set_x(float3 *x);

private:
    bool x_managed_by_gui{false};
};

#endif // PARTICLES_H_