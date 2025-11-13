#ifndef SOLVERS_SESPH_CUH_
#define SOLVERS_SESPH_CUH_

#include "common.h"
#include "kernels.cuh"
#include "particles.cuh"
#include "buffer.cuh"
#include "datastructure/uniformgrid.cuh"
#include "scene/sample_boundary.cuh"

template <IsKernel K, Resort R> class SESPH {
private:
public:
    /// @brief Kernel function
    const K W;
    /// @brief particle spacing h
    const float h;
    /// @brief number of particles
    uint N;
    /// @brief kinematic viscosity, with units of [L^2/T]
    float nu;
    /// @brief density buffer
    DeviceBuffer<float>& rho;
    /// @brief acceleration buffer (x-component)
    DeviceBuffer<float>& ax;
    /// @brief acceleration buffer (y-component)
    DeviceBuffer<float>& ay;
    /// @brief acceleration buffer (z-component)
    DeviceBuffer<float>& az;
    /// @brief Stiffness coefficient for the state equation
    float k { 1000. };
    /// @brief Gravitational acceleration
    float3 g { v3(0.f, -9.81f, 0.f) };
    /// @brief rest density
    float rho_0;

    SESPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
        DeviceBuffer<float>& _rho, DeviceBuffer<float>& _ax,
        DeviceBuffer<float>& _ay, DeviceBuffer<float>& _az)
        : W(_W)
        , N(_N)
        , nu(_nu)
        , h(_h)
        , rho(_rho)
        , ax(_ax)
        , ay(_ay)
        , az(_az)
        , rho_0(_rho_0)
    {
        // ensure that the buffer can hold all densitites
        rho.resize(_N);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
    };
    ~SESPH() {};

    void step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    /// disallow copying
    SESPH(const SESPH&) = delete;
    /// disallow assignment
    SESPH& operator=(const SESPH&) = delete;
};

#endif // SOLVERS_SESPH_CUH_