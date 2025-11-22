#ifndef SOLVERS_SESPH_CUH_
#define SOLVERS_SESPH_CUH_

#include "common.h"
#include "kernels.cuh"
#include "particles.cuh"
#include "buffer.cuh"
#include "datastructure/uniformgrid.cuh"
#include "scene/scene.cuh"

template <IsKernel K, Resort R> class SESPH {
private:
    /// @brief Kernel function
    const K W;
    /// @brief particle spacing h
    const float h;
    /// @brief number of particles
    uint N;
    /// @brief kinematic viscosity, with units of [L^2/T]
    float nu;
    /// @brief density buffer
    DeviceBuffer<float>& ρ;
    /// @brief acceleration buffer (x-component)
    DeviceBuffer<float>& ax;
    /// @brief acceleration buffer (y-component)
    DeviceBuffer<float>& ay;
    /// @brief acceleration buffer (z-component)
    DeviceBuffer<float>& az;
    /// @brief Gravitational acceleration
    float3 g;
    /// @brief rest density
    float rho_0;
    /// @brief stiffness coefficient k
    double k;

public:
    SESPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
        DeviceBuffer<float>& _rho, DeviceBuffer<float>& _ax,
        DeviceBuffer<float>& _ay, DeviceBuffer<float>& _az,
        float3 _g = v3(0.f, -9.81f, 0.f))
        : W(_W)
        , N(_N)
        , nu(_nu)
        , h(_h)
        , ρ(_rho)
        , ax(_ax)
        , ay(_ay)
        , az(_az)
        , rho_0(_rho_0)
        , k(500.)
        , g(_g)
    {
        // ensure that the buffer can hold all densitites
        ρ.resize(_N);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
        Log::Success("SESPH Solver initialized");
    };

    void step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    ~SESPH() {};

    /// disallow copying
    SESPH(const SESPH&) = delete;
    /// disallow assignment
    SESPH& operator=(const SESPH&) = delete;
};

#endif // SOLVERS_SESPH_CUH_