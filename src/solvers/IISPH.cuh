#ifndef SOLVERS_IISPH_CUH_
#define SOLVERS_IISPH_CUH_

#include "log.h"
#include "buffer.cuh"
#include "kernels.cuh"

template <IsKernel K, Resort R> class IISPH {
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
    float3 g { v3(0.f, -9.81f, 0.f) };
    /// @brief rest density
    float ρ₀;
    /// @brief maximum accepted average density as a factor of rest density
    /// ρ₀
    const float eta_rho_max { 1.001 };
    /// @brief minimum iteration count
    const uint min_iter { 3 };

public:
    IISPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
        DeviceBuffer<float>& _rho, DeviceBuffer<float>& _ax,
        DeviceBuffer<float>& _ay, DeviceBuffer<float>& _az)
        : W(_W)
        , N(_N)
        , nu(_nu)
        , h(_h)
        , ρ(_rho)
        , ax(_ax)
        , ay(_ay)
        , az(_az)
        , ρ₀(_rho_0)
    {
        // ensure that the buffer can hold all densitites
        ρ.resize(_N);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
        Log::Success("IISPH Solver initialized");
    };

    uint step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    ~IISPH() {};

    /// disallow copying
    IISPH(const IISPH&) = delete;
    /// disallow assignment
    IISPH& operator=(const IISPH&) = delete;
};

#endif // SOLVERS_IISPH_CUH_