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
    /// @brief pressure buffer (this buffer is owned by the solver and therefore
    /// guaranteed to persist across steps)
    DeviceBuffer<float>& p;
    /// @brief IISPH diagonal element
    DeviceBuffer<float>& a_ii;
    /// @brief IISPH density invariance source term
    DeviceBuffer<float>& s_i;
    /// @brief bitpacked indicators of booleans: does the predicted density
    /// violate the constraint?
    DeviceBuffer<uint32_t>& ρ_err_threshold;
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
    /// @brief predicted density deviation threshold in units of rest density ρ₀
    const float eta_rho_max { 0.005 };
    /// @brief minimum iteration count
    const uint min_iter { 3 };
    /// @brief Jacobi weight ∈ (0,2)
    const float ω { 0.5f };
    /// @brief Whether or not to use a warm start, i.e. halve pressure values
    /// instead of resetting to zero in every timestep
    const bool warmstart { true };

public:
    IISPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
        DeviceBuffer<float>& _rho, DeviceBuffer<float>& _a_ii,
        DeviceBuffer<float>& _s_i, DeviceBuffer<float>& _ax,
        DeviceBuffer<float>& _ay, DeviceBuffer<float>& _az,
        DeviceBuffer<float>& _p, DeviceBuffer<uint32_t>& _ρ_err_threshold)
        : W(_W)
        , N(_N)
        , a_ii(_a_ii)
        , s_i(_s_i)
        , p(_p)
        , ρ_err_threshold(_ρ_err_threshold)
        , nu(_nu)
        , h(_h)
        , ρ(_rho)
        , ax(_ax)
        , ay(_ay)
        , az(_az)
        , ρ₀(_rho_0)
    {
        // ensure that the buffer can hold all densitites
        p.resize(_N);
        ρ.resize(_N);
        a_ii.resize(_N);
        s_i.resize(_N);
        // number of blocks of 32 bits required to store N bits: (unsigned ceil
        // divide, initialize to zero)
        ρ_err_threshold.resize((_N + 31) / 32, 0);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
        Log::Success("IISPH Solver initialized");
    };

    uint step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    ~IISPH() { };

    /// disallow copying
    IISPH(const IISPH&) = delete;
    /// disallow assignment
    IISPH& operator=(const IISPH&) = delete;
};

#endif // SOLVERS_IISPH_CUH_