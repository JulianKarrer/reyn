#ifndef SOLVERS_SESPH_CUH_
#define SOLVERS_SESPH_CUH_

#include "common.h"
#include "kernels.cuh"
#include "particles.cuh"
#include "buffer.cuh"
#include "datastructure/uniformgrid.cuh"

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
    /// @brief scalar buffer
    DeviceBuffer<float>& rho;
    /// @brief Stiffness coefficient for the state equation
    float k { 50. };
    /// @brief Gravitational acceleration
    float3 g { v3(0.f, -9.81f, 0.f) };
    /// @brief rest density
    float rho_0 { 1. };

    SESPH(K _W, uint _N, DeviceBuffer<float>& _rho, float _nu, const float _h)
        : W(_W)
        , N(_N)
        , nu(_nu)
        , h(_h)
        , rho(_rho)
    {
        // ensure that the buffer can hold all densitites
        rho.resize(_N);
    };
    ~SESPH() {};

    void step(Particles& state, const UniformGrid<R> grid, const float dt);

    /// disallow copying
    SESPH(const SESPH&) = delete;
    /// disallow assignment
    SESPH& operator=(const SESPH&) = delete;
};

#endif // SOLVERS_SESPH_CUH_