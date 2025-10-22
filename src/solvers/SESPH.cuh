#ifndef SOLVERS_SESPH_CUH_
#define SOLVERS_SESPH_CUH_

// #include <thrust/device_vector.h>
#include "common.h"
#include "kernels.cuh"
#include "particles.cuh"
#include "buffer.cuh"

template <IsKernel K>
class SESPH 
{
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
    DeviceBuffer<float> rho;
    /// @brief Stiffness coefficient for the state equation
    float k{2000.};
    /// @brief Gravitational acceleration
    float3 g{v3(0.f, -9.81f, 0.f)};
    /// @brief rest density
    float rho_0{1000.};

    SESPH(K _W, uint _N, float _nu, const float _h) : W(_W), N(_N), nu(_nu), h(_h), rho(_N) 
    {};
    ~SESPH(){};

    void compute_accelerations(Particles& state, float dt);


    SESPH(const SESPH &) = delete;
    SESPH &operator=(const SESPH &) = delete;
};

#endif // SOLVERS_SESPH_CUH_