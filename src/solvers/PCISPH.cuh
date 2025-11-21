#ifndef SOLVERS_PCISPH_CUH_
#define SOLVERS_PCISPH_CUH_

///
///@file PCISPH.cuh
///@author Julian Karrer (github.com/JulianKarrer)
///@brief
///@version 0.1
///@date 2025-11-21
///
///@copyright Copyright (c) 2025
///
///

#include "common.h"
#include "buffer.cuh"
#include "kernels.cuh"

template <IsKernel K, Resort R> class PCISPH {
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
    DeviceBuffer<float>& rho;
    /// @brief acceleration buffer (x-component)
    DeviceBuffer<float>& ax;
    /// @brief acceleration buffer (y-component)
    DeviceBuffer<float>& ay;
    /// @brief acceleration buffer (z-component)
    DeviceBuffer<float>& az;
    /// @brief Gravitational acceleration
    float3 g { v3(0.f, -9.81f, 0.f) };
    /// @brief rest density
    float rho_0;
    /// @brief PCISPH stiffness coefficient δ from the PCISPH paper
    /// [Solenthaler, Pajarola 2009], except for factor 1/Δt² which is unknown
    /// at the time of calling this function
    const double δ;
    /// @brief maximum accepted average density as a factor of rest density
    /// rho_0
    const float eta_rho_max { 1.01 };
    /// @brief minimum iteration count
    const uint min_iter { 5 };

    /// @brief Function for pre-calculating PCISPH stifness for a template
    /// particle, sampled on a regular grid with particle spacing h
    /// @param rho_0 rest density of the fluid
    /// @param W kernel function used
    /// @param h fluid particle spacing
    /// @param κ factor between kernel support radius and particle spacing,
    /// (usually κ=2 for the implemented kernels, since they have 2h kernel
    /// support)
    /// @return stiffness coefficient
    static double k_c_pcisph(
        const double rho_0, const K W, const double h, const int κ = 2)
    {
        const double m { rho_0 * h * h * h };
        const float h_f { (float)h };
        float3 dW = v3(0.);
        /// Σ( || ∇W_ij ||²)
        double sum_of_sq = 0.;
        for (int x { -κ }; x <= κ; ++x) {
            for (int y { -κ }; y <= κ; ++y) {
                for (int z { -κ }; z <= κ; ++z) {
                    const float3 x_j { v3(
                        (float)x * h_f, (float)y * h_f, (float)z * h_f) };
                    const float3 dW_ij { W.nabla(x_j) };
                    std::cout << "x_j |x: " << x_j.x << "|y: " << x_j.y
                              << "|z: " << x_j.z << std::endl;
                    std::cout << "dW_ij |x: " << dW_ij.x << "|y: " << dW_ij.y
                              << "|z: " << dW_ij.z << std::endl;
                    dW = dW + dW_ij;
                    sum_of_sq += (double)dot(dW_ij, dW_ij);
                }
            }
        }
        /// || Σ ∇W_ij ||²
        const double sq_of_sum { (double)dot(dW, dW) };
        std::cout << sum_of_sq << "|" << sq_of_sum << std::endl;
        const double m_over_rho_0 { (m / rho_0) };
        const double beta { 2. * m_over_rho_0 };
        std::cout << beta << std::endl;
        return (float)(1. / (beta * (sq_of_sum + sum_of_sq)));
    }

public:
    PCISPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
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
        , δ(k_c_pcisph(_rho_0, _W, _h, 2))
    {
        // ensure that the buffer can hold all densitites
        rho.resize(_N);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
    };

    void step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    ~PCISPH() {};

    /// disallow copying
    PCISPH(const PCISPH&) = delete;
    /// disallow assignment
    PCISPH& operator=(const PCISPH&) = delete;
};

#endif // SOLVERS_PCISPH_CUH_
