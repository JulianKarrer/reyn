#ifndef SOLVERS_PCISPH_CUH_
#define SOLVERS_PCISPH_CUH_

#include "log.h"
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
    /// @brief PCISPH stiffness coefficient δΔt² from the PCISPH paper
    /// [Solenthaler, Pajarola 2009], except for factor 1/Δt² which is unknown
    /// at the time of calling this function
    const double δΔt²;
    /// @brief maximum accepted average density as a factor of rest density
    /// ρ₀
    const float eta_rho_max { 0.001f };
    /// @brief minimum iteration count
    const uint min_iter { 3 };

    /// @brief Function for pre-calculating PCISPH stifness for a template
    /// particle, sampled on a regular grid with particle spacing h
    /// @param ρ₀ rest density of the fluid
    /// @param W kernel function used
    /// @param h fluid particle spacing
    /// @param κ factor between kernel support radius and particle spacing,
    /// (usually κ=2 for the implemented kernels, since they have 2h kernel
    /// support)
    /// @return stiffness coefficient
    static double k_c_pcisph(
        const double ρ₀, const K W, const double h, const int κ = 2)
    {
        const float h_f { static_cast<float>(h) };
        float3 sumdW = v3(0.);
        /// Σ( || ∇W_ij ||²)
        double sum_of_sq = 0.;
        for (int x { -κ }; x <= κ; ++x) {
            for (int y { -κ }; y <= κ; ++y) {
                for (int z { -κ }; z <= κ; ++z) {
                    const float3 x_j { v3(static_cast<float>(x) * h_f,
                        static_cast<float>(y) * h_f,
                        static_cast<float>(z) * h_f) };
                    const float3 dW_ij { W.nabla(-x_j) };
                    sumdW += dW_ij;
                    sum_of_sq += static_cast<double>(dot(dW_ij, dW_ij));
                }
            }
        }
        /// || Σ ∇W_ij ||²
        const double sq_of_sum { static_cast<double>(dot(sumdW, sumdW)) };
        // m / ρ₀ = V₀ = (h³ * ρ₀)/ρ₀ = h³
        const double V₀ { h * h * h };
        const double beta { 2. * V₀ * V₀ };
        Log::Info("PCISPH template particle stats:\n\t| Σ ∇W_ij ||² = {}\n\tΣ( "
                  "|| ∇W_ij ||²) = {}\n\t β = {}",
            sq_of_sum, sum_of_sq, beta);
        return static_cast<float>(1. / (beta * (sq_of_sum + sum_of_sq)));
    }

public:
    PCISPH(K _W, uint _N, float _nu, const float _h, const float _rho_0,
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
        , δΔt²(k_c_pcisph(_rho_0, _W, _h, 2))
    {
        // ensure that the buffer can hold all densitites
        ρ.resize(_N);
        ax.resize(_N);
        ay.resize(_N);
        az.resize(_N);
        Log::Success("PCISPH Solver initialized");
    };

    uint step(Particles& state, const UniformGrid<R> grid,
        const BoundarySamples& bdy, const float dt);

    ~PCISPH() {};

    /// disallow copying
    PCISPH(const PCISPH&) = delete;
    /// disallow assignment
    PCISPH& operator=(const PCISPH&) = delete;
};

#endif // SOLVERS_PCISPH_CUH_
