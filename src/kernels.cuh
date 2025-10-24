#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include "vector_helper.cuh"
#include <cmath>

/// @brief Base class of kernel functions for static polymorphism using CRTP
/// design. Any kernel function should
///
/// - be implemented as a class providing the `w` and `dw` shape function
/// methods,
///
/// - use this base class constructor with the appropriate value for the
/// normalization constant alpha.
///
/// For details on the structure of the kernel from a shape function and
/// normalization factor, see Appendix A of Stefan Band's PhD Thesis.
/// @tparam Derived
template <typename Derived> class Kernel {
private:
    const float _h_bar_inv;
    const float _W_scale;
    const float _dW_scale;

protected:
    /// The base constructor is protected, instantiate subclasses instead.
    ///
    /// Pre-compute constant values in the constructor:
    /// - the kernel function is scaled by alpha / h^d
    /// - the kernel function gradient is scaled by alpha / h^(d+1)
    __host__ __device__ __forceinline__ Kernel(float h_bar, const float alpha)
        : _h_bar_inv(1.0f / h_bar)
        , _W_scale(alpha / (h_bar * h_bar * h_bar))
        , _dW_scale(alpha / (h_bar * h_bar * h_bar * h_bar))
    {
    }

public:
    __host__ __device__ __forceinline__ float operator()(float3 dx) const
    {
        const float len { norm(dx) };
        // sat helps with potential NaNs due to overflow when q is huge
        // which arguably should not happen but is covered for robustness.
        // sat conveniently clamps to q=1, which implies a kernel value of zero,
        // as intended.
        const float q { sat(len * _h_bar_inv) };
        return _W_scale * static_cast<const Derived*>(this)->w(q);
    }

    __host__ __device__ __forceinline__ float3 nabla(float3 dx) const
    {
        // compute dot product with itself (squared distance) using fused
        // multiply add intrinsics
        const float len_2 { dot(dx, dx) };
        // take a fast square root
        const float len { sqrtf(len_2) };
        const float one_over_len { 1.f / len };
        // then use isfinite to check for infinities and NaNs, in which case the
        // distance was zero this incurrs one conditional
        const float q { len * _h_bar_inv };
        const float scale { _dW_scale * static_cast<const Derived*>(this)->dw(q)
            * one_over_len };
        // account for division by zero in normalization of zero-length vector:
        // -> in this case the result should be the zero vector
        const float safe_scale { std::isfinite(scale) ? scale : 0.f };
        return safe_scale * dx;
    }
};

/// Create a concept that allows templating kernel launches on some
/// implementation of the `Kernel` class, while abstracting the specific type of
/// kernel function used, at no runtime cost.
template <typename T>
concept IsKernel = std::is_base_of_v<Kernel<T>, T>;

// Kernel function implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @brief Wendland C2 kernel function [Wendland 95, taken from Stefan Band]
class C2 : public Kernel<C2> {
public:
    __host__ __device__ __forceinline__ C2(float h_bar)
        : Kernel(h_bar, 21.f / 2.f * M_1_PI) {};
    __host__ __device__ __forceinline__ static float w(float q)
    {
        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t2 { 4.f * q + 1.f };
        const float t1_2 { t1 * t1 };
        return t1_2 * t1_2 * t2;
    };
    __host__ __device__ __forceinline__ static float dw(float q)
    {
        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t2 { -20.f * q };
        return t1 * t1 * t1 * t2;
    };
};

/// @brief Cubic B-Spline kernel function [Monaghan 92, taken from Stefan Band]
class B3 : public Kernel<B3> {
public:
    __host__ __device__ __forceinline__ B3(float h_bar)
        : Kernel(h_bar, 16.f * M_1_PI) {};
    __host__ __device__ __forceinline__ static float w(float q)
    {
        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t2 { sat(0.5f - q) };
        return t1 * t1 * t1 - 4.f * t2 * t2 * t2;
    };
    __host__ __device__ __forceinline__ static float dw(float q)
    {

        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t2 { sat(0.5f - q) };
        return -3.f * t1 * t1 + 12.f * t2 * t2;
    };
};

/// @brief Wendland C6 kernel function [Wendland 95, taken from Stefan Band]
class W6 : public Kernel<W6> {
public:
    __host__ __device__ __forceinline__ W6(float h_bar)
        : Kernel(h_bar, 1365.f / 64.f * M_1_PI) {};
    __host__ __device__ __forceinline__ static float w(float q)
    {
        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t1_2 { t1 * t1 };
        const float t1_4 { t1_2 * t1_2 };
        const float t1_8 { t1_4 * t1_4 };
        const float q_2 { q * q };
        const float t2 { 32.f * q_2 * q + 25.f * q_2 + 8.f * q + 1.f };
        return t1_8 * t2;
    };
    __host__ __device__ __forceinline__ static float dw(float q)
    {
        const float t1 { fmaxf(0.f, 1.f - q) };
        const float t1_2 { t1 * t1 };
        const float t1_4 { t1_2 * t1_2 };
        const float t1_7 { t1_4 * t1_2 * t1 };

        const float t2 { (16.f * q * q + 7.f * q + 1.f) * (-22.f * q) };
        return t1_7 * t2;
    };
};

/// @brief Double Cosine Kernel function [Yang, Peng, Liu 2013]
class COS : public Kernel<COS> {
public:
    __host__ __device__ __forceinline__ COS(float h_bar)
        : Kernel(h_bar, M_PI / (4.f * M_PI * M_PI - 30.f)) {};
    __host__ __device__ __forceinline__ static float w(float q)
    {
        constexpr float PI { static_cast<float>(M_PI) };
        const float pi_s { PI * fminf(1.f, q) };
        return 4.f * cosf(pi_s) + cosf(2.f * pi_s) + 3.f;
    };
    __host__ __device__ __forceinline__ static float dw(float q)
    {
        constexpr float PI { static_cast<float>(M_PI) };
        const float pi_s { PI * fminf(1.f, q) };
        constexpr float two_pi { 2.f * M_PI };
        return -2.f * two_pi * sinf(pi_s) - sinf(2.f * pi_s) * two_pi;
    };
};

/// Macro used for explicit template instantiation, so that code templated on
/// kernel function types is generated in all relevant translation units. ADD
/// NEW KERNELS TO THE MACRO :
#define FOREACH_KERNEL(X)                                                      \
    X(C2)                                                                      \
    X(B3)                                                                      \
    X(W6)                                                                      \
    X(COS)

#endif // KERNELS_CUH_
