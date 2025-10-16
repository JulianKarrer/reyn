#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include "vector_helper.cuh"

/// @brief Base class of kernel f unctions for static polymorphism using CRTP design.
/// Any kernel function should
///
/// - be implemented as a class providing the `w` and `dw` shape function methods,
///
/// - use this base class constructor with the appropriate value for the normalization constant alpha.
///
/// For details on the structure of the kernel from a shape function and normalization factor, see Appendix A of Stefan Band's PhD Thesis.
/// @tparam Derived
template <typename Derived>
class Kernel
{
private:
    const float _h_bar_inv;
    const float _W_scale;
    const float _dW_scale;

public:
    /// pre-compute constant values in the constructor:
    /// the kernel function is scaled by alpha / h^d
    /// the kernel function gradient is scaled by alpha / h^(d+1)
    __host__ __device__ __forceinline__ Kernel(float h_bar, const float alpha) : _h_bar_inv(1.0f / h_bar), _W_scale(alpha / (h_bar * h_bar * h_bar)), _dW_scale(alpha / (h_bar * h_bar * h_bar * h_bar))
    {
    }

    __host__ __device__ __forceinline__ float operator()(float3 dx) const
    {
        const float len{norm(dx)};
        const float q{len * _h_bar_inv};
        return _W_scale * static_cast<const Derived *>(this)->w(q);
    }

    __host__ __device__ __forceinline__ float3 nabla(float3 dx) const
    {
        // compute dot product with itself (squared distance) using fused multiply add intrinsics
        const float len_2{dot(dx, dx)};
        // take a fast square root
        const float len{sqrtf(len_2)};
        // compute inverse square root using intrinsics
        // may return +/- infty
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1ga71ee45580cbeeea206297f0112aff42c
        const float one_over_len{rsqrtf(len_2)};
        // then use isfinite to check for infinities and NaNs, in which case the distance was zero
        // this incurrs one conditional
        const float safe_one_over_len{isfinite(one_over_len) ? one_over_len : 0.f};
        const float q{len * _h_bar_inv};
        // account for division by zero in normalization of zero-length vector:
        // -> in this case the result should be the zero vector
        return _dW_scale * static_cast<const Derived *>(this)->dw(q) * (dx * safe_one_over_len);
    }

    // prevent copying
    // Kernel(const Kernel &) = delete;
    // Kernel &operator=(const Kernel &) = delete;
};

/// @brief Cubic Spline Kernel function [Monaghan, taken from Stefan Band]
class C3 : public Kernel<C3>
{
public:
    __host__ __device__ __forceinline__
    C3(float h_bar) : Kernel(h_bar, 16.f * M_1_PI) {};
    __host__ __device__ __forceinline__ float w(float q) const;
    __host__ __device__ __forceinline__ float dw(float q) const;

    __host__ __device__ __forceinline__ float operator()(float3 dx) const;
    __host__ __device__ __forceinline__ float3 nabla(float3 dx) const;
};

#endif // KERNELS_CUH_
