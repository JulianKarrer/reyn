#ifndef VECTOR_CUH_
#define VECTOR_CUH_

#include <cuda_runtime.h>

// CONSTRUCTOR SHORTHANDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A shorthand constructor for `float3`, alias for `make_float3`
inline __host__ __device__ float3 v3(
    const float& x, const float& y, const float& z)
{
    return make_float3(x, y, z);
};
inline __host__ __device__ float3 v3(const float& x)
{
    return make_float3(x, x, x);
};

inline __host__ __device__ float3 v3(const uint& i, const float* __restrict__ x,
    const float* __restrict__ y, const float* __restrict__ z)
{
    return make_float3(x[i], y[i], z[i]);
};

inline __host__ __device__ void store_v3(const float3& vec, const uint& i,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z)
{
    x[i] = vec.x;
    y[i] = vec.y;
    z[i] = vec.z;
};

// BINARY ARITHMATIC OPERATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
};

inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
};

inline __host__ __device__ float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline __host__ __device__ float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

/// divide a 3-vector by a float to obtain an integer 3-vector by rounding down
/// with `floorf` and casting to integers in each component.
inline __host__ __device__ int3 floor_div(const float3& a, const float& b)
{
    return make_int3(
        (int)floorf(a.x / b), (int)floorf(a.y / b), (int)floorf(a.z / b));
};

/// divide a 3-vector by a float to obtain an integer 3-vector by rounding up
/// with `ceilf` and casting to integers in each component.
inline __host__ __device__ int3 ceil_div(const float3& a, const float& b)
{
    return make_int3(
        (int)ceilf(a.x / b), (int)ceilf(a.y / b), (int)ceilf(a.z / b));
};

/// @brief compute the dot product of two vectors
inline __host__ __device__ float dot(const float3& a, const float3& b)
{
    return fmaf(a.x, b.x, fmaf(a.y, b.y, (a.z * b.z)));
};

// division with scalar
inline __host__ __device__ float3 operator/(const float3& a, const float& b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
};

// multiplication with scalar (commutative)
inline __host__ __device__ float3 operator*(const float3& a, const float& b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
};
inline __host__ __device__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(b.x * a, b.y * a, b.z * a);
};

// UNARY ARITHMATIC OPERATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @brief NOTE: this function might behave differently in `__device__` and
/// `__host__` code due to alternate implementations depending on intrinsics
/// availablility!
///
/// saturate a float, i.e. clamp it to a [0.f ; 1.f] range (inclusive)
inline __host__ __device__ float sat(const float& a)
{
#ifdef __CUDA_ARCH__
    return __saturatef(a);
#else
    return fmin(fmax(a, 0.f), 1.f);
#endif
};

/// @brief NOTE: this function might behave differently in `__device__` and
/// `__host__` code due to alternate implementations depending on intrinsics
/// availablility!
///
/// take the 3D euclidean norm of a vector
inline __host__ __device__ float norm(const float3& a)
{
#ifdef __CUDA_ARCH__
    return norm3df(a.x, a.y, a.z);
#else
    return fsqrt(dot(a, a));
#endif
};

/// @brief Round down each component of a `float3` to the next highest integer
/// smaller than the component
/// @param a `float3` to round down
/// @return `int3` with highest components less than or equal to the respective
/// component of the original vector
inline __host__ __device__ int3 floor(const float3& a)
{
    return make_int3((int)floorf(a.x), (int)floorf(a.y), (int)floorf(a.z));
};

/// @brief Round down each component of a `float3` to the next highest integer
/// smaller than the component
/// @param a `float3` to round down
/// @return `float3` with highest components less than or equal to the
/// respective component of the original vector
inline __host__ __device__ float3 floorf(const float3& a)
{
    return v3(floorf(a.x), floorf(a.y), floorf(a.z));
};

// BINARY PREDICATES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @brief the disjunction (OR) of element-wise application of <=
/// @return a boolean indicating if a <= b in EVERY component or not
inline __host__ __device__ bool operator<=(const float3& a, const float3& b)
{
    return (a.x <= b.x) || (a.y <= b.y) || (a.z <= b.z);
};

/// @brief the disjunction (OR) of element-wise application of <=
/// @return a boolean indicating if a <= b in EVERY component or not
inline __host__ __device__ bool operator<=(const float3& a, const float& b)
{
    return (a.x <= b) || (a.y <= b) || (a.z <= b);
};

/// @brief the disjunction (OR) of element-wise application of >=
/// @return a boolean indicating if a >= b in EVERY component or not
inline __host__ __device__ bool operator>=(const float3& a, const float3& b)
{
    return (a.x >= b.x) || (a.y >= b.y) || (a.z >= b.z);
};

/// @brief the disjunction (OR) of element-wise application of >=
/// @return a boolean indicating if a >= b in EVERY component or not
inline __host__ __device__ bool operator>=(const float3& a, const float& b)
{
    return (a.x >= b) || (a.y >= b) || (a.z >= b);
};

#endif // VECTOR_CUH_