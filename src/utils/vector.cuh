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

/// A shorthand constructor for `float3` using the same float for each of the
/// three components
inline __host__ __device__ float3 v3(const float& x)
{
    return make_float3(x, x, x);
};

/// A shorthand for converting a `double3` to a `float3`
inline __host__ __device__ float3 v3(const double3& v)
{
    return make_float3(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z));
};

/// A shorthand constructor for `double3`, alias for `make_double3`
inline __host__ __device__ double3 dv3(
    const double& x, const double& y, const double& z)
{
    return make_double3(x, y, z);
};

/// @brief Constructor for a `float3` that loads each compoenent from the `i`th
/// entry provided in the accompanying pointers in order to construct a vector
/// from quantities stored in SoA format
/// @param i index of vector to load
/// @param x pointer to x-components
/// @param y pointer to y-components
/// @param z pointer to z-components
/// @return a single `float3` constructed from three load operations
inline __host__ __device__ float3 v3(const uint& i, const float* __restrict__ x,
    const float* __restrict__ y, const float* __restrict__ z)
{
    return make_float3(x[i], y[i], z[i]);
};

/// @brief Constructor for a `double3` that loads each compoenent from the `i`th
/// entry provided in the accompanying pointers in order to construct a vector
/// from quantities stored in SoA format
/// @param i index of vector to load
/// @param x pointer to x-components
/// @param y pointer to y-components
/// @param z pointer to z-components
/// @return a single `double3` constructed from three load operations
inline __host__ __device__ double3 dv3(const uint& i,
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z)
{
    return make_double3(x[i], y[i], z[i]);
};

inline __host__ __device__ void store_v3(const float3& vec, const uint& i,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z)
{
    x[i] = vec.x;
    y[i] = vec.y;
    z[i] = vec.z;
};

/// @brief Constructor for a `float3` that performs conversion from double to
/// float and loads each component from the `i`th entry provided in the
/// accompanying pointers in order to construct a vector from quantities stored
/// in SoA format
/// @param i index of vector to load
/// @param x pointer to x-components (double)
/// @param y pointer to y-components (double)
/// @param z pointer to z-components (double)
/// @return a single `float3` constructed from three load operations
inline __host__ __device__ float3 v3(const uint& i,
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z)
{
    return make_float3(static_cast<float>(x[i]), static_cast<float>(y[i]),
        static_cast<float>(z[i]));
};

// BINARY ARITHMATIC OPERATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
};

inline __host__ __device__ double3 operator-(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
};

inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
};

inline __host__ __device__ double3 operator+(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
};

inline __host__ __device__ float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline __host__ __device__ float2& operator+=(float2& a, const float2& b)
{
    a.x += b.x;
    a.y += b.y;
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
    return make_int3(static_cast<int>(floorf(a.x / b)),
        static_cast<int>(floorf(a.y / b)), static_cast<int>(floorf(a.z / b)));
};

/// divide a 3-vector by a float to obtain an integer 3-vector by rounding up
/// with `ceilf` and casting to integers in each component.
inline __host__ __device__ int3 ceil_div(const float3& a, const float& b)
{
    return make_int3(static_cast<int>(ceilf(a.x / b)),
        static_cast<int>(ceilf(a.y / b)), static_cast<int>(ceilf(a.z / b)));
};

/// @brief compute the dot product of two vectors
inline __host__ __device__ float dot(const float3& a, const float3& b)
{
    return fmaf(a.x, b.x, fmaf(a.y, b.y, (a.z * b.z)));
};

/// @brief compute the dot product of two vectors
inline __host__ __device__ double dot(const double3& a, const double3& b)
{
    return fma(a.x, b.x, fma(a.y, b.y, (a.z * b.z)));
};

/// @brief compute the cross product of two vectors
inline __host__ __device__ double3 cross(const double3& a, const double3& b)
{
    return make_double3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
};

/// @brief compute the cross product of two vectors
inline __host__ __device__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
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
/// @brief Element-wise multiplication of two vectors, i.e. Hadamard product ⊗
inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
};
/// @brief Element-wise multiplication of two vectors, i.e. Hadamard product ⊗
inline __host__ __device__ double3 operator*(const double3& a, const double3& b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
};

inline __host__ __device__ float2 operator*(const float2& a, const float& b)
{
    return make_float2(a.x * b, a.y * b);
};
inline __host__ __device__ float2 operator*(const float& a, const float2& b)
{
    return make_float2(b.x * a, b.y * a);
};

inline __host__ __device__ double3 operator*(const double3& a, const double& b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
};
inline __host__ __device__ double3 operator*(const double& a, const double3& b)
{
    return make_double3(b.x * a, b.y * a, b.z * a);
};

// UNARY ARITHMATIC OPERATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @brief Return the product of the components in `a`
/// @param a input `float3`
/// @return product of the components in `a`
inline __host__ __device__ float prod(const float3& a)
{
    return a.x * a.y * a.z;
};

/// @brief NOTE: this function might behave differently in `__device__` and
/// `__host__` code due to alternate implementations depending on intrinsics
/// availablility!
///
/// saturate a `float`, i.e. clamp it to a [0.f ; 1.f] range (inclusive)
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
/// saturate a `float2`, i.e. clamp each component to a [0.f ; 1.f] range
/// (inclusive)
inline __host__ __device__ float2 sat(const float2& a)
{
    return make_float2(sat(a.x), sat(a.y));
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
/// @brief NOTE: this function might behave differently in `__device__` and
/// `__host__` code due to alternate implementations depending on intrinsics
/// availablility!
///
/// take the 3D euclidean norm of a vector
inline __host__ __device__ double norm(const double3& a)
{
#ifdef __CUDA_ARCH__
    return norm3d(a.x, a.y, a.z);
#else
    return sqrt(dot(a, a));
#endif
};

/// @brief Compute the element-wise minimum
inline __host__ __device__ float3 min(const float3& a, const float3& b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

/// @brief Compute the element-wise maximum
inline __host__ __device__ float3 max(const float3& a, const float3& b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
/// @brief Compute the element-wise maximum
inline __host__ __device__ double3 max(const double3& a, const double3& b)
{
    return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

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

/// @brief Unary minus operation on `float3`
/// @param a vector to negate
/// @return a negated vector, with flipped sign in each component
inline __host__ __device__ float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
};

// BINARY PREDICATES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @brief the disjunction (OR) of element-wise application of <=
/// @return a boolean indicating if a <= b in ANY component or not
inline __host__ __device__ bool operator<=(const float3& a, const float3& b)
{
    return (a.x <= b.x) || (a.y <= b.y) || (a.z <= b.z);
};

/// @brief the disjunction (OR) of element-wise application of <=
/// @return a boolean indicating if a <= b in ANY component or not
inline __host__ __device__ bool operator<=(const float3& a, const float& b)
{
    return (a.x <= b) || (a.y <= b) || (a.z <= b);
};

/// @brief the disjunction (OR) of element-wise application of >=
/// @return a boolean indicating if a >= b in ANY component or not
inline __host__ __device__ bool operator>=(const float3& a, const float3& b)
{
    return (a.x >= b.x) || (a.y >= b.y) || (a.z >= b.z);
};

/// @brief the disjunction (OR) of element-wise application of >=
/// @return a boolean indicating if a >= b in ANY component or not
inline __host__ __device__ bool operator>=(const float3& a, const float& b)
{
    return (a.x >= b) || (a.y >= b) || (a.z >= b);
};

#endif // VECTOR_CUH_