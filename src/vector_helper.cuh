#ifndef VECTOR_CUH_
#define VECTOR_CUH_

/// A shorthand constructor for `float3`, alias for `make_float3`
inline __host__ __device__ float3 v3(const float &x, const float &y, const float &z)
{
    return make_float3(x, y, z);
};
inline __host__ __device__ float3 v3(const float &x)
{
    return make_float3(x, x, x);
};

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
};

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
};

/// @brief Divide a 3-vector by a float to obtain an integer 3-vector by rounding down with `floorf` and casting to integers in each component.
inline __host__ __device__ int3 operator/(const float3 &a, const float &b)
{
    return make_int3((int)floorf(a.x / b), (int)floorf(a.y / b), (int)floorf(a.z / b));
};

#endif // VECTOR_CUH_