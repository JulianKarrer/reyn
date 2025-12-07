#ifndef GEOMETRY_HELPER_CUH_
#define GEOMETRY_HELPER_CUH_

#include "vector.cuh"

/// @brief Compute the closest point to `p` on a triangle formed by `a`, `b` and
/// `c`
///
/// Algorithm from "Real-Time Collision Detection" by Christer Ericson
/// @param p query point, the closest point on the triangle to which is sought
/// @param a first vertex of the triangle
/// @param b second vertex of the triangle
/// @param c third vertex of the triangle
/// @return the closest point to `p` in the triangle formed by `a`, `b` and `c`
__device__ static inline float3 closest_point_on_triangle(
    const float3 p, const float3 a, const float3 b, const float3 c)
{
    // this is taken basically verbatim from Christer Ericson, including the
    // helpful comments
    const float3 ab { b - a };
    const float3 ac { c - a };
    const float3 ap { p - a };
    // vertex region outside a
    const float d1 { dot(ab, ap) };
    const float d2 { dot(ac, ap) };
    if (d1 <= 0.f && d2 <= 0.f)
        return a; // barycentric (1,0,0)
    // vertex region outside b
    const float3 bp { p - b };
    const float d3 { dot(ab, bp) };
    const float d4 { dot(ac, bp) };
    if (d3 >= 0.0f && d4 <= d3)
        return b; // barycentric (0,1,0)
    // if in edge region ab, project
    const float vc { d1 * d4 - d3 * d2 };
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return a + v * ab; // barycentric (1-v,v,0)
    }
    // vertex region outside c
    const float3 cp { p - c };
    const float d5 { dot(ab, cp) };
    const float d6 { dot(ac, cp) };
    if (d6 >= 0.0f && d5 <= d6)
        return c; // barycentric (0,0,1)
    // if in edge region ac, project
    const float vb { d5 * d2 - d1 * d6 };
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = { d2 / (d2 - d6) };
        return a + w * ac; // barycentric (1-w,0,w)
    }
    // if in edge region bc, project
    const float va { d3 * d6 - d5 * d4 };
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w { (d4 - d3) / ((d4 - d3) + (d5 - d6)) };
        return b + w * (c - b); // barycentric (0,1-w,w)
    }
    // projected inside face region, use uvw to compute
    const float denom { 1.0f / (va + vb + vc) };
    const float v { vb * denom };
    const float w { vc * denom };
    return a + ab * v + ac * w;
};

/// @brief Compute the closest point to `p` on a triangle formed by `a`, `b` and
/// `c`
///
/// Algorithm from "Real-Time Collision Detection" by Christer Ericson
/// @param p query point, the closest point on the triangle to which is sought
/// @param a first vertex of the triangle
/// @param b second vertex of the triangle
/// @param c third vertex of the triangle
/// @return the closest point to `p` in the triangle formed by `a`, `b` and `c`

///@brief Compute the closest point to `p` on a triangle formed by the specified
/// face, indexing into the given vertex coordinate buffers
///
///@param p query point (the closest point on the triangle to which shall be
/// computed)
///@param face vector of indices into the vertex coordinate buffers representing
/// a single triangle.
///@param vxs vertex buffer of x-coordinates
///@param vys vertex buffer of y-coordinates
///@param vzs vertex buffer of z-coordinates
///@return closest point to `p` on the triangle indexed by `face` in
///`vxs,vys,vzs`
__device__ static inline float3 closest_point_on_triangle(const float3 p,
    const uint3 face, const double* __restrict__ vxs,
    const double* __restrict__ vys, const double* __restrict__ vzs)
{
    const float3 a { v3(face.x, vxs, vys, vzs) };
    const float3 b { v3(face.y, vxs, vys, vzs) };
    const float3 c { v3(face.z, vxs, vys, vzs) };
    // this is taken basically verbatim from Christer Ericson, including the
    // helpful comments
    const float3 ab { b - a };
    const float3 ac { c - a };
    const float3 ap { p - a };
    // vertex region outside a
    const float d1 { dot(ab, ap) };
    const float d2 { dot(ac, ap) };
    if (d1 <= 0.f && d2 <= 0.f)
        return a; // barycentric (1,0,0)
    // vertex region outside b
    const float3 bp { p - b };
    const float d3 { dot(ab, bp) };
    const float d4 { dot(ac, bp) };
    if (d3 >= 0.0f && d4 <= d3)
        return b; // barycentric (0,1,0)
    // if in edge region ab, project
    const float vc { d1 * d4 - d3 * d2 };
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return a + v * ab; // barycentric (1-v,v,0)
    }
    // vertex region outside c
    const float3 cp { p - c };
    const float d5 { dot(ab, cp) };
    const float d6 { dot(ac, cp) };
    if (d6 >= 0.0f && d5 <= d6)
        return c; // barycentric (0,0,1)
    // if in edge region ac, project
    const float vb { d5 * d2 - d1 * d6 };
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = { d2 / (d2 - d6) };
        return a + w * ac; // barycentric (1-w,0,w)
    }
    // if in edge region bc, project
    const float va { d3 * d6 - d5 * d4 };
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w { (d4 - d3) / ((d4 - d3) + (d5 - d6)) };
        return b + w * (c - b); // barycentric (0,1-w,w)
    }
    // projected inside face region, use uvw to compute
    const float denom { 1.0f / (va + vb + vc) };
    const float v { vb * denom };
    const float w { vc * denom };
    return a + ab * v + ac * w;
};

#endif // GEOMETRY_HELPER_CUH_
