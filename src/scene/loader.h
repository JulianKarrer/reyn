#ifndef SCENE_LOADER_CUH_
#define SCENE_LOADER_CUH_

#include <filesystem>
#include <cuda_runtime.h>
#include <vector>

/// @brief Structure containing x,y,z components of vertices as `double`
/// respectively and `uint3` of indices into those buffers representing the
/// faces of a triangular surface mesh
struct Mesh {
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<uint3> faces;

    size_t vertex_count() const { return xs.size(); }
    size_t face_count() const { return faces.size(); }
};

/// @brief Load a `Mesh` from a Wavefront OBJ file using `tinyobjloader`
/// @param path path to the OBJ file
/// @return a `Mesh` instance
Mesh load_mesh_from_obj(const std::filesystem::path& path);

#endif // SCENE_LOADER_CUH_