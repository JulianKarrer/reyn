#include "scene/loader.h"

#include <format>
#include <iostream>

#define TINYOBJLOADER_USE_DOUBLE
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using namespace tinyobj;

Mesh load_mesh_from_obj(const std::filesystem::path& path)
{
    const std::string pathstr { path.string() };
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    if (!tinyobj::LoadObj(
            &attrib, &shapes, &materials, &err, pathstr.c_str())) {
        throw std::runtime_error(std::format(
            "Error occured in mesh loading process via tinyobjloader: {}",
            err));
    };

    // compute number of vertices in attribute buffer
    const size_t vertex_count { attrib.vertices.size() / 3 };

    // iterate all shapes in file to find faces
    std::vector<uint3> faces;
    for (const auto& shape : shapes) {
        // skip shapes containing "fluid" or "bounds" in the name
        if (shape.name.find("fluid") != std::string::npos
            || shape.name.find("bounds") != std::string::npos)
            continue;

        // for each face in the respective mesh
        const mesh_t mesh = shape.mesh;
        for (uint i { 0 }; i < mesh.num_face_vertices.size(); ++i) {
            // if any of the faces are not triangulated, abort
            if (mesh.num_face_vertices[i] != 3) {
                throw std::runtime_error(std::format(
                    "Non-triangulated face encountered in loading of mesh {}",
                    pathstr));
            }
            // otherwise, we can assume that there are three indices per face
            const int idx { mesh.indices[3 * i + 0].vertex_index };
            const int idy { mesh.indices[3 * i + 1].vertex_index };
            const int idz { mesh.indices[3 * i + 2].vertex_index };
            // if any of the indices exceeds the bounds of a uint, abort
            if (idx < 0 || idx >= vertex_count || idy < 0 || idy >= vertex_count
                || idz < 0 || idz >= vertex_count) {
                throw std::runtime_error(std::format(
                    "Invalid vertex id encountered in face {} of the mesh {}",
                    i, pathstr));
            }
            // add the face consisting of three indices of vertices to the
            // buffer
            faces.push_back(make_uint3(idx, idy, idz));
        }
    }

    // allocate vectors for quantitites in mesh
    if (attrib.vertices.size() % 3 != 0) {
        throw std::runtime_error(std::format(
            "Vertex count not divisible by 3 encountered in {}", pathstr));
    }
    std::vector<double> xs(vertex_count);
    std::vector<double> ys(vertex_count);
    std::vector<double> zs(vertex_count);

    for (uint i { 0 }; i < vertex_count; ++i) {
        xs[i] = attrib.vertices[i * 3 + 0];
        ys[i] = attrib.vertices[i * 3 + 1];
        zs[i] = attrib.vertices[i * 3 + 2];
    }

    return Mesh { xs, ys, zs, faces };
}