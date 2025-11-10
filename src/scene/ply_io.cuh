#ifndef SCENE_PLYIO_CPP_
#define SCENE_PLYIO_CPP_
#include <filesystem>
#include "buffer.cuh"

/// @brief Writes the positions in the given buffers to the specified output
/// file using the Polygon File Format (PLY) as described in:
/// https://en.wikipedia.org/wiki/PLY_(file_format)
/// @param path file to write to
/// @param xs x-components of the positions to save
/// @param ys y-components of the positions to save
/// @param zs z-components of the positions to save
void save_to_ply(const std::filesystem::path& path,
    const DeviceBuffer<float>& xs, const DeviceBuffer<float>& ys,
    const DeviceBuffer<float>& zs);

#endif // SCENE_PLYIO_CPP_