#ifndef SCENE_SAMPLE_CUH_
#define SCENE_SAMPLE_CUH_

#include "scene/loader.h"
#include "buffer.cuh"
#include <thrust/device_vector.h>

struct BoundarySamples {
    DeviceBuffer<float> xs;
    DeviceBuffer<float> ys;
    DeviceBuffer<float> zs;
};

BoundarySamples sample_mesh(const Mesh mesh, const float h);

#endif // SCENE_SAMPLE_CUH_