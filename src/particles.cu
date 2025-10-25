#include "particles.cuh"
#include "gui.h"
#include <thrust/gather.h>

/// @brief When using a GUI to manage positions through sharing a buffer with
/// the OpenGL VBO used for visualization and mapping and unmapping the buffer
/// for use with CUDA, this method must be called by the GUI in every time step
/// to ensure the correct positions buffer is used. Apart from the internals of
/// the GUI, this function should not be used, hence it outputs an error message
/// and causes a crash if not used on a `Particles` instance initialized using a
/// `GUI` instance.
/// @param x the updated pointer to the buffer of particle positions
void Particles::set_x(float3* x)
{
    if (!gui) {
        throw std::runtime_error(
            "Attempted to call set_x on Particles instance that is not managed "
            "by a GUI. Changing where the pointer to positions x points to "
            "does not make sense if the pointer is not managed by a mapping a "
            "OpenGL VBO but instead created by the Particles constructor.");
    } else {
        this->x.update_raw_ptr(x);
    }
};

Particles::Particles(GUI* _gui, float _rho_0)
    : rho_0(_rho_0)
    , gui(_gui)
    , v(_gui->N)
    , m(_gui->N)
    , x(_gui)
    , tmp3(1)
    , tmp(1) {};

Particles::Particles(const int N, float _rho_0)
    : rho_0(_rho_0)
    , v(N)
    , m(N)
    , x(N)
    , tmp3(1)
    , tmp(1) {};

void Particles::resize_uninit(uint N)
{
    x.resize(N);
    v.resize(N);
    m.resize(N);
}

__global__ void _reorder_f3(const uint* map, float3* __restrict__ src,
    float3* __restrict__ dst, const uint N)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    dst[i] = src[map[i]];
}

void Particles::gather(const DeviceBuffer<uint>& sorted)
{
    // resize the temporary buffers for resorting to fit all particles
    if (tmp3.size() != sorted.size())
        tmp3.resize(sorted.size());
    if (tmp.size() != sorted.size())
        tmp.resize(sorted.size());

    // gather velocities
    thrust::gather(sorted.get().begin(), sorted.get().end(), v.get().begin(),
        tmp3.get().begin());
    tmp3.get().swap(v.get());

    // gather masses
    thrust::gather(sorted.get().begin(), sorted.get().end(), m.get().begin(),
        tmp.get().begin());
    tmp.get().swap(m.get());

    // gather positions
    if (gui) {
        uint N { (uint)x.size() };
        _reorder_f3<<<BLOCKS(N), BLOCK_SIZE>>>(
            sorted.ptr(), x.ptr(), tmp3.ptr(), N);
        cudaMemcpy(
            x.ptr(), tmp3.ptr(), N * sizeof(float3), cudaMemcpyDeviceToDevice);
    } else {
        thrust::gather(sorted.get().begin(), sorted.get().end(),
            x.get().begin(), tmp3.get().begin());
        tmp3.get().swap(x.get());
    }
}
