#include "particles.cuh"
#include "gui.cuh"
#include <thrust/gather.h>

Particles::Particles(GUI* _gui, float _rho_0)
    : rho_0(_rho_0)
    , gui(_gui)
    , vx(_gui->N)
    , vy(_gui->N)
    , vz(_gui->N)
    , m(_gui->N)
    , xx(_gui)
    , xy(_gui)
    , xz(_gui)
{
    // now that a state exists, tell the GUI to call `set_x` with the
    // appropriate pointer to a CUDA mapped position VBO
    _gui->initialize_buffers(*this);
};

Particles::Particles(const int N, float _rho_0)
    : rho_0(_rho_0)
    , vx(N)
    , vy(N)
    , vz(N)
    , m(N)
    , xx(N)
    , xy(N)
    , xz(N) {};

void Particles::resize_uninit(uint N)
{
    if (gui) {
        gui->resize_mapped_buffers(N, *this);
    } else {
        xx.resize(N);
        xy.resize(N);
        xz.resize(N);
    }
    vx.resize(N);
    vy.resize(N);
    vz.resize(N);
    m.resize(N);
}

__global__ void _reorder(const uint* map, float* __restrict__ src,
    float* __restrict__ dst, const uint N)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    dst[i] = src[map[i]];
}

void Particles::gather(
    const DeviceBuffer<uint>& sorted, DeviceBuffer<float>& tmp)
{
    // resize the temporary buffers for resorting to fit all particles
    if (tmp.size() <= sorted.size()) {
        tmp.resize(sorted.size());
    }

    // gather velocities
    for (auto v : { &vx, &vy, &vz }) {
        thrust::gather(sorted.get().begin(), sorted.get().end(),
            v->get().begin(), tmp.get().begin());
        tmp.get().swap(v->get());
    }

    // gather masses
    thrust::gather(sorted.get().begin(), sorted.get().end(), m.get().begin(),
        tmp.get().begin());
    tmp.get().swap(m.get());

    // gather positions
    for (auto x : { &xx, &xy, &xz }) {
        if (gui) {
            uint N { (uint)xx.size() };
            // if the buffer is externally managed, manually gather
            _reorder<<<BLOCKS(N), BLOCK_SIZE>>>(
                sorted.ptr(), x->ptr(), tmp.ptr(), N);
            // in this case, a copy is required because thrust device_buffer and
            // a manually managed float* cannot be trivially pointer-swapped
            cudaMemcpy(x->ptr(), tmp.ptr(), N * sizeof(float),
                cudaMemcpyDeviceToDevice);
        } else {
            // otherwise thrust may be used
            thrust::gather(sorted.get().begin(), sorted.get().end(),
                x->get().begin(), tmp.get().begin());
            tmp.get().swap(x->get());
        }
    }
}
