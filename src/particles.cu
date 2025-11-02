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

void Particles::reorder(
    const DeviceBuffer<uint>& sorted, DeviceBuffer<float>& tmp)
{
    // resize the temporary buffers for resorting to fit all particles
    if (tmp.size() <= sorted.size()) {
        tmp.resize(sorted.size());
    }

    // reorder velocities
    vx.reorder(sorted, tmp);
    vy.reorder(sorted, tmp);
    vz.reorder(sorted, tmp);

    // reorder masses
    m.reorder(sorted, tmp);

    // reorder positions
    xx.reorder(sorted, tmp);
    xy.reorder(sorted, tmp);
    xz.reorder(sorted, tmp);
}
