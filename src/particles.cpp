#include "particles.h"
#include "gui.h"

/// @brief When using a GUI to manage positions through sharing a buffer with the OpenGL VBO used for visualization and mapping and unmapping the buffer for use with CUDA, this method must be called by the GUI in every time step to ensure the correct positions buffer is used. Apart from the internals of the GUI, this function should not be used, hence it outputs an error message and causes a crash if not used on a `Particles` instance initialized using a `GUI` instance.
/// @param x the updated pointer to the buffer of particle positions
void Particles::set_x(float3 *x)
{
    if (!gui)
    {
        throw std::runtime_error("Attempted to call set_x on Particles instance that is not managed by a GUI. Changing where the pointer to positions x points to does not make sense if the pointer is not managed by a mapping a OpenGL VBO but instead created by cudaMalloc in the Particles constructor.");
    }
    else
    {
        this->x = x;
    }
};

Particles::Particles(GUI *_gui, float _h, float _rho_0)
    : h(_h), rho_0(_rho_0), gui(_gui)
{
    CUDA_CHECK(cudaMalloc((void **)&v, sizeof(float3) * gui->N));
    CUDA_CHECK(cudaMalloc((void **)&m, sizeof(float) * gui->N));
};

Particles::Particles(const int N, float _h, float _rho_0)
    : h(_h), rho_0(_rho_0)
{
    CUDA_CHECK(cudaMalloc((void **)&x, sizeof(float3) * N));
    CUDA_CHECK(cudaMalloc((void **)&v, sizeof(float3) * N));
    CUDA_CHECK(cudaMalloc((void **)&m, sizeof(float) * N));
};

void Particles::resize_uninit(uint N)
{
    if (gui)
    {
        // resize position buffer owned by OpenGL context
        this->x = gui->resize_mapped_buffer(N);
    }
    else
    {
        // resize position buffer owned by this
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaMalloc((void **)&x, sizeof(float3) * N));
    };
    // in either case, resize mass and velocity arrays
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaMalloc((void **)&v, sizeof(float3) * N));
    CUDA_CHECK(cudaMalloc((void **)&m, sizeof(float) * N));
}

Particles::~Particles()
{
    if (!gui)
    {
        CUDA_CHECK(cudaFree(x));
    }
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(m));
}
