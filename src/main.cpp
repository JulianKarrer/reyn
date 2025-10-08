#include "gui.h"
#include "particles.h"
#include "kernel.cuh"

void step(Particles &state, int N)
{
    static double time{0.};

    time += 0.0001;
    // launch CUDA kernel
    launch_kernel(state.x, time, N);
}

int main()
{
    int N{100};
    auto on_failure{
        []()
        {
        std::cout << "Error initializing GUI, exiting.\n"
                  << std::endl;
        exit(1); }};
    GUI gui(N, 1280, 720, on_failure, true);
    Particles state(gui, N, 0.1, 1.);

    gui.run(&step, state);
    return 0;
}