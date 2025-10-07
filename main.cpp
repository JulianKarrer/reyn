#include "gui.h"
#include "kernel.cuh"

void simulation(float3 *x, int N)
{
    static double time{0.};
    time += 0.0001;
    // launch CUDA kernel
    launch_kernel(x, time, N);
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
    gui.run(&simulation);
    return 0;
}