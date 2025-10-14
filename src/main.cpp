#include "gui.h"
#include "particles.h"
#include "scene.cuh"

void step(Particles &state, int N)
{
    static double time{0.};

    time += 0.0001;
    // launch CUDA kernels:
    // TODO
}

void init(Particles &state, int N)
{
    Scene scene(1000, v3(-.5), v3(.5), v3(-1.), v3(1.), 1000., state);
}

int main()
{
    int N{1};
    auto on_failure{
        []()
        {
        std::cout << "Error initializing GUI, exiting.\n"
                  << std::endl;
        exit(1); }};
    GUI gui(N, 1280, 720, on_failure, true);
    Particles state(&gui, 0.1, 1.);
    std::cout << "now running:" << std::endl;

    gui.run(&step, &init, state);
    return 0;
}