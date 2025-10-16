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
    try
    {
        GUI gui(1280, 720, true);
        Particles state(&gui, 0.1, 1.);
        gui.run(&step, &init, state);
    }
    catch (std::exception const &e)
    {
        // print any errors thrown to stderr and terminate with error
        std::cerr << e.what() << std::endl;
    }
    catch (...)
    {
        // catch-all: detect non-standard exceptions and terminate with error
        std::cerr << "An unspecified exception has occured. Terminating." << std::endl;
    }

    return 0;
}