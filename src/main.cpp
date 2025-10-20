#include "gui.h"
#include "particles.h"
#include "scene.cuh"
#include "kernels.cuh"
#include "solvers/SESPH.cuh"

void step(Particles &state, int N)
{
    const float dt{0.001};
    static double time{0.};
    static const B3 W(0.1);
    static const Scene scene(10000, v3(-.5), v3(.5), v3(-.5, -1., -.5), v3(.5, 1., .5), 1000., state);
    static SESPH<B3> solver(W, scene.N, 0.001f, scene.h);

    solver.compute_accelerations(state, dt);
    scene.hard_enforce_bounds(state);

    time += dt;
}

void init(Particles &state, int N)
{
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