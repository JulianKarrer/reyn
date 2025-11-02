#include "gui.cuh"
#include "particles.cuh"
#include "scene.cuh"
#include "kernels.cuh"
#include "solvers/SESPH.cuh"
#include "datastructure/uniformgrid.cuh"
#include "vector_helper.cuh"

int main()
{
    try {
        // Setup in required order
        // GUI -> Particles -> Scene -> everything else
        GUI gui(1280, 720, false);
        std::cout << "gui initialized" << std::endl;
        Particles state(&gui, 1.);
        // Particles state(10000, 1.);
        std::cout << "state initialized" << std::endl;

        const float dt { 0.0008 };
        const Scene scene(1000000, v3(-1.), v3(0.), v3(-1), v3(1.), 1., state);
        std::cout << "scene initialized, h=" << scene.h << std::endl;
        const B3 W(2.f * scene.h);
        const uint N { scene.N };

        double time { 0. };
        UniformGridBuilder uniform_grid(
            scene.bound_min, scene.bound_max, 2. * scene.h);
        std::cout << "grid initialized" << std::endl;
        // single tmp buffer shared for every operation
        DeviceBuffer<float> tmp(N);
        SESPH<B3, Resort::yes> solver(W, N, 0.005f, scene.h);
        std::cout << "solver initialized" << std::endl;

        // MAIN LOOP
        std::cout << "fully initialized" << std::endl;
        while (gui.update_or_exit(state, scene)) {
            // update the acceleration datastructure
            const auto grid { uniform_grid.construct_and_reorder(
                2. * scene.h, tmp, state) };

            // then invoke the fluid solver
            solver.step(state, grid, dt);

            // enforce boundary conditions by clamping since the grid relies
            // on scene bounds being strict for memory safety
            scene.hard_enforce_bounds(state);

            time += dt;
        }

    } catch (std::exception const& e) {
        // print any errors thrown to stderr and terminate with error
        std::cerr << e.what() << std::endl;
    } catch (...) {
        // catch-all: detect non-standard exceptions and terminate with error
        std::cerr << "An unspecified exception has occured. Terminating."
                  << std::endl;
    }

    return 0;
}
