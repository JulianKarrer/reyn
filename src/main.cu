#include "gui.cuh"
#include "particles.cuh"
#include "scene/scene.cuh"
#include "kernels.cuh"
#include "solvers/PCISPH.cuh"
#include "solvers/IISPH.cuh"
#include "datastructure/uniformgrid.cuh"
#include "utils/vector.cuh"
#include "scene/loader.h"
#include "scene/sample_boundary.cuh"
#include "scene/ply_io.cuh"
#include "timestep/cfl.cuh"
#include "log.h"
#include <signal.h>

// instead of abnormally exiting, make CTRL+C (SIGINT) set a flag to end the
// simulation and exit gracefully
bool interrupted { false };
void sigint_handler(int s) { interrupted = true; }

/// @brief Custom literal operator to make specifying large numbers of particles
/// more convenient
/// @param x number to multiply by a million
/// @return a million times the prefix
unsigned operator""_million(const unsigned long long x) { return 1000000 * x; };

int main()
{
    // register the cusomt SIGINT handler
    signal(SIGINT, sigint_handler);
    try {
        // setup in required order
        // GUI -> Particles -> Scene -> everything else
        GUI gui(1280, 720, false);
        Particles state(&gui, 1.);

        // tmp buffers to share across operations
        DeviceBuffer<float> tmp1(1);
        DeviceBuffer<float> tmp2(1);
        DeviceBuffer<float> tmp3(1);
        DeviceBuffer<float> tmp4(1);
        DeviceBuffer<float> tmp5(1);
        DeviceBuffer<float> tmp6(1);
        DeviceBuffer<float> tmp7(1);

        Scene scene { Scene::from_obj(
            "scenes/dragonbox.obj", 90_million, 1., state, tmp1, 3., 1.) };
        gui.set_boundary_to_render(&scene.bdy);

        // initialize kernel function and solver
        const B3 W(2.f * scene.h);
        auto solver { IISPH<B3, Resort::yes>(W, scene.N, 0.001f, scene.h,
            scene.ρ₀, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7) };

        // MAIN LOOP
        Log::Success("Fully initialized, starting main loop.");
        double time { 0. };
        uint iters { 0 };
        float dt { 0.0005 };
        while (gui.update_or_exit(state, scene.h, dt, iters, &tmp1)) {
            if (interrupted) // catch CTRL+C on POSIX
                gui.exit();

            dt = simple_dt_controller(
                iters, dt, 1.0, scene.h, state, v3(0., -9.81, 0.), 5);

            // get an updated acceleration datastructure
            const auto grid { scene.get_grid(state, tmp1) };

            // then invoke the fluid solver
            iters = solver.step(state, grid, scene.bdy, dt);

            // enforce boundary conditions by clamping since the grid relies
            // on scene bounds being strict for memory safety
            scene.hard_enforce_bounds(state);

            time += dt;
        }

    } catch (std::exception const& e) {
        // print any errors thrown to stderr and terminate with error
        Log::Error("An Error has occured and was caught:\n{}", e.what());
    } catch (...) {
        // catch-all: detect non-standard exceptions and terminate with error
        Log::Error("An unspecified exception has occured. Terminating.");
    }
    Log::Success("Exiting Application");
    return 0;
}
