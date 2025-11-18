#include "gui.cuh"
#include "particles.cuh"
#include "scene/scene.cuh"
#include "kernels.cuh"
#include "solvers/SESPH.cuh"
#include "datastructure/uniformgrid.cuh"
#include "vector_helper.cuh"
#include "scene/loader.h"
#include "scene/sample_boundary.cuh"
#include "scene/ply_io.cuh"
#include "timestep/cfl.cuh"

int main()
{
    try {
        // Setup in required order
        // GUI -> Particles -> Scene -> everything else
        GUI gui(1280, 720, false);
        std::cout << "gui initialized" << std::endl;
        Particles state(&gui, 1.);
        std::cout << "state initialized" << std::endl;

        // tmp buffers to share across operations
        DeviceBuffer<float> tmp1(1);
        DeviceBuffer<float> tmp2(1);
        DeviceBuffer<float> tmp3(1);
        DeviceBuffer<float> tmp4(1);

        Scene scene(
            "scenes/cube.obj", 1000000, v3(-1.), v3(0.), 1., state, tmp1, 3.);
        gui.set_boundary_to_render(&scene.bdy);
        std::cout << "scene initialized, h=" << scene.h << std::endl;

        const B3 W(2.f * scene.h);
        const uint N { scene.N };
        double time { 0. };

        SESPH<B3, Resort::yes> solver(
            W, N, 0.001f, scene.h, scene.rho_0, tmp1, tmp2, tmp3, tmp4);
        std::cout << "solver initialized" << std::endl;

        std::cout << "fluid avg mass " << state.m.sum() / (float)state.m.size()
                  << std::endl;

        // MAIN LOOP
        std::cout << "fully initialized" << std::endl;
        while (gui.update_or_exit(state, scene.h, &tmp1)) {
            // const float dt { cfl_time_step(
            //     0.1, scene.h, state, v3(0., -9.81, 0.)) };
            const float dt { 0.0001 };

            // get an updated acceleration datastructure
            const auto grid { scene.get_grid(state, tmp1) };
            // const auto grid { grid_builder.construct_and_reorder(
            //     2.f * scene.h, tmp1, state) };

            // then invoke the fluid solver
            solver.step(state, grid, scene.bdy, dt);

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
