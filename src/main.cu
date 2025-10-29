#include "gui.cuh"
#include "particles.cuh"
#include "scene.cuh"
#include "kernels.cuh"
#include "solvers/SESPH.cuh"
#include "datastructure/uniformgrid.cuh"
#include "vector_helper.cuh"

void step(Particles& state, const int N, const Scene scene)
{
    const float dt { 0.0008 };
    static double time { 0. };
    static const B3 W(2.f * scene.h);
    static UniformGridBuilder uniform_grid(
        scene.bound_min, scene.bound_max, 2. * scene.h);
    static SESPH<B3, Resort::yes> solver(W, N, 0.005f, scene.h);

    // update the acceleration datastructure
    const auto grid { uniform_grid.construct_and_reorder(state) };
    // then invoke the fluid solver
    solver.compute_accelerations(state, grid, dt);

    time += dt;
}

void init(Particles& state, const int N, const Scene scene) { }

int main()
{
    try {
        GUI gui(1280, 720, false);
        Particles state(&gui, 1.);

        state.set_x(gui.map_buffer());
        const Scene scene(1000000, v3(-1.), v3(0.), v3(-1), v3(1.), 1., state);
        gui.unmap_buffer();

        gui.run(&step, &init, state, scene);
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
