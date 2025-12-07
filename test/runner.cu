#include "gui.cuh"
#include "particles.cuh"
#include "scene/scene.cuh"
#include "kernels.cuh"
#include "solvers/PCISPH.cuh"
#include "datastructure/uniformgrid.cuh"
#include "utils/vector.cuh"
#include "scene/loader.h"
#include "scene/sample_boundary.cuh"
#include "scene/ply_io.cuh"
#include "timestep/cfl.cuh"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

TEST_CASE("Generate Docs Video")
{
#ifdef BENCH
    // output video settings
    const float total_time { 3.0 };
    const float frame_time { 1. / 60. };
    float last_frame { -total_time };
    uint current_frame { 0 };

    // initial camera settings
    GUI gui(1920, 1080, false, 60., false); // do not maximize window
    gui.phi = -M_PI / 2.0f;
    gui.theta = M_PI / 2.0f * 0.75;
    gui.camera_offset = glm::vec3(0., -.2, 0.);
    gui.camera_radius_init = 4.f;
    gui.show_boundary = true;
    gui.stopped = false;
    gui.colour_scale = 10.;
    gui.colour_map_selector = 0;
    gui.attribute_visualized = 1;

    Particles state(&gui, 1.);
    DeviceBuffer<float> tmp1(1);
    DeviceBuffer<float> tmp2(1);
    DeviceBuffer<float> tmp3(1);
    DeviceBuffer<float> tmp4(1);
    Scene scene { Scene::from_obj(
        "scenes/dragonbox.obj", 1000000, 1., state, tmp1, 3., 1.) };
    gui.set_boundary_to_render(&scene.bdy);
    const B3 W(2.f * scene.h);
    const uint N { scene.N };
    double time { 0. };
    auto solver { PCISPH<B3, Resort::yes>(
        W, N, 0.001f, scene.h, scene.ρ₀, tmp1, tmp2, tmp3, tmp4) };
    while (gui.update_or_exit(state, scene.h, &tmp1)) {
        const float dt { 0.0005 };
        const auto grid { scene.get_grid(state, tmp1) };
        solver.step(state, grid, scene.bdy, dt);
        scene.hard_enforce_bounds(state);
        time += dt;
        std::cout << time << " / " << total_time << std::endl;
        // regularly use screenshot function
        if (time > last_frame + frame_time) {
            std::cout << "RENDERING" << std::endl;
            last_frame = time;
            gui.phi = -M_PI / 2.0f + 2 * M_PI * time / total_time;
            gui.update_view();
            gui.screenshot(std::format("out/{:05d}.bmp", current_frame));
            current_frame += 1;
        }
        // exit if total time has been exceeded
        if (time >= total_time) {
            gui.exit();
        }
    }
    return;
#endif
}