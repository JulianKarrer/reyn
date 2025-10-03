#include "gui.h"
#include "kernel.cuh"

int main()
{
    int N{100};

    auto on_failure{
        []()
        {
        std::cout << "Error initializing GUI, exiting.\n"
                  << std::endl;
        exit(1); }};

    GUI gui{GUI(N, 1280, 720, on_failure, true)};

    while (true)
    {
        if (gui.exit_requested())
            break;

        double time{glfwGetTime()};
        float4 *d_vertices{gui.get_buffer()};

        // launch CUDA kernel
        launch_kernel(d_vertices, time, N);

        gui.show_updated();
    }
    return 0;
}