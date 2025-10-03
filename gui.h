#ifndef GUI_H_
#define GUI_H_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <functional>
#include "common.h"
#include <chrono>
#include <algorithm>

/// @brief Object managing a graphical user interface.
/// Uses GLFW for cross-platform windowing and sets up
/// OpenGL <-> CUDA interop for displaying the contents
/// of a buffer of positions managed by CUDA as a point
/// cloud.
class GUI
{
public:
    /// @brief Construct a GUI window using GLFW and OpenGL for display of the particles.
    /// @param N initial number of particles
    /// @param init_w initial window width
    /// @param init_h initial window height
    /// @param on_failure a parameter-less void callback to execute when initialization of the GUI fails, such that e.g. MPI can be finalized before exiting the application
    /// @param enable_vsync whether or not the GUI framerate should be limited by the V-synced refresh rate
    GUI(
        const int N,
        int init_w,
        int init_h,
        std::function<void()> on_failure,
        bool enable_vsync = false);
    ~GUI();

    /// @brief Query whether the GUI has requested the application to close
    /// @return `bool` indicating whether a close has been requested
    bool exit_requested();

    float4 *get_buffer();

    void show_updated();

    int window_width, window_height;

private:
    /// current number of particles
    int N;

    // internals for GUI
    // whether the user is currently pressing the cursor with left click
    bool exit_pressed{false};
    bool pressing{false};

    // GLFW resources
    GLFWwindow *window;

    // ImGui resources
    ImGuiIO *io;
    ImFont *font;

    // OpenGL resources
    GLuint shader_program, vao, vbo;

    // CUDA resources
    cudaGraphicsResource *cuda_vbo_resource = nullptr;

    // internal functions for setup and events
    /// Compile the fragment and vertex shaders required for visalization by OpenGL
    GLuint compile_shader();
    /// Process the input events provided by GLFW
    void glfw_process_input();
    ///  Update and manage the ImGui contents
    void imgui_draw();
};

#endif // GUI_H_