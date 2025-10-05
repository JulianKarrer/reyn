#ifndef GUI_H_
#define GUI_H_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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

    /// @brief Update thje projection matrix representing the current camera frustum, which depends on the fov, aspect ratio, near and far planes.
    ///
    /// Updates the private internal `proj` variable and should be called initially, as well as when window width or height may have changed.
    void update_proj();

    float3 *get_buffer();

    void show_updated();

    /// Current width of the GUI window
    int window_width;
    /// Current height of the GUI window
    int window_height;
    /// The multiplier applied to the initial radius based on scroll position
    float radius_scroll_factor{1.f};
    /// @brief The scroll speed as the fraction of `radius_scroll_factor` that is added or taken away per line scrolled
    float scroll_speed{0.05f};

private:
    /// current number of particles
    int N;

    // internals for GUI
    bool exit_pressed{false};
    /// @brief whether the user is currently pressing the cursor with left click
    bool dragging{false};
    /// @brief x-position in normalized coordinates [0;1]^2 of where the current mouse dragging operation started
    float drag_start_x{0.};
    /// @brief y-position in normalized coordinates [0;1]^2 of where the current mouse dragging operation started
    float drag_start_y{0.};
    /// @brief The initial or base radius of the camera around the `camera_target` position in spherical cooridnates, before scrolling
    float radius_init{10.f};

    // GLFW resources
    GLFWwindow *window;

    // ImGui resources
    ImGuiIO *io;
    ImFont *font;

    // OpenGL resources
    GLuint shader_program, vao, vbo;
    float fov{45.0};
    glm::mat4 proj;
    /// @brief Get a view matrix, representing the orientation of the camera in world space, depending on camera positions and where the camera is pointed
    /// @return view matrix for use in vertex shader as a uniform
    glm::mat4 get_view();
    /// @brief The φ-angle of the camera in spherical coordinates
    float phi{M_PI / 2.0f};
    float d_phi{0.f};
    /// @brief The θ-angle of the camera in spherical coordinates
    float theta{M_PI / 2.0f};
    float d_theta{0.f};
    /// @brief The position that the camera is looking directly at
    glm::vec3 camera_target{glm::vec3(0.f)};

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