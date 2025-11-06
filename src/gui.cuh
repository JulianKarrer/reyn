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
#include "common.h"
#include <chrono>
using namespace std::literals;
#include <thread>
#include <atomic>

// forwards declaration to avoid circular depenedency
class Particles;
class Scene;
class UniformGridBuilder;
struct BoundarySamples;

enum class Resort : bool;
template <Resort Resorted> struct UniformGrid;

/// @brief Object managing a graphical user interface.
/// Uses GLFW for cross-platform windowing and sets up
/// OpenGL <-> CUDA interop for displaying the contents
/// of a buffer of positions managed by CUDA as a point
/// cloud.
class GUI {
public:
    /// @brief Construct a GUI window using GLFW and OpenGL for display of the
    /// particles.
    /// @param N initial number of particles
    /// @param init_w initial window width
    /// @param init_h initial window height
    /// @param on_failure a parameter-less void callback to execute when
    /// initialization of the GUI fails, such that e.g. MPI can be finalized
    /// before exiting the application
    /// @param enable_vsync whether or not the GUI framerate should be limited
    /// by the V-synced refresh rate
    /// @param target_fps the target maximum frame rate of the GUI, which is
    /// used for throttling in case the simulation runs at a higher rate than
    /// what is required by the GUI
    GUI(int init_w, int init_h, bool enable_vsync = false,
        double target_fps = 60.);
    ~GUI();

    /// @brief Method called by the `Particles` class on construction with a
    /// reference to itself if constructed using a GUI instance to make sure
    /// that the positions buffer managed by the GUI to enable zero-copy
    /// visualization of particle positions is mapped for use by CUDA and ready
    /// to go
    /// @param state `Particles` instance wrapping the VBO of particle positions
    /// allocated by the GUI
    void initialize_buffers(Particles& state);

    /// @brief Update the GUI, rendering particles to screen, handling inputs
    /// and updating the UI if enough time has passed since the last update,
    /// otherwise exits early.
    ///
    /// Can be used as a `while` condition of the main loop since the return
    /// value signals whether the closing of the application has been requested
    /// through the GUI
    /// @param state current `Particles` state to display
    /// @param scene current `Scene` being rendered, read for getting the
    /// particle spacing/radius etc.
    /// @return `false` if exit of the application is requested by the user
    /// through the GUI, `true` otherwise
    bool update_or_exit(Particles& state, const Scene scene);

    /// @brief Resize the particle positions buffer. Must be called while mapped
    /// for use by CUDA.
    /// @param N desired number of particles
    /// @param state state containing position `DeviceBuffer`s to update
    /// @returns pointer to the resized and mapped buffer.
    void resize_mapped_buffers(uint N, Particles& state);

    void set_boundary_to_render(const BoundarySamples* samples);

    /// @brief Update the projection matrix representing the current camera
    /// frustum, which depends on the fov, aspect ratio, near and far planes.
    ///
    /// Updates the private internal `proj` variable and should be called
    /// initially, as well as when window width or height may have changed.
    void _update_proj();
    /// @brief Recompute and update the `view` matrix, required once initially
    /// and every time the camera might have changed.
    void update_view();
    /// @brief Current width of the GUI window
    int _window_width;
    /// @brief Current height of the GUI window
    int _window_height;
    /// @brief The multiplier applied to the initial radius based on scroll
    /// position
    float _radius_scroll_factor { 1.f };
    /// @brief The scroll speed as the fraction of `radius_scroll_factor` that
    /// is added or taken away per line scrolled
    float _scroll_speed { 0.05f };
    ///  @brief Current number of particles
    uint N { 1 };

    ///  @brief Current number of boundary particles
    uint N_bdy { 0 };

    // explicitly forbid a copy constructor for safety, since only one GUI
    // instance may ever exist
    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

private:
    /// @brief Map the position VBO for use by CUDA and upodate pointers to the
    /// buffer of particle positions in the `Particles` state
    void map_buffers(Particles& state);

    /// @brief Unmap the position vertex buffer from CUDA and enable its use by
    /// OpenGL for rendering
    void unmap_buffers();

    /// @brief whether or not the position buffer is currently mapped for access
    /// by CUDA
    bool pos_cuda_mapped { false };

    /// @brief whether or not the colour buffer is currently mapped for access
    /// by CUDA
    bool col_cuda_mapped { false };
    /// @brief whether the gui should request to fill the colour buffer and use
    /// it to colour each particle
    bool use_per_particle_colour { true };

    /// @brief whether or not the `set_boundary_to_render` method has been
    /// successfully called and boundary particles may be rendered
    bool has_boundary { false };
    /// @brief whether or not to render boundary particles, if present
    bool show_boundary { false };
    /// @brief the RGB colour of the boundary particles
    float bdy_colour[3] { 0.5, 0.5, 0.5 };
    /// @brief Relative size of a boundary particle to a fluid particle for
    /// visualization
    float bdy_particle_display_size_factor { 0.1 };

    // internals for GUI
    /// @brief Query whether the GUI has requested the application to close
    std::atomic<bool> exit_requested { ATOMIC_VAR_INIT(false) };
    /// @brief Whether the GUI should run at the maximum frame rate,
    /// unthrottled. This is generally not recommended for computationally
    /// undemanding scenes since the simulation and rendering will essentially
    /// run in lockstep, reducing simulation FPS
    bool fps_gui_sim_coupled { false };
    /// @brief Target maximum FPS in case throttling is required
    double target_fps { 60. };
    /// @brief Measuered frames per second of the simulation
    double sim_fps { 0. };
    /// @brief A atomic boolean set by a timer thread, indicating whether enough
    /// time has elapsed for the GUI (main) thread to render an update to
    /// screen. Otherwise, more simulation steps will be performed before the
    /// next GUI update is rendered.
    std::atomic<bool> should_render { false };
    /// @brief Timer thread used to request a re-render at regular intervals
    std::thread timer;
    /// @brief Whether to draw all particles or not (i.e. only render UI etc.)
    bool show_fluid { true };

    /// @brief a private struct to represent the state of clicking a holding a
    /// mouse button to drag
    struct DragState {
        /// @brief whether the user is currently holding the respective mouse
        /// button
        bool dragging { false };
        /// @brief x-position in normalized coordinates [0;1]^2 of where the
        /// current dragging operation started
        float start_x { 0. };
        /// @brief y-position in normalized coordinates [0;1]^2 of where the
        /// current dragging operation of the left mouse button started
        float start_y { 0. };

        /// @brief Update the state of dragging a pressed mouse button using
        /// current inputs
        /// @param pressed whether the respective mouse button is pressed
        /// @param x the current normalized x-position of the cursor
        /// @param dx reference to a variable storing the offset in `out_x` due
        /// to dragging
        /// @param x_scale the scale of change in `out_x` per window width of
        /// dragging
        /// @param out_x the output variable that is updated when changes in
        /// `dx` due to dragging are commited
        /// @param y  the current normalized y-position of the cursor
        /// @param dy reference to a variable storing the offset in `out_y` due
        /// to dragging
        /// @param y_scale the scale of change in `out_y` per window height of
        /// dragging
        /// @param out_y the output variable that is updated when changes in
        /// `dy`  due to draggingare commited
        /// @param camera_needs_update set to true if the camera must be updated
        /// to reflect a change in `dx`, `dy`, `out_x` or `out_y`
        void update(bool pressed, float x, float& dx, float x_scale,
            float& out_x, float y, float& dy, float y_scale, float& out_y,
            bool& camera_needs_update)
        {
            if (!dragging && pressed) {
                // start dragging
                start_x = x;
                start_y = y;
                dragging = true;
            } else if (dragging && pressed) {
                // update dragging
                camera_needs_update = true;
                dx = x_scale * (start_x - x);
                dy = y_scale * (start_y - y);
            } else if (dragging && !pressed) {
                // stop dragging
                camera_needs_update = true;
                start_x = 0.;
                start_y = 0.;
                dragging = false;
                out_x += dx;
                out_y += dy;
                dx = 0.;
                dy = 0;
            };
        };
    };
    /// @brief The state of dragging the left mouse button, as described in
    /// `DragState`
    DragState left_drag { false, 0., 0. };
    /// @brief The state of dragging the right mouse button, as described in
    /// `DragState`
    DragState right_drag { false, 0., 0. };

    /// @brief The initial or base radius of the camera around the
    /// `camera_target` position in spherical cooridnates, before scrolling
    float radius_init { 5.f };

    // GLFW resources
    GLFWwindow* window;

    // ImGui resources
    ImGuiIO* io;
    ImFont* font;

    // OpenGL resources
    GLuint col_vbo;
    GLuint shader_program_fld, vao, x_vbo, y_vbo, z_vbo;
    GLuint shader_program_bdy, vao_bdy, x_vbo_bdy, y_vbo_bdy, z_vbo_bdy;

    float fov { 45.0 };
    glm::mat4 proj;
    /// @brief The view matrix, representing the orientation of the camera in
    /// world space, depending on camera positions and where the camera is
    /// pointed
    /// @return view matrix for use in vertex shader as a uniform
    glm::mat4 view;
    /// @brief The φ-angle of the camera in spherical coordinates
    float phi { -M_PI / 2.0f };
    /// @brief Current offset to the φ-angleof the camera in spherical
    /// coordinates due to dragging
    float d_phi { 0.f };
    /// @brief The θ-angle of the camera in spherical coordinates
    float theta { M_PI / 2.0f };
    /// @brief Current offset to the θ-angle of the camera in spherical
    /// coordinates due to dragging
    float d_theta { 0.f };
    /// @brief The position that the camera is looking directly at
    glm::vec3 camera_target { glm::vec3(0.f) };
    /// @brief An offset to the position of the camera
    glm::vec3 camera_offset { glm::vec3(0.f) };

    /// @brief Current rightwards contribution to the camera offset due to
    /// dragging
    float d_right { 0. };
    /// @brief Current upwards contribution to the camera offset due to dragging
    float d_up { 0. };

    float offset_right { 0. };
    float offset_up { 0. };

    // colour mapping and per-particle colouring

    /// @brief Map the colour VBO for use by CUDA and obtain a pointer to the
    /// buffer of one scalar per particle for use in the fragment shader
    float* map_colour_buffer();
    /// @brief Unmap the colour buffer from CUDA and enable its use by OpenGL
    /// for rendering
    void unmap_colour_buffer();
    /// @brief The scalar used for colour mapping is scaled by the inverse of
    /// this value, to be adjusted intuitively to the maximum value of whatever
    /// quantity should be visualized.
    float colour_scale { 10. };
    /// @brief Which colour map to use, 0 is the default
    int colour_map_selector { 0 };
    /// @brief value in \f$[0;100]\f$ that interpolates between diffuse
    /// lambertian and pure ambient / no shading
    float shading_strength { 5. };

    // CUDA resources
    cudaGraphicsResource* cuda_x_vbo_resource = nullptr;
    cudaGraphicsResource* cuda_y_vbo_resource = nullptr;
    cudaGraphicsResource* cuda_z_vbo_resource = nullptr;
    cudaGraphicsResource* cuda_col_vbo_resource = nullptr;

    // internal functions for setup and events
    /// Compile the fragment and vertex shaders required for visualization
    /// by OpenGL
    /// @param vertex_shader vertex shader to compile
    /// @param fragment_shader fragment shader to compile
    GLuint compile_shaders(
        const char* vertex_shader, const char* fragment_shader);

    /// @brief Process user input events provided by GLFW, in particular cursor
    /// events that enable clicking and dragging to use camera orbital controls.
    void glfw_process_input();

    /// @brief Create the VBO and VAO for positions and register the buffer with
    /// CUDA
    /// @param N The number of `float3` to fit in the positions buffer
    void create_and_register_buffers(uint N);

    /// @brief Delete the VBO and VAO for positions and unregister the buffer
    /// from CUDA.
    void destroy_and_deregister_buffers();

    ///  Update and manage the ImGui contents
    void imgui_draw();

    /// @brief Set up ImGui style, based on 'Deep Dark Theme' by janekb04
    void setup_imgui_style();

    // main functions for running the gui
    /// @brief Update the GUI, rendering the current particles to screen.
    void update(float h);
};

#endif // GUI_H_