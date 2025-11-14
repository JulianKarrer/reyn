#include "gui.cuh"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <chrono>
using namespace std::literals;
#include "particles.cuh"
#include "scene/scene.cuh"
#include "scene/sample_boundary.cuh"

// constants
const char* FONT_PATH { "res/JBM.ttf" };

// shaders

/// Vertex shader for creating billboard spheres from glPoints, used for
/// visualizing the fluid
const char* VERTEX_SHADER_FLUID = R"GLSL(
    #version 330 core
    layout (location = 0) in float x;
    layout (location = 1) in float y;
    layout (location = 2) in float z;
    layout (location = 3) in float col;
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform vec2 viewport; // viewport size
    uniform float radius; // particle radius

    flat out vec4 centre_vs;
    out float colour;

    void main() {
        // pass the colour vertex attribute through to the fragment shader
        colour = col;

        // compute the centre of the sphere in view space and clip space
        centre_vs = view * vec4(x,y,z, 1.0);
        gl_Position = proj * centre_vs;

        // if the sphere is behind the camera, disregard it by setting point size to zero
        float dist = -centre_vs.z;
        if (dist <= 0.0) {
            gl_PointSize = 0.0;
            return;
        }
        // compute the size of the sphere from the projection matrix, distance and 
        // viewport height
        // https://stackoverflow.com/questions/907756/
        float size_pixels = (proj[1][1] * radius * viewport.y) / dist;
        gl_PointSize = size_pixels;
    }
)GLSL";

/// Vertex shader for creating billboard spheres from glPoints, used for
/// visualizing the boundary
const char* VERTEX_SHADER_BOUNDARY = R"GLSL(
    #version 330 core
    layout (location = 0) in float x;
    layout (location = 1) in float y;
    layout (location = 2) in float z;
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform vec2 viewport; // viewport size
    uniform float radius; // particle radius

    flat out vec4 centre_vs;

    void main() {
        // same function as in VERTEX_SHADER_FLUID, except no colour is passed on
        centre_vs = view * vec4(x,y,z, 1.0);
        gl_Position = proj * centre_vs;
        float dist = -centre_vs.z;
        if (dist <= 0.0) {
            gl_PointSize = 0.0;
            return;
        }
        float size_pixels = (proj[1][1] * radius * viewport.y) / dist;
        gl_PointSize = size_pixels;
    }
)GLSL";

/// Fragment shader inspired by Simon Green's 2010 GDC presentation
/// for creating billboard spheres from glPoints
/// https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf
const char* FRAGMENT_SHADER_FLUID = R"GLSL(
    #version 330 core

    uniform float radius; // particle radius in world space
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform int use_colour; // one if per-particle colours from aCol buffer should be used, 0 otherwise
    uniform float colour_scale; // scale the colour scalar by this factor before mapping
    uniform int colour_map_selector; // the colour map to use, where 0 is the default
    uniform float shading_strength; // interpolates between using diffuse lambertian shading or not

    flat in vec4 centre_vs; // view-space position of the centre of the particle
    in float colour;
    out vec4 FragColor;

    vec3 colour_map(float x){
        // in this case, revert the input
        x = clamp(1.0 - x * colour_scale, 0., 1.);
        float r = 0.;
        float g = 0.;
        float b = 0.;
        if (colour_map_selector == 1){
            //https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/IDL_CB-RdYiBu.frag
            // red
            if (x < 0.09790863520700754) {
                r= 5.14512820512820E+02 * x + 1.64641025641026E+02;
            } else if (x < 0.2001887081633112) {
                r= 2.83195402298854E+02 * x + 1.87288998357964E+02;
            } else if (x < 0.3190117539655621) {
                r= 9.27301587301214E+01 * x + 2.25417989417999E+02;
            } else if (x < 0.500517389125164) {
                r= 255.0;
            } else if (x < 0.6068377196788788) {
                r= -3.04674876847379E+02 * x + 4.07495073891681E+02;
            } else if (x < 0.9017468988895416) {
                r= (1.55336390191951E+02 * x - 7.56394659038288E+02) * x + 6.24412733169483E+02;
            } else {
                r= -1.88350769230735E+02 * x + 2.38492307692292E+02;
            }
            //green
            if (x < 0.09638568758964539) {
                g = 4.81427692307692E+02 * x + 4.61538461538488E-01;
            } else if (x < 0.4987066686153412) {
                g = ((((3.25545903568267E+04 * x - 4.24067109461319E+04) * x + 1.83751375886345E+04) * x - 3.19145329617892E+03) * x + 8.08315127034676E+02) * x - 1.44611527812961E+01;
            } else if (x < 0.6047312345537269) {
                g = -1.18449917898218E+02 * x + 3.14234811165860E+02;
            } else if (x < 0.7067635953426361) {
                g = -2.70822112753102E+02 * x + 4.06379036672115E+02;
            } else {
                g = (-4.62308723214883E+02 * x + 2.42936159122279E+02) * x + 2.74203431802418E+02;
            }
            //blue 
            if (x < 0.09982818011951204) {
                b = 1.64123076923076E+01 * x + 3.72646153846154E+01;
            } else if (x < 0.2958717460833126) {
                b = 2.87014675052409E+02 * x + 1.02508735150248E+01;
            } else if (x < 0.4900527540014758) {
                b = 4.65475113122167E+02 * x - 4.25505279034673E+01;
            } else if (x < 0.6017014681258838) {
                b = 5.61032967032998E+02 * x - 8.93789173789407E+01;
            } else if (x < 0.7015737100463595) {
                b = -1.51655677655728E+02 * x + 3.39446886446912E+02;
            } else if (x < 0.8237156500567735) {
                b = -2.43405347593559E+02 * x + 4.03816042780725E+02;
            } else {
                b = -3.00296889157305E+02 * x + 4.50678495922638E+02;
            }

        } else {

            // https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/IDL_CB-Spectral.frag
            // red
            if (x < 0.09752005946586478) {
                r = 5.63203907203907E+02 * x + 1.57952380952381E+02;
            } else if (x < 0.2005235116443438) {
                r = 3.02650769230760E+02 * x + 1.83361538461540E+02;
            } else if (x < 0.2974133397506856) {
                r = 9.21045429665647E+01 * x + 2.25581007115501E+02;
            } else if (x < 0.5003919130598823) {
                r = 9.84288115246108E+00 * x + 2.50046722689075E+02;
            } else if (x < 0.5989021956920624) {
                r = -2.48619704433547E+02 * x + 3.79379310344861E+02;
            } else if (x < 0.902860552072525) {
                r = ((2.76764884219295E+03 * x - 6.08393126459837E+03) * x + 3.80008072407485E+03) * x - 4.57725185424742E+02;
            } else {
                r = 4.27603478260530E+02 * x - 3.35293188405479E+02;
            }
            // green
            if (x < 0.09785836420571035) {
                g = 6.23754529914529E+02 * x + 7.26495726495790E-01;
            } else if (x < 0.2034012006283468) {
                g = 4.60453201970444E+02 * x + 1.67068965517242E+01;
            } else if (x < 0.302409765476316) {
                g = 6.61789401709441E+02 * x - 2.42451282051364E+01;
            } else if (x < 0.4005965758690823) {
                g = 4.82379130434784E+02 * x + 3.00102898550747E+01;
            } else if (x < 0.4981907026473237) {
                g = 3.24710622710631E+02 * x + 9.31717541717582E+01;
            } else if (x < 0.6064345916502067) {
                g = -9.64699507389807E+01 * x + 3.03000000000023E+02;
            } else if (x < 0.7987472620841592) {
                g = -2.54022986425337E+02 * x + 3.98545610859729E+02;
            } else {
                g = -5.71281628959223E+02 * x + 6.51955082956207E+02;
            }
            // blue
            if (x < 0.0997359608740309) {
                b = 1.26522393162393E+02 * x + 6.65042735042735E+01;
            } else if (x < 0.1983790695667267) {
                b = -1.22037851037851E+02 * x + 9.12946682946686E+01;
            } else if (x < 0.4997643530368805) {
                b = (5.39336225400169E+02 * x + 3.55461986381562E+01) * x + 3.88081126069087E+01;
            } else if (x < 0.6025972254407099) {
                b = -3.79294261294313E+02 * x + 3.80837606837633E+02;
            } else if (x < 0.6990141388105746) {
                b = 1.15990231990252E+02 * x + 8.23805453805459E+01;
            } else if (x < 0.8032653181119567) {
                b = 1.68464957265204E+01 * x + 1.51683418803401E+02;
            } else if (x < 0.9035796343050095) {
                b = 2.40199023199020E+02 * x - 2.77279202279061E+01;
            } else {
                b = -2.78813846153774E+02 * x + 4.41241538461485E+02;
            }
        };
        
        
        return clamp(vec3(r, g, b) / 255., 0., 1.);
    }


    void main() {
        // rescale fragment coordinate to [-1;1]^2
        vec2 coord = gl_PointCoord * 2. - 1.;
        float length_squared = dot(coord, coord);
        // discard fragments outside a circle
        if (length_squared > 1.) discard; 
        // calculate the normal in view space
        float z = sqrt(1. - length_squared);
        vec3 normal_vs = vec3(coord, z);

        // calculate the position of the fragment view space
        vec3 frag_pos_vs = centre_vs.xyz + normal_vs * radius;
        // then project to clip space to obtain correct depth
        vec4 frag_pos_cs = proj * vec4(frag_pos_vs, 1.);
        float z_cs = frag_pos_cs.z / frag_pos_cs.w;
        // remap depth from [-1;1] to [0;1] and write to depth buffer
        gl_FragDepth = z_cs * 0.5 + 0.5;

        // shading
        float diffuse = max(0.0, normalize(normal_vs).z);
        vec3 albedo = (
            (use_colour == 1) ? 
            colour_map(colour) :       // if using per-particle colours
            normalize(vec3(135.,206.,250.)) // default colour
        );
        FragColor = vec4(mix(albedo, albedo * diffuse, shading_strength), 1.0);
    }

)GLSL";

/// Fragment shader similar to `FRAGMENT_SHADER_FLUID` but with simpler shading,
/// less uniforms and a single colour, used for visualization of the boundary
/// particles
const char* FRAGMENT_SHADER_BOUNDARY = R"GLSL(
    #version 330 core

    uniform float radius; // particle radius in world space
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform vec4 colour; 

    flat in vec4 centre_vs; // view-space position of the centre of the particle
    out vec4 FragColor;

    void main() {
        // same code as in FRAGMENT_SHADER_FLUID
        vec2 coord = gl_PointCoord * 2. - 1.;
        float length_squared = dot(coord, coord);
        if (length_squared > 1.) discard;
        float z = sqrt(1. - length_squared);
        vec3 normal_vs = vec3(coord, z);
        vec3 frag_pos_vs = centre_vs.xyz + normal_vs * radius;
        vec4 frag_pos_cs = proj * vec4(frag_pos_vs, 1.);
        float z_cs = frag_pos_cs.z / frag_pos_cs.w;
        gl_FragDepth = z_cs * 0.5 + 0.5;
        float diffuse = max(0.0, normalize(normal_vs).z);
        FragColor = diffuse * colour;
    }

)GLSL";

// OpenGL helper functions

/// @brief OpenGL convenience function for checking the success of shader
/// compilation and retrieving and displaying errors if any occured. See
/// https://learnopengl.com/Getting-started/Shaders
/// @param shader shader to check
/// @param message message to display on failure
static void _opengl_check_compile(GLuint shader, const char* message)
{
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cout << message << "\n" << log << std::endl;
    }
}
/// @brief OpenGL convenience function for checking the success of program
/// linking and retrieving and displaying errors if any occured. See
/// https://learnopengl.com/Getting-started/Shaders
/// @param program program to check
/// @param message message to display on failure
static void _opengl_check_link(GLuint program, const char* message)
{
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        std::cout << message << "\n" << log << std::endl;
    }
}

GLuint GUI::compile_shaders(
    const char* vertex_shader, const char* fragment_shader)
{
    // compile the shaders, checking for errors
    GLuint vert_shader { glCreateShader(GL_VERTEX_SHADER) };
    glShaderSource(vert_shader, 1, &vertex_shader, NULL);
    glCompileShader(vert_shader);
    _opengl_check_compile(vert_shader, "Vertex shader compilation failed:");

    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag_shader, 1, &fragment_shader, NULL);
    glCompileShader(frag_shader);
    _opengl_check_compile(frag_shader, "Fragment shader compilation failed:");

    // build program from vertex and fragment shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);
    _opengl_check_link(program, "Shader program linking failed:");

    // delete shaders, return the program id
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    return program;
};

void GUI::_update_proj()
{
    proj = glm::perspective(glm::radians(fov), // fov
        (float)_window_width / (float)_window_height, // aspect ratio
        1e-3f, // near plane
        1e3f // far plane
    );
};

float wrap_around(float val, float lower, float upper)
{
    while (val > upper) {
        val -= upper - lower;
    }
    while (val < lower) {
        val += upper - lower;
    }
    return val;
}

void GUI::update_view()
{
    // https://learnopengl.com/Getting-started/Camera
    // compute camera position and view direction from spherical coordinates,
    // wrapping overflowing values around
    const float phi_cur { wrap_around(phi + d_phi, 0., 2. * M_PI) };
    // for theta, clamp instead of wrapping around
    constexpr float eps { 0.00001 };
    theta = std::clamp(theta, 0.f, M_PIf);
    const float theta_cur { std::clamp(theta + d_theta, eps, M_PIf - eps) };
    const float radius { radius_init * _radius_scroll_factor };
    const glm::vec3 camera_position { glm::vec3(
        radius * sinf(theta_cur) * cosf(phi_cur), radius * cosf(theta_cur),
        radius * sinf(theta_cur) * sinf(phi_cur)) };
    const glm::vec3 camera_dir_rev { camera_position - camera_target };
    // standard up direction is positive y
    const glm::vec3 world_up { glm::vec3(0.f, 1.f, 0.f) };
    // compute the up and right unit vectors with respect to the cameras view
    const glm::vec3 right = glm::cross(world_up, camera_dir_rev);
    const glm::vec3 up { glm::normalize(glm::cross(camera_dir_rev, right)) };
    // add result of current right click drag to camera offset
    if (!(offset_right == 0.f && offset_up == 0.f)) {
        camera_offset
            += offset_right * glm::normalize(right) + offset_up * world_up;
        offset_right = 0.f;
        offset_up = 0.f;
    };
    const glm::vec3 offset { camera_offset + d_right * glm::normalize(right)
        + d_up * world_up };
    // update the view matrix using the `glm::lookAt` function
    view = glm::lookAt(camera_position + offset, camera_target + offset, up);
};

// CALLBACKS
/// Callback reacting to user resize of the window, adjusting the OpenGL
/// viewport, updating the `_window_width` and `_window_height` variables and
/// updating the camera projection matrix.
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    const auto gui = (GUI*)glfwGetWindowUserPointer(window);
    if (gui) {
        gui->_window_width = width;
        gui->_window_height = height;
        gui->_update_proj();
    }
    glViewport(0, 0, width, height);
};

/// Callback reacting to user scroll, which changes the radius of the orbital
/// controls
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    const auto gui = (GUI*)glfwGetWindowUserPointer(window);
    if (gui) {
        gui->_radius_scroll_factor = std::clamp(gui->_radius_scroll_factor
                * (1.0f - (float)yoffset * gui->_scroll_speed),
            0.01f, 100.f);
        // update the camera
        gui->update_view();
    }
}

void GUI::glfw_process_input()
{
    // update current window size
    glfwGetFramebufferSize(window, &_window_width, &_window_height);

    // only react to cursor events if window is focused
    if (glfwGetWindowAttrib(window, GLFW_FOCUSED)
        && glfwGetWindowAttrib(window, GLFW_FOCUSED)
        && !ImGui::GetIO().WantCaptureMouse) {
        // query normalized cursor position in [0.0; 1.0] x [0.0; 1.0]
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        xpos = std::clamp(xpos / _window_width, 0., 1.);
        ypos = std::clamp(ypos / _window_height, 0., 1.);

        // query cursor state and update camera
        bool camera_needs_update { false };

        // adjust viewing angle if dragging left mouse button
        const bool left_pressed {
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS
        };
        left_drag.update(left_pressed, xpos, d_phi, -2.f * M_PI, phi, ypos,
            d_theta, M_PI, theta, camera_needs_update);

        // adjust camera and target position if dragging right mouse button
        const bool right_pressed {
            glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
        };
        right_drag.update(right_pressed, xpos, d_right, 1., offset_right, ypos,
            d_up, 1., offset_up, camera_needs_update);

        // if the camera was changed in any way, recompute the view matrix and
        // update `view`
        if (camera_needs_update)
            update_view();
    }
};

// QUERY WHETHER CLOSE WAS REQUESTED

// CONSTRUCTOR

GUI::GUI(int init_w, int init_h, bool enable_vsync, double target_fps)
{
    // save parameters
    this->_window_width = init_w;
    this->_window_height = init_h;
    this->_window_height = init_h;
    this->target_fps = target_fps;
    _update_proj();

    // initialize GLFW with the OpenGL version 3.3
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // create a window
    const float main_scale { ImGui_ImplGlfw_GetContentScaleForMonitor(
        glfwGetPrimaryMonitor()) };
    this->window = glfwCreateWindow((int)(init_w * main_scale),
        (int)(init_h * main_scale), "REYN", NULL, NULL);

    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    // set current context
    glfwMakeContextCurrent(window);
    // set update interval: 1 is vsync, 0 as as fast as possible
    glfwSwapInterval(static_cast<int>(enable_vsync));
    // save the pointer to this gui object in the window instance for access via
    // `glfwGetWindowUserPointer` in callbacks
    glfwSetWindowUserPointer(window, (void*)(this));
    // set callbacks for resizeing and scrolling
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    // maximize the window
    glfwMaximizeWindow(window);

    // SET UP IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    this->io = &ImGui::GetIO();
    (void)io;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // keyboard controls
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // gamepad controls
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable; // enable docking
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // multi viewport
    // set style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    // manage scaling and viewports
    style.ScaleAllSizes(main_scale);
    style.FontScaleDpi = main_scale;
    io->ConfigDpiScaleFonts = true;
    io->ConfigDpiScaleViewports = true;
    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    // custom imgui theme
    setup_imgui_style();

    // load font
    style.FontSizeBase = 16.0f;
    io->Fonts->AddFontDefault();
    font = io->Fonts->AddFontFromFileTTF(FONT_PATH, style.FontSizeBase);
    // initialize ImGui
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // set up glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("Failed to initialize GLAD");

    // set clear colour and initial viewport
    glViewport(0, 0, init_w, init_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // enable point rendering features
    glEnable(GL_PROGRAM_POINT_SIZE); // enable varying, programmable point sizes
    glEnable(GL_DEPTH_TEST); // enable depth testing

    // create shader programs
    shader_program_fld
        = compile_shaders(VERTEX_SHADER_FLUID, FRAGMENT_SHADER_FLUID);
    shader_program_bdy
        = compile_shaders(VERTEX_SHADER_BOUNDARY, FRAGMENT_SHADER_BOUNDARY);

    // trigger the initial computation of the view matrix representing the
    // camera this is recomputed on-demand in the `glfw_process_input` function
    // whenever the camera is adjusted through user input
    update_view();

    // create a buffer and register it for use with CUDA
    create_and_register_buffers(N);

    // start a timer in another thread that periodically sets an atomic bool to
    // true to signal the main thread to update and render the GUI at the target
    // FPS
    timer = std::thread([this]() {
        const auto wait_time { 1s / this->target_fps };
        while (!exit_requested.load()) {
            should_render.store(true);
            std::this_thread::sleep_for(wait_time);
        }
    });
}

void GUI::create_and_register_buffers(uint N)
{
    // create VBO for positions
    glGenBuffers(1, &x_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, x_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &y_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, y_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &z_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, z_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // create vbo for colours
    glGenBuffers(1, &col_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // expose VBOs to CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_x_vbo_resource, x_vbo, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_y_vbo_resource, y_vbo, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_z_vbo_resource, z_vbo, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_col_vbo_resource, col_vbo, cudaGraphicsMapFlagsWriteDiscard));

    // create VAO:
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    // - bind position vbos
    glBindBuffer(GL_ARRAY_BUFFER, x_vbo);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, y_vbo);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, z_vbo);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    // - bind colour vbo
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
}

void GUI::destroy_and_deregister_buffers()
{
    // de-register VBO from use by CUDA
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_x_vbo_resource));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_y_vbo_resource));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_z_vbo_resource));

    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_col_vbo_resource));

    // delete VBO and VAO buffers
    glDeleteBuffers(1, &x_vbo);
    glDeleteBuffers(1, &y_vbo);
    glDeleteBuffers(1, &z_vbo);

    glDeleteBuffers(1, &col_vbo);
    glDeleteVertexArrays(1, &vao);
}

void GUI::map_buffers(Particles& state)
{
    // map the buffer for CUDA access
    pos_cuda_mapped = true;

    size_t _num_bytes;

    // map x position buffer and update xx pointer in state
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_x_vbo_resource, 0));
    float* x_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&x_ptr, &_num_bytes, cuda_x_vbo_resource));
    state.xx.update_raw_ptr(x_ptr, N);

    // map y position buffer and update xy pointer in state
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_y_vbo_resource, 0));
    float* y_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&y_ptr, &_num_bytes, cuda_y_vbo_resource));
    state.xy.update_raw_ptr(y_ptr, N);

    // map z position buffer and update xz pointer in state
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_z_vbo_resource, 0));
    float* z_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&z_ptr, &_num_bytes, cuda_z_vbo_resource));
    state.xz.update_raw_ptr(z_ptr, N);
}

float* GUI::map_colour_buffer()
{
    // map the buffer for CUDA access
    col_cuda_mapped = true;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_col_vbo_resource, 0));

    float* colours = nullptr;
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&colours, &num_bytes, cuda_col_vbo_resource));

    return colours;
}

void GUI::unmap_buffers()
{
    pos_cuda_mapped = false;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_x_vbo_resource, 0))
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_y_vbo_resource, 0))
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_z_vbo_resource, 0))
}

void GUI::unmap_colour_buffer()
{
    col_cuda_mapped = false;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_col_vbo_resource, 0));
}

void GUI::resize_mapped_buffers(uint N_new, Particles& state)
{
    // require that the positions buffer be currently mapped for usage by CUDA,
    // such that a CUDA-valid pointer can be returned after remapping
    if (!pos_cuda_mapped)
        throw std::runtime_error(
            "resize_mapped_buffers called on an unmapped buffer");

    // unmap the buffer
    unmap_buffers();

    // clean up CUDA binding of vbo
    destroy_and_deregister_buffers();

    // set the new number of particles
    this->N = N_new;

    // create the buffer with the correct, new size
    create_and_register_buffers(N_new);

    // map the buffer for use by CUDA and return the resulting pointer
    map_buffers(state);
}

void GUI::initialize_buffers(Particles& state) { map_buffers(state); };

void GUI::set_boundary_to_render(const BoundarySamples* samples)
{
    // get boundary sample count
    N_bdy = samples->xs.size();
    // set flags for rendering in `GUI::update` and cleanup in `GUI::~GUI()`
    has_boundary = true;

    // same procedure as in create_and_register_buffers:

    // create VBOs
    glGenBuffers(1, &x_vbo_bdy);
    glBindBuffer(GL_ARRAY_BUFFER, x_vbo_bdy);
    glBufferData(
        GL_ARRAY_BUFFER, N_bdy * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &y_vbo_bdy);
    glBindBuffer(GL_ARRAY_BUFFER, y_vbo_bdy);
    glBufferData(
        GL_ARRAY_BUFFER, N_bdy * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &z_vbo_bdy);
    glBindBuffer(GL_ARRAY_BUFFER, z_vbo_bdy);
    glBufferData(
        GL_ARRAY_BUFFER, N_bdy * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // register them with cuda
    cudaGraphicsResource* cuda_x = nullptr;
    cudaGraphicsResource* cuda_y = nullptr;
    cudaGraphicsResource* cuda_z = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_x, x_vbo_bdy, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_y, y_vbo_bdy, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_z, z_vbo_bdy, cudaGraphicsMapFlagsWriteDiscard));

    // create vao and bind positions
    glGenVertexArrays(1, &vao_bdy);
    glBindVertexArray(vao_bdy);
    glBindBuffer(GL_ARRAY_BUFFER, x_vbo_bdy);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, y_vbo_bdy);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, z_vbo_bdy);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    // map the buffers for the memory transfer from cuda
    size_t _num_bytes;
    float* x_ptr = nullptr;
    float* y_ptr = nullptr;
    float* z_ptr = nullptr;

    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_x, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&x_ptr, &_num_bytes, cuda_x));
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_y, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&y_ptr, &_num_bytes, cuda_y));
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_z, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&z_ptr, &_num_bytes, cuda_z));

    // actual transfer: Memcpy
    // use multiple streams to overlap these three copies
    cudaStream_t cuda_stream_x, cuda_stream_y, cuda_stream_z;
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_x));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_y));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_z));
    CUDA_CHECK(cudaMemcpyAsync(x_ptr, samples->xs.ptr(), N_bdy * sizeof(float),
        cudaMemcpyDeviceToDevice, cuda_stream_x))
    CUDA_CHECK(cudaMemcpyAsync(y_ptr, samples->ys.ptr(), N_bdy * sizeof(float),
        cudaMemcpyDeviceToDevice, cuda_stream_y))
    CUDA_CHECK(cudaMemcpyAsync(z_ptr, samples->zs.ptr(), N_bdy * sizeof(float),
        cudaMemcpyDeviceToDevice, cuda_stream_z))
    CUDA_CHECK(cudaDeviceSynchronize()); // synchronize the streams here

    // unmap buffers again
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_x, 0))
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_y, 0))
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_z, 0))

    // unregister the resources from cuda
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_x));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_y));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_z));
};

bool GUI::update_or_exit(
    Particles& state, const float h, DeviceBuffer<float>* rho)
{
    // declare a static variable for measuring how much time has passed since
    // the last re-render
    static auto prev { std::chrono::steady_clock::now() };
    // update the fps counter of the simulation, regardless of whether the GUI
    // will re-render in this invocation
    const auto now { std::chrono::steady_clock::now() };
    sim_fps = 1000ms / (now - prev);
    prev = now;

    // if an  exit was requested by the GUI, pass that information on to the
    // caller and return immediately
    if (exit_requested.load())
        return false;
    // if the `timer` thread has not deemed it time to re-render yet, also
    // return early but with return value indicating no exit from the
    // application
    if (!fps_gui_sim_coupled && !should_render.load())
        return true;

    // otherwise, render:

    // first, fill the colour buffer
    if (use_per_particle_colour) {
        float* col_buf = map_colour_buffer();
        const size_t N_fld { state.xx.size() };
        // display velocities
        switch (attribute_visualized) {
        case 0: {
            // visualize densities
            if (rho) {
                auto rho_raw { rho->ptr() };
                thrust::transform(thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(N_fld),
                    thrust::device_pointer_cast(col_buf),
                    [rho_raw] __device__(size_t i) { return rho_raw[i]; });
            }
            break;
        }
        default: {
            // visualize velocities
            auto vx { state.vx.ptr() };
            auto vy { state.vy.ptr() };
            auto vz { state.vz.ptr() };
            thrust::transform(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator<size_t>(N_fld),
                thrust::device_pointer_cast(col_buf),
                [vx, vy, vz] __device__(
                    size_t i) { return norm(v3(vx[i], vy[i], vz[i])); });
            break;
        };
        };

        unmap_colour_buffer();
    }

    // then unmap the particle position buffer from CUDA so that OpenGL can use
    // it as a VBO for drawing spheres
    unmap_buffers();

    // now the main update to the GUI can happen, drawing to the screen,
    // processing inputs and handling interaction with UI elements
    update(h);

    // repeat the update while the simulation is stopped
    while (stopped) {
        std::this_thread::sleep_for(1s / 60.);
        update(h);
    }

    // before returning, remap the positions buffers and reflect that change in
    // the particle state in case the pointer has changed
    map_buffers(state);

    // finally, reset the flag for needing to render until the timer sets it
    // again to throttle rendering to the `target_fps`
    should_render.store(false);

    // the program should not be exited right now, return true
    return true;
}

void GUI::update(float h)
{
    // process inputs
    glfw_process_input();
    // clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (show_fluid) {
        // OpenGL rendering commands
        glUseProgram(shader_program_fld);

        // send uniforms
        glUniformMatrix4fv(
            glGetUniformLocation(shader_program_fld, "view"), // location
            1, // count
            GL_FALSE, // transpose
            glm::value_ptr(view) // value
        );
        glUniformMatrix4fv(
            glGetUniformLocation(shader_program_fld, "proj"), // location
            1, // count
            GL_FALSE, // transpose
            glm::value_ptr(proj) // value
        );
        glUniform2f(
            glGetUniformLocation(shader_program_fld, "viewport"), // location
            (float)_window_width, // value 1
            (float)_window_height // value 2
        );
        glUniform1f(
            glGetUniformLocation(shader_program_fld, "radius"), // location
            h / 2.f // value
        );
        glUniform1i(
            glGetUniformLocation(shader_program_fld, "use_colour"), // location
            use_per_particle_colour ? 1 : 0 // value
        );
        glUniform1f(glGetUniformLocation(
                        shader_program_fld, "colour_scale"), // location
            1. / colour_scale // value
        );
        glUniform1i(glGetUniformLocation(shader_program_fld,
                        "colour_map_selector"), // location
            colour_map_selector // value
        );
        glUniform1f(glGetUniformLocation(
                        shader_program_fld, "shading_strength"), // location
            sat(shading_strength / 100.f) // value
        );

        glBindVertexArray(vao);

        glDrawArrays(GL_POINTS, 0, N);
        glBindVertexArray(0);
    }
    if (has_boundary && show_boundary) {
        // same as above but with fewer uniforms
        glUseProgram(shader_program_bdy);
        glUniformMatrix4fv(glGetUniformLocation(shader_program_bdy, "view"), 1,
            GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shader_program_bdy, "proj"), 1,
            GL_FALSE, glm::value_ptr(proj));
        glUniform2f(glGetUniformLocation(shader_program_bdy, "viewport"),
            (float)_window_width, (float)_window_height);
        glUniform1f(glGetUniformLocation(shader_program_bdy, "radius"),
            h / 2.f * bdy_particle_display_size_factor);
        glUniform4f(glGetUniformLocation(shader_program_bdy, "colour"),
            bdy_colour[0], bdy_colour[1], bdy_colour[2], bdy_colour[3]);

        glBindVertexArray(vao_bdy);

        glDrawArrays(GL_POINTS, 0, N_bdy);
        glBindVertexArray(0);
    }

    // poll for window events
    glfwPollEvents();
    if (glfwWindowShouldClose(window)) {
        exit_requested.store(true);
        // make sure the exit is processed if stalling for a stopped simulation
        stopped = false;
    }

    // update ImGUI
    imgui_draw();

    // swap back and front buffers
    glfwSwapBuffers(window);
}

void GUI::imgui_draw()
{
    // update the FPS count in the window title using the ImGui framerate
    // counter
    const int max_fps_str_size { 40 };
    char fps_str[max_fps_str_size];
    snprintf(fps_str, max_fps_str_size, "REYN | FPS %5.1f / %5.1f",
        io->Framerate, sim_fps);
    glfwSetWindowTitle(window, fps_str);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // // docking setup
    // ImGuiID id;
    // ImGui::DockSpaceOverViewport(id, ImGui::GetMainViewport(),
    // ImGuiDockNodeFlags_PassthruCentralNode);

    // use custom font
    ImGui::PushFont(font);

    // ImGui::ShowDemoWindow();

    // start of contents ~~~~~
    ImGui::Begin("SETTINGS");
    if (ImGui::Button("Exit")) {
        exit_requested.store(true);
        stopped = false;
    }
    ImGui::Checkbox("Simulation Stopped", &stopped);
    ImGui::Text("GUI interval %.3fms (%.1f FPS)", 1000.0f / io->Framerate,
        io->Framerate);
    ImGui::Text("SIM interval %.3fms (%.1f FPS)", 1000.0f / sim_fps, sim_fps);

    const float slider_width { 0.4f };
    if (ImGui::CollapsingHeader(
            "Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("run GUI and SIM at same rate", &fps_gui_sim_coupled);
        ImGui::Checkbox("show fluid particles", &show_fluid);
        ImGui::Checkbox("show boundary particles", &show_boundary);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * slider_width);
        if (ImGui::InputFloat("base camera radius", &radius_init, 0.1f, 1.0f))
            update_view();
        ImGui::Checkbox("per-particle colours", &use_per_particle_colour);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * slider_width);
        ImGui::SliderFloat(
            "shading strength", &shading_strength, 0.f, 100.f, "%.0f%");
        ImGui::InputFloat(
            "colour mapping max", &colour_scale, 1.f, 5.f, "%.0f");
        ImGui::Combo("colour map", &colour_map_selector, "Spectral\0CB-RdYiBu");
        ImGui::Combo(
            "attribute visualized", &attribute_visualized, "Density\0Velocity");
        ImGui::SliderFloat(
            "boundary size", &bdy_particle_display_size_factor, 0.0f, 1.0f);
        ImGui::ColorEdit4("boundary colour", bdy_colour);
    }

    // end of contents ~~~~~~~

    ImGui::PopFont();
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

GUI::~GUI()
{
    // join the timer thread to prevent leaks
    timer.join();
    // unmap, unregister VBOs from CUDA and delete them
    unmap_buffers();
    destroy_and_deregister_buffers();
    // only destroy boundary buffers if need be
    if (has_boundary) {
        glDeleteBuffers(1, &x_vbo_bdy);
        glDeleteBuffers(1, &y_vbo_bdy);
        glDeleteBuffers(1, &z_vbo_bdy);
        glDeleteVertexArrays(1, &vao_bdy);
        glDeleteProgram(shader_program_bdy);
    }
    // clearn up OpenGL
    glDeleteProgram(shader_program_fld);
    // clean up GLFW
    glfwTerminate();
}

void GUI::setup_imgui_style()
{
    // 'Deep Dark Theme' by janekb04
    auto& style = ImGui::GetStyle();
    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.6f;
    style.WindowPadding = ImVec2(8.0f, 8.0f);
    style.WindowRounding = 7.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = ImVec2(32.0f, 32.0f);
    style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_Left;
    style.ChildRounding = 4.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 4.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(5.0f, 2.0f);
    style.FrameRounding = 3.0f;
    style.FrameBorderSize = 1.0f;
    style.ItemSpacing = ImVec2(6.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 6.0f);
    style.CellPadding = ImVec2(6.0f, 6.0f);
    style.IndentSpacing = 25.0f;
    style.ColumnsMinSpacing = 6.0f;
    style.ScrollbarSize = 15.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabMinSize = 10.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;
    style.TabBorderSize = 1.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled]
        = ImVec4(0.49803922f, 0.49803922f, 0.49803922f, 1.0f);
    style.Colors[ImGuiCol_WindowBg]
        = ImVec4(0.09803922f, 0.09803922f, 0.09803922f, 1.0f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[ImGuiCol_PopupBg]
        = ImVec4(0.1882353f, 0.1882353f, 0.1882353f, 0.92f);
    style.Colors[ImGuiCol_Border]
        = ImVec4(0.1882353f, 0.1882353f, 0.1882353f, 0.29f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.24f);
    style.Colors[ImGuiCol_FrameBg]
        = ImVec4(0.047058824f, 0.047058824f, 0.047058824f, 0.54f);
    style.Colors[ImGuiCol_FrameBgHovered]
        = ImVec4(0.1882353f, 0.1882353f, 0.1882353f, 0.54f);
    style.Colors[ImGuiCol_FrameBgActive]
        = ImVec4(0.2f, 0.21960784f, 0.22745098f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive]
        = ImVec4(0.05882353f, 0.05882353f, 0.05882353f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg]
        = ImVec4(0.13725491f, 0.13725491f, 0.13725491f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg]
        = ImVec4(0.047058824f, 0.047058824f, 0.047058824f, 0.54f);
    style.Colors[ImGuiCol_ScrollbarGrab]
        = ImVec4(0.3372549f, 0.3372549f, 0.3372549f, 0.54f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered]
        = ImVec4(0.4f, 0.4f, 0.4f, 0.54f);
    style.Colors[ImGuiCol_ScrollbarGrabActive]
        = ImVec4(0.5568628f, 0.5568628f, 0.5568628f, 0.54f);
    style.Colors[ImGuiCol_CheckMark]
        = ImVec4(0.32941177f, 0.6666667f, 0.85882354f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab]
        = ImVec4(0.3372549f, 0.3372549f, 0.3372549f, 0.54f);
    style.Colors[ImGuiCol_SliderGrabActive]
        = ImVec4(0.5568628f, 0.5568628f, 0.5568628f, 0.54f);
    style.Colors[ImGuiCol_Button]
        = ImVec4(0.047058824f, 0.047058824f, 0.047058824f, 0.54f);
    style.Colors[ImGuiCol_ButtonHovered]
        = ImVec4(0.1882353f, 0.1882353f, 0.1882353f, 0.54f);
    style.Colors[ImGuiCol_ButtonActive]
        = ImVec4(0.2f, 0.21960784f, 0.22745098f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.0f, 0.0f, 0.0f, 0.52f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.0f, 0.0f, 0.0f, 0.36f);
    style.Colors[ImGuiCol_HeaderActive]
        = ImVec4(0.2f, 0.21960784f, 0.22745098f, 0.33f);
    style.Colors[ImGuiCol_Separator]
        = ImVec4(0.2784314f, 0.2784314f, 0.2784314f, 0.29f);
    style.Colors[ImGuiCol_SeparatorHovered]
        = ImVec4(0.4392157f, 0.4392157f, 0.4392157f, 0.29f);
    style.Colors[ImGuiCol_SeparatorActive]
        = ImVec4(0.4f, 0.4392157f, 0.46666667f, 1.0f);
    style.Colors[ImGuiCol_ResizeGrip]
        = ImVec4(0.2784314f, 0.2784314f, 0.2784314f, 0.29f);
    style.Colors[ImGuiCol_ResizeGripHovered]
        = ImVec4(0.4392157f, 0.4392157f, 0.4392157f, 0.29f);
    style.Colors[ImGuiCol_ResizeGripActive]
        = ImVec4(0.4f, 0.4392157f, 0.46666667f, 1.0f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.0f, 0.0f, 0.0f, 0.52f);
    style.Colors[ImGuiCol_TabHovered]
        = ImVec4(0.13725491f, 0.13725491f, 0.13725491f, 1.0f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.2f, 0.2f, 0.2f, 0.36f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.0f, 0.0f, 0.52f);
    style.Colors[ImGuiCol_TabUnfocusedActive]
        = ImVec4(0.13725491f, 0.13725491f, 0.13725491f, 1.0f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered]
        = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.52f);
    style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.0f, 0.0f, 0.0f, 0.52f);
    style.Colors[ImGuiCol_TableBorderLight]
        = ImVec4(0.2784314f, 0.2784314f, 0.2784314f, 0.29f);
    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);
    style.Colors[ImGuiCol_TextSelectedBg]
        = ImVec4(0.2f, 0.21960784f, 0.22745098f, 1.0f);
    style.Colors[ImGuiCol_DragDropTarget]
        = ImVec4(0.32941177f, 0.6666667f, 0.85882354f, 1.0f);
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight]
        = ImVec4(1.0f, 0.0f, 0.0f, 0.7f);
    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.2f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.35f);
}
