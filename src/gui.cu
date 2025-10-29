#include "gui.cuh"
#include "particles.cuh"
#include "scene.cuh"
#include "datastructure/uniformgrid.cuh"

// constants
const char* FONT_PATH { "res/JBM.ttf" };

// shaders

/// Vertex shader for creating billboard spheres from glPoints
const char* VERTEX_SHADER = R"GLSL(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in float aCol;
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform vec2 viewport; // viewport size
    uniform float radius; // particle radius

    flat out vec4 centre_vs;
    out float colour;

    void main() {
        // pass the colour vertex attribute through to the fragment shader
        colour = aCol;

        // compute the centre of the sphere in view space and clip space
        centre_vs = view * vec4(aPos, 1.0);
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

/// Fragment shader inspired by Simon Green's 2010 GDC presentation
/// for creating billboard spheres from glPoints
/// https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf
const char* FRAGMENT_SHADER = R"GLSL(
    #version 330 core

    uniform float radius; // particle radius in world space
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform int use_colour; // one if per-particle colours from aCol buffer should be used, 0 otherwise
    uniform float colour_scale; // scale the colour scalar by this factor before mapping
    uniform int colour_map_selector; // the colour map to use, where 0 is the default

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
        // FragColor = vec4(albedo * max(.3, diffuse), 1.0);
        FragColor = vec4(albedo * diffuse, 1.0);

        // vec3 light_vs = normalize((view * vec4(normalize(-light_dir), 0.0)).xyz);
        // float lambertian = max(0.0, dot(normal_vs, light_vs));
        // vec3 view_dir = normalize(-frag_pos_vs); // camera at origin in view-space
        // vec3 reflect_dir = reflect(-light_vs, normal_vs);
        // float spec = pow(max(0.0, dot(view_dir, reflect_dir)), 64.0); // shininess is exponent
        // vec3 albedo = normalize(vec3(135.,206.,250.));
        // vec3 ambient = 0.5 * albedo;
        // vec3 color = ambient + (0.9 * lambertian) * albedo + 0.6 * spec;
        // FragColor = vec4(color, 1.0);
    }

)GLSL";

// OpenGL helper functions

/// OpenGL error checking macro that generalizes using a getIv command for a
/// specified flag such as `GL_COMPILE_STATUS` or `GL_LINK_STATUS`, checking for
/// success of the operation and retrieving and displaying errors if any
/// occured. See https://learnopengl.com/Getting-started/Shaders
#define OPENGL_CHECK(command, flag, shader, message)                           \
    {                                                                          \
        int success;                                                           \
        char log[512];                                                         \
        command(shader, flag, &success);                                       \
        if (!success) {                                                        \
            glGetShaderInfoLog(shader, 512, NULL, log);                        \
            std::cout << message << "\n" << log << std::endl;                  \
        }                                                                      \
    }

GLuint GUI::compile_shader(void)
{
    // compile the shaders, checking for errors
    GLuint vertex_shader { glCreateShader(GL_VERTEX_SHADER) };
    glShaderSource(vertex_shader, 1, &VERTEX_SHADER, NULL);
    glCompileShader(vertex_shader);
    OPENGL_CHECK(glGetShaderiv, GL_COMPILE_STATUS, vertex_shader,
        "Vertex shader compilation failed:")

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &FRAGMENT_SHADER, NULL);
    glCompileShader(fragment_shader);
    OPENGL_CHECK(glGetShaderiv, GL_COMPILE_STATUS, fragment_shader,
        "Fragment shader compilation failed:")

    // build program from vertex and fragment shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    OPENGL_CHECK(glGetProgramiv, GL_LINK_STATUS, fragment_shader,
        "Shader program linking failed:")

    // delete shaders, return the program id
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
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
    const float theta_cur { wrap_around(theta + d_theta, 0, M_PI) };
    const float phi_cur { wrap_around(phi + d_phi, 0, 2 * M_PI) };
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

    // create shader program
    shader_program = compile_shader();

    // trigger the initial computation of the view matrix representing the
    // camera this is recomputed on-demand in the `glfw_process_input` function
    // whenever the camera is adjusted through user input
    update_view();

    create_and_register_buffer(N);
}

void GUI::create_and_register_buffer(uint N)
{
    // create VBO for positions
    glGenBuffers(1, &pos_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);

    // create vbo for colours
    glGenBuffers(1, &col_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // expose VBOs to CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_pos_vbo_resource, pos_vbo, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_col_vbo_resource, col_vbo, cudaGraphicsMapFlagsWriteDiscard));

    // create VAO:
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    // - bind position vbo
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
    glEnableVertexAttribArray(0);
    // - bind colour vbo
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void GUI::destroy_and_deregister_buffer()
{
    // de-register VBO from use by CUDA
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pos_vbo_resource));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_col_vbo_resource));
    // delete VBO and VAO buffers
    glDeleteBuffers(1, &pos_vbo);
    glDeleteBuffers(1, &col_vbo);
    glDeleteVertexArrays(1, &vao);
}

float3* GUI::map_buffer()
{
    // map the buffer for CUDA access
    pos_cuda_mapped = true;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pos_vbo_resource, 0));

    float3* vertices = nullptr;
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void**)&vertices, &num_bytes, cuda_pos_vbo_resource));

    return vertices;
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

void GUI::unmap_buffer()
{
    pos_cuda_mapped = false;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pos_vbo_resource, 0));
}

void GUI::unmap_colour_buffer()
{
    col_cuda_mapped = false;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_col_vbo_resource, 0));
}

float3* GUI::resize_mapped_buffer(uint N_new)
{
    // require that the positions buffer be currently mapped for usage by CUDA,
    // such that a CUDA-valid pointer can be returned after remapping
    if (!pos_cuda_mapped)
        throw std::runtime_error(
            "resize_mapped_buffer called on an unmapped buffer");

    // unmap the buffer
    unmap_buffer();

    // clean up CUDA binding of vbo
    destroy_and_deregister_buffer();

    // set the new number of particles
    this->N = N_new;

    // create the buffer with the correct, new size
    create_and_register_buffer(N_new);

    // map the buffer for use by CUDA and return the resulting pointer
    return map_buffer();
}

void GUI::run(std::function<void(Particles&, int, const Scene)> step,
    std::function<void(Particles&, int, const Scene)> init, Particles& state,
    const Scene scene)
{
    // start a timer in another thread that periodically sets an atomic bool to
    // true to signal the main thread to update and render the GUI at the target
    // FPS
    std::thread timer([this]() {
        const auto wait_time { 1s / target_fps };
        while (!exit_requested.load()) {
            should_render.store(true);
            std::this_thread::sleep_for(wait_time);
        }
    });

    // initialize a time stamp used for measuring the FPS of the simulation
    auto prev { std::chrono::steady_clock::now() };

    // main loop:
    static bool first_run { true };
    while (!exit_requested.load()) {
        float3* x { map_buffer() };
        state.set_x(x);

        // run initialization function on the first run
        if (first_run) {
            init(state, N, scene);
            first_run = false;
        }

        // conduct as many simulation steps as possible before an update to the
        // GUI is requested by the timer thread
        while (!should_render.load()) {
            // rebuild the acceleration datastructure
            // inner simulation loop is here, use the callback
            step(state, N, scene);
            // hard enforce boundaries
            scene.hard_enforce_bounds(state);
            // update simulation fps, slowly interpolating towards the new value
            const auto now { std::chrono::steady_clock::now() };
            sim_fps = 1000ms / (now - prev);
            prev = now;
        }

        // conditionally call back to request filling the colour buffer
        if (use_per_particle_colour) {
            float* col_buf = map_colour_buffer();
            thrust::transform(state.v.get().begin(), state.v.get().end(),
                col_buf, [] __device__(float3 const v) { return norm(v); });
            unmap_colour_buffer();
        }

        // unmap position buffer for use by OpenGL and update the GUI
        unmap_buffer();
        update(scene.h);
        should_render.store(false);
    }
    timer.join();
}

void GUI::update(float h)
{
    // process inputs
    glfw_process_input();
    // clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // OpenGL rendering commands
    glUseProgram(shader_program);

    // send uniforms
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), // location
        1, // count
        GL_FALSE, // transpose
        glm::value_ptr(view) // value
    );
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "proj"), // location
        1, // count
        GL_FALSE, // transpose
        glm::value_ptr(proj) // value
    );
    glUniform2f(glGetUniformLocation(shader_program, "viewport"), // location
        (float)_window_width, // value 1
        (float)_window_height // value 2
    );
    glUniform1f(glGetUniformLocation(shader_program, "radius"), // location
        h / 2.f // value
    );
    glUniform1i(glGetUniformLocation(shader_program, "use_colour"), // location
        use_per_particle_colour ? 1 : 0 // value
    );
    glUniform1f(
        glGetUniformLocation(shader_program, "colour_scale"), // location
        1. / colour_scale // value
    );
    glUniform1i(
        glGetUniformLocation(shader_program, "colour_map_selector"), // location
        colour_map_selector // value
    );

    glBindVertexArray(vao);

    glDrawArrays(GL_POINTS, 0, N);
    glBindVertexArray(0);

    // poll for window events
    glfwPollEvents();
    if (glfwWindowShouldClose(window))
        exit_requested.store(true);

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

    // start of contents ~~~~~
    ImGui::Begin("SETTINGS");
    ImGui::Text("GUI Frame time %.3fms (%.1f FPS)", 1000.0f / io->Framerate,
        io->Framerate);
    ImGui::Text("SIM Frame time %.3fms (%.1f FPS)", 1000.0f / sim_fps, sim_fps);
    if (ImGui::Button("Exit"))
        exit_requested.store(true);
    if (ImGui::InputFloat("Base Camera Radius", &radius_init, 0.1f, 1.0f))
        update_view();
    ImGui::Checkbox("Use per-particle colours", &use_per_particle_colour);
    ImGui::InputFloat(
        "Colour mapping maximum value", &colour_scale, 1.f, 5.f, "%.0f");
    ImGui::Combo("Colour map", &colour_map_selector, "Spectral\0CB-RdYiBu");

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
    destroy_and_deregister_buffer();
    // clearn up OpenGL
    glDeleteProgram(shader_program);
    // clean up GLFW
    glfwTerminate();
}
