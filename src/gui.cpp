#include "gui.h"
#include "particles.h"
#include "scene.cuh"


// constants
const char *FONT_PATH{"res/JBM.ttf"};

// shaders

/// Vertex shader for creating billboard spheres from glPoints
const char *VERTEX_SHADER = R"GLSL(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform vec2 viewport; // viewport size
    uniform float radius; // particle radius

    flat out vec4 centre_vs;

    void main() {
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
const char *FRAGMENT_SHADER = R"GLSL(
    #version 330 core

    uniform vec3 light_dir; // direction of the light source in world space
    uniform float radius; // particle radius in world space
    uniform mat4 view; // view matrix
    uniform mat4 proj; // projection matrix
    uniform float shininess; // shininess for blinn-phong shading

    flat in vec4 centre_vs; // view-space position of the centre of the particle
    out vec4 FragColor;

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
        vec3 albedo = normalize(vec3(135.,206.,250.));
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

/// OpenGL error checking macro that generalizes using a getIv command for a specified flag
/// such as `GL_COMPILE_STATUS` or `GL_LINK_STATUS`, checking for success of the operation
/// and retrieving and displaying errors if any occured.
/// See https://learnopengl.com/Getting-started/Shaders
#define OPENGL_CHECK(command, flag, shader, message)    \
    {                                                   \
        int success;                                    \
        char log[512];                                  \
        command(shader, flag, &success);                \
        if (!success)                                   \
        {                                               \
            glGetShaderInfoLog(shader, 512, NULL, log); \
            std::cout << message << "\n"                \
                      << log << std::endl;              \
        }                                               \
    }

GLuint GUI::compile_shader(void)
{
    // compile the shaders, checking for errors
    GLuint vertex_shader{glCreateShader(GL_VERTEX_SHADER)};
    glShaderSource(vertex_shader, 1, &VERTEX_SHADER, NULL);
    glCompileShader(vertex_shader);
    OPENGL_CHECK(glGetShaderiv, GL_COMPILE_STATUS, vertex_shader, "Vertex shader compilation failed:")

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &FRAGMENT_SHADER, NULL);
    glCompileShader(fragment_shader);
    OPENGL_CHECK(glGetShaderiv, GL_COMPILE_STATUS, fragment_shader, "Fragment shader compilation failed:")

    // build program from vertex and fragment shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    OPENGL_CHECK(glGetProgramiv, GL_LINK_STATUS, fragment_shader, "Shader program linking failed:")

    // delete shaders, return the program id
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
};

void GUI::_update_proj()
{
    proj = glm::perspective(
        glm::radians(fov),                            // fov
        (float)_window_width / (float)_window_height, // aspect ratio
        1e-3f,                                        // near plane
        1e3f                                          // far plane
    );
};

float wrap_around(float val, float lower, float upper)
{
    while (val > upper)
    {
        val -= upper - lower;
    }
    while (val < lower)
    {
        val += upper - lower;
    }
    return val;
}

glm::mat4 GUI::get_view()
{
    // https://learnopengl.com/Getting-started/Camera
    // compute camera position and view direction from spherical coordinates,
    // wrapping overflowing values around
    const float theta_cur{wrap_around(theta + d_theta, 0, M_PI)};
    const float phi_cur{wrap_around(phi + d_phi, 0, 2 * M_PI)};
    const float radius{radius_init * _radius_scroll_factor};
    const glm::vec3 camera_position{glm::vec3(
        radius * sinf(theta_cur) * cosf(phi_cur),
        radius * cosf(theta_cur),
        radius * sinf(theta_cur) * sinf(phi_cur))};
    const glm::vec3 camera_dir_rev{camera_position - camera_target};
    // standard up direction is positive y
    const glm::vec3 world_up{glm::vec3(0.f, 1.f, 0.f)};
    // compute the up and right unit vectors with respect to the cameras view
    const glm::vec3 right = glm::cross(world_up, camera_dir_rev);
    const glm::vec3 up{glm::cross(camera_dir_rev, right)};
    // return the view matrix using the `glm::lookAt` function
    return glm::lookAt(camera_position, camera_target, glm::normalize(up));
};

// CALLBACKS
/// Callback reacting to user resize of the window, adjusting the OpenGL viewport, updating the `_window_width` and `_window_height` variables and updating the camera projection matrix.
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    const auto gui = (GUI *)glfwGetWindowUserPointer(window);
    if (gui)
    {
        gui->_window_width = width;
        gui->_window_height = height;
        gui->_update_proj();
    }
    glViewport(0, 0, width, height);
};

/// Callback reacting to user scroll, which changes the radius of the orbital controls
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    const auto gui = (GUI *)glfwGetWindowUserPointer(window);
    if (gui)
    {
        gui->_radius_scroll_factor = std::clamp(gui->_radius_scroll_factor * (1.0f - (float)yoffset * gui->_scroll_speed), 0.01f, 100.f);
    }
}

void GUI::glfw_process_input()
{
    // update current window size
    glfwGetFramebufferSize(window, &_window_width, &_window_height);

    // only react to cursor events if window is focused
    if (
        glfwGetWindowAttrib(window, GLFW_FOCUSED) &&
        glfwGetWindowAttrib(window, GLFW_FOCUSED) &&
        !ImGui::GetIO().WantCaptureMouse)
    {
        // query normalized cursor position in [0.0; 1.0] x [0.0; 1.0]
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        xpos = std::clamp(xpos / _window_width, 0., 1.);
        ypos = std::clamp(ypos / _window_height, 0., 1.);

        // query cursor state
        const int state{glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)};
        const bool pressed{state == GLFW_PRESS};
        if (dragging == false && pressed)
        {
            // start dragging
            dragging = true;
            drag_start_x = xpos;
            drag_start_y = ypos;
        }
        else if (dragging == true && pressed)
        {
            // update dragging
            d_phi = -2.f * M_PI * (drag_start_x - xpos);
            d_theta = M_PI * (drag_start_y - ypos);
        }
        else if (dragging == true && !pressed)
        {
            // stop dragging
            dragging = false;
            phi += d_phi;
            d_phi = 0.;
            theta += d_theta;
            d_theta = 0;
        }
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
    const float main_scale{ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor())};
    this->window = glfwCreateWindow((int)(init_w * main_scale), (int)(init_h * main_scale), "REYN", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    // set current context
    glfwMakeContextCurrent(window);
    // set update interval: 1 is vsync, 0 as as fast as possible
    glfwSwapInterval(static_cast<int>(enable_vsync));
    // save the pointer to this gui object in the window instance for access via
    // `glfwGetWindowUserPointer` in callbacks
    glfwSetWindowUserPointer(window, (void *)(this));
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
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // gamepad controls
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // enable docking
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // multi viewport
    // set style
    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    // manage scaling and viewports
    style.ScaleAllSizes(main_scale);
    style.FontScaleDpi = main_scale;
    io->ConfigDpiScaleFonts = true;
    io->ConfigDpiScaleViewports = true;
    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
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
    glEnable(GL_DEPTH_TEST);         // enable depth testing

    // create shader program
    shader_program = compile_shader();

    create_and_register_buffer(N);
}

void GUI::create_and_register_buffer(uint N)
{
    // create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // expose VBO to CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    // create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void GUI::destroy_and_deregister_buffer()
{
    // de-registerVBO from use by CUDA
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    // delete VBO and VAO buffers
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

float3 *GUI::map_buffer()
{
    // map the buffer for CUDA access
    cuda_mapped = true;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));

    float3 *vertices = nullptr;
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&vertices, &num_bytes, cuda_vbo_resource));

    return vertices;
}

void GUI::unmap_buffer()
{
    cuda_mapped = false;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

float3 *GUI::resize_mapped_buffer(uint N_new)
{
    // require that the positions buffer be currently mapped for usage by CUDA,
    // such that a CUDA-valid pointer can be returned after remapping
    if (!cuda_mapped)
        throw std::runtime_error("resize_mapped_buffer called on an unmapped buffer");

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

void GUI::run(std::function<void(Particles &, int)> step, std::function<void(Particles &, int)> init, Particles &state, const Scene &scene)
{
    // start a timer in another thread that periodically sets an atomic bool to true to signal the main thread to update and render the GUI at the target FPS
    std::thread timer([this]()
                      { 
        const auto wait_time {1s / target_fps};
        while (!exit_requested.load()){
            should_render.store(true); 
            std::this_thread::sleep_for(wait_time);
        } });

    // initialize a time stamp used for measuring the FPS of the simulation
    auto prev{std::chrono::steady_clock::now()};

    // main loop:
    static bool first_run{true};
    while (!exit_requested.load())
    {
        float3 *x{map_buffer()};
        state.set_x(x);

        if (first_run)
        {
            // run initialization function on the first run
            init(state, N);
            first_run = false;
        }

        while (!should_render.load())
        {
            // inner simulation loop is here, use the callback
            step(state, N);
            // hard enforce boundaries
            scene.hard_enforce_bounds(state);
            // update simulation fps, slowly interpolating towards the new value
            const auto now{std::chrono::steady_clock::now()};
            sim_fps = 1000ms / (now - prev);
            prev = now;
        }
        unmap_buffer();
        update(state.h);
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
    glUniformMatrix4fv(
        glGetUniformLocation(shader_program, "view"), // location
        1,                                            // count
        GL_FALSE,                                     // transpose
        glm::value_ptr(get_view())                    // value
    );
    glUniformMatrix4fv(
        glGetUniformLocation(shader_program, "proj"), // location
        1,                                            // count
        GL_FALSE,                                     // transpose
        glm::value_ptr(proj)                          // value
    );
    glUniform2f(
        glGetUniformLocation(shader_program, "viewport"), // location
        (float)_window_width,                             // value 1
        (float)_window_height                             // value 2
    );
    glUniform1f(
        glGetUniformLocation(shader_program, "radius"), // location
        h / 2.f                                         // value
    );
    glUniform3fv(
        glGetUniformLocation(shader_program, "light_dir"), // location
        1,                                                 // count
        glm::value_ptr(light_direction));

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
    // update the FPS count in the window title using the ImGui framerate counter
    const int max_fps_str_size{40};
    char fps_str[max_fps_str_size];
    snprintf(fps_str, max_fps_str_size, "REYN | FPS %.1f / %.1f", io->Framerate, sim_fps);
    glfwSetWindowTitle(window, fps_str);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // // docking setup
    // ImGuiID id;
    // ImGui::DockSpaceOverViewport(id, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    // use custom font
    ImGui::PushFont(font);

    // start of contents ~~~~~
    ImGui::Begin("SETTINGS");
    ImGui::Text("GUI Frame time %.3fms (%.1f FPS)", 1000.0f / io->Framerate, io->Framerate);
    ImGui::Text("SIM Frame time %.3fms (%.1f FPS)", 1000.0f / sim_fps, sim_fps);
    if (ImGui::Button("Exit"))
        exit_requested.store(true);
    ImGui::InputFloat("Base Camera Radius", &radius_init, 0.1f, 1.0f);

    float light_dir[3]{light_direction.x, light_direction.y, light_direction.z};
    ImGui::SliderFloat3("Light Direction", light_dir, -1., 1.);
    light_direction.x = light_dir[0];
    light_direction.y = light_dir[1];
    light_direction.z = light_dir[2];

    // end of contents ~~~~~~~

    ImGui::PopFont();
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
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
