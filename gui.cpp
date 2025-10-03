#include "gui.h"

// constants
const char *FONT_PATH{"res/JBM.ttf"};

// shaders
const char *VERTEX_SHADER = R"GLSL(
    #version 330 core
    layout (location = 0) in vec4 aPos;
    void main() {
        gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        gl_PointSize = 10.0;
    }
)GLSL";

const char *FRAGMENT_SHADER = R"GLSL(
    #version 330 core
    out vec4 FragColor;
    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) {
            discard;
        }
        FragColor = vec4(1.0, 0.5, 0.2, 1.0); // orange
    }
)GLSL";

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

// CALLBACKS

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
};

void GUI::glfw_process_input()
{
    // update current window size
    glfwGetFramebufferSize(window, &window_width, &window_height);

    // only react to cursor events if window is focused
    if (
        glfwGetWindowAttrib(window, GLFW_FOCUSED) &&
        glfwGetWindowAttrib(window, GLFW_FOCUSED) &&
        !ImGui::GetIO().WantCaptureMouse)
    {
        // query normalized cursor position in [0.0; 1.0] x [0.0; 1.0]
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        xpos = std::clamp(xpos / window_width, 0., 1.);
        ypos = std::clamp(ypos / window_height, 0., 1.);

        // query cursor state
        int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        pressing = (state == GLFW_PRESS);

        // debug output
        std::cout << "x:" << xpos << " y:" << ypos << " pressed:" << pressing << std::endl;
    }
};

// QUERY WHETHER CLOSE WAS REQUESTED

bool GUI::exit_requested()
{
    return exit_pressed || glfwWindowShouldClose(window);
}

// CONSTRUCTOR

GUI::GUI(const int N, int init_w, int init_h, std::function<void()> on_failure, bool enable_vsync)
{
    // save parameters
    this->N = N;
    this->window_width = init_w;
    this->window_height = init_h;

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
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        on_failure();
        return;
    }
    // set current context and register resize handler
    glfwMakeContextCurrent(window);
    glfwSwapInterval(static_cast<int>(enable_vsync));
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // SET UP IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    this->io = &ImGui::GetIO();
    (void)io;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows
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
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        on_failure();
        return;
    }

    // set clear colour and initial viewport
    glViewport(0, 0, init_w, init_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // enable point rendering features
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // create shader program
    shader_program = compile_shader();

    // create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // expose VBO to CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    // create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

float4 *GUI::get_buffer()
{
    // process inputs
    glfw_process_input();
    // clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // map the buffer for CUDA access
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));

    float4 *vertices = nullptr;
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&vertices, &num_bytes, cuda_vbo_resource));

    return vertices;
}

void GUI::show_updated()
{
    // unmap vertex buffer from CUDA for use by OpenGL
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    // OpenGL rendering commands
    glUseProgram(shader_program);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, N);
    glBindVertexArray(0);

    // poll for window events
    glfwPollEvents();

    // update ImGUI
    imgui_draw();

    // swap back and front buffers
    glfwSwapBuffers(window);
}

void GUI::imgui_draw()
{
    // update the FPS count in the window title using the ImGui framerate counter
    const int max_fps_str_size{25};
    char fps_str[max_fps_str_size];
    snprintf(fps_str, max_fps_str_size, "REYN - GUI %.1f FPS", io->Framerate);
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
    ImGui::Text("Frame time %.3fµs (%.1f FPS)", 1000.0f / io->Framerate, io->Framerate);
    if (ImGui::Button("Exit"))
        exit_pressed = true;

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
    // clean up CUDA binding of vbo
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    // clean up OpenGL
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shader_program);
    // clean up GLFW
    glfwTerminate();
}
