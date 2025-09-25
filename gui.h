#ifndef GUI_H_
#define GUI_H_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <functional>
#include "common.h"

/// @brief Object managing a graphical user interface.
/// Uses GLFW for cross-platform windowing and sets up
/// OpenGL <-> CUDA interop for displaying the contents
/// of a buffer of positions managed by CUDA as a point
/// cloud.
class GUI
{
public:
    GUI(const int N, int init_w, int init_h, std::function<void()> on_failure);
    ~GUI();
    bool exit_requested();
    float4 *get_buffer();
    void show_updated();
    int window_width, window_height;

private:
    int N;
    // GLFW resources
    GLFWwindow *window;
    // OpenGL resources
    GLuint shader_program, vao, vbo;
    // CUDA resources
    cudaGraphicsResource *cuda_vbo_resource = nullptr;
    // internal functions for setup and events
    GLuint compile_shader();
    void glfw_process_input();
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
};

#endif // GUI_H_