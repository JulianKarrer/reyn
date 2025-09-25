# use recent CMAKE version and define project name and languages
cmake_minimum_required(VERSION 3.16)
project(reyn LANGUAGES CUDA CXX C ) # C is required for glad

# check if cuda is available
include(CheckLanguage)
check_language(CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set CUDA compute capability, 88 for 4060Ti
# https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 89)

# fetch content
include(FetchContent)

# fetch GLFW 3.4
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG "3.4"
)
FetchContent_MakeAvailable(glfw)

# set up GLAD
add_library(glad STATIC deps/glad/src/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/deps/glad/include)

# require OpenGL
find_package(OpenGL REQUIRED)

# define executable
add_executable(${PROJECT_NAME} 
    kernel.cu
    gui.cpp
    main.cpp
)

# handle cuda compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# link libraries (GLFW, OpenGL)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    glad
    glfw 
    OpenGL::GL
)

# ensure cuda runtime is there
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)