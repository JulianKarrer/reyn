# use recent CMAKE version and define project name and languages
cmake_minimum_required(VERSION 3.6)
project(reyn LANGUAGES CUDA CXX C ) # C is required for glad

# check if cuda is available
include(CheckLanguage)
check_language(CUDA)

# set optimization level
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_BUILD_TYPE_RELEASE "Release" CACHE STRING "Release build type")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")

# use the scccache compiler wrapper for caching to speed up compilation
find_program(SCCACHE_PROGRAM sccache)
if (SCCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    message(STATUS "Sccache found: ${SCCACHE_PROGRAM}")
else ()
    message(STATUS "Sccache not found")
endif ()

# set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set CUDA compute capability, 89 for 4060Ti
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
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# fetch ImGUI
FetchContent_Populate(imgui
  URL https://github.com/ocornut/imgui/archive/docking.zip
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/imgui
)

add_library(imgui_glfw STATIC
  build/_deps/imgui/imgui.cpp
  build/_deps/imgui/imgui_draw.cpp
  build/_deps/imgui/imgui_demo.cpp
  build/_deps/imgui/imgui_tables.cpp
  build/_deps/imgui/imgui_widgets.cpp

  build/_deps/imgui/backends/imgui_impl_glfw.cpp
  build/_deps/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_glfw
PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/build/_deps/imgui
  ${CMAKE_CURRENT_LIST_DIR}/build/_deps/imgui/backends
)

target_link_libraries(imgui_glfw PUBLIC glfw)

find_package(X11 REQUIRED)
target_link_libraries(imgui_glfw PUBLIC ${X11_LIBRARIES})

# fetch GLM
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

# get thrust
find_package(Thrust REQUIRED)
thrust_create_target(Thrust)

# fetch doctest
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)

# define executable
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(REYN_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/gui.cpp
    ${SRC_DIR}/particles.cpp
    ${SRC_DIR}/scene.cu
    ${SRC_DIR}/kernels.cu
    ${SRC_DIR}/solvers/SESPH.cu
    )
set(REYN_HEADERS
    ${SRC_DIR}/common.h
    ${SRC_DIR}/gui.h
    ${SRC_DIR}/particles.h
    ${SRC_DIR}/scene.cuh
    ${SRC_DIR}/vector_helper.cuh
    ${SRC_DIR}/kernels.cuh
    ${SRC_DIR}/solvers/SESPH.cuh
    )
add_executable(${PROJECT_NAME}
    ${REYN_SOURCES}
    ${REYN_HEADERS}
)

# disable doctest in normal compilation target
target_compile_definitions(${PROJECT_NAME} PRIVATE DOCTEST_CONFIG_DISABLE)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
)

# handle cuda compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ensure cuda runtime is there
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# link libraries (GLFW, OpenGL, ...)
target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC 
    glfw
    imgui_glfw 
    glad
    glm::glm
    OpenGL::GL
    doctest::doctest
    Thrust
)

# TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file(GLOB_RECURSE USER_TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp" "${CMAKE_SOURCE_DIR}/test/*.cu")

set(REYN_SOURCES_NO_MAIN ${REYN_SOURCES})
list(REMOVE_ITEM REYN_SOURCES_NO_MAIN "${SRC_DIR}/main.cpp")
add_executable(tests
    test/runner.cpp
    ${REYN_SOURCES_NO_MAIN}
    ${REYN_HEADERS}
    ${USER_TEST_SOURCES}
)

set_target_properties(tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(tests PRIVATE
    ${SRC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(tests PRIVATE
    doctest::doctest
    CUDA::cudart
    ${X11_LIBRARIES}
    glfw
    imgui_glfw
    glad
    glm::glm
    OpenGL::GL
)

enable_testing()
add_test(NAME unit_tests COMMAND tests)


# MISC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# supress GLM volatile warning
foreach(tgt ${PROJECT_NAME} tests)
    if(TARGET ${tgt})
        target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=3012>)
        target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-volatile> )
    endif()
endforeach()
