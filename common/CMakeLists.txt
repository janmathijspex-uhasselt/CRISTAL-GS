cmake_minimum_required(VERSION 3.5)
project(common_lib)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(RTRenderer REQUIRED)

add_library(common_lib STATIC
    src/camera_utils.cpp
    src/geometry.cpp
    src/pose_generator.cpp
)

target_include_directories(common_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(common_lib PUBLIC
    RTRenderer::RTRenderer
    DeviceFactory
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
