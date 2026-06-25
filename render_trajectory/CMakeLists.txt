cmake_minimum_required(VERSION 3.5)

project(render_trajectory LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(RTRenderer REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(render_trajectory
    main.cpp
    )

target_include_directories(render_trajectory PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(render_trajectory
    RTRenderer::RTRenderer
    DeviceFactory
    common_lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

install(TARGETS render_trajectory
    RUNTIME DESTINATION bin
)