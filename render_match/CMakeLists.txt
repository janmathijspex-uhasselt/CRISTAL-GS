cmake_minimum_required(VERSION 3.5)

project(render_match LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(RTRenderer REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(3rdparty/xfeat_cpp)
include_directories(3rdparty/xfeat_cpp/include)

add_executable(render_match
    main.cpp
    feature/featuretracker.h
    feature/xfeattracker.h feature/xfeattracker.cpp
    trackingstate.h
    utils.h utils.cpp
    localizer.h localizer.cpp
    map/map.h
    map/xfeatmap.h map/xfeatmap.cpp
    map/cubemapgenerator.h map/cubemapgenerator.cpp
    feature/xfeatframe.h
    data/trajectoryframe.h data/trajectoryframe.cpp
    )

target_include_directories(render_match PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(render_match
    RTRenderer::RTRenderer
    DeviceFactory
    common_lib
    xfeat
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

install(TARGETS render_match
    RUNTIME DESTINATION bin
)
