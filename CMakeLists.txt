cmake_minimum_required(VERSION 3.5)

project(CRISTAL-GS LANGUAGES CXX)

# change cpp version to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/DeviceFactory)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/DeviceFactory/include)

set(BUILD_SHARED_LIBS off)

add_subdirectory(${CMAKE_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_SOURCE_DIR}/prebuild_localize)
# add_subdirectory(${CMAKE_SOURCE_DIR}/render_trajectory)
# add_subdirectory(${CMAKE_SOURCE_DIR}/render_match)
