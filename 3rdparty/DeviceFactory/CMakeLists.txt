cmake_minimum_required(VERSION 3.13)
project(DeviceFactory VERSION 0.1.0 LANGUAGES CXX)

# Find dependencies
find_package(OpenCV 4 REQUIRED)

option(WITH_REALSENSE2 "Use RealSense2 SDK" ON)
if(WITH_REALSENSE2)
    find_package(realsense2 QUIET CONFIG)

    if(NOT realsense2_FOUND)
        message(STATUS "librealsense2 not found — fetching and building from source")
        include(FetchContent)

        # Declare librealsense
        FetchContent_Declare(
            librealsense
            GIT_REPOSITORY https://github.com/realsenseai/librealsense.git
            GIT_TAG v2.57.4
        )

        set(BUILD_EXAMPLES OFF CACHE BOOL "Disable building examples")
        set(BUILD_GRAPHICAL_EXAMPLES OFF CACHE BOOL "Disable graphical examples")
        set(BUILD_WITH_CUDA OFF CACHE BOOL "Disable CUDA support")
        set(BUILD_WITH_OPENMP OFF CACHE BOOL "Disable OpenMP support")
        set(BUILD_UNIT_TESTS OFF CACHE BOOL "Disable unit tests")
        set(INSTALL_CPP_EXAMPLES OFF CACHE BOOL "Do not install C++ examples")

        FetchContent_MakeAvailable(librealsense)
    endif()
endif()

option(WITH_FFMPEG "Include FFMPEG decoder" OFF)
if(WITH_FFMPEG)
    find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h REQUIRED)
    find_library(AVCODEC_LIBRARY avcodec REQUIRED)

    find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h REQUIRED)
    find_library(AVFORMAT_LIBRARY avformat REQUIRED)

    find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h REQUIRED)
    find_library(AVUTIL_LIBRARY avutil REQUIRED)

    find_library(SWSCALE_LIBRARY swscale REQUIRED)
endif()

# Collect source files
set(DEVICE_FACTORY_SOURCES
    src/Device.cpp
    src/DeviceFactory.cpp
    src/CameraCalibration.cpp
    src/CVVideoCaptureDevice.cpp
    src/CVImageCaptureDevice.cpp
)

if(WITH_REALSENSE2)
    list(APPEND DEVICE_FACTORY_SOURCES src/RealSense2Device.cpp)
endif()

if(WITH_FFMPEG)
    list(APPEND DEVICE_FACTORY_SOURCES src/FFMPEGDevice.cpp)
endif()

# Create shared library
add_library(DeviceFactory SHARED ${DEVICE_FACTORY_SOURCES} ${DEVICE_FACTORY_HEADERS})

# Include directories
target_include_directories(DeviceFactory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/DeviceFactory
        $<INSTALL_INTERFACE:include/DeviceFactory>
        ${OpenCV_INCLUDE_DIRS}
)

# Compile definitions
if(WITH_REALSENSE2)
    target_compile_definitions(DeviceFactory PUBLIC WITH_REALSENSE2)
endif()

if(WITH_FFMPEG)
    target_compile_definitions(DeviceFactory PUBLIC WITH_FFMPEG)
endif()

# Link dependencies
target_link_libraries(DeviceFactory
    PUBLIC
        ${OpenCV_LIBS}
)

if(WITH_REALSENSE2)
    target_link_libraries(DeviceFactory PUBLIC realsense2)
endif()

if(WITH_FFMPEG)
    target_include_directories(DeviceFactory PRIVATE
        ${AVCODEC_INCLUDE_DIR}
        ${AVFORMAT_INCLUDE_DIR}
        ${AVUTIL_INCLUDE_DIR}
    )
    target_link_libraries(DeviceFactory
        PRIVATE
            ${AVCODEC_LIBRARY}
            ${AVFORMAT_LIBRARY}
            ${AVUTIL_LIBRARY}
            ${SWSCALE_LIBRARY}
    )
endif()

include(GNUInstallDirs)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)


file(GLOB DEVICE_FACTORY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/DeviceFactory/*.h")
install(FILES ${DEVICE_FACTORY_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DeviceFactory
)

install(TARGETS DeviceFactory
    EXPORT DeviceFactoryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DeviceFactory
)

install(EXPORT DeviceFactoryTargets
    FILE DeviceFactoryTargets.cmake
    NAMESPACE DeviceFactory::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceFactory
)


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/DeviceFactoryConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeviceFactoryConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DeviceFactoryConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceFactory
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DeviceFactoryConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/DeviceFactoryConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceFactory
)


