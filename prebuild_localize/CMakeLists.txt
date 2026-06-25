cmake_minimum_required(VERSION 3.5)
project(Prebuild_localize)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===================================================================
# Find common dependencies
# ===================================================================

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(RTRenderer REQUIRED)
find_package(PCL REQUIRED)

# ===================================================================
# Find stella_vslam (optional)
# ===================================================================

find_package(stella_vslam QUIET)
if(stella_vslam_FOUND)
    set(STELLA_AVAILABLE TRUE)
    message(STATUS "Stella VSLAM found - will build stella executables")
else()
    set(STELLA_AVAILABLE FALSE)
    message(STATUS "Stella VSLAM not found - stella executables disabled")
endif()

# ===================================================================
# Find ORB-SLAM3 (optional)
# ===================================================================

find_package(ORB_SLAM3 QUIET)
if(NOT ORB_SLAM3_FOUND)
    # Fallback to manual path if not found via find_package
    set(ORB_SLAM3_ROOT_DIR "/home/jvanherck/lib/ORB_SLAM3" CACHE PATH "Path to ORB-SLAM3")
    if(EXISTS "${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so")
        set(ORB_SLAM3_INCLUDE_DIRS 
            ${ORB_SLAM3_ROOT_DIR}/include
            ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
            ${ORB_SLAM3_ROOT_DIR}/Thirdparty/Sophus
            ${ORB_SLAM3_ROOT_DIR}
        )
        set(ORB_SLAM3_LIBRARIES ${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so)
        set(DBOW2_LIBRARY ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)
        set(G2O_LIBRARY ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib/libg2o.so)
        set(ORB_SLAM3_AVAILABLE TRUE)
        message(STATUS "ORB-SLAM3 found at ${ORB_SLAM3_ROOT_DIR} - will build orb executables")
    else()
        set(ORB_SLAM3_AVAILABLE FALSE)
        message(STATUS "ORB-SLAM3 not found - orb executables disabled")
    endif()
else()
    set(ORB_SLAM3_AVAILABLE TRUE)
    message(STATUS "ORB-SLAM3 found via find_package - will build orb executables")
endif()

# ===================================================================
# Additional dependencies for ORB-SLAM3
# ===================================================================

if(ORB_SLAM3_AVAILABLE)
    find_package(Pangolin QUIET)
    find_package(Boost REQUIRED COMPONENTS serialization)
    find_package(OpenSSL REQUIRED)
endif()

# ===================================================================
# Check that at least one backend is available
# ===================================================================

if(NOT STELLA_AVAILABLE AND NOT ORB_SLAM3_AVAILABLE)
    message(FATAL_ERROR "At least one SLAM backend (stella_vslam or ORB-SLAM3) must be available!")
endif()

# ===================================================================
# Common source files (shared by all executables)
# ===================================================================

set(COMMON_SOURCES_MAP
    src/map/main.cpp
    src/utils.cpp
    include/utils.h
    include/map/frame_cache.h
)

set(COMMON_SOURCES_LOCALIZE
    src/localize/main.cpp
    src/utils.cpp
    include/utils.h
)

# ===================================================================
# Build Stella VSLAM executable
# ===================================================================

if(STELLA_AVAILABLE)
# =================
# Map executable
# =================	
    add_executable(map_stella
        ${COMMON_SOURCES_MAP}
        include/map/stella_backend.h
        include/map/pcl_gs_alignment.h
        src/map/stella_backend.cpp
    )

    target_compile_features(map_stella PRIVATE cxx_std_17)
    target_compile_definitions(map_stella PRIVATE STELLA_AVAILABLE ${PCL_DEFINITIONS})

    target_include_directories(map_stella PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        include/
    )

    target_link_libraries(map_stella PRIVATE
        RTRenderer::RTRenderer
        DeviceFactory
        common_lib
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        stella_vslam::stella_vslam
    )

    install(TARGETS map_stella RUNTIME DESTINATION bin)

# =======================
# Localize executable
# =======================
    find_package(iridescence_viewer REQUIRED)

    add_executable(localize_stella
        ${COMMON_SOURCES_LOCALIZE}
    )

    target_compile_features(localize_stella PRIVATE cxx_std_17)
    target_compile_definitions(localize_stella PRIVATE 
        STELLA_AVAILABLE
        HAVE_IRIDESCENCE_VIEWER
    )

    target_include_directories(localize_stella PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        include/
    )

    target_link_libraries(localize_stella PRIVATE
        DeviceFactory
        common_lib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        stella_vslam::stella_vslam
        iridescence_viewer::iridescence_viewer
    )

    install(TARGETS localize_stella RUNTIME DESTINATION bin)
endif()

# ===================================================================
# Build ORB-SLAM3 executable
# ===================================================================

if(ORB_SLAM3_AVAILABLE)
# =================
# Map executable
# =================
    add_executable(map_orb
        ${COMMON_SOURCES_MAP}
        include/map/orbslam3_backend.h
        src/map/orbslam3_backend.cpp
    )

    target_compile_features(map_orb PRIVATE cxx_std_17)
    target_compile_options(map_orb PRIVATE -march=native)
    target_compile_definitions(map_orb PRIVATE ORBSLAM3_AVAILABLE)

    target_include_directories(map_orb PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${ORB_SLAM3_INCLUDE_DIRS}
        include/
    )

    target_link_libraries(map_orb PRIVATE
        RTRenderer::RTRenderer
        DeviceFactory
        common_lib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${ORB_SLAM3_LIBRARIES}
        ${DBOW2_LIBRARY}
        ${G2O_LIBRARY}
        Boost::serialization
        OpenSSL::Crypto
    )

    if(Pangolin_FOUND)
        target_link_libraries(map_orb PRIVATE ${Pangolin_LIBRARIES})
    endif()

    install(TARGETS map_orb RUNTIME DESTINATION bin)
    
# =======================
# Localize executable
# =======================
    add_executable(localize_orb
        ${COMMON_SOURCES_LOCALIZE}
    )

    target_compile_features(localize_orb PRIVATE cxx_std_17)
    target_compile_options(localize_orb PRIVATE -march=native)
    target_compile_definitions(localize_orb PRIVATE ORBSLAM3_AVAILABLE)

    target_include_directories(localize_orb PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${ORB_SLAM3_INCLUDE_DIRS}
        include/
    )

    target_link_libraries(localize_orb PRIVATE
        DeviceFactory
        common_lib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${ORB_SLAM3_LIBRARIES}
        ${DBOW2_LIBRARY}
        ${G2O_LIBRARY}
    )

    if(Pangolin_FOUND)
        target_link_libraries(localize_orb PRIVATE ${Pangolin_LIBRARIES})
    endif()

    install(TARGETS localize_orb RUNTIME DESTINATION bin)
endif()


# ===================================================================
# Print configuration summary
# ===================================================================

message(STATUS "")
message(STATUS "=== Configuration ===")
message(STATUS "  stella: ${STELLA_AVAILABLE}")
message(STATUS "  orb:    ${ORB_SLAM3_AVAILABLE}")
message(STATUS "")
