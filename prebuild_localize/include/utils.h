#pragma once

#include <filesystem>
#include <string>
#include "DeviceFactory/CameraCalibration.h"

/**
 * @brief Generate ORB-SLAM3 config file from template and camera intrinsics
 * Always outputs PinHole camera type. Fisheye frames must be undistorted before SLAM.
 */
bool createORBSLAM3Config(const std::filesystem::path& template_file,
                         const std::filesystem::path& intrinsics_file,
                         const std::filesystem::path& output_file, bool use_map = true);

/**
 * @brief Generate Stella VSLAM config file from template and camera intrinsics
 * Always outputs perspective camera model. Fisheye frames must be undistorted before SLAM.
 */
bool createStellaConfig(const std::filesystem::path& template_file,
                        const std::filesystem::path& intrinsics_file,
                        const std::filesystem::path& output_file);
