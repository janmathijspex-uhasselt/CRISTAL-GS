#include "utils.h"
#include <opencv2/core/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

bool createORBSLAM3Config(const std::filesystem::path& template_file,
                         const std::filesystem::path& intrinsics_file,
                         const std::filesystem::path& output_file, bool use_map)
{
    // Load camera calibration
    CameraCalibration calib;
    if (!calib.loadCalibration(intrinsics_file.string())) {
        std::cerr << "Failed to load camera intrinsics from: " << intrinsics_file << std::endl;
        return false;
    }

    // Read template file
    std::ifstream template_stream(template_file);
    if (!template_stream.is_open()) {
        std::cerr << "Failed to open template file: " << template_file << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << template_stream.rdbuf();
    std::string content = buffer.str();
    template_stream.close();

    if(!use_map)
    {
        // set save map file to directory containing the output_file with name "default.osa"
        std::filesystem::path map_output_path = output_file.parent_path() / "default.osa";
        size_t map_file_pos = content.find("System.SaveAtlasToFile: \"\"");
        if (map_file_pos != std::string::npos) {
            content.replace(map_file_pos, 26, "System.SaveAtlasToFile: \"" + map_output_path.string() + "\"");

            std::cout << "Set ORB-SLAM3 config to save map to: " << map_output_path << std::endl;
        }
    }

    // Create camera parameters section
    std::stringstream camera_params;
    camera_params << std::fixed << std::setprecision(10);
    
    // Get calibration parameters
    double fx = calib.getFocalLengthX();
    double fy = calib.getFocalLengthY();
    double cx = calib.getPrincipalPointX();
    double cy = calib.getPrincipalPointY();
    int width = calib.getWidth();
    int height = calib.getHeight();
    const auto& dists = calib.getDistortionParameters();

    camera_params << "Camera1.fx: " << fx << "\n";
    camera_params << "Camera1.fy: " << fy << "\n";
    camera_params << "Camera1.cx: " << cx << "\n";
    camera_params << "Camera1.cy: " << cy << "\n";

    // Always PinHole model: k1, k2, p1, p2, k3
    // Fisheye frames are undistorted before being fed to SLAM
    camera_params << "Camera1.k1: " << 0.0 << "\n";
    camera_params << "Camera1.k2: " << 0.0 << "\n";
    camera_params << "Camera1.p1: " << 0.0 << "\n";
    camera_params << "Camera1.p2: " << 0.0 << "\n";
    camera_params << "Camera1.k3: " << 0.0 << "\n";
    camera_params << "\nCamera.width: " << width << "\n";
    camera_params << "Camera.height: " << height << "\n";

    // Find insertion point using clearer markers
    size_t start_marker = content.find("# ===== CAMERA PARAMETERS START =====");
    size_t end_marker = content.find("# ===== CAMERA PARAMETERS END =====");
    
    if (start_marker != std::string::npos && end_marker != std::string::npos) {
        // Move to after the start marker line
        start_marker = content.find('\n', start_marker);
        if (start_marker != std::string::npos) {
            start_marker += 1;
            // Erase the content between markers and insert new parameters
            content.erase(start_marker, end_marker - start_marker);
            content.insert(start_marker, camera_params.str() + "\n\n");
        }
    } else {
        std::cerr << "Warning: Could not find camera parameter markers in template file" << std::endl;
        std::cerr << "Camera parameters may not be inserted correctly" << std::endl;
    }

    // Write output file
    std::ofstream output_stream(output_file);
    if (!output_stream.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    output_stream << content;
    output_stream.close();

    std::cout << "ORB-SLAM3 config file created successfully: " << output_file << std::endl;
    return true;
}

bool createStellaConfig(const std::filesystem::path& template_file,
                        const std::filesystem::path& intrinsics_file,
                        const std::filesystem::path& output_file)
{
    // Load camera calibration
    CameraCalibration calib;
    if (!calib.loadCalibration(intrinsics_file.string())) {
        std::cerr << "Failed to load camera intrinsics from: " << intrinsics_file << std::endl;
        return false;
    }

    // Read template file
    std::ifstream template_stream(template_file);
    if (!template_stream.is_open()) {
        std::cerr << "Failed to open template file: " << template_file << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << template_stream.rdbuf();
    std::string content = buffer.str();
    template_stream.close();

    // Create camera parameters section
    // Always perspective model - fisheye frames are undistorted before being fed to SLAM
    std::stringstream camera_params;
    camera_params << std::fixed << std::setprecision(10);
    
    // Get calibration parameters
    double fx = calib.getFocalLengthX();
    double fy = calib.getFocalLengthY();
    double cx = calib.getPrincipalPointX();
    double cy = calib.getPrincipalPointY();
    int width = calib.getWidth();
    int height = calib.getHeight();
    const auto& dists = calib.getDistortionParameters();

    camera_params << "  fx: " << fx << "\n";
    camera_params << "  fy: " << fy << "\n";
    camera_params << "  cx: " << cx << "\n";
    camera_params << "  cy: " << cy << "\n";
    camera_params << "\n";
    camera_params << "  k1: " << 0.0 << "\n";
    camera_params << "  k2: " << 0.0 << "\n";
    camera_params << "  p1: " << 0.0 << "\n";
    camera_params << "  p2: " << 0.0 << "\n";
    camera_params << "  k3: " << 0.0 << "\n";
    camera_params << "\n";
    camera_params << "  fps: 30.0\n";
    camera_params << "  cols: " << width << "\n";
    camera_params << "  rows: " << height << "\n";

    // Find insertion point using clearer markers
    size_t start_marker = content.find("# ===== CAMERA PARAMETERS START =====");
    size_t end_marker = content.find("# ===== CAMERA PARAMETERS END =====");
    
    if (start_marker != std::string::npos && end_marker != std::string::npos) {
        // Move to after the start marker line
        start_marker = content.find('\n', start_marker);
        if (start_marker != std::string::npos) {
            start_marker += 1;
            // Erase the content between markers and insert new parameters
            content.erase(start_marker, end_marker - start_marker);
            content.insert(start_marker, camera_params.str());
        }
    } else {
        std::cerr << "Warning: Could not find camera parameter markers in template file" << std::endl;
        std::cerr << "Camera parameters may not be inserted correctly" << std::endl;
    }

    // Write output file
    std::ofstream output_stream(output_file);
    if (!output_stream.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    output_stream << content;
    output_stream.close();

    std::cout << "Stella VSLAM config file created successfully: " << output_file << std::endl;
    return true;
}