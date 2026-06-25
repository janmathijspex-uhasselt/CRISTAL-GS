#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iostream>
#include "RTRenderer/project_cloud.h"
#include "RTRenderer/cloudreader.h"
#include "RTRenderer/CameraCalibration.h"
#include "DeviceFactory.h"
#include "Device.h"
#include "common/camera_utils.h"

struct TrajectoryEntry {
    double timestamp;
    cv::Matx44d pose;
};

TrajectoryEntry parseTrajectoryLine(const std::string& line) {
    std::istringstream iss(line);
    TrajectoryEntry entry;
    double qw, qx, qy, qz, tx, ty, tz;
    int camera_id;

    iss >> entry.timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

    cv::Quatd q(qw, qx, qy, qz);
    cv::Matx33d R = q.toRotMat3x3();

    entry.pose = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i) {
        entry.pose(i, 0) = R(i, 0);
        entry.pose(i, 1) = R(i, 1);
        entry.pose(i, 2) = R(i, 2);
        entry.pose(i, 3) = (i == 0 ? tx : (i == 1 ? ty : tz));
    }
    entry.pose = entry.pose.inv();

    return entry;
}

std::vector<TrajectoryEntry> readOrderedTrajectoryFile(const std::string& filename) {
    std::vector<TrajectoryEntry> traj_list;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        TrajectoryEntry entry = parseTrajectoryLine(line);
        traj_list.push_back(entry);
    }

    return traj_list;
}

cv::Mat createOverlayPanel(
    const cv::Mat& img1,
    const cv::Mat& img2,
    double alpha = 0.5)
{
    CV_Assert(img1.size() == img2.size());
    CV_Assert(img1.type() == img2.type());

    double beta = 1.0 - alpha;

    cv::Mat overlay1;
    cv::addWeighted(img1, alpha, img2, beta, 0.0, overlay1);

    cv::Mat panel;
    cv::hconcat(std::vector<cv::Mat>{img1, img2, overlay1}, panel);

    return panel;
}


int main(int argc, char**argv)
{
    if(argc < 6)
    {
        std::cerr << "Missing required parameter:" << std::endl;
        std::cerr << "render_trajectory folder cloud video_name calibration_file trajectory_file" << std::endl;
        std::cerr << "folder: should point to folder containing all data" << std::endl;
        std::cerr << "cloud: filename of the point cloud relative to folder" << std::endl;
        std::cerr << "video_name: filename of the video to localize relative to folder" << std::endl;
        std::cerr << "calibration_file: filename of the calibration file relative to folder" << std::endl;
        std::cerr << "trajectory_file: filename of the estimated trajectory file relative to folder" << std::endl;
        exit(-1);
    }

    std::filesystem::path folder = argv[1];
    std::filesystem::path video_name = argv[3];
    std::filesystem::path path_to_cloud = folder / std::filesystem::path(argv[2]);
    std::filesystem::path video_path = folder / video_name;
    std::filesystem::path calib_path = folder / std::filesystem::path(argv[4]);
    std::filesystem::path traj_path = folder / std::filesystem::path(argv[5]);

    DeviceFactory::DeviceFactory df;
    auto device_cam = df.createDevices("CVVideoCapture", video_path.string(), calib_path);
    CameraCalibration calibration = device_cam->getCalibration();

    // Setup undistortion (handles both fisheye and pinhole distortion)
    CameraUndistorter undistorter(calibration);

    // Get effective calibration aligned to 16-pixel boundary (required by renderer)
    CameraCalibration effectiveCalib = undistorter.isActive()
        ? undistorter.getUndistortedCalibration() : calibration;
    effectiveCalib = getAlignedCalibration(effectiveCalib, 16);

    auto grid = CloudReader::loadCloud(path_to_cloud, folder / ".cache");
    std::shared_ptr<ProjectCloud> projector = std::make_shared<ProjectCloud>(grid, "trt_" + std::to_string(effectiveCalib.getWidth()) + "x" + std::to_string(effectiveCalib.getHeight()) + ".ts");

    auto traj = readOrderedTrajectoryFile(traj_path.string());
    int current_traj_idx = 0;

    std::filesystem::create_directories(folder / "visualization");

    while(true)
    {
        cv::Mat frame; double ts;
        device_cam->captureImages(frame, ts);

        while(current_traj_idx < traj.size() && traj[current_traj_idx].timestamp < ts - 0.01)
        {
            ++current_traj_idx;
        }

        if(frame.rows == 0)
        {
            std::cerr << "Failed to capture the next frame" << std::endl;
            break;
        }

        if(traj[current_traj_idx].timestamp <= ts + 0.01 && traj[current_traj_idx].timestamp >= ts - 0.01)
        {
            if (frame.size() != cv::Size(calibration.getWidth(), calibration.getHeight())) 
            {
                cv::resize(frame, frame, cv::Size(calibration.getWidth(), calibration.getHeight()));
            }

            // Undistort and center-crop to aligned dimensions
            undistorter.undistort(frame);
            centerCrop(frame, effectiveCalib);

            cv::Mat render = cv::Mat(cv::Size(effectiveCalib.getWidth(), effectiveCalib.getHeight()), CV_8UC3);
            projector->computeFull(effectiveCalib, traj[current_traj_idx].pose, &render, nullptr);

            cv::Mat blended = createOverlayPanel(frame, render);
            cv::imwrite(folder / "visualization" / (std::to_string(ts) + ".png"), blended);
            ++current_traj_idx;
        }
        else if (traj[current_traj_idx].timestamp < ts - 0.01)
        {
            std::cerr << "Warning: No trajectory entry found for frame at timestamp: " << std::fixed << std::setprecision(15) << ts << "s (closest entry at " << traj[current_traj_idx].timestamp << "s)" << std::endl;
        }
        else
        {
            std::cerr << "Warning: Trajectory entry timestamp: " << std::fixed << std::setprecision(15) << traj[current_traj_idx].timestamp << "s is ahead of frame timestamp: " << ts << "s" << std::endl;
        }
    }

    return 0;
}
