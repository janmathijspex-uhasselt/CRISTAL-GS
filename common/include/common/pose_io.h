#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

/**
 * @brief Save a list of camera poses (world-to-camera Matx44d) to a TUM-format file.
 *
 * Each line: timestamp tx ty tz qx qy qz qw
 * Poses are stored as camera-to-world (inverted from the w2c input).
 */
inline bool savePoses(const std::filesystem::path& path,
                      const std::vector<cv::Matx44d>& poses)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open pose file for writing: " << path << std::endl;
        return false;
    }

    ofs << "# Saved camera poses (TUM format: timestamp tx ty tz qx qy qz qw)\n";
    ofs << "# Poses are camera-to-world\n";
    ofs << std::fixed << std::setprecision(15);

    for (size_t i = 0; i < poses.size(); ++i) {
        // w2c -> c2w
        cv::Matx44d c2w = poses[i].inv();

        cv::Vec3d t(c2w(0, 3), c2w(1, 3), c2w(2, 3));
        cv::Matx33d R;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R(r, c) = c2w(r, c);
        cv::Quatd q = cv::Quatd::createFromRotMat(R);

        double timestamp = static_cast<double>(i) / 30.0;
        ofs << std::setprecision(9) << timestamp << " "
            << t[0] << " " << t[1] << " " << t[2] << " "
            << q.x << " " << q.y << " " << q.z << " " << q.w << "\n";
    }

    ofs.close();
    std::cout << "Saved " << poses.size() << " poses to: " << path << std::endl;
    return true;
}

/**
 * @brief Load camera poses from a TUM-format file.
 *
 * Returns world-to-camera (w2c) Matx44d poses (inverted from the c2w file).
 */
inline std::vector<cv::Matx44d> loadPoses(const std::filesystem::path& path)
{
    std::vector<cv::Matx44d> poses;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open pose file: " << path << std::endl;
        return poses;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;

        cv::Quatd q(qw, qx, qy, qz);
        cv::Matx33d R = q.toRotMat3x3();

        // Build c2w
        cv::Matx44d c2w = cv::Matx44d::eye();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                c2w(r, c) = R(r, c);
        c2w(0, 3) = tx;
        c2w(1, 3) = ty;
        c2w(2, 3) = tz;

        // Return as w2c
        poses.push_back(c2w.inv());
    }

    std::cout << "Loaded " << poses.size() << " poses from: " << path << std::endl;
    return poses;
}
