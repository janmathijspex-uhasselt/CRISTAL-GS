#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

/**
 * @brief Writes a standardised metadata comment header to a trajectory file.
 *
 * All fields are optional — pass empty strings for fields that don't apply.
 */
inline void writeTrajectoryHeader(std::ofstream& ofs,
                                  const std::string& pipeline,
                                  const std::string& backend,
                                  const std::string& point_cloud,
                                  const std::string& video,
                                  const std::string& rendering_mode,
                                  const std::string& extra = "")
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);

    ofs << "# CRISTAL trajectory file\n";
    ofs << "# created: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";

    if (!pipeline.empty())       ofs << "# pipeline: "       << pipeline << "\n";
    if (!backend.empty())        ofs << "# backend: "        << backend << "\n";
    if (!point_cloud.empty())    ofs << "# point_cloud: "    << point_cloud << "\n";
    if (!video.empty())          ofs << "# video: "          << video << "\n";
    if (!rendering_mode.empty()) ofs << "# rendering_mode: " << rendering_mode << "\n";
    if (!extra.empty())          ofs << "# " << extra << "\n";

    ofs << "# format: TUM (timestamp tx ty tz qx qy qz qw)\n";
    ofs << "#\n";
}
