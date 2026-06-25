#pragma once

#include <opencv2/opencv.hpp>
#include "Octreegrid.h"

namespace PoseGenerator {

/**
 * @brief Opens an interactive GUI to select scan positions in the point cloud,
 *        then returns the raw 3D positions (no pose generation).
 */
std::vector<cv::Point3f> getPositions(std::unordered_map<int, OctreeGrid::Block> grid);

/**
 * @brief Opens an interactive GUI to select scan positions in the point cloud,
 *        then converts each position into multiple camera poses
 *        (tilted faces with random Z-rotation).
 */
std::vector<cv::Matx44d> getPoses(std::unordered_map<int, OctreeGrid::Block> grid);

}
