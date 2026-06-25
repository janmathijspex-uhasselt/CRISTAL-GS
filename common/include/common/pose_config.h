#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

constexpr int CONFIG_WINDOW_WIDTH       = 1920;
constexpr int CONFIG_WINDOW_HEIGHT      = 1080;
constexpr char CONFIG_WINDOW_NAME[]     = "Ortho";
constexpr float CONFIG_DILATE_DISTANCE  = 0.05f;
constexpr float CONFIG_SLICE_MIN_FLOAT  = 0.5f;
constexpr float CONFIG_SLICE_MAX_FLOAT  = 1.70f;
constexpr int   CONFIG_TRACKBAR_RES     = 100;

const std::vector<double> CONFIG_TILT_ANGLES_DEG = { 20.0, -5.0 };
constexpr double CONFIG_RANDOM_ROTATION_MIN  = 0.0;
constexpr double CONFIG_RANDOM_ROTATION_MAX  = 2.0 * CV_PI;
