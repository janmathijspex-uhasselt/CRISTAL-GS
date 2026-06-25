#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>

/**
 * @brief Converts a cv::Matx44d to Eigen::Matrix4d.
 */
Eigen::Matrix4d convertToEigen(cv::Matx44d mat);

/**
 * @brief Computes a world-to-camera (look-at) transformation matrix.
 */
cv::Matx44d lookAt(const cv::Vec3d& position, const cv::Vec3d& target, const cv::Vec3d& up);
