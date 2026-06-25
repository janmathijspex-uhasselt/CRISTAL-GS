#include "common/camera_utils.h"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>

CameraUndistorter::CameraUndistorter(const CameraCalibration& calib)
    : m_newCalib(calib)
{
    const auto& dists = calib.getDistortionParameters();

    // Check if any distortion coefficient is non-zero
    bool hasDistortion = false;
    for (double d : dists) {
        if (std::abs(d) > 1e-12) { hasDistortion = true; break; }
    }
    if (!hasDistortion) return;

    m_active = true;

    cv::Matx33d K = calib.getIntrinsicsMatrix();
    cv::Size imageSize(calib.getWidth(), calib.getHeight());

    if (calib.isFishEye()) {
        // Fisheye (Kannala-Brandt): 4 coefficients
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
        for (int i = 0; i < std::min(static_cast<int>(dists.size()), 4); i++)
            D.at<double>(i) = dists[i];

        cv::fisheye::initUndistortRectifyMap(
            cv::Mat(K), D, cv::Matx33d::eye(), cv::Mat(K), imageSize, CV_16SC2, m_map1, m_map2);

        std::cout << "CameraUndistorter: Initialized remap maps for "
                  << imageSize.width << "x" << imageSize.height << " fisheye camera" << std::endl;
    } else {
        // Pinhole: k1, k2, p1, p2, k3
        cv::Mat D = cv::Mat::zeros(1, std::max(static_cast<int>(dists.size()), 5), CV_64F);
        for (int i = 0; i < static_cast<int>(dists.size()); i++)
            D.at<double>(i) = dists[i];

        cv::initUndistortRectifyMap(
            cv::Mat(K), D, cv::Mat(), cv::Mat(K), imageSize, CV_16SC2, m_map1, m_map2);

        std::cout << "CameraUndistorter: Initialized remap maps for "
                  << imageSize.width << "x" << imageSize.height << " pinhole camera" << std::endl;
    }

    m_newCalib.setDistortionParameters({0.0, 0.0, 0.0, 0.0, 0.0});
    m_newCalib.setFishEye(false);
}

void CameraUndistorter::undistort(cv::Mat& frame) const
{
    if (!m_active) return;

    cv::Mat undistorted;
    cv::remap(frame, undistorted, m_map1, m_map2, cv::INTER_LINEAR);
    frame = undistorted;
}

CameraCalibration getAlignedCalibration(const CameraCalibration& calib, int alignment)
{
    int w = calib.getWidth();
    int h = calib.getHeight();
    int newW = (w / alignment) * alignment;
    int newH = (h / alignment) * alignment;

    if (newW == w && newH == h)
        return calib;

    // Crop offsets (symmetric from each side)
    double cropX = (w - newW) / 2.0;
    double cropY = (h - newH) / 2.0;

    CameraCalibration aligned = calib;
    aligned.setWidth(newW);
    aligned.setHeight(newH);

    // Shift principal point by the crop offset
    cv::Matx33d K = calib.getIntrinsicsMatrix();
    K(0, 2) -= cropX;  // cx
    K(1, 2) -= cropY;  // cy
    aligned.setIntrinsicsMatrix(K);

    std::cout << "Aligned calibration: " << w << "x" << h
              << " -> " << newW << "x" << newH
              << " (cropped " << (w - newW) << "x" << (h - newH) << " pixels)" << std::endl;

    return aligned;
}

void centerCrop(cv::Mat& frame, const CameraCalibration& calib)
{
    int targetW = calib.getWidth();
    int targetH = calib.getHeight();

    if (frame.cols == targetW && frame.rows == targetH)
        return;

    int x = (frame.cols - targetW) / 2;
    int y = (frame.rows - targetH) / 2;
    frame = frame(cv::Rect(x, y, targetW, targetH)).clone();
}
