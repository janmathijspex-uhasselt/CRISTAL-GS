#pragma once

#include <opencv2/core.hpp>
#include "DeviceFactory/CameraCalibration.h"

/**
 * @brief Undistorts camera frames using precomputed remap maps.
 * 
 * Supports both fisheye (Kannala-Brandt, 4 coefficients) and
 * pinhole (k1, k2, p1, p2, k3) distortion models.
 * Only activates when distortion coefficients are non-zero.
 */
class CameraUndistorter {
public:
    explicit CameraUndistorter(const CameraCalibration& calib);

    bool isActive() const { return m_active; }

    void undistort(cv::Mat& frame) const;

    CameraCalibration getUndistortedCalibration() const { return m_newCalib; }

private:
    bool m_active = false;
    cv::Mat m_map1, m_map2;
    CameraCalibration m_newCalib;
};

/**
 * @brief Returns a calibration with width/height rounded down to the nearest
 *        multiple of @p alignment, and principal point shifted accordingly.
 */
CameraCalibration getAlignedCalibration(const CameraCalibration& calib, int alignment = 16);

/**
 * @brief Center-crops @p frame to match the dimensions in @p calib.
 *        No-op if dimensions already match.
 */
void centerCrop(cv::Mat& frame, const CameraCalibration& calib);
