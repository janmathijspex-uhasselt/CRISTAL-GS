#ifndef FEATURETRACKER
#define FEATURETRACKER

#include <opencv2/opencv.hpp>
#include "CameraCalibration.h"
#include "../utils.h"

class FeatureTracker
{
protected:
    int min_number_matches = 10;
    int iterations = 1000;
    double reprojection_error = 3.0;
    double confidence = 0.995;

    int find_filtered_pose(cv::Mat& mkpts_cam, cv::Mat& mkpts_pcd, cv::Mat& points_3d, CameraCalibration& calibration, cv::Matx44d& w2cam)
    {
        // filter outliers
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(mkpts_pcd,mkpts_cam,calibration.getIntrinsicsMatrix(),cv::USAC_MAGSAC,confidence,reprojection_error, iterations,mask);
        cv::Mat indices;
        cv::findNonZero(mask, indices);

        if(indices.rows < min_number_matches)
        {
            return -1;
        }

        cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask);
        cv::Mat mkpts_cam_filtered = Utils::filterRowsByMask(mkpts_cam, mask);

        cv::Mat rvec, tvec; std::vector<int> inliers;
        cv::solvePnPRansac(points_3d_filtered, mkpts_cam_filtered, calibration.getIntrinsicsMatrix(), calibration.getDistortionParameters(), rvec, tvec, false, iterations, reprojection_error, confidence, inliers);

        points_3d_filtered = Utils::filterRowsByIndices(points_3d_filtered, inliers);
        mkpts_cam_filtered = Utils::filterRowsByIndices(mkpts_cam_filtered, inliers);

        if(cv::norm(tvec) > 2.00f) // more than 2 m off since last pose, is obviously wrong pose
        {
            return -1;
        }

        cv::Matx44d pcd2cam; // cam(t0) naar cam(t0 + 1)
        Utils::make_transform(rvec, tvec, pcd2cam);

        w2cam = pcd2cam * w2cam;

        return points_3d_filtered.rows;
    }

public:
    virtual int find_pose(cv::Mat& img_cam, CameraCalibration& calibration, cv::Matx44d& w2cam) = 0;
    virtual void update_keyframe(cv::Mat& img_pcd, cv::Mat& depth_pcd, CameraCalibration& intrinsics, cv::Matx44d& w2cam) = 0;

protected:
    std::mutex kf_mutex;

};

#endif // FEATUREDETECTOR_H
