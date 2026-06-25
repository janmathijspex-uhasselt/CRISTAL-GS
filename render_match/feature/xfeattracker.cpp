#include "xfeattracker.h"
#include "../utils.h"

XFeatTracker::XFeatTracker(std::shared_ptr<XFeat::XFDetector> detector): FeatureTracker(), detector(detector)
{

}

int XFeatTracker::find_pose(cv::Mat &img_cam, CameraCalibration& calibration, cv::Matx44d& w2cam)
{
    XfeatFrame last_keyframe_local;
    {
        std::lock_guard<std::mutex> lock(kf_mutex);
        last_keyframe_local = last_frame_rendered;
    }

    auto frame_tensor = detector->parseInput(img_cam);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    if(features["descriptors"].size(0) < min_number_matches || last_keyframe_local.features["descriptors"].size(0) < min_number_matches)
    {
        return -1;
    }

    // match features between render and camera frame
    torch::Tensor idxs_cam, idxs_kf;
    detector->match(features["descriptors"], last_keyframe_local.features["descriptors"], idxs_cam, idxs_kf, -1.0);

    torch::Tensor mkpts_cam_tensor = features["keypoints"].index({idxs_cam});
    torch::Tensor mkpts_kf_tensor = last_keyframe_local.features["keypoints"].index({idxs_kf});

    cv::Mat mkpts_cam = detector->tensorToMat(mkpts_cam_tensor);
    cv::Mat mkpts_kf = detector->tensorToMat(mkpts_kf_tensor);

    cv::Mat points_3d = Utils::selectByIndicesTensor(last_keyframe_local.features_3d, idxs_kf);

    w2cam = last_keyframe_local.w2cam;
    int ret =  find_filtered_pose(mkpts_cam, mkpts_kf, points_3d, calibration, w2cam);
    Utils::showMatches(last_keyframe_local.render,img_cam, mkpts_kf, mkpts_cam);

    return ret;
}

void XFeatTracker::update_keyframe(cv::Mat &img_pcd, cv::Mat &depth_pcd, CameraCalibration &calibration, cv::Matx44d &w2cam)
{
    auto frame_tensor = detector->parseInput(img_pcd);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    cv::Mat points_3d, mask_nan;
    Utils::project_3d(detector->tensorToMat(features["keypoints"]), depth_pcd, calibration.getIntrinsicsMatrix(), points_3d, mask_nan);

    cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask_nan);

    std::vector<int64_t> valid_indices;
    for (int i = 0; i < mask_nan.rows; ++i) {
        if (mask_nan.at<uchar>(i)) {
            valid_indices.push_back(i);
        }
    }

    at::Tensor indices = torch::tensor(valid_indices, torch::dtype(torch::kLong));

    features["keypoints"] = features["keypoints"].index({indices});
    features["descriptors"] = features["descriptors"].index({indices});

    {
        std::lock_guard<std::mutex> lock(kf_mutex);

        last_frame_rendered = XfeatFrame{w2cam, features, calibration, points_3d_filtered, img_pcd};
    }
}


