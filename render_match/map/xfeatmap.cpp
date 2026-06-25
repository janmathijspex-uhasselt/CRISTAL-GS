#include "xfeatmap.h"
#include "../utils.h"
#include <fstream>

XFeatMap::XFeatMap(std::shared_ptr<XFeat::XFDetector> detector, CameraCalibration calibration): Map(), detector(detector), calibration(calibration)
{
}

bool XFeatMap::find_pose(cv::Mat &req_frame, cv::Matx44d &w2cam)
{
    auto frame_tensor = detector->parseInput(req_frame);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    auto& req_desc = features["descriptors"];
    auto& req_kpts = features["keypoints"];

    for(auto& kf : keyframes)
    {
        auto& desc = kf.features["descriptors"];
        auto& kpts = kf.features["keypoints"];

        // acquire matches
        torch::Tensor idxs_req, idxs;
        detector->match(req_desc, desc, idxs_req, idxs, -1.0);

        if(idxs.size(0) < min_number_matches)
        {
            continue;
        }

        torch::Tensor mkpts_req_tensor = req_kpts.index({idxs_req});
        torch::Tensor mkpts_tensor = kpts.index({idxs});

        cv::Mat mkpts_req = detector->tensorToMat(mkpts_req_tensor);
        cv::Mat mkpts = detector->tensorToMat(mkpts_tensor);

        // filter outliers
        cv::Mat mask;
        cv::findHomography(mkpts_req, mkpts, cv::USAC_MAGSAC, reprojection_error, mask, iterations, confidence);

        cv::Mat indices;
        cv::findNonZero(mask, indices);

        if(indices.rows < min_number_matches)
        {
            continue;
        }

        cv::Mat points_3d = Utils::selectByIndicesTensor(kf.features_3d, idxs);
        cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask);
        cv::Mat mkpts_req_filtered = Utils::filterRowsByMask(mkpts_req, mask);

        cv::Mat rvec, tvec;
        cv::solvePnPRansac(points_3d_filtered, mkpts_req_filtered, calibration.getIntrinsicsMatrix(), calibration.getDistortionParameters(), rvec, tvec, false, iterations/*, false, 2000, 2.0, 0.995*/);

        cv::Matx44d pcd2cam;
        Utils::make_transform(rvec, tvec, pcd2cam);

        w2cam = pcd2cam * kf.w2cam;

        return true;
    }
    return false;
}

void XFeatMap::add_frame(cv::Mat &frame, cv::Mat& depth, cv::Matx44d w2cam, CameraCalibration calibration)
{
    auto frame_tensor = detector->parseInput(frame);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    cv::Mat points_3d, mask_nan;
    Utils::project_3d(detector->tensorToMat(features["keypoints"]), depth, calibration.getIntrinsicsMatrix(), points_3d, mask_nan);

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

    keyframes.push_back(XfeatFrame(w2cam, features, calibration, points_3d_filtered));
}

void XFeatMap::save_keyframes(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    size_t num_keyframes = keyframes.size();
    out.write(reinterpret_cast<const char*>(&num_keyframes), sizeof(size_t));
    for (const auto& kf : keyframes) {
        kf.save(out);
    }

    out.close();
}

void XFeatMap::load_keyframes(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    size_t num_keyframes;
    in.read(reinterpret_cast<char*>(&num_keyframes), sizeof(size_t));
    keyframes.resize(num_keyframes);
    for (auto& kf : keyframes) {
        kf.load(in);
    }

    in.close();
}
