#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include "CameraCalibration.h"
#include "torch/torch.h"

namespace Utils{
    void make_transform(const cv::Mat& rvec, const cv::Mat& tvec, cv::Matx44d& T);
    cv::Mat filterRowsByMask(const cv::Mat& data, const cv::Mat& mask);
    cv::Mat filterRowsByIndices(const cv::Mat& data, const std::vector<int> idxs);
    cv::Mat selectByIndicesTensor(const cv::Mat &mat, const at::Tensor &indices);
    void project_3d(const cv::Mat &mkpts_pcd, const cv::Mat &depth_pcd, const cv::Matx33d& intrinsics, cv::Mat& points3D, cv::Mat &mask);

    void visualize_side_by_side(const cv::Mat& img1, const cv::Mat& img2, const std::string& windowName = "Side-by-side");
    void showMatches(
        const cv::Mat& img_cam,
        const cv::Mat& img_kf,
        const cv::Mat& mkpts_cam,
        const cv::Mat& mkpts_kf,
        int max_width = 1920,
        int max_height = 1080
    );
}
#endif // UTILS_H
