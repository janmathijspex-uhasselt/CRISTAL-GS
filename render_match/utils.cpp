#include "utils.h"
#include <random>
#include <fstream>
#include <chrono>

namespace Utils {
    void make_transform(const cv::Mat& rvec, const cv::Mat& tvec, cv::Matx44d& T)
    {
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);

        T(0, 0) = R(0, 0); T(0, 1) = R(0, 1); T(0, 2) = R(0, 2);
        T(1, 0) = R(1, 0); T(1, 1) = R(1, 1); T(1, 2) = R(1, 2);
        T(2, 0) = R(2, 0); T(2, 1) = R(2, 1); T(2, 2) = R(2, 2);

        T(0, 3) = tvec.at<double>(0);
        T(1, 3) = tvec.at<double>(1);
        T(2, 3) = tvec.at<double>(2);

        T(3, 0) = 0.0;
        T(3, 1) = 0.0;
        T(3, 2) = 0.0;
        T(3, 3) = 1.0;
    }

    cv::Mat filterRowsByMask(const cv::Mat& data, const cv::Mat& mask) {
        cv::Mat indices, output;
        cv::findNonZero(mask, indices);
        for (const auto& pt : cv::Mat_<cv::Point>(indices)) {
            output.push_back(data.row(pt.y));
        }
        return output;
    }

    cv::Mat filterRowsByIndices(const cv::Mat& data, const std::vector<int> idxs)
    {
        cv::Mat output;
        for (const auto& pt : idxs) {
            output.push_back(data.row(pt));
        }
        return output;
    }

    cv::Mat selectByIndicesTensor(const cv::Mat &mat, const at::Tensor &indices)
    {
        TORCH_CHECK(indices.dim() == 1, "Indices must be a 1D tensor");

        at::Tensor idx_tensor = indices.to(at::kCPU).contiguous();

        const int numRows = idx_tensor.size(0);
        cv::Mat selected(numRows, mat.cols, mat.type());

        auto indices_acc = idx_tensor.accessor<int64_t, 1>();

        for (int i = 0; i < numRows; ++i)
        {
            int idx = static_cast<int>(indices_acc[i]);
            mat.row(idx).copyTo(selected.row(i));
        }

        return selected;
    }

    void project_3d(const cv::Mat &mkpts_pcd, const cv::Mat &depth_pcd, const cv::Matx33d& intrinsics, cv::Mat& points3D, cv::Mat &mask)
    {
        mask = cv::Mat(mkpts_pcd.rows, 1, CV_8UC1, cv::Scalar(1));
        points3D = cv::Mat(mkpts_pcd.rows, 1, CV_32FC3);

        int rows = depth_pcd.rows;
        int cols = depth_pcd.cols;

        for(int i = 0; i < mkpts_pcd.rows; ++i)
        {
            int u = static_cast<int>(mkpts_pcd.at<float>(i, 0));
            int v = static_cast<int>(mkpts_pcd.at<float>(i, 1));

            // Skip out-of-bounds
            if (u < 0 || u >= cols || v < 0 || v >= rows) {
                points3D.at<cv::Vec3f>(i) = cv::Vec3f{NAN, NAN, NAN};
                mask.at<uchar>(i) = 0;
                continue;
            }

            float d = depth_pcd.at<float>(v, u);

            // If invalid, try to average valid neighbors in 3x3 window
            if (d <= 0)
            {
                float sum = 0.0f;
                int count = 0;

                for (int dy = -2; dy <= 2; ++dy)
                {
                    for (int dx = -2; dx <= 2; ++dx)
                    {
                        int uu = u + dx;
                        int vv = v + dy;

                        if (uu >= 0 && uu < cols && vv >= 0 && vv < rows)
                        {
                            float d_neigh = depth_pcd.at<float>(vv, uu);
                            if (d_neigh > 0)
                            {
                                sum += d_neigh;
                                count++;
                            }
                        }
                    }
                }

                if (count > 0)
                {
                    d = sum / count;
                }
                else
                {
                    points3D.at<cv::Vec3f>(i) = cv::Vec3f{NAN, NAN, NAN};
                    mask.at<uchar>(i) = 0;
                    continue;
                }
            }

            float x = (u - intrinsics(0,2)) * d / intrinsics(0,0);
            float y = (v - intrinsics(1,2)) * d / intrinsics(1,1);
            float z = d;

            points3D.at<cv::Vec3f>(i) = cv::Vec3f{x, y, z};
        }
    }

    void visualize_side_by_side(const cv::Mat& img1, const cv::Mat& img2, const std::string& windowName) {
        cv::Mat img1_resized = img1;
        cv::Mat img2_resized = img2;

        if (img1.rows != img2.rows) {
            float scale = static_cast<float>(img1.rows) / img2.rows;
            cv::resize(img2, img2_resized, cv::Size(), scale, scale);
        }

        if (img1_resized.channels() == 1) {
            cv::cvtColor(img1_resized, img1_resized, cv::COLOR_GRAY2BGR);
            cv::cvtColor(img2_resized, img2_resized, cv::COLOR_GRAY2BGR);
        }

        cv::Mat stacked;
        cv::hconcat(img1_resized, img2_resized, stacked);

        cv::imshow(windowName, stacked);
    }

    void showMatches(
        const cv::Mat& img_cam,
        const cv::Mat& img_kf,
        const cv::Mat& mkpts_cam,
        const cv::Mat& mkpts_kf,
        int max_width,
        int max_height
    ) {
        // Convert grayscale → color for visualization
        cv::Mat img_cam_color, img_kf_color;
        if (img_cam.channels() == 1)
            cv::cvtColor(img_cam, img_cam_color, cv::COLOR_GRAY2BGR);
        else
            img_cam_color = img_cam.clone();

        if (img_kf.channels() == 1)
            cv::cvtColor(img_kf, img_kf_color, cv::COLOR_GRAY2BGR);
        else
            img_kf_color = img_kf.clone();

        // Put images side by side
        cv::Mat canvas(std::max(img_cam_color.rows, img_kf_color.rows),
                       img_cam_color.cols + img_kf_color.cols,
                       CV_8UC3, cv::Scalar(0,0,0));
        img_cam_color.copyTo(canvas(cv::Rect(0, 0, img_cam_color.cols, img_cam_color.rows)));
        img_kf_color.copyTo(canvas(cv::Rect(img_cam_color.cols, 0, img_kf_color.cols, img_kf_color.rows)));

        // Draw matches
        for (int i = 0; i < mkpts_cam.rows; i++) {
            cv::Point2f pt_cam(mkpts_cam.at<float>(i, 0), mkpts_cam.at<float>(i, 1));
            cv::Point2f pt_kf(mkpts_kf.at<float>(i, 0) + img_cam_color.cols, mkpts_kf.at<float>(i, 1));

            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(canvas, pt_cam, 3, color, -1, cv::LINE_AA);
            cv::circle(canvas, pt_kf, 3, color, -1, cv::LINE_AA);
            cv::line(canvas, pt_cam, pt_kf, color, 1, cv::LINE_AA);
        }

        // Compute resize scale
        double scale_x = (double)max_width / canvas.cols;
        double scale_y = (double)max_height / canvas.rows;
        double scale = std::min(1.0, std::min(scale_x, scale_y)); // only shrink, don’t enlarge

        cv::Mat resized;
        if (scale < 1.0)
            cv::resize(canvas, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        else
            resized = canvas;

        cv::imshow("Matches", resized);
        cv::waitKey(1);
    }
}
