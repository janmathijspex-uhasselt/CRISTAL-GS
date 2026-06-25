#ifndef XFEATMAP_H
#define XFEATMAP_H

#include "map.h"
#include "XFeat.h"
#include "RTRenderer/CameraCalibration.h"
#include "../feature/xfeatframe.h"

class XFeatMap: public Map
{
public:
    XFeatMap(std::shared_ptr<XFeat::XFDetector> detector, CameraCalibration calibration);

    bool find_pose(cv::Mat& frame, cv::Matx44d& w2cam);
    void add_frame(cv::Mat& frame, cv::Mat& depth, cv::Matx44d w2cam, CameraCalibration calibration);

    void save_keyframes(const std::string& filename) const;
    void load_keyframes(const std::string& filename);
    int getKeyframeCount() const override { return static_cast<int>(keyframes.size()); }

private:
    std::vector<XfeatFrame> keyframes;
    std::shared_ptr<XFeat::XFDetector> detector;
    CameraCalibration calibration;

    cv::Mat selectByIndicesTensor(const cv::Mat& mat, const at::Tensor& indices);
};

#endif // XFEATMAP_H
