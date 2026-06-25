#ifndef CUBEMAPGENERATOR_H
#define CUBEMAPGENERATOR_H

#include <opencv2/opencv.hpp>
#include "project_cloud.h"
#include "map.h"

class CubemapGenerator
{
public:
    CubemapGenerator(std::shared_ptr<ProjectCloud> projector, bool useNeural = true);

    void renderCubemapToMap(const cv::Vec3d& position, std::shared_ptr<Map> map);

private:
    cv::Size cubemapSize = cv::Size{1440,1440};
    CameraCalibration calibration;

    std::shared_ptr<ProjectCloud> projector;
    bool useNeural_;

    cv::Matx44d lookAt(const cv::Vec3d& position, const cv::Vec3d& target, const cv::Vec3d& up);

};

#endif // CUBEMAPGENERATOR_H
