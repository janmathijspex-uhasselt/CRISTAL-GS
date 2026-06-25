#ifndef MAP_H
#define MAP_H

#include <opencv2/opencv.hpp>
#include "RTRenderer/CameraCalibration.h"

class Map
{
public:
    virtual bool find_pose(cv::Mat& frame, cv::Matx44d& w2cam) = 0;
    virtual void add_frame(cv::Mat& frame, cv::Mat& depth, cv::Matx44d w2cam, CameraCalibration calibration) = 0;
    virtual void save_keyframes(const std::string& filename) const = 0;
    virtual void load_keyframes(const std::string& filename) = 0;
    virtual int getKeyframeCount() const = 0;

protected:
    int min_number_matches = 15;
    int iterations = 2000;
    double reprojection_error = 2.0;
    double confidence = 0.995;
};


#endif // MAP_H
