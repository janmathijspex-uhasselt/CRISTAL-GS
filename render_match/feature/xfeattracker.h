#ifndef XFEATTRACKER_H
#define XFEATTRACKER_H

#include "featuretracker.h"
#include "XFeat.h"
#include "xfeatframe.h"

class XFeatTracker: public FeatureTracker
{
public:
    XFeatTracker(std::shared_ptr<XFeat::XFDetector> detector);

    int find_pose(cv::Mat &img_cam, CameraCalibration& intrinsics, cv::Matx44d& w2cam);
    void update_keyframe(cv::Mat& img_pcd, cv::Mat& depth_pcd, CameraCalibration& intrinsics, cv::Matx44d& w2cam);

private:
    std::shared_ptr<XFeat::XFDetector> detector;

    XfeatFrame last_frame_rendered;
    XfeatFrame previous_frame_cam;
};

#endif // XFEATDETECTOR_H
