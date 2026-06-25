#include "trajectoryframe.h"
#include "opencv2/core/quaternion.hpp"
#include "opencv2/calib3d.hpp"

cv::Quatd TrajectoryFrame::getQuaternion() const
{
    cv::Matx33d R(pose.get_minor<3, 3>(0, 0));
    cv::Vec3d t(pose(0, 3), pose(1, 3), pose(2, 3));

    cv::Matx33d Rt = R.t();
    cv::Matx44d Rt_pose = cv::Matx44d::eye();
    Rt_pose.get_minor<3, 3>(0, 0) = Rt;
    Rt_pose(0, 3) = -Rt(0, 0)*t[0] - Rt(0, 1)*t[1] - Rt(0, 2)*t[2];
    Rt_pose(1, 3) = -Rt(1, 0)*t[0] - Rt(1, 1)*t[1] - Rt(1, 2)*t[2];
    Rt_pose(2, 3) = -Rt(2, 0)*t[0] - Rt(2, 1)*t[1] - Rt(2, 2)*t[2];

    cv::Matx33d qR = pose.get_minor<3, 3>(0, 0);
    cv::Quatd q = cv::Quatd::createFromRotMat(qR);
    return q;
}

cv::Vec3d TrajectoryFrame::getPosition() const
{
    cv::Vec3d t(pose(0, 3), pose(1, 3), pose(2, 3));
    return t;
}
