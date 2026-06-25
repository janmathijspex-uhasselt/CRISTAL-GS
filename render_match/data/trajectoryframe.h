#pragma once
#include <opencv2/core.hpp>
#include "opencv2/core/quaternion.hpp"
#include <ostream>

class TrajectoryFrame
{
public:
    TrajectoryFrame(int frame_nr, const cv::Matx44d& pose, int nr_matches, double timestamp)
        : frame_nr(frame_nr), pose(pose.inv()), nr_matches(nr_matches), timestamp{timestamp} {}

    cv::Quatd getQuaternion() const;
    cv::Vec3d getPosition() const;
    inline double getTimestamp() const{return timestamp;};
    inline int getFrameNr() const {return frame_nr;};
    inline int getMatches() const {return nr_matches;};

private:
    int frame_nr;
    int nr_matches;
    double timestamp;
    cv::Matx44d pose; // 4x4 transformation matrix (w2c or c2w)
};
