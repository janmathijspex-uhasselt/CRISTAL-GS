#include "common/geometry.h"

Eigen::Matrix4d convertToEigen(cv::Matx44d mat)
{
    Eigen::Matrix4d eigenMat;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            eigenMat(i, j) = mat(i,j);
    return eigenMat;
}

cv::Matx44d lookAt(const cv::Vec3d &position, const cv::Vec3d &target, const cv::Vec3d &up)
{
    cv::Vec3d z = (target - position);
    z /= cv::norm(z);

    cv::Vec3d x = up.cross(z);
    x /= cv::norm(x);

    cv::Vec3d y = z.cross(x);  // this gives a down direction

    cv::Matx44d mat = cv::Matx44d::eye();

    // Row-major for OpenCV: each row is an axis
    mat(0, 0) = x[0]; mat(0, 1) = x[1]; mat(0, 2) = x[2]; mat(0, 3) = -x.dot(position);
    mat(1, 0) = y[0]; mat(1, 1) = y[1]; mat(1, 2) = y[2]; mat(1, 3) = -y.dot(position);
    mat(2, 0) = z[0]; mat(2, 1) = z[1]; mat(2, 2) = z[2]; mat(2, 3) = -z.dot(position);

    return mat;
}
