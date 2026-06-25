#include "cubemapgenerator.h"

CubemapGenerator::CubemapGenerator(std::shared_ptr<ProjectCloud> projector, bool useNeural): projector(projector), useNeural_(useNeural)
{
    calibration.setIntrinsicsMatrix(cv::Matx33d(cubemapSize.width/2.0, 0, cubemapSize.width/2.0, 0, cubemapSize.height/2.0, cubemapSize.height/2.0, 0.0,0.0,1.0));
    calibration.setWidth(cubemapSize.width);
    calibration.setHeight(cubemapSize.height);
}

    int i = 0;
void CubemapGenerator::renderCubemapToMap(const cv::Vec3d &position, std::shared_ptr<Map> map)
{
    struct Face {
        std::string name;
        cv::Vec3d direction;
        cv::Vec3d up;
    };

    std::vector<Face> faces = {
        {"posx", { 1,  0,  0}, {0, 0, -1}},
        {"negx", {-1,  0,  0}, {0, 0, -1}},
        {"posy", { 0,  1,  0}, {0, 0, -1}},
        {"negy", { 0, -1,  0}, {0, 0, -1}},
//        {"posz", { 0,  0,  1}, {0, -1, 0}},
//        {"negz", { 0,  0, -1}, {0, -1, 0}}
    };

    for(const auto& face: faces)
    {
        cv::Vec3d target = position + face.direction;
        cv::Matx44d w2cam = lookAt(position, target, face.up);

        cv::Mat color = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_8UC3);
        cv::Mat depth = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_32F);

        if (useNeural_)
            projector->computeFull(calibration, w2cam, &color, &depth);
        else
            projector->computeRGBD(calibration, w2cam, &color, &depth);

        cv::Mat depthMask = depth > 0;

        cv::dilate(depthMask, depthMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 9);
        cv::erode(depthMask, depthMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 9);

        cv::Mat colorResult;
        color.copyTo(colorResult, depthMask);

        cv::imwrite(std::string("/home/jvanherck/.pcl_cache/keyframes/") + std::to_string(i++) + std::string(".jpg"), color);
        map->add_frame(color, depth, w2cam, calibration);
    }
}

cv::Matx44d CubemapGenerator::lookAt(const cv::Vec3d &position, const cv::Vec3d &target, const cv::Vec3d &up)
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




