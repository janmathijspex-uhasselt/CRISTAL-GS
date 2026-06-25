#ifndef CVIMAGECAPTUREDEVICE_H
#define CVIMAGECAPTUREDEVICE_H

#include "Device.h"
#include <mutex>
#include <condition_variable>

namespace DeviceFactory{
class CVImageCaptureDevice : public Device
{
public:
    CVImageCaptureDevice();

    void captureImages(cv::Mat &color, double &timestamp);
    void captureImages(cv::Mat& color, cv::Mat& depth, double& timestamp);

    std::shared_ptr<Device> createInstance();
    bool init(const std::string path, const DeviceProperties &properties, const std::string& calibrationFile = "");
    void listAvailableDevices();

    std::string getDriver() { return "CVImageCapture"; }

private:
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::string m_path;
    long m_frame;
};
}

#endif // CVVIDEOCAPTUREDEVICE_H
