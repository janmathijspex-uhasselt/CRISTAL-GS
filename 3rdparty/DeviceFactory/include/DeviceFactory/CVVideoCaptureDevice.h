#ifndef CVVIDEOCAPTUREDEVICE_H
#define CVVIDEOCAPTUREDEVICE_H

#include "Device.h"
#include <mutex>
#include <condition_variable>
namespace DeviceFactory{
class CVVideoCaptureDevice : public Device
{
public:
    CVVideoCaptureDevice();

    void captureImages(cv::Mat &color, double &timestamp);
    void captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp);

    std::shared_ptr<Device> createInstance();

    bool init(const std::string ID, const DeviceProperties &properties, const std::string& calibrationFile = "");

    void listAvailableDevices();

    std::string getDriver() { return "CVVideoCapture"; }

    virtual int numberOfFrames() const;

    std::vector<std::string> split(const std::string &s, char delim);

private:
    bool openCamera(cv::VideoCapture &vc, const std::string &ID);
    void setSupportedOutputs();

    std::chrono::high_resolution_clock::time_point m_startTime;
    cv::VideoCapture m_videoCaptureColor;
    cv::VideoCapture m_videoCaptureDepth;

    void setResolution();

    cv::Mat BGR8BitToGrayscale16Bit(cv::Mat image);

    int m_framesInSequence;
    int m_frameID;
};
}

#endif // CVVIDEOCAPTUREDEVICE_H
