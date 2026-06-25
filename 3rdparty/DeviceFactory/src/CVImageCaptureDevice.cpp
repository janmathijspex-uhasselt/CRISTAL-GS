#include "CVImageCaptureDevice.h"
#include <chrono>

namespace DeviceFactory {
CVImageCaptureDevice::CVImageCaptureDevice()
{

}

void CVImageCaptureDevice::captureImages(cv::Mat &color, double &timestamp)
{
    cv::Mat depth;
    captureImages(color, depth, timestamp);
    
}

void CVImageCaptureDevice::captureImages(cv::Mat& color, cv::Mat& depth, double& timestamp) {
    timestamp = 0.0;

    std::stringstream ss;
    ss << m_path << std::setw(5) << std::setfill('0') << m_frame << ".jpg";
    color = cv::imread(ss.str());


    std::stringstream ssDepth;
    ssDepth << m_path << std::setw(5) << std::setfill('0') << m_frame << ".exr";
    depth = cv::imread(ssDepth.str(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    depth.convertTo(depth, CV_16U);


    auto m_current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_current - m_startTime);
    timestamp = duration.count() / 1000.0;

    m_frame += 1;
}

std::shared_ptr<Device> CVImageCaptureDevice::createInstance()
{
    return std::make_shared<CVImageCaptureDevice>();
}

bool CVImageCaptureDevice::init(const std::string path, const DeviceProperties &properties, const std::string &calibrationFile)
{
    setInitInfo(path, properties, calibrationFile);

    m_path = path;
    m_frame = 0;

    bool isOpen = false;


    std::stringstream ss;
    cv::Mat initial;
    ss << m_path << std::setw(5) << std::setfill('0') << 0 << ".jpg";
    initial = cv::imread(ss.str());

    if (initial.empty())
        isOpen = false;
    else
        isOpen = true;

    if (isOpen) {
        setWidth(initial.cols);
        setHeight(initial.rows);

        if (!calibrationFile.empty())
        {
            CameraCalibration calibration;
            calibration.loadCalibration(calibrationFile);
            setCalibration(calibration);
            std::cout << "Using calibration: " << calibration.getIntrinsicsMatrix() << std::endl;
        }
        else {
            std::cerr << "CVImageCaptureDevice: no calibration file given" << std::endl;
            return false;
        }

        setSupportedOutput(SUPPORTED_OUTPUTS::BGR);
        setFisheye(false);
        m_startTime = std::chrono::high_resolution_clock::now();
    }
    else
    {
        std::cerr << "Failed to image device in folder " << path << std::endl;
    }



    return isOpen;
}

void CVImageCaptureDevice::listAvailableDevices()
{
    std::cout << "--------------------------------" << std::endl;
    std::cout << "path to image file sequence (RGBD), e.g. %03d.jpg and %03d.exr starting from 0" << std::endl;
}
}
