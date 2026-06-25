#include "Device.h"

namespace DeviceFactory{
Device::Device()
{

}

void Device::captureImages(cv::Mat &color, double &timestamp)
{
    std::cerr << "Warning: Device::captureImages(cv::Mat &color, double &timestamp) not implemented for device" << std::endl;
}

void Device::captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp)
{
    std::cerr << "Warning: captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp) not implemented for device" << std::endl;
}

int Device::getWidth() const
{
    return m_imgWidth;
}

int Device::getHeight() const
{
    return m_imgHeight;
}

bool Device::supportsOutput(SUPPORTED_OUTPUTS output)
{
    return (m_supportedOutputs & output) == output;
}

bool Device::isFisheye()
{
    std::cerr << "Device::isFisheye is deprecated. Please use CameraCalibration::isFishEye instead." << std::endl;
    return m_calibration.isFishEye();
}

CameraCalibration Device::getCalibration()
{
    return m_calibration;//.getScaledCalibration(getWidth(), getHeight());
}

void Device::reset()
{
    stop();
    init(m_deviceID, m_initProperties, m_initCalibrationFile);
}

void Device::setCalibration(CameraCalibration calibration)
{
    m_calibration = calibration;
}

void Device::setSupportedOutput(SUPPORTED_OUTPUTS output)
{
    m_supportedOutputs = output;
}

void Device::setFisheye(bool isfisheye)
{
    std::cerr << "Device::setFisheye is deprecated. Please use CameraCalibration::setFishEye instead." << std::endl;
    m_calibration.setFishEye(isfisheye);
}

void Device::setWidth(int width)
{
    m_imgWidth = width;
}

void Device::setHeight(int height)
{
    m_imgHeight = height;
}

void Device::setInitInfo(const std::string &ID, const DeviceProperties &properties, const std::string &calibrationFile)
{
    m_deviceID = ID;
    m_initProperties = properties;
    m_initCalibrationFile = calibrationFile;
}
}
