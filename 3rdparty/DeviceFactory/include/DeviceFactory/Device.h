#ifndef DEVICE_H
#define DEVICE_H

#include <opencv2/opencv.hpp>
#include "CameraCalibration.h"
#include <unordered_map>



namespace DeviceFactory{
typedef std::map<std::string, std::string> DeviceProperties;

class Device
{
public:

    enum SUPPORTED_OUTPUTS
    {
        BGR = 0x0001,
        DEPTH = 0x0002,
        BGR_DEPTH = 0x0003,
        IMU = 0x0004,
        BGR_DEPTH_IMU = 0x0007
    };

    Device();
    virtual ~Device() {};

    /**
     * @brief captureImages
     * @param color
     * @param timestamp The timestamp in seconds
     */
    virtual void captureImages(cv::Mat& color, double& timestamp);
    virtual void captureImages(cv::Mat& color, cv::Mat& depth, double& timestamp);

    /**
     * @brief stop Stop the camera
     */
    virtual void stop() {};

    int getWidth() const;
    int getHeight() const;

    bool supportsOutput(SUPPORTED_OUTPUTS output);

    bool isFisheye();

    CameraCalibration getCalibration();

    /**
     * @brief reset Reset the camera
     */
    virtual void reset();
public:
    // Functions to be called by factory
    virtual std::shared_ptr<Device> createInstance() = 0;

    virtual bool init(const std::string ID, const DeviceProperties& properties, const std::string& calibrationFile = "") = 0;

    virtual void listAvailableDevices() = 0;
    virtual std::string getDriver() { return "Device"; }

    virtual int numberOfFrames() const { return std::numeric_limits<int>::max(); }

    virtual std::string getDeviceId() const { return m_deviceID; }

protected:
    void setCalibration(CameraCalibration calibration);

    void setSupportedOutput(SUPPORTED_OUTPUTS output);

    void setFisheye(bool isfisheye);
    void setWidth(int width);
    void setHeight(int height);

    void setInitInfo(const std::string &ID, const DeviceProperties& properties, const std::string& calibrationFile);

    CameraCalibration m_calibration;
    int m_imgWidth, m_imgHeight;    
    SUPPORTED_OUTPUTS m_supportedOutputs;

    std::string m_deviceID;
    DeviceProperties m_initProperties;
    std::string m_initCalibrationFile;

    std::deque<double> timestamps;
};

typedef std::shared_ptr<Device> DevicePtr;
}
#endif // DEVICE_H
