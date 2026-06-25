#include "DeviceFactory.h"

#ifdef WITH_REALSENSE2
#include "RealSense2Device.h"
#endif
#ifdef WITH_XIMEA
#include "XimeaDevice.h"
#endif
#ifdef WITH_FFMPEG
#include "FFMPEGDevice.h"
#endif
#include "CVVideoCaptureDevice.h"
#include "CVImageCaptureDevice.h"

namespace DeviceFactory{

DeviceFactory::DeviceFactory()
{
#ifdef WITH_REALSENSE2
    registerFactoryDevice(std::make_shared<RealSense2Device>());
#endif
#ifdef WITH_XIMEA
    registerFactoryDevice(std::make_shared<XimeaDevice>());
#endif
#ifdef WITH_FFMPEG
    registerFactoryDevice(std::make_shared<FFMPEGDevice>());
#endif
    registerFactoryDevice(std::make_shared<CVVideoCaptureDevice>());
    registerFactoryDevice(std::make_shared<CVImageCaptureDevice>());
}

std::shared_ptr<Device> DeviceFactory::createDevices(const std::string &driver, const std::string &device_ID, const std::string calibration_file)
{
    return createDevices(driver, device_ID, DeviceProperties(), calibration_file);
}

std::shared_ptr<Device> DeviceFactory::createFirstDevice(const std::string &driver, const std::string calibration_file)
{
    for(auto device: m_factoryDevices)
    {
        std::cout << device.second->getDriver() << "   " << device.second->getDeviceId() << std::endl;
        if(device.second->getDriver() == driver)
        {
            return createDevices(driver, device.second->getDeviceId(), calibration_file);
        }
    }
}

std::shared_ptr<Device> DeviceFactory::createDevices(const std::string &driver, const std::string &device_ID, const DeviceProperties &properties, const std::string calibration_file)
{
    auto it = m_factoryDevices.find(driver);
    if ( it != m_factoryDevices.end() )
    {
        std::shared_ptr<Device> device = it->second->createInstance();
        if (device->init(device_ID, properties, calibration_file))
            return device;
        else
            return nullptr;
    }
    else
        return nullptr;
}

void DeviceFactory::registerFactoryDevice(std::shared_ptr<Device> device)
{
    std::string type = device->getDriver();
    m_factoryDevices[type] = device;
}

void DeviceFactory::listAvailableDevices()
{
    std::cout << "Available Devices: " << std::endl;
    for (auto device : m_factoryDevices)
    {
        std::cout << "================================================" << std::endl;
        std::cout << "Driver: " << device.second->getDriver() << std::endl;
        device.second->listAvailableDevices();
    }
}

void DeviceFactory::listAvailableDrivers()
{
    std::cout << "Available Drivers: " << std::endl;
    for (auto device : m_factoryDevices)
    {
        std::cout << device.second->getDriver() << std::endl;
    }
}
}
