#ifndef DEVICEFACTORY_H
#define DEVICEFACTORY_H

#include "Device.h"
#include <memory>
namespace DeviceFactory{
class DeviceFactory
{
public:
    DeviceFactory();

    std::shared_ptr<Device> createFirstDevice(const std::string &driver, const std::string calibration_file);
    std::shared_ptr<Device> createDevices(const std::string& driver, const std::string& device_ID, const std::string calibration_file = "");
    std::shared_ptr<Device> createDevices(const std::string& driver, const std::string& device_ID, const DeviceProperties& properties, const std::string calibration_file = "");

    void listAvailableDevices();
    void listAvailableDrivers();

private:
    void registerFactoryDevice(std::shared_ptr<Device> device);

    std::map< std::string, std::shared_ptr<Device> > m_factoryDevices;

};
}

#endif // DEVICEFACTORY_H
