#ifdef WITH_REALSENSE2

#ifndef REALSENSE2DEVICE_H
#define REALSENSE2DEVICE_H

#include "Device.h"
#include "librealsense2/rs.hpp"
#include <mutex>
#include <condition_variable>

namespace DeviceFactory{
class RealSense2Device : public Device
{
public:

    enum FRAME_DATA_TYPE
    {
        BGR_DEPTH,
        ACCEL,
        GYRO,
    };

    RealSense2Device();
    virtual ~RealSense2Device();

    virtual std::shared_ptr<Device> createInstance();

    // Device interface
public:

    bool init(const std::string ID, const DeviceProperties& properties, const std::string& calibrationFile = "");

    void captureImages(cv::Mat &color, double &timestamp);
    void captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp);

    virtual std::string getDriver() { return "RealSense2"; }


    virtual void listAvailableDevices();

    virtual std::string getDeviceId() const;

    void initCalibration(rs2_intrinsics &intrinsics, const std::string &calibrationFile);

    void addHandler(std::function<void(FRAME_DATA_TYPE frame_data_type, double timestamp, float x, float y, float z)> callback)
    {
        handlers.emplace_back(callback);
    }

private:
    bool profileChanged(const std::vector<rs2::stream_profile> &current, const std::vector<rs2::stream_profile> &prev);
    rs2_stream findStreamToAlign(const std::vector<rs2::stream_profile> &streams);
    rs2_option getSensorOptions(const rs2::sensor &sensor);
    void imu_callback(const rs2::frame &frame);
    bool getDevice(const std::string serial, rs2::context &ctx, rs2::device &out_device);

    rs2::frameset m_fsLast;

    rs2::context m_ctx;
    rs2_stream m_align_to;
    rs2::align* m_align;
    rs2::pipeline_profile m_pipe_profile;
    rs2::device m_selected_device;

    std::mutex m_imu_mutex;


    std::condition_variable m_cond_image_rec;

    //std::vector<double> m_v_accel_timestamp;
    //std::vector<rs2_vector> m_v_accel_data;
    //std::vector<double> m_v_gyro_timestamp;
    //std::vector<rs2_vector> m_v_gyro_data;

    std::vector< std::function<void(FRAME_DATA_TYPE frame_data_type, double timestamp, float x, float y, float z)>> handlers;

    double m_prev_accel_timestamp = 0;
    rs2_vector m_prev_accel_data;
    double m_current_accel_timestamp = 0;
    rs2_vector m_current_accel_data;
    std::vector<double> m_v_accel_timestamp_sync;
    std::vector<rs2_vector> m_v_accel_data_sync;

    double m_timestamp_image;
    bool m_image_ready;
    int m_count_im_buffer; // count dropped frames


    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline m_pipe;

    void setAutoExposure(const DeviceProperties &properties);
    void setAutoWhiteBalance(const DeviceProperties &properties);
    void setWhiteBalance(const DeviceProperties &properties);
    void setExposure(const DeviceProperties& properties);

};

}
#endif // REALSENSE2DEVICE_H

#endif
