#ifdef WITH_REALSENSE2

#include "RealSense2Device.h"
#include <chrono>


#define WITH_IMU

namespace DeviceFactory{
RealSense2Device::RealSense2Device() : m_align(nullptr)
{
    m_timestamp_image = -1.0;
    m_image_ready = false;
    m_count_im_buffer = 0;
}

RealSense2Device::~RealSense2Device() {
    //m_pipe.stop();
}


std::shared_ptr<Device> RealSense2Device::createInstance()
{
    return std::make_shared<RealSense2Device>();
}

rs2_option RealSense2Device::getSensorOptions(const rs2::sensor& sensor)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    std::cout << "Sensor supports \"" << sensor.get_info(rs2_camera_info::RS2_CAMERA_INFO_NAME) << "\" the following options:\n" << std::endl;

    // The following loop shows how to iterate over all available options
    // Starting from 0 until RS2_OPTION_COUNT (exclusive)
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
    {
        rs2_option option_type = static_cast<rs2_option>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << "  " << i << ": " << option_type;

        // To control an option, use the following api:

        // First, verify that the sensor actually supports this option
        if (sensor.supports(option_type))
        {
            std::cout << std::endl;

            // Get a human readable description of the option
            const char* description = sensor.get_option_description(option_type);
            std::cout << "       Description   : " << description << std::endl;

            // Get the current value of the option
            float current_value = sensor.get_option(option_type);
            std::cout << "       Current Value : " << current_value << std::endl;

            //To change the value of an option, please follow the change_sensor_option() function
        }
        else
        {
            std::cout << " is not supported" << std::endl;
        }
    }

    uint32_t selected_sensor_option = 0;
    return static_cast<rs2_option>(selected_sensor_option);
}


rs2_stream RealSense2Device::findStreamToAlign(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (const rs2::stream_profile &sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}


bool RealSense2Device::profileChanged(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

void RealSense2Device::listAvailableDevices()
{
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();

    for (const auto &device : devices)
    {
        std::cout << "--------------------------------" << std::endl;
        if (device.supports(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER))
            std::cout << "ID: " << device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;

        if (device.supports(rs2_camera_info::RS2_CAMERA_INFO_NAME))
            std::cout << "Device Name: " << device.get_info(rs2_camera_info::RS2_CAMERA_INFO_NAME) << std::endl;

        std::cout << "Device properties" << std::endl;
        std::cout << "\t[\"filename\"] = path to input/output file name. Empy to not defines for live streaming" << std::endl;
        std::cout << "\t[\"rw\"] = \"r\" for reading from file \"w\" for writing to file." << std::endl;


    }
}

std::string RealSense2Device::getDeviceId() const
{
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    for(const auto&device: devices)
    {
        if (device.supports(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER))
            return device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER);
    }
    return "";
}


bool RealSense2Device::getDevice(const std::string serial, rs2::context& ctx, rs2::device& out_device)
{
    // Select device based on name
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;

    // Check if there are devices and select default device
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return false;
    }

    // Search device based on serial number
    bool deviceFound = false;
    for (int i = 0; i < devices.size() && !deviceFound; ++i)
    {
        auto device = devices[i];
        if (device.supports(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER))
            if (device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER) == serial)
            {
                selected_device = device;
                deviceFound = true;
            }
    }

    if (deviceFound)
    {
        out_device = selected_device;
        if (out_device.supports(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER))
            std::cout << "Selected Device " << out_device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
        return true;
    }
    else
    {
        std::cerr << "No device found with ID '" << serial << "' for driver '" << getDriver() << "'. " << std::endl;
        return false;
    }
}

void RealSense2Device::imu_callback(const rs2::frame& frame)
{
    
    // first extract IMU data
    auto motion = frame.as<rs2::motion_frame>();
    if (rs2::motion_frame motion = frame.as<rs2::motion_frame>()) {

        if (motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get the timestamp of the current frame
            double ts = motion.get_timestamp();// *1e-3;
            // Get gyro measures
            rs2_vector gyro_data = motion.get_motion_data();
            for (const auto &handler : handlers)
                handler(FRAME_DATA_TYPE::GYRO, ts, gyro_data.x, gyro_data.y, gyro_data.z);


        }

        if (motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            double ts = motion.get_timestamp();// *1e-3;
            rs2_vector accel_data = motion.get_motion_data();
            for (const auto &handler : handlers)
                handler(FRAME_DATA_TYPE::ACCEL, ts, accel_data.x, accel_data.y, accel_data.z);
        }
    }



    

    if(rs2::frameset fs = frame.as<rs2::frameset>())
    {
        std::unique_lock<std::mutex> lock(m_imu_mutex);
        m_count_im_buffer++;

        double new_timestamp_image = fs.get_timestamp()*1e-3;
        if(abs(m_timestamp_image-new_timestamp_image)<0.001){
            m_count_im_buffer--;
            return;
        }

        if (profileChanged(m_pipe.get_active_profile().get_streams(), m_pipe_profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            m_pipe_profile = m_pipe.get_active_profile();
            m_align_to = findStreamToAlign(m_pipe_profile.get_streams());

            delete m_align;
            m_align = new rs2::align(m_align_to);
        }

        //Align depth and rgb takes long time, move it out of the interruption to avoid losing IMU measurements
        m_fsLast = fs;

        m_timestamp_image = fs.get_timestamp() * 1e-3;
        m_image_ready = true;

        lock.unlock();
        m_cond_image_rec.notify_all();
        for (const auto &handler : handlers)
            handler(FRAME_DATA_TYPE::BGR_DEPTH, new_timestamp_image, 0.0f, 0.0f, 0.0f);

        
        /*
        //Get processed aligned frame
        auto processed = align.process(fs);


        // Trying to get both other and aligned depth frames
        rs2::video_frame color_frame = processed.first(align_to);
        rs2::depth_frame depth_frame = processed.get_depth_frame();
        //If one of them is unavailable, continue iteration
        if (!depth_frame || !color_frame) {
            cout << "Not synchronized depth and image\n";
            return;
        }


        imCV = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);
        depthCV = cv::Mat(cv::Size(width_img, height_img), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP);

        cv::Mat depthCV_8U;
        depthCV.convertTo(depthCV_8U,CV_8U,0.01);
        cv::imshow("depth image", depthCV_8U);*/

      /*  m_timestamp_image = fs.get_timestamp()*1e-3;
        
        
        while(m_v_gyro_timestamp.size() > m_v_accel_timestamp_sync.size())
        {
            int index = m_v_accel_timestamp_sync.size();
            double target_time = m_v_gyro_timestamp[index];

            m_v_accel_data_sync.push_back(m_current_accel_data);
            m_v_accel_timestamp_sync.push_back(target_time);
        }*/

        
    }
};

void RealSense2Device::setAutoExposure(const DeviceProperties& properties)
{
    auto itInner = properties.find("auto-exposure");
    if (itInner != properties.end() && itInner->second == "0")
    {
        std::vector<rs2::sensor> sensors = m_selected_device.query_sensors();

        // We can now iterate the sensors and print their names
        for (rs2::sensor sensor : sensors)
        {
            if (sensor.is<rs2::color_sensor>())
            {
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0);
            }
        }
    }
}

void RealSense2Device::setAutoWhiteBalance(const DeviceProperties& properties)
{
    auto itInner = properties.find("auto-white-balance");
    if (itInner != properties.end() && itInner->second == "0")
    {
        std::vector<rs2::sensor> sensors = m_selected_device.query_sensors();

        // We can now iterate the sensors and print their names
        for (const rs2::sensor &sensor : sensors)
        {
            if (sensor.is<rs2::color_sensor>())
            {
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE))
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 0.0);
            }
        }
    }
}

void RealSense2Device::setWhiteBalance(const DeviceProperties& properties)
{
    auto itInner = properties.find("white-balance");
    if (itInner != properties.end())
    {
        std::istringstream iss(itInner->second);
        float whiteBalance;
        iss >> whiteBalance;

        std::vector<rs2::sensor> sensors = m_selected_device.query_sensors();

        // We can now iterate the sensors and print their names
        for (const rs2::sensor &sensor : sensors)
        {
            if (sensor.is<rs2::color_sensor>())
            {
                if (sensor.supports(RS2_OPTION_WHITE_BALANCE))
                {
                    sensor.set_option(RS2_OPTION_WHITE_BALANCE, whiteBalance);
                }
            }
        }
    }
}

void RealSense2Device::setExposure(const DeviceProperties& properties)
{
    auto itInner = properties.find("exposure");
    if (itInner != properties.end())
    {
        std::istringstream iss(itInner->second);
        float exposure;
        iss >> exposure;

        std::vector<rs2::sensor> sensors = m_selected_device.query_sensors();

        // We can now iterate the sensors and print their names
        for (const rs2::sensor &sensor : sensors)
        {
            if (sensor.is<rs2::color_sensor>())
            {
                if (sensor.supports(RS2_OPTION_EXPOSURE))
                {
                    sensor.set_option(RS2_OPTION_EXPOSURE, exposure);
                }
            }
        }
    }
}

bool RealSense2Device::init(const std::string ID, const DeviceProperties& properties, const std::string& calibrationFile)
{
    setInitInfo(ID, properties, calibrationFile);

    // Read properties
    std::string filename;
    std::string rw;
    auto itInner = properties.find("filename");
    if (itInner != properties.end()) {
        filename = itInner->second;
        auto itRW = properties.find("rw");
        if (itRW != properties.end()) {
            rw = itRW->second;
        }

    }


    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;


    if (!ID.empty() && std::all_of(ID.begin(), ID.end(), ::isdigit))
    {
        // ID is a serial number of a camera to live capture

        //Get device
        if (!getDevice(ID, m_ctx, m_selected_device))
            return false;

        //Enable the selected device
        if (m_selected_device.supports(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER))
        {
            std::string serial = m_selected_device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER);
            cfg.enable_device(serial);
        }

        // RGB stream
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
       // cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

        // Depth stream
        cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
       // cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        // ID is not a serial number, but a filename to stream from
        if (!rw.empty() && rw.compare("w") == 0)
            cfg.enable_record_to_file(filename);
    
#ifdef WITH_IMU
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
        cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

#endif


    }
    else {
        // ID is not a serial number, but a filename to stream from
        if (!rw.empty() && rw.compare("w") == 0)
            cfg.enable_record_to_file(filename);
        else
            cfg.enable_device_from_file(ID);

    }


    setAutoExposure(properties);
    setAutoWhiteBalance(properties);
    setWhiteBalance(properties);
    setExposure(properties);

      std::vector<rs2::sensor> sensors = m_selected_device.query_sensors();
      int index = 0;

      // We can now iterate the sensors and print their names
      for (rs2::sensor sensor : sensors)
          if (sensor.supports(RS2_CAMERA_INFO_NAME))
          {
              ++index;
              getSensorOptions(sensor);
          }




   
    
    // start and stop just to get necessary profile
    m_pipe_profile = m_pipe.start(cfg);
    std::cout << "DEBUG: init complete" << std::endl;
    m_pipe.stop();

    
    // Align depth and RGB frames
    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    m_align_to = findStreamToAlign(m_pipe_profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    m_align = new rs2::align(m_align_to);
    //rs2::frameset fsSLAM;


    m_pipe_profile = m_pipe.start(cfg, std::bind(&RealSense2Device::imu_callback, this, std::placeholders::_1));
  /*  rs2::device device = m_pipe.get_active_profile().get_device();
    rs2::playback playback = device.as<rs2::playback>();
    playback.set_real_time(false);*/

    rs2::stream_profile cam_stream = m_pipe_profile.get_stream(RS2_STREAM_COLOR);
    rs2_intrinsics intrinsics_cam = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();

    int width_img = intrinsics_cam.width;
    int height_img = intrinsics_cam.height;
    setWidth(width_img);
    setHeight(height_img);

    initCalibration(intrinsics_cam, calibrationFile);

    setSupportedOutput(SUPPORTED_OUTPUTS::BGR_DEPTH);

    m_calibration.setFishEye(false);

    return true;
}

void RealSense2Device::initCalibration(rs2_intrinsics& intrinsics_cam, const std::string &calibrationFile)
{
    if (calibrationFile.empty())
    {
        CameraCalibration calib;
        calib.setWidth(getWidth());
        calib.setHeight(getHeight());

        cv::Matx33d K = cv::Matx33d::eye();
        K(0, 0) = intrinsics_cam.fx;
        K(1, 1) = intrinsics_cam.fy;
        K(0, 2) = intrinsics_cam.ppx;
        K(1, 2) = intrinsics_cam.ppy;
        calib.setIntrinsicsMatrix(K);

        std::vector<double> dists = {intrinsics_cam.coeffs[0], intrinsics_cam.coeffs[1], intrinsics_cam.coeffs[2], intrinsics_cam.coeffs[3], intrinsics_cam.coeffs[4]};
        calib.setDistortionParameters(dists);
        setCalibration(calib);

        std::cout << " fx = " << intrinsics_cam.fx << std::endl;
        std::cout << " fy = " << intrinsics_cam.fy << std::endl;
        std::cout << " cx = " << intrinsics_cam.ppx << std::endl;
        std::cout << " cy = " << intrinsics_cam.ppy << std::endl;
        std::cout << " height = " << intrinsics_cam.height << std::endl;
        std::cout << " width = " << intrinsics_cam.width << std::endl;
        std::cout << " Distortion Coeff = " << intrinsics_cam.coeffs[0] << ", " << intrinsics_cam.coeffs[1] << ", " <<
        intrinsics_cam.coeffs[2] << ", " << intrinsics_cam.coeffs[3] << ", " << intrinsics_cam.coeffs[4] << ", " << std::endl;
        std::cout << " Model = " << intrinsics_cam.model << std::endl;
    }
    else
    {
        CameraCalibration calibration;
        calibration.loadCalibration(calibrationFile);
        setCalibration(calibration);
    }
}

void RealSense2Device::captureImages(cv::Mat &color, double& timestamp)
{
    cv::Mat depth;
    captureImages(color, depth, timestamp);
}

void RealSense2Device::captureImages(cv::Mat &color, cv::Mat &depth, double& timestamp)
{
    rs2::frameset fs;

    {
        std::unique_lock<std::mutex> lk(m_imu_mutex);
        if(!m_image_ready)
            m_cond_image_rec.wait(lk);

        fs = m_fsLast;

        if(m_count_im_buffer>1)
            std::cout << m_count_im_buffer -1 << " dropped frs\n";
        m_count_im_buffer = 0;

        timestamp = m_timestamp_image;
        m_image_ready = false;
    }

    // Perform alignment here
    auto processed = m_align->process(fs);

    // Trying to get both other and aligned depth frames
    rs2::video_frame color_frame = processed.first(m_align_to);
    rs2::depth_frame depth_frame = processed.get_depth_frame();

    int w = getWidth();
    int h = getHeight();

    color = cv::Mat(cv::Size(w, h), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP).clone();
    depth = cv::Mat(cv::Size(w, h), CV_16U, (void*)(depth_frame.get_data()), cv::Mat::AUTO_STEP).clone();

}

}
#endif
