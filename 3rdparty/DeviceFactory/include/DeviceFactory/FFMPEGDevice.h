#ifndef FFMPEGDEVICE_H
#define FFMPEGDEVICE_H

#include "Device.h"
#include "opencv2/core/mat.hpp"
#include <string>

class AVFrame;
class AVFormatContext;
class AVStream;
class SwsContext;
class AVCodecContext;

namespace DeviceFactory{

class FFMPEGDevice : public Device
{
public:
    class FFMPEGDeviceStream
    {
    public:
        FFMPEGDeviceStream();
        bool open(const std::string& file);

        cv::Mat getFrame();

        void close();


    private:
        AVFrame* decframe;
        AVFormatContext* inctx = nullptr;
        AVStream* vstrm;
        AVFrame* frame;
        std::vector<uint8_t> framebuf;
        SwsContext* swsctx;

        AVCodecContext* codecctx;

        int vstrm_idx;
        unsigned nb_frames = 0;
        bool end_of_stream = false;
        int got_pic = 0;

    public:
        int dst_width;
        int dst_height;
        int numberOfFrames;
        bool isOpened() const;
        bool seek(uint64_t frame);
    };

public:
    FFMPEGDevice();

    void captureImages(cv::Mat &color, double &timestamp);
    void captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp);

    std::shared_ptr<Device> createInstance();

    bool init(const std::string ID, const DeviceProperties &properties, const std::string& calibrationFile = "");

    void listAvailableDevices();

    std::string getDriver() { return "FFMPEG"; }

    virtual int numberOfFrames() const;

private:
    bool openCamera(FFMPEGDeviceStream &vc, const std::string &ID);
    void setSupportedOutputs();
    std::vector<std::string> split(const std::string &s, char delim);

    std::chrono::high_resolution_clock::time_point m_startTime;
    FFMPEGDeviceStream m_videoCaptureColor;
    FFMPEGDeviceStream m_videoCaptureDepth;

    void setResolution();

    cv::Mat BGR8BitToGrayscale16Bit(cv::Mat image);

    int m_framesInSequence;
    int m_frameID;
};
}

#endif // FFMPEGDEVICE_H
