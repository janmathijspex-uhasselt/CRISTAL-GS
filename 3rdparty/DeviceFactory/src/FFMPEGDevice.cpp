#include "FFMPEGDevice.h"

#include <iostream>
#include <fstream>

#include <vector>
// FFmpeg
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace DeviceFactory{
FFMPEGDevice::FFMPEGDeviceStream::FFMPEGDeviceStream()
{
    inctx = nullptr;
    decframe = nullptr;
}

bool FFMPEGDevice::FFMPEGDeviceStream::open(const std::string &file)
{
    const char* infile = file.c_str();
    int ret;

    // open input file context
    inctx = nullptr;
    ret = avformat_open_input(&inctx, infile, nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avformat_open_input(\"" << infile << "\"): ret=" << ret;
        return false;
    }

    // retrieve input stream information
    ret = avformat_find_stream_info(inctx, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avformat_find_stream_info: ret=" << ret;
        return false;
    }

    // find best video stream
    const AVCodec* vcodec = nullptr;
    ret = av_find_best_stream(inctx, AVMEDIA_TYPE_VIDEO, -1, -1, &vcodec, 0);
    if (ret < 0) {
        std::cerr << "fail to av_find_best_stream: ret=" << ret;
        return false;
    }
    vstrm_idx = ret;
    vstrm = inctx->streams[vstrm_idx];

    // allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(vcodec);
    if (!codec_ctx) {
        std::cerr << "failed to allocate AVCodecContext\n";
        return false;
    }

    // copy codec parameters from stream to context
    ret = avcodec_parameters_to_context(codec_ctx, vstrm->codecpar);
    if (ret < 0) {
        std::cerr << "fail to avcodec_parameters_to_context: ret=" << ret;
        return false;
    }

    // open codec
    ret = avcodec_open2(codec_ctx, vcodec, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avcodec_open2: ret=" << ret;
        return false;
    }

    codecctx = codec_ctx; // store it for later use

    // print info
    std::cout
        << "infile: " << infile << "\n"
        << "format: " << inctx->iformat->name << "\n"
        << "vcodec: " << vcodec->name << "\n"
        << "size:   " << codecctx->width << 'x' << codecctx->height << "\n"
        << "fps:    " << av_q2d(vstrm->avg_frame_rate) << " [fps]\n"
        << "length: " << av_rescale_q(vstrm->duration, vstrm->time_base, {1,1000}) / 1000. << " [sec]\n"
        << "pixfmt: " << av_get_pix_fmt_name(codecctx->pix_fmt) << "\n"
        << "frame:  " << vstrm->nb_frames << "\n"
        << std::flush;

    numberOfFrames = vstrm->nb_frames;

    // setup scaler
    dst_width = codecctx->width;
    dst_height = codecctx->height;
    const AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;

    swsctx = sws_getCachedContext(
        nullptr, codecctx->width, codecctx->height, codecctx->pix_fmt,
        dst_width, dst_height, dst_pix_fmt,
        SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsctx) {
        std::cerr << "fail to sws_getCachedContext";
        return false;
    }

    std::cout << "output: " << dst_width << 'x' << dst_height << ',' << av_get_pix_fmt_name(dst_pix_fmt) << std::endl;

    // allocate frame
    frame = av_frame_alloc();
    decframe = av_frame_alloc();

    int buffer_size = av_image_get_buffer_size(dst_pix_fmt, dst_width, dst_height, 1);
    framebuf = std::vector<uint8_t>(buffer_size);
    av_image_fill_arrays(frame->data, frame->linesize, framebuf.data(), dst_pix_fmt, dst_width, dst_height, 1);

    return true;
}

bool FFMPEGDevice::FFMPEGDeviceStream::seek(uint64_t frame)
{
    // Convert frame number to timestamp using the stream's time_base
    int64_t seekTarget = av_rescale_q(frame, {1, static_cast<int>(av_q2d(vstrm->r_frame_rate))}, vstrm->time_base);

    if (av_seek_frame(inctx, vstrm_idx, seekTarget, AVSEEK_FLAG_BACKWARD) < 0)
        return false;

    // Flush decoder buffers
    avcodec_flush_buffers(codecctx);
    return true;
}

bool FFMPEGDevice::FFMPEGDeviceStream::isOpened() const
{
    return (decframe != nullptr);
}

cv::Mat FFMPEGDevice::FFMPEGDeviceStream::getFrame()
{
    cv::Mat result;

    if (!isOpened())
        return result;

    AVPacket pkt;
    av_init_packet(&pkt);

    while (true) {
        if (!end_of_stream) {
            int ret = av_read_frame(inctx, &pkt);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    end_of_stream = true;
                    av_packet_unref(&pkt);
                    pkt.data = nullptr;
                    pkt.size = 0;
                } else {
                    std::cerr << "fail to av_read_frame: ret=" << ret << std::endl;
                    return {};
                }
            } else {
                if (pkt.stream_index != vstrm_idx) {
                    av_packet_unref(&pkt);
                    continue;
                }
            }
        }

        // Send packet to decoder
        int ret = avcodec_send_packet(codecctx, &pkt);
        av_packet_unref(&pkt);
        if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
            std::cerr << "fail to avcodec_send_packet: ret=" << ret << std::endl;
            return {};
        }

        // Receive decoded frame
        ret = avcodec_receive_frame(codecctx, decframe);
        if (ret == AVERROR(EAGAIN)) {
            continue; // Need more packets
        } else if (ret == AVERROR_EOF) {
            return {}; // Decoder flushed
        } else if (ret < 0) {
            std::cerr << "fail to avcodec_receive_frame: ret=" << ret << std::endl;
            return {};
        }

        // Frame successfully decoded
        sws_scale(swsctx,
                  decframe->data, decframe->linesize,
                  0, decframe->height,
                  frame->data, frame->linesize);

        cv::Mat image(dst_height, dst_width, CV_8UC3, framebuf.data(), frame->linesize[0]);
        result = image.clone();

        std::cout << nb_frames << '\r' << std::flush;
        ++nb_frames;
        break;
    }

    return result;
}

void FFMPEGDevice::FFMPEGDeviceStream::close()
{
    av_frame_free(&decframe);
    av_frame_free(&frame);

    if (codecctx) {
        avcodec_free_context(&codecctx);
        codecctx = nullptr;
    }

    if (inctx) {
        avformat_close_input(&inctx);
        inctx = nullptr;
    }

    sws_freeContext(swsctx);
    swsctx = nullptr;

    vstrm = nullptr;
    vstrm_idx = -1;
    end_of_stream = false;
    nb_frames = 0;
}


FFMPEGDevice::FFMPEGDevice()
{
    m_framesInSequence = -1;
    m_frameID = 0;

}

void FFMPEGDevice::captureImages(cv::Mat &color, double &timestamp)
{
    timestamp = 0.0;
    if (m_videoCaptureColor.isOpened())
    {
        color = m_videoCaptureColor.getFrame();
        auto m_current = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_current - m_startTime);
        timestamp = duration.count()/1000.0;
    }
}

cv::Mat FFMPEGDevice::BGR8BitToGrayscale16Bit(cv::Mat image)
{
    assert(image.type() == CV_8UC3);

    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    channels.pop_back();

    cv::Mat BG_Image;
    cv::merge(channels, BG_Image);

    cv::Mat grayscale_Image = cv::Mat(image.rows, image.cols, CV_16UC1, BG_Image.ptr());

    return grayscale_Image.clone();
}


void FFMPEGDevice::captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp)
{
    timestamp = 0.0;

    if (m_framesInSequence == -1 || (m_frameID < m_framesInSequence))
    {
        if (m_videoCaptureColor.isOpened() && m_videoCaptureDepth.isOpened())
        {
            color = m_videoCaptureColor.getFrame();

            cv::Mat tmpDepth;
            tmpDepth = m_videoCaptureDepth.getFrame();;


            // Reset if one is empty
            if (color.empty() || tmpDepth.empty())
            {
                color = cv::Mat();
                depth = cv::Mat();
                std::cerr << "Empty_Internal: " << color.empty() << " " << tmpDepth.empty() << std::endl;
            }
            else
            {
                depth = BGR8BitToGrayscale16Bit(tmpDepth);
            }
        }
        ++m_frameID;
    }
    else
    {
        color = cv::Mat();
        depth = cv::Mat();
    }
    auto m_current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_current - m_startTime);
    timestamp = duration.count()/1000.0;
}

std::vector<std::string> FFMPEGDevice::split(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim))
    {
        result.push_back(item);
    }

    return result;
}

std::shared_ptr<Device> FFMPEGDevice::createInstance()
{
    return std::make_shared<FFMPEGDevice>();
}

bool FFMPEGDevice::openCamera(FFMPEGDeviceStream &vc, const std::string& ID)
{
    bool isOpen = false;
    if (!ID.empty())
    {
        isOpen = vc.open(ID);
    }

    if (!isOpen)
    {
        std::cerr << "Failed to open camera with ID '" << ID << "'" << std::endl;
    }
    return isOpen;
}

void FFMPEGDevice::setResolution()
{
    if (m_videoCaptureColor.isOpened())
    {
        cv::Mat initial;
        unsigned int width = m_videoCaptureColor.dst_width;
        unsigned int height = m_videoCaptureColor.dst_height;
        setWidth(width);
        setHeight(height);
    }
    else if (m_videoCaptureDepth.isOpened())
    {
        cv::Mat initial;
        unsigned int width = m_videoCaptureDepth.dst_width;
        unsigned int height = m_videoCaptureDepth.dst_height;
        setWidth(width);
        setHeight(height);
    }
}

void FFMPEGDevice::setSupportedOutputs()
{
    if (m_videoCaptureColor.isOpened() && m_videoCaptureDepth.isOpened())
    {
        setSupportedOutput(SUPPORTED_OUTPUTS::BGR_DEPTH);
    }
    else if (m_videoCaptureColor.isOpened())
    {
        setSupportedOutput(SUPPORTED_OUTPUTS::BGR);
    }
    else if (m_videoCaptureDepth.isOpened())
    {
        setSupportedOutput(SUPPORTED_OUTPUTS::DEPTH);
    }

}

bool FFMPEGDevice::init(const std::string ID, const DeviceProperties &properties, const std::string &calibrationFile)
{
    m_framesInSequence = -1;
    m_frameID = 0;

    setInitInfo(ID, properties, calibrationFile);

    // Split ID string by seperator
    std::vector<std::string> subParts = split(ID, ';');
    for (int i = 0; i < subParts.size(); ++i)
    {
        bool isOpen = false;

        std::string cameraString = subParts[i];
        auto subStrings = split(cameraString, '#');
        std::string path = subStrings[0];
        if (subStrings.size() > 1)
        {
            if (subStrings[1] == "RGB") // Process as RGB camera
            {
                isOpen = openCamera(m_videoCaptureColor, path);
            }
            else if (subStrings[1] == "Depth") // Process as Depth camera
            {
                isOpen = openCamera(m_videoCaptureDepth, path);
            }
            else if (subStrings[1] == "Frames")
            {
                std::string framesString = subStrings[0];
                bool isNumber = true;
                for (unsigned int c = 0; c < framesString.size() && isNumber; ++c)
                    isNumber = std::isdigit(c);

                if (isNumber)
                {
                    std::istringstream iss(subStrings[0]);
                    int count;
                    iss >> count;
                    m_framesInSequence = count;
                    isOpen = true;
                }
                else // If not a number: assume file
                {
                    std::ifstream fframes;
                    fframes.open(framesString);
                    if (fframes.is_open())
                    {
                        int count;
                        fframes >> count;
                        m_framesInSequence = count;
                        isOpen = true;
                    }
                    else
                    {
                        std::cerr << "Failed to open frames file: '" << framesString << "'" << std::endl;
                    }

                }

            }
            else
            {
                std::cerr << "Unkown camera type '" << subStrings[1] << "'" << std::endl;
                return false;
            }
        }
        else // No type defined: open as RGB
        {
            isOpen = openCamera(m_videoCaptureColor, path);
        }

        if (!isOpen)
        {
            std::cerr << "Failed to open camera. " << std::endl;
            return false;
        }
    }

    setResolution();

    if (!calibrationFile.empty())
    {
        CameraCalibration calibration;
        calibration.loadCalibration(calibrationFile);
        setCalibration(calibration);
    }

    setSupportedOutputs();

    m_startTime = std::chrono::high_resolution_clock::now();
    return true;
}

void FFMPEGDevice::listAvailableDevices()
{
    std::cout << "--------------------------------" << std::endl;
    std::cout << "<path to video file> (assumes RGB)" << std::endl;
    std::cout << "<path to video file>#RGB (processes as RGB)" << std::endl;
    std::cout << "<path to video file>#Depth (processes as Depth image)" << std::endl;
    std::cout << "<path to video file 1>#RGB;<path to video file 2>#Depth (two cameras, first as RGB, second as Depth - this assumes the same resolution)" << std::endl;
    std::cout << "<path to video file 1>#RGB;<path to video file 2>#Depth;<frames>#Frames (see above, #Frames defines the number of frames in the sequence)" << std::endl;

    std::cout << "--------------------------------" << std::endl;
}

int FFMPEGDevice::numberOfFrames() const
{
    if (m_framesInSequence > 0)
    {
        return m_framesInSequence;
    }
    else
    {
        if (m_videoCaptureColor.isOpened())
        {
            int frames = int(m_videoCaptureColor.numberOfFrames);
            if (frames > 0)
                return frames;
            else
                return std::numeric_limits<int>::max();
        }
        else if (m_videoCaptureDepth.isOpened())
        {
            int frames = int(m_videoCaptureDepth.numberOfFrames);
            if (frames > 0)
                return frames;
            else
                return std::numeric_limits<int>::max();
        }
    }
    return std::numeric_limits<int>::max();
}

}
