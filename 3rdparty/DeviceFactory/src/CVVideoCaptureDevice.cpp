#include "CVVideoCaptureDevice.h"
#include <filesystem>
#include <fstream>

namespace DeviceFactory{

CVVideoCaptureDevice::CVVideoCaptureDevice()
{
    m_framesInSequence = -1;
    m_frameID = 0;

}

void CVVideoCaptureDevice::captureImages(cv::Mat &color, double &timestamp)
{
    timestamp = 0.0;
    if (m_videoCaptureColor.isOpened())
    {
        m_videoCaptureColor >> color;
        auto m_current = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_current - m_startTime);
        timestamp = duration.count()/1000.0;
    }

    if(!timestamps.empty())
    {
        timestamp = timestamps.front();
        timestamps.pop_front();
    }
}

cv::Mat CVVideoCaptureDevice::BGR8BitToGrayscale16Bit(cv::Mat image)
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


void CVVideoCaptureDevice::captureImages(cv::Mat &color, cv::Mat &depth, double &timestamp)
{
    timestamp = 0.0;

    if (m_framesInSequence == -1 || (m_frameID < m_framesInSequence))
    {
        if (m_videoCaptureColor.isOpened() && m_videoCaptureDepth.isOpened())
        {
            m_videoCaptureColor >> color;

            cv::Mat tmpDepth;
            m_videoCaptureDepth >> tmpDepth;

            std::cerr << "Empty_Internal: " << color.empty() << " " << tmpDepth.empty() << std::endl;
            // Reset if one is empty
            if (color.empty() || tmpDepth.empty())
            {
                color = cv::Mat();
                depth = cv::Mat();
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

    if(!timestamps.empty())
    {
        timestamp = timestamps.front();
        timestamps.pop_front();
    }
}

std::vector<std::string> CVVideoCaptureDevice::split(const std::string &s, char delim)
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

std::shared_ptr<Device> CVVideoCaptureDevice::createInstance()
{
    return std::make_shared<CVVideoCaptureDevice>();
}

bool CVVideoCaptureDevice::openCamera(cv::VideoCapture& vc, const std::string& ID)
{
    bool isOpen = false;
    if (!ID.empty() && std::isdigit(ID[0]))
    {
        std::istringstream iss(ID);
        int intID;
        iss >> intID;
        isOpen = vc.open(intID);
    }
    else
    {
        isOpen = vc.open(ID);
    }

    if (!isOpen)
    {
        std::cerr << "Failed to open camera with ID '" << ID << "'" << std::endl;
    }
    return isOpen;
}

void CVVideoCaptureDevice::setResolution()
{
    if (m_videoCaptureColor.isOpened())
    {
        cv::Mat initial;
        unsigned int width = m_videoCaptureColor.get(cv::CAP_PROP_FRAME_WIDTH);
        unsigned int height = m_videoCaptureColor.get(cv::CAP_PROP_FRAME_HEIGHT);
        setWidth(width);
        setHeight(height);
    }
    else if (m_videoCaptureDepth.isOpened())
    {
        cv::Mat initial;
        unsigned int width = m_videoCaptureDepth.get(cv::CAP_PROP_FRAME_WIDTH);
        unsigned int height = m_videoCaptureDepth.get(cv::CAP_PROP_FRAME_HEIGHT);
        setWidth(width);
        setHeight(height);
    }
}

void CVVideoCaptureDevice::setSupportedOutputs()
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

bool CVVideoCaptureDevice::init(const std::string ID, const DeviceProperties &properties, const std::string &calibrationFile)
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
                std::istringstream iss(subStrings[0]);
                int count;
                iss >> count;
                m_framesInSequence = count;
                isOpen = true;
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
            std::filesystem::path fsPath {path};
            auto tsPath = fsPath.parent_path() / (fsPath.stem().string() + "_timestamps.txt");

            if(std::filesystem::exists(tsPath))
            {
                std::ifstream file(tsPath);
                if (!file.is_open()) {
                    std::cerr << "Failed to open " << tsPath << "\n";
                }

                double val;
                while (file >> val) {  // reads line by line, or whitespace-separated
                    timestamps.push_back(val);
                }
            }
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

void CVVideoCaptureDevice::listAvailableDevices()
{
    std::cout << "--------------------------------" << std::endl;
    std::cout << "<path to video file> (assumes RGB)" << std::endl;
    std::cout << "<path to video file>#RGB (processes as RGB)" << std::endl;
    std::cout << "<path to video file>#Depth (processes as Depth image)" << std::endl;
    std::cout << "<path to video file 1>#RGB;<path to video file 2>#Depth (two cameras, first as RGB, second as Depth - this assumes the same resolution)" << std::endl;
    std::cout << "<path to video file 1>#RGB;<path to video file 2>#Depth;<frames>#Frames (see above, #Frames defines the number of frames in the sequence)" << std::endl;

    std::cout << "--------------------------------" << std::endl;
    std::cout << "<path to image file sequence> (e.g. img_%02d.jpg starting from 0)" << std::endl;
}

int CVVideoCaptureDevice::numberOfFrames() const
{
    if (m_framesInSequence > 0)
    {
        return m_framesInSequence;
    }
    else
    {
        if (m_videoCaptureColor.isOpened())
        {
            int frames = int(m_videoCaptureColor.get(cv::CAP_PROP_FRAME_COUNT));
            if (frames > 0)
                return frames;
            else
                return std::numeric_limits<int>::max();
        }
        else if (m_videoCaptureDepth.isOpened())
        {
            int frames = int(m_videoCaptureDepth.get(cv::CAP_PROP_FRAME_COUNT));
            if (frames > 0)
                return frames;
            else
                return std::numeric_limits<int>::max();
        }
    }
    return std::numeric_limits<int>::max();
}
}

