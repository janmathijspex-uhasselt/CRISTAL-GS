#include <iostream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "DeviceFactory/DeviceFactory.h"
#include "DeviceFactory/Device.h"
#include "DeviceFactory/CameraCalibration.h"
#include "localizer.h"
#include "common/camera_utils.h"
#include "common/trajectory_io.h"
#include <torch/torch.h>
#include <csignal>

bool run = true;
void signalHandler(int)
{
    std::cout << "Caught keyboard interrupt." << std::endl;
    run = false;
}

void printUsage(const char* progName)
{
    std::cerr << "Usage: " << progName << " <mode> [options] <arguments...>\n\n";

    std::cerr << "Modes:\n";
    std::cerr << "  render    - Render cubemap frames from point cloud to cache\n";
    std::cerr << "  map       - Build XFeat feature map from cached cubemap frames\n";
    std::cerr << "  localize  - Localize a video against a pre-built feature map\n";
    std::cerr << "  full      - Render, build map, and localize in one step\n\n";

    std::cerr << "Options:\n";
    std::cerr << "  --no-neural       Use point projection instead of neural rendering\n";
    std::cerr << "  --poses <path>    Load map-building poses from file (render/full modes)\n\n";

    std::cerr << "Arguments per mode:\n";
    std::cerr << "  render:    " << progName << " render   [opts] <folder> <cloud> <calibration>\n";
    std::cerr << "  map:       " << progName << " map      [opts] <folder> <cloud> <calibration>\n";
    std::cerr << "  localize:  " << progName << " localize [opts] <folder> <cloud> <video> <calibration>\n";
    std::cerr << "  full:      " << progName << " full     [opts] <folder> <cloud> <video> <calibration>\n\n";

    std::cerr << "Arguments:\n";
    std::cerr << "  folder         Folder containing all data\n";
    std::cerr << "  cloud          Point cloud filename relative to folder\n";
    std::cerr << "  video          Video filename relative to folder\n";
    std::cerr << "  calibration    Calibration filename relative to folder\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "render" && mode != "map" && mode != "localize" && mode != "full") {
        std::cerr << "Unknown mode: " << mode << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Parse optional flags
    bool useNeural = true;
    std::string posesFile;
    int argIdx = 2;
    while (argIdx < argc && argv[argIdx][0] == '-') {
        std::string arg = argv[argIdx];
        if (arg == "--no-neural") {
            useNeural = false;
        } else if (arg == "--poses" && argIdx + 1 < argc) {
            posesFile = argv[++argIdx];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
        ++argIdx;
    }

    // Determine required positional arguments based on mode
    bool needsVideo = (mode == "localize" || mode == "full");
    int requiredArgs = needsVideo ? 4 : 3;  // folder cloud [video] calibration

    if (argc - argIdx < requiredArgs) {
        std::cerr << "Missing required arguments for '" << mode << "' mode.\n";
        printUsage(argv[0]);
        return 1;
    }

    // Parse positional arguments
    std::filesystem::path folder   = argv[argIdx];
    std::filesystem::path pcl_name = argv[argIdx + 1];
    std::filesystem::path video_name;
    std::filesystem::path calib_arg;

    if (needsVideo) {
        video_name = argv[argIdx + 2];
        calib_arg  = argv[argIdx + 3];
    } else {
        calib_arg  = argv[argIdx + 2];
    }

    std::filesystem::path path_to_cloud = folder / pcl_name;
    std::filesystem::path calib_path    = folder / calib_arg;
    std::filesystem::path cacheDir      = folder / ".cache";
    std::filesystem::create_directories(cacheDir);

    // ==== Load calibration & prepare effective calibration ====
    CameraCalibration effectiveCalib;
    std::shared_ptr<DeviceFactory::Device> device_cam;
    std::unique_ptr<CameraUndistorter> undistorter;

    if (needsVideo) {
        DeviceFactory::DeviceFactory df;
        std::filesystem::path color = folder / video_name;
        device_cam = df.createDevices("CVVideoCapture", color.string(), calib_path);
        CameraCalibration calibration = device_cam->getCalibration();

        undistorter = std::make_unique<CameraUndistorter>(calibration);
        effectiveCalib = undistorter->isActive()
            ? undistorter->getUndistortedCalibration() : calibration;
    } else {
        CameraCalibration calibration;
        calibration.loadCalibration(calib_path.string());
        effectiveCalib = calibration;
    }
    effectiveCalib = getAlignedCalibration(effectiveCalib, 16);

    // ==== INIT localizer ====
    Localizer localizer{effectiveCalib, cacheDir, useNeural};
    localizer.setCloudStem(pcl_name.stem().string());

    // ==== Execute mode ====
    if (mode == "render") {
        localizer.renderToCache(path_to_cloud, posesFile);
        return 0;
    }

    if (mode == "map") {
        localizer.buildMapFromCache();
        return 0;
    }

    if (mode == "localize") {
        if (!localizer.loadMap()) {
            std::cerr << "Feature map not found: " << localizer.getMapBinPath() << "\n";
            std::cerr << "Run 'render' and 'map' modes first." << std::endl;
            return 1;
        }
        localizer.initProjector(path_to_cloud);
        localizer.start();

        std::signal(SIGINT, signalHandler);
        auto trackingStart = std::chrono::high_resolution_clock::now();
        while (run) {
            double ts_cam;  cv::Mat frame_cam;
            device_cam->captureImages(frame_cam, ts_cam);
            if (frame_cam.rows <= 0) break;

            if (frame_cam.size() != cv::Size(effectiveCalib.getWidth(), effectiveCalib.getHeight())) 
            {
                cv::resize(frame_cam, frame_cam, cv::Size(effectiveCalib.getWidth(), effectiveCalib.getHeight()));
            }

            undistorter->undistort(frame_cam);
            centerCrop(frame_cam, effectiveCalib);

            localizer.process_frame(ts_cam, frame_cam);
        }
        auto trackingEnd = std::chrono::high_resolution_clock::now();
        auto trackingMs = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();

        int processed = localizer.getProcessedFrames();
        int tracked = localizer.getTrackedFrames();
        double avgTrackingMs = processed > 0 ? static_cast<double>(trackingMs) / processed : 0.0;

        std::cout << "\n=== Processing Statistics ===" << std::endl;
        std::cout << "Processed frames: " << processed << std::endl;
        std::cout << "Tracked frames: " << tracked << std::endl;
        if (processed > 0) {
            std::cout << "Tracking rate: " << (100.0 * tracked / processed) << "%" << std::endl;
            std::cout << "Average tracking time: " << avgTrackingMs << " ms/frame" << std::endl;
        }

        std::string renderTag = useNeural ? "neural" : "projected";
        std::string trajName  = video_name.stem().string() + "_rm_"
            + pcl_name.stem().string() + "_" + renderTag + ".txt";
        auto trajPath = folder / "estimated" / trajName;
        localizer.save_trajectory(trajPath, "TUM",
                                  pcl_name.string(), video_name.string(), renderTag);

        // Write metadata sidecar
        std::ofstream meta(trajPath.replace_extension(".meta").string());
        writeTrajectoryHeader(meta, "render_match", "xfeat",
                              pcl_name.string(), video_name.string(), renderTag);
        meta << "# num_keyframes: " << localizer.getKeyframeCount() << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta << "# processed_frames: " << processed << "\n";
        meta << "# tracked_frames: " << tracked << "\n";
        meta << "# tracking_rate: " << std::fixed << std::setprecision(2) << (100.0 * tracked / processed) << "%\n";
        meta.close();

        return 0;
    }

    if (mode == "full") {
        // Build map if not already cached
        if (!localizer.loadMap()) {
            localizer.buildMap(path_to_cloud, posesFile);
        }
        localizer.initProjector(path_to_cloud);
        localizer.start();

        std::signal(SIGINT, signalHandler);
        auto trackingStart = std::chrono::high_resolution_clock::now();
        while (run) {
            double ts_cam;  cv::Mat frame_cam;
            device_cam->captureImages(frame_cam, ts_cam);
            if (frame_cam.rows <= 0) break;

            undistorter->undistort(frame_cam);
            centerCrop(frame_cam, effectiveCalib);

            localizer.process_frame(ts_cam, frame_cam);
        }
        auto trackingEnd = std::chrono::high_resolution_clock::now();
        auto trackingMs = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();

        int processed = localizer.getProcessedFrames();
        int tracked = localizer.getTrackedFrames();
        double avgTrackingMs = processed > 0 ? static_cast<double>(trackingMs) / processed : 0.0;

        std::cout << "\n=== Processing Statistics ===" << std::endl;
        std::cout << "Processed frames: " << processed << std::endl;
        std::cout << "Tracked frames: " << tracked << std::endl;
        if (processed > 0) {
            std::cout << "Tracking rate: " << (100.0 * tracked / processed) << "%" << std::endl;
            std::cout << "Average tracking time: " << avgTrackingMs << " ms/frame" << std::endl;
        }

        std::string renderTag = useNeural ? "neural" : "projected";
        std::string trajName  = video_name.stem().string() + "_rm_"
            + pcl_name.stem().string() + "_" + renderTag + ".txt";
        auto trajPath = folder / "estimated" / trajName;
        localizer.save_trajectory(trajPath, "TUM",
                                  pcl_name.string(), video_name.string(), renderTag);

        // Write metadata sidecar
        std::ofstream meta(trajPath.replace_extension(".meta").string());
        writeTrajectoryHeader(meta, "render_match", "xfeat",
                              pcl_name.string(), video_name.string(), renderTag);
        meta << "# num_keyframes: " << localizer.getKeyframeCount() << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta << "# processed_frames: " << processed << "\n";
        meta << "# tracked_frames: " << tracked << "\n";
        meta << "# tracking_rate: " << std::fixed << std::setprecision(2) << (100.0 * tracked / processed) << "%\n";
        meta.close();

        return 0;
    }

    return 0;
}
