/**
 * @file main.cpp
 * @brief Unified video localization using SLAM backends (ORB-SLAM3 or Stella VSLAM).
 * 
 * This application localizes a video stream against a pre-built map or builds a new map.
 * 
 * Usage:
 *   ./localize_<backend> [backend-options] <folder> <video> <calib> [map]
 */

#ifdef STELLA_AVAILABLE
#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/util/yaml.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/publish/frame_publisher.h"
#include "iridescence_viewer/viewer.h"
#endif

#ifdef ORBSLAM3_AVAILABLE
#include <System.h>
#include <Atlas.h>
#endif

#include "DeviceFactory/DeviceFactory.h"
#include "DeviceFactory/Device.h"
#include "DeviceFactory/CameraCalibration.h"
#include "utils.h"
#include "common/camera_utils.h"
#include "common/trajectory_io.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <thread>
#include <mutex>

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " [options] <folder> <video> <calib> [map]\n\n";
    
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "Options for ORB-SLAM3:\n";
    std::cerr << "  --voc <path>      Path to ORB vocabulary file (ORBvoc.txt)\n";
    std::cerr << "  --config <path>   Path to ORB-SLAM3 config template file\n\n";
#endif

#ifdef STELLA_AVAILABLE
    std::cerr << "Options for Stella VSLAM:\n";
    std::cerr << "  --voc <path>      Path to ORB vocabulary file (orb_vocab.fbow)\n";
    std::cerr << "  --config <path>   Path to Stella config file (config.yaml)\n\n";
#endif

    std::cerr << "Required arguments:\n";
    std::cerr << "  folder   Base folder containing all data\n";
    std::cerr << "  video    Video file relative to folder\n";
    std::cerr << "  calib    Camera calibration file relative to folder\n\n";
    
    std::cerr << "Optional arguments:\n";
    std::cerr << "  map      Map file in <folder>/.cache (omit to create new map)\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "           For Stella: map.msg\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "           For ORB-SLAM3: map.osa\n";
#endif
    
    std::cerr << "\nExamples:\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "  # Localize with pre-built Stella map:\n";
    std::cerr << "  " << progName << " --voc /path/orb_vocab.fbow --config /path/config.yaml \\\n";
    std::cerr << "      /data/scene video.mp4 calib.txt map.msg\n\n";
    std::cerr << "  # Build new Stella map:\n";
    std::cerr << "  " << progName << " --voc /path/orb_vocab.fbow --config /path/config.yaml \\\n";
    std::cerr << "      /data/scene video.mp4 calib.txt\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "  # Localize with pre-built ORB-SLAM3 map:\n";
    std::cerr << "  " << progName << " --voc /path/ORBvoc.txt --config /path/template.yaml \\\n";
    std::cerr << "      /data/scene video.mp4 calib.txt map.osa\n\n";
    std::cerr << "  # Build new ORB-SLAM3 map:\n";
    std::cerr << "  " << progName << " --voc /path/ORBvoc.txt --config /path/template.yaml \\\n";
    std::cerr << "      /data/scene video.mp4 calib.txt\n";
#endif
}

#ifdef STELLA_AVAILABLE
int runStella(const std::filesystem::path& vocPath,
              const std::filesystem::path& configPath,
              const std::filesystem::path& folder,
              const std::filesystem::path& videoPath,
              const std::filesystem::path& calibFile,
              const std::filesystem::path& mapFile,
              bool useMap) {
    
    std::filesystem::path cacheDir = folder / ".cache";
    std::filesystem::create_directories(cacheDir);

    // Initialize device
    DeviceFactory::DeviceFactory df;
    auto device = df.createDevices("CVVideoCapture", videoPath.string(), calibFile.string());
    CameraCalibration calibration = device->getCalibration();

    // Setup undistortion if needed (handles both fisheye and pinhole distortion)
    CameraUndistorter undistorter(calibration);

    // Get effective calibration (undistorted if needed), then align to 16-pixel boundary
    CameraCalibration effectiveCalib = undistorter.isActive()
        ? undistorter.getUndistortedCalibration() : calibration;
    effectiveCalib = getAlignedCalibration(effectiveCalib, 16);

    // Save effective calibration for config generation
    std::filesystem::path effectiveCalibFile = cacheDir / "effective_calib.txt";
    effectiveCalib.saveCalibration(effectiveCalibFile.string());

    // Generate config from template
    std::filesystem::path stellaConfig = cacheDir / "stella_config.yaml";
    if (!createStellaConfig(configPath, effectiveCalibFile, stellaConfig)) {
        std::cerr << "Failed to create Stella config file" << std::endl;
        return 1;
    }

    // Load and configure Stella
    std::shared_ptr<stella_vslam::config> cfg;
    try {
        cfg = std::make_shared<stella_vslam::config>(stellaConfig);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    spdlog::set_level(spdlog::level::info);

    // Build SLAM system
    auto slam = std::make_shared<stella_vslam::system>(cfg, vocPath);
    
    if (useMap) {
        if (!slam->load_map_database(mapFile)) {
            std::cerr << "Failed to load map: " << mapFile << std::endl;
            return EXIT_FAILURE;
        }
        slam->startup(false);
        slam->disable_mapping_module();
        slam->disable_loop_detector();
        std::cout << "Loaded map: " << mapFile << std::endl;
    } else {
        slam->startup(true);
        std::cout << "Building new map..." << std::endl;
    }

    // Setup viewer
    std::mutex mtx_pause, mtx_terminate, mtx_step;
    bool is_paused = false;
    bool terminate_is_requested = false;
    unsigned int step_count = 0;

    auto iri_viewer = std::make_shared<iridescence_viewer::viewer>(
        stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "IridescenceViewer"),
        slam->get_frame_publisher(),
        slam->get_map_publisher());

    iri_viewer->add_checkbox("Pause", [&is_paused, &mtx_pause](bool check) {
        std::lock_guard<std::mutex> lock(mtx_pause);
        is_paused = check;
    });
    iri_viewer->add_button("Step", [&step_count, &mtx_step] {
        std::lock_guard<std::mutex> lock(mtx_step);
        step_count++;
    });
    iri_viewer->add_button("Reset", [&slam] {
        slam->request_reset();
    });
    iri_viewer->add_button("Save and exit", [&is_paused, &mtx_pause, &terminate_is_requested, &mtx_terminate, &iri_viewer] {
        std::lock_guard<std::mutex> lock1(mtx_pause);
        is_paused = false;
        std::lock_guard<std::mutex> lock2(mtx_terminate);
        terminate_is_requested = true;
        iri_viewer->request_terminate();
    });
    iri_viewer->add_close_callback([&is_paused, &mtx_pause, &terminate_is_requested, &mtx_terminate] {
        std::lock_guard<std::mutex> lock1(mtx_pause);
        is_paused = false;
        std::lock_guard<std::mutex> lock2(mtx_terminate);
        terminate_is_requested = true;
    });

    // Processing thread
    int processed_frames = 0;
    int tracked_frames = 0;
    auto trackingStart = std::chrono::high_resolution_clock::now();
    std::thread thread([&]() {
        while (true) {
            // Handle pause
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(mtx_pause);
                    if (!is_paused) break;
                }
                {
                    std::lock_guard<std::mutex> lock(mtx_step);
                    if (step_count > 0) {
                        step_count--;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }

            cv::Mat frame;
            double timestamp;
            device->captureImages(frame, timestamp);
            if (frame.empty()) break;
            processed_frames++;
            
            if (frame.size() != cv::Size(calibration.getWidth(), calibration.getHeight())) {
                cv::resize(frame, frame, cv::Size(calibration.getWidth(), calibration.getHeight()));
            }

            // Undistort (no-op if no distortion) and center-crop to aligned dimensions
            undistorter.undistort(frame);
            centerCrop(frame, effectiveCalib);

            slam->feed_monocular_frame(frame, timestamp);

            if(slam->get_frame_publisher()->get_tracking_state() == "Tracking")
            {
                tracked_frames++;
            }

            {
                std::lock_guard<std::mutex> lock(mtx_terminate);
                if (terminate_is_requested) break;
            }

            while (slam->loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }

            {
                std::lock_guard<std::mutex> lock(mtx_terminate);
                if (terminate_is_requested) break;
            }
        }
        iri_viewer->request_terminate();
    });

    iri_viewer->run();
    thread.join();
    slam->shutdown();

    // Print statistics
    auto trackingEnd = std::chrono::high_resolution_clock::now();
    auto trackingMs = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
    double avgTrackingMs = processed_frames > 0 ? static_cast<double>(trackingMs) / processed_frames : 0.0;

    std::cout << "\n=== Processing Statistics ===" << std::endl;
    std::cout << "Processed frames: " << processed_frames << std::endl;
    std::cout << "Tracked frames: " << tracked_frames << std::endl;
    if (processed_frames > 0) {
        std::cout << "Tracking rate: " << (100.0 * tracked_frames / processed_frames) << "%" << std::endl;
        std::cout << "Average tracking time: " << avgTrackingMs << " ms/frame" << std::endl;
    }

    // Save results
    std::filesystem::create_directories(folder / "estimated");

    // Query map stats
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>> all_keyfrms;
    unsigned int num_keyframes = slam->get_map_publisher()->get_keyframes(all_keyfrms);
    std::vector<std::shared_ptr<stella_vslam::data::landmark>> all_lms;
    std::set<std::shared_ptr<stella_vslam::data::landmark>> local_lms;
    unsigned int num_landmarks = slam->get_map_publisher()->get_landmarks(all_lms, local_lms);

    if (useMap) {
        std::string trajName = videoPath.stem().string() + "_stella_" + mapFile.stem().string() + ".txt";
        std::filesystem::path trajPath = folder / "estimated" / trajName;
        slam->save_frame_trajectory(trajPath, "TUM");

        // Write metadata sidecar
        std::ofstream meta(trajPath.replace_extension(".meta").string());
        writeTrajectoryHeader(meta, "prebuild_localize", "stella_vslam",
                              "", videoPath.filename().string(), "neural",
                              "map: " + mapFile.filename().string());
        meta << "# num_keyframes: " << num_keyframes << "\n";
        meta << "# num_landmarks: " << num_landmarks << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta << "# processed_frames: " << processed_frames << "\n";
        meta << "# tracked_frames: " << tracked_frames << "\n";
        meta << "# tracking_rate: " << std::fixed << std::setprecision(2) << (100.0 * tracked_frames / processed_frames) << "%\n";
        meta.close();

        std::cout << "Trajectory saved: " << trajPath << std::endl;
    } else {
        slam->save_map_database(cacheDir / "default.msg");

        // Write metadata sidecar for built map
        std::ofstream meta(cacheDir / "default_stella.meta");
        writeTrajectoryHeader(meta, "stella_vslam_map", "stella_vslam",
                              "", videoPath.filename().string(), "");
        meta << "# num_keyframes: " << num_keyframes << "\n";
        meta << "# num_landmarks: " << num_landmarks << "\n";
        meta << "# map_construction_time_ms: " << trackingMs << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta.close();

        std::cout << "Map saved to: " << (cacheDir / "default.msg") << std::endl;
    }

    return 0;
}
#endif

#ifdef ORBSLAM3_AVAILABLE
int runOrbSlam3(const std::filesystem::path& vocPath,
                const std::filesystem::path& configTemplate,
                const std::filesystem::path& folder,
                const std::filesystem::path& videoPath,
                const std::filesystem::path& calibFile,
                const std::filesystem::path& mapFile,
                bool useMap) {
    
    std::filesystem::path cacheDir = folder / ".cache";
    std::filesystem::create_directories(cacheDir);

    // Initialize device
    DeviceFactory::DeviceFactory df;
    auto device = df.createDevices("CVVideoCapture", videoPath.string(), calibFile.string());
    CameraCalibration calibration = device->getCalibration();

    // Setup undistortion if needed (handles both fisheye and pinhole distortion)
    CameraUndistorter undistorter(calibration);

    // Get effective calibration (undistorted if needed), then align to 16-pixel boundary
    CameraCalibration effectiveCalib = undistorter.isActive()
        ? undistorter.getUndistortedCalibration() : calibration;
    effectiveCalib = getAlignedCalibration(effectiveCalib, 16);

    // Save effective calibration for config generation
    std::filesystem::path effectiveCalibFile = cacheDir / "effective_calib.txt";
    effectiveCalib.saveCalibration(effectiveCalibFile.string());

    // Generate config
    std::filesystem::path orbConfig = cacheDir / "orbslam3_config.yaml";
    if (!createORBSLAM3Config(configTemplate, effectiveCalibFile, orbConfig, useMap)) {
        std::cerr << "Failed to create ORB-SLAM3 config file" << std::endl;
        return 1;
    }

    // Initialize ORB-SLAM3
    std::string loadFile = useMap ? mapFile.string() : std::string();
    ORB_SLAM3::System slam(vocPath.string(), orbConfig.string(),
                           ORB_SLAM3::System::MONOCULAR, false, 0, std::string(), loadFile, !useMap);

    // ORB_SLAM3::Verbose::SetTh(ORB_SLAM3::Verbose::eLevel::VERBOSITY_DEBUG);

    if (useMap) {
        slam.ActivateLocalizationMode();
        std::cout << "Localization mode activated with map: " << mapFile << std::endl;
    } else {
        std::cout << "Building new map..." << std::endl;
    }

    int frame_count = 0;
    int tracked_frames = 0;
    auto trackingStart = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat frame;
        double timestamp;
        device->captureImages(frame, timestamp);
        
        if (frame.empty() || slam.isShutDown()) {
            break;
        }

        if (frame.size() != cv::Size(calibration.getWidth(), calibration.getHeight())) {
            cv::resize(frame, frame, cv::Size(calibration.getWidth(), calibration.getHeight()));
        }

        // Undistort (no-op if no distortion) and center-crop to aligned dimensions
        undistorter.undistort(frame);
        centerCrop(frame, effectiveCalib);

        slam.TrackMonocular(frame, timestamp);

        if(slam.GetTrackingState() == ORB_SLAM3::Tracking::eTrackingState::OK){
            tracked_frames++;
        }
        frame_count++;
    }

    // Print statistics
    auto trackingEnd = std::chrono::high_resolution_clock::now();
    auto trackingMs = std::chrono::duration_cast<std::chrono::milliseconds>(trackingEnd - trackingStart).count();
    double avgTrackingMs = frame_count > 0 ? static_cast<double>(trackingMs) / frame_count : 0.0;

    std::cout << "\n=== Processing Statistics ===" << std::endl;
    std::cout << "Processed frames: " << frame_count << std::endl;
    std::cout << "Tracked frames: " << tracked_frames << std::endl;
    if (frame_count > 0) {
        std::cout << "Tracking rate: " << (100.0 * tracked_frames / frame_count) << "%" << std::endl;
        std::cout << "Average tracking time: " << avgTrackingMs << " ms/frame" << std::endl;
    }

    if(!slam.isShutDown())
    {
        slam.Shutdown();
    }

    while (!slam.AllShutdown()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Shutdown succeeded" << std::endl;

    // Save results
    std::filesystem::create_directories(folder / "estimated");

    if (useMap) {
        std::string trajName = videoPath.stem().string() + "_orb_" + mapFile.stem().string() + ".txt";
        std::filesystem::path trajPath = folder / "estimated" / trajName;
        slam.SaveTrajectoryTUM(trajPath);

        // Write metadata sidecar
        std::ofstream meta(trajPath.replace_extension(".meta").string());
        writeTrajectoryHeader(meta, "prebuild_localize", "orbslam3",
                              "", videoPath.filename().string(), "neural",
                              "map: " + mapFile.filename().string());
        meta << "# num_keyframes: " << slam.GetAtlas()->GetAllKeyFrames().size() << "\n";
        meta << "# num_landmarks: " << slam.GetAtlas()->GetAllMapPoints().size() << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta << "# processed_frames: " << frame_count << "\n";
        meta << "# tracked_frames: " << tracked_frames << "\n";
        meta << "# tracking_rate: " << std::fixed << std::setprecision(2) << (100.0 * tracked_frames / frame_count) << "%\n";
        meta.close();

        std::cout << "Trajectory saved: " << trajPath << std::endl;
    } else {
        // Write metadata sidecar for built map
        std::ofstream meta(cacheDir / "default_orbslam3.meta");
        writeTrajectoryHeader(meta, "orbslam3_map", "orbslam3",
                              "", videoPath.filename().string(), "");
        meta << "# num_keyframes: " << slam.GetAtlas()->GetAllKeyFrames().size() << "\n";
        meta << "# num_landmarks: " << slam.GetAtlas()->GetAllMapPoints().size() << "\n";
        meta << "# map_construction_time_ms: " << trackingMs << "\n";
        meta << "# avg_tracking_time_ms: " << std::fixed << std::setprecision(2) << avgTrackingMs << "\n";
        meta.close();
    }

    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse optional arguments
    std::string vocPath;
    std::string configPath;
    int argIdx = 1;

    while (argIdx < argc && argv[argIdx][0] == '-') {
        std::string arg = argv[argIdx];
        if (arg == "--voc" && argIdx + 1 < argc) {
            vocPath = argv[++argIdx];
        } else if (arg == "--config" && argIdx + 1 < argc) {
            configPath = argv[++argIdx];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        ++argIdx;
    }

    // Check required arguments
    if (argc - argIdx < 3) {
        std::cerr << "Missing required arguments.\n";
        printUsage(argv[0]);
        return 1;
    }

    if (vocPath.empty() || configPath.empty()) {
        std::cerr << "Missing --voc and/or --config options.\n";
        printUsage(argv[0]);
        return 1;
    }

    std::filesystem::path folder = argv[argIdx];
    std::filesystem::path videoFile = folder / argv[argIdx + 1];
    std::filesystem::path calibFile = folder / argv[argIdx + 2];
    std::filesystem::path cacheDir = folder / ".cache";

    bool useMap = (argc - argIdx > 3);
    std::filesystem::path mapFile;
    if (useMap) {
        mapFile = cacheDir / argv[argIdx + 3];
    }

    std::cout << "Folder: " << folder << std::endl;
    std::cout << "Video: " << videoFile << std::endl;
    std::cout << "Calibration: " << calibFile << std::endl;
    if (useMap) {
        std::cout << "Map: " << mapFile << std::endl;
    } else {
        std::cout << "Mode: Building new map" << std::endl;
    }

#ifdef STELLA_AVAILABLE
    return runStella(vocPath, configPath, folder, videoFile, calibFile, mapFile, useMap);
#endif

#ifdef ORBSLAM3_AVAILABLE
    return runOrbSlam3(vocPath, configPath, folder, videoFile, calibFile, mapFile, useMap);
#endif

    std::cerr << "No SLAM backend available!" << std::endl;
    return 1;
}
