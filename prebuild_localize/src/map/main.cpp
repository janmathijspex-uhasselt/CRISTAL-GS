/**
 * @file main.cpp
 * @brief Unified map builder supporting multiple SLAM backends with frame caching.
 *
 * This application can:
 * 1. Render frames from a point cloud and save them to a cache
 * 2. Load cached frames and build a map with a specific backend
 *
 * This allows building maps with different SLAM backends using the exact same frames.
 *
 * Usage:
 *   Render frames only (no SLAM):
 *     ./prebuild_map_<backend> render <folder> <calib> <cloud>
 *
 *   Build map from cached frames:
 *     ./prebuild_map_<backend> build [backend-options] <folder> <calib>
 *
 *   Render and build in one step:
 *     ./prebuild_map_<backend> full [backend-options] <folder> <calib> <cloud>
 */

#include "map/slam_backend.h"
#include "map/frame_cache.h"
#include "map/pcl_gs_alignment.h"
#include "common/pose_generator.h"
#include "common/pose_io.h"
#include "common/geometry.h"
#include "common/camera_utils.h"
#include "common/trajectory_io.h"

#ifdef STELLA_AVAILABLE
#include "map/stella_backend.h"
#endif

#ifdef ORBSLAM3_AVAILABLE
#include "map/orbslam3_backend.h"
#endif

#include "RTRenderer/cloudreader.h"
#include "RTRenderer/project_cloud.h"
#include "DeviceFactory/CameraCalibration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <memory>
#include <vector>
#include <cstdlib>
#include <algorithm>

namespace Align = PclGsAlignment;

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <mode> <calib> [options]\n\n";
    std::cerr << "Modes:\n";
    std::cerr << "  render  - Render and cache frames only (no SLAM), requires <cloud>\n";
    std::cerr << "  build   - Build map from cached frames (no rendering)\n";
    std::cerr << "  full    - Render frames and build map in one step, requires <cloud>\n\n";

    std::cerr << "Options (render/full modes):\n";
    std::cerr << "  --pcl <path>      Pointcloud file\n";
    std::cerr << "  --gs <path>       Gaussian splat folder\n";
    std::cerr << "  --lod <int>       Level of detail of gaussian splat. Default: 0 [IGNORED when --gs is NOT given]\n";
    std::cerr << "  --poses <path>    Load camera poses from file instead of interactive GUI\n";
    std::cerr << "  --gs-repo <path>    Path to gaussian splatting repository. Default: /home/pex/Werk/gaussian-splatting/ \n";
    std::cerr << "  --conda-env <name>    Name of conda environment. Default: gaussian_splatting \n";
    std::cerr << "  --alignment <path>    Load alignment transform between pcl and gs from file, if not given calculated automatically. [ONLY possible when both --pcl and --gs are given]\n";
    std::cerr << "  --no-neural       Use point projection (computeRGBD) instead of neural rendering. [ONLY possible when --pcl is given]\n\n";

#ifdef STELLA_AVAILABLE
    std::cerr << "Options for Stella VSLAM (build/full modes):\n";
    std::cerr << "  --features <orb|liftfeat>   Feature type for keypoints/descriptors. Default: orb\n";
    std::cerr << "  --liftfeat-model <path>     LiftFeat ONNX model (.onnx.data must be next to it).\n";
    std::cerr << "                              Default: " << PREBUILD_CONFIGS_DIR << "/liftfeat_model/liftfeat_core.onnx\n\n";
#else
    std::cerr << "  --features / --liftfeat-model are accepted but ignored (ORB-SLAM3 only supports ORB)\n\n";
#endif

#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "Options for ORB-SLAM3 (build/full modes):\n";
    std::cerr << "  --voc <path>      Path to ORB vocabulary file (ORBvoc.txt)\n";
    std::cerr << "  --config <path>   Path to ORB-SLAM3 config template file\n\n";
#endif

    std::cerr << "Required arguments:\n";
    std::cerr << "  calib    Camera calibration file\n";
    std::cerr << "  At least one or both:\n";
    std::cerr << "    --pcl <path>   Pointcloud file\n";
    std::cerr << "    --gs <path>    Gaussian splat folder\n\n";

    std::cerr << "Examples:\n";
    std::cerr << "  # Render only using Gaussian Splatting\n";
    std::cerr << "  " << progName
              << " render calib.txt --gs /data/scene_gs --lod 2\n\n";

    std::cerr << "  # Build only using a point cloud\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "  " << progName
              << " build calib.txt --pcl cloud.ply\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "  " << progName
              << " build calib.txt --pcl cloud.ply "
              << "--voc /path/ORBvoc.txt --config /path/template.yaml\n";
#endif

    std::cerr << "\n  # Full pipeline using both Gaussian Splatting and Point Cloud\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "  " << progName << " full calib.txt --pcl cloud.ply --gs /data/scene_gs " << "--lod 2\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "  " << progName
              << " full calib.txt --pcl cloud.ply --gs /data/scene_gs "
              << "--lod 2 --voc /path/ORBvoc.txt "
              << "--config /path/template.yaml\n";
#endif
}

Eigen::Matrix4d calculateAlignment(std::filesystem::path pclFile, std::filesystem::path cacheDir, std::filesystem::path gsFolder, const int lod = 0) {
    // =========================================================================
    // TODO: PCL ↔ GS alignment (use this block when both --pcl and --gs are given)
    // =========================================================================

    // --- Step 1: Load data ---
    auto pclCloud = Align::loadPCLCloud(pclFile, cacheDir);
    auto gsCloud  = Align::loadGSCentroids(gsFolder, lod);
    //
    //   // Load full Gaussian data for covariance-informed GICP (recommended):
    auto gsData   = Align::loadGSData(gsFolder, lod);

    // --- Step 2: Run alignment ---
    // Tune voxelLeaf / normalRadius / fpfhRadius to your scene scale:
    Eigen::Matrix4d pcl2gs = Align::alignPCLtoGS(
        pclCloud, gsCloud, gsData,
          /*voxelLeaf=*/   0.2f,   // 20 cm voxel downsampling
          /*normalRadius=*/0.5f,   // 50 cm normal estimation radius
          /*fpfhRadius=*/  1.0f    // 1 m FPFH feature radius
    );

    // Or load a previously saved transform instead:
    // Eigen::Matrix4d pcl2gs = loadAlignmentTransform(alignmentFile);

    return pcl2gs;
}

void setupColmapDataset(
    const std::filesystem::path& colmapDir,
    const CameraCalibration& calib)  // in pixels
{
    std::filesystem::path sparseDir = colmapDir / "sparse" / "0";
    std::filesystem::path imagesDir = colmapDir / "images";
    std::filesystem::create_directories(sparseDir);
    std::filesystem::create_directories(imagesDir);

    // --- cameras.txt ---
    // Formaat: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    // SIMPLE_PINHOLE: fx, cx, cy
    // PINHOLE:        fx, fy, cx, cy  ← gebruik dit als fx != fy
    {
        std::ofstream f(sparseDir / "cameras.txt");
        f << "# Camera list with one line of data per camera:\n";
        f << "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        f << "# Number of cameras: 1\n";

        double cx = calib.getPrincipalPointX();
        double cy = calib.getPrincipalPointY();
        double fx = calib.getFocalLengthX();
        double fy = calib.getFocalLengthY();
        int width = calib.getWidth();
        int height = calib.getHeight();

        f << "1 PINHOLE "
          << width << " " << height << " "
          << fx << " " << fy << " "
          << cx << " " << cy << "\n";
    }

    // --- points3D.txt --- (leeg, enkel header)
    {
        std::ofstream f(sparseDir / "points3D.txt");
        f << "# 3D point list with one line of data per point:\n";
        f << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n";
        f << "# Number of points: 0\n";
    }
}

bool renderFrames(const std::string& pclFile,
                  const std::string& gsFolder,
                  const int& lod,
                  const std::filesystem::path& cacheDir,
                  const CameraCalibration& calib,
                  FrameCache& cache,
                  const std::string& posesFile = "",
                  const std::string& alignmentFile = "",
                  const std::string& condaEnv = "gaussian_splatting",
                  const std::string& gsRepo = "/home/pex/Werk/gaussian-splatting/",
                  bool useNeural = true) {

    // PCL params
    std::filesystem::path cloudFile;
    std::unordered_map<int, OctreeGrid::Block> grid;
    std::string engineName = std::string();

    // GS params
    int bestIter = -1;

    std::cout << "\n=== Rendering Frames ===" << std::endl;

    // Init pcl rendering stuff
    if (!pclFile.empty()) {
        cloudFile = std::filesystem::path(pclFile);
        grid = CloudReader::loadCloud(cloudFile, cacheDir);

        std::cout << "Loading point cloud from: " << cloudFile << std::endl;
        std::string renderTag = useNeural ? "neural" : "projected";
        std::cout << "Rendering mode: " << renderTag << std::endl;

        if (useNeural) engineName = std::string("trt_") + std::to_string(calib.getWidth()) + "x" + std::to_string(calib.getHeight()) + ".ts";
    }
    if (!gsFolder.empty()) {
        // Find highest iteration point cloud in GS folder
        std::filesystem::path gs_plyDir = std::filesystem::path(gsFolder) / "point_cloud";
        std::string ply_file_name = "point_cloud" + ((lod != 0) ? ("_" + std::to_string(lod)) : "") + ".ply";

        std::filesystem::path bestPly;
        for (const auto& entry : std::filesystem::directory_iterator(gs_plyDir)) {
            if (!entry.is_directory()) continue;
            std::string dirName = entry.path().filename().string();
            // Match "iteration_XXXXX"
            if (dirName.rfind("iteration_", 0) == 0) {
                int iter = std::stoi(dirName.substr(10));
                if (iter > bestIter) {
                    std::filesystem::path candidate = entry.path() / ply_file_name;
                    if (std::filesystem::exists(candidate)) {
                        bestIter = iter;
                        bestPly = candidate;
                    }
                }
            }
        }
        if (bestPly.empty()) {
            std::cerr << "No point_cloud.ply found in: " << gs_plyDir << std::endl;
            return false;
        }
        std::cout << "Using GS point cloud: " << bestPly << " (iteration " << bestIter << ")" << std::endl;

        if (pclFile.empty()) {
            cloudFile = bestPly;
            grid = CloudReader::loadCloud(cloudFile, cacheDir);
        }

        std::cout << "Loading gaussian splat from: " << cloudFile << std::endl;
    }

    std::vector<cv::Matx44d> trajectory_pcl;
    if (!posesFile.empty()) {
        std::cout << "Loading camera poses from: " << posesFile << std::endl;
        trajectory_pcl = loadPoses(posesFile);
        if (trajectory_pcl.empty()) {
            std::cerr << "Failed to load poses or file is empty: " << posesFile << std::endl;
            return false;
        }
    } else {
        std::cout << "Generating camera poses interactively..." << std::endl;
        trajectory_pcl = PoseGenerator::getPoses(grid);
        // Save the generated poses so they can be reused later
        savePoses(cacheDir / "generated_poses.txt", trajectory_pcl);
    }
    std::cout << "Using " << trajectory_pcl.size() << " poses." << std::endl;

    // Setup colmap dataset
    std::filesystem::path colmapDir = cacheDir / "colmap";
    std::filesystem::path sparseDir = colmapDir / "sparse" / "0";
    std::filesystem::path imagesDir = colmapDir / "images";

    setupColmapDataset(colmapDir, calib);

    // After taking the trajectory_pcl calculate transform and apply to camera trajectory_pcl
    cv::Mat dummy(calib.getHeight(), calib.getWidth(), CV_8UC3, cv::Scalar(0,0,0));

    cv::Matx44d gs2pcl, pcl2gs_cv;
    Eigen::Matrix4d pcl2gs = Eigen::Matrix4d::Identity();
    if (!gsFolder.empty()) {
        if (!pclFile.empty()) {
            std::cout << "Calculating pcl2gs transform..." << std::endl;
            pcl2gs = calculateAlignment(pclFile, cacheDir, gsFolder, lod);
            cv::eigen2cv(pcl2gs, pcl2gs_cv);
            cv::invert(pcl2gs_cv, gs2pcl);
        }

        // COLMAP file aanmaken
        std::string filename = sparseDir.string() + "/images.txt";
        std::ofstream colmap_file(filename);
        if (!colmap_file.is_open()) {
            std::cerr << "Can't open file: " << filename << std::endl;
        }

        // COLMAP header
        colmap_file << "# Image list with two lines of data per image:\n";
        colmap_file << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n";
        colmap_file << "# POINTS2D[] as (X, Y, POINT3D_ID)\n";

        int totalFrames = static_cast<int>(trajectory_pcl.size());
        int padWidth = std::max(static_cast<int>(std::to_string(std::max(totalFrames - 1, 0)).length()), 5);

        int i = 0;
        for (auto pose_pcl : trajectory_pcl) {
            cv::Matx44d pose_gs = pose_pcl;

            if (!pclFile.empty()) {
                pose_gs = pose_pcl * gs2pcl;
            }

            // extract R,t
            cv::Matx33d R_cv = pose_gs.get_minor<3,3>(0,0);
            cv::Vec3d t(pose_gs(0,3), pose_gs(1,3), pose_gs(2,3));

            // naar quaternion via Eigen
            Eigen::Matrix3d R;
            cv::cv2eigen(R_cv, R);
            Eigen::Quaterniond q(R);

            std::ostringstream ossName;
            ossName << std::setw(padWidth) << std::setfill('0') << i;
            std::string imgName = ossName.str() + ".jpg";

            colmap_file << (i + 1) << " "
                 << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                 << t[0] << " " << t[1] << " " << t[2] << " "
                 << 1 << " " // camera id
                 << imgName << "\n";

            colmap_file << "\n";

            // Save image per colmap pose
            cv::imwrite(imagesDir.string() + "/" + imgName, dummy);

            i++;
        }
        colmap_file.flush();
        colmap_file.close();
    }

    if (!cache.initForWriting()) {
        std::cerr << "Failed to initialize frame cache for writing" << std::endl;
        return false;
    }

    // Render pcl images
    std::vector<FrameCache::Frame> pclFrames;
    if (!pclFile.empty()) {
        ProjectCloud projector(grid, engineName);
        int frameIdx = 0;
        for (const auto& w2cam : trajectory_pcl) {
            std::cout << "Rendering frame " << frameIdx << "/" << trajectory_pcl.size() << "\r" << std::flush;

            cv::Mat rgb(calib.getHeight(), calib.getWidth(), CV_8UC3);
            cv::Mat depth(calib.getHeight(), calib.getWidth(), CV_32F);

            if (useNeural)
                projector.computeFull(calib, w2cam, &rgb, &depth);
            else
                projector.computeRGBD(calib, w2cam, &rgb, &depth);

            double timestamp = static_cast<double>(frameIdx) / 30.0;  // Simulated 30 FPS
            Eigen::Matrix4d pose_cw;
            cv::cv2eigen(w2cam, pose_cw);

            pclFrames.push_back(FrameCache::Frame(rgb, depth, pose_cw, timestamp));

            ++frameIdx;

            // OLD
            /*
            if (!cache.saveFrame(rgb, depth, pose_cw, timestamp)) {
                 std::cerr << "\nFailed to save frame " << frameIdx << std::endl;
                 return false;
            }
            */
        }
        std::cout << std::endl;
    }

    // Render gs images
    std::vector<FrameCache::Frame> gsFrames;
    if (!gsFolder.empty()) {
        std::cout << "Rendering gaussian splat..." << std::endl;
        std::string slash = gsRepo.ends_with('/') ? "" : "/";
        std::string cmd = "conda run -n " + condaEnv + " python " + gsRepo + slash + "render.py -m " + gsFolder + " -s " + colmapDir.string();
        int ret = std::system(cmd.c_str());

        if (ret != 0) {
            std::cerr << "Failed to render gs images" << std::endl;
            return false;
        }

        // Load images from folder
        std::filesystem::path gsRenders = std::filesystem::path(gsFolder) / "train" / ("ours_" + std::to_string(bestIter)) / "renders";
        std::filesystem::path gsDepthImages = std::filesystem::path(gsFolder) / "train" / ("ours_" + std::to_string(bestIter)) / "depth_images";

        int frameIdx = 0;
        for (const auto& w2cam : trajectory_pcl) {
            std::ostringstream oss;
            oss << std::setw(5) << std::setfill('0') << frameIdx;
            std::string filename = oss.str();

            std::filesystem::path rgbPath   = gsRenders     / (filename + ".png");
            std::filesystem::path depthPath = gsDepthImages / (filename + ".exr");

            cv::Mat rgb   = cv::imread(rgbPath.string(),   cv::IMREAD_COLOR);
            cv::Mat depth = cv::imread(depthPath.string(), cv::IMREAD_ANYDEPTH);

            if (rgb.empty() || depth.empty()) {
                std::cerr << "Failed to load frame " << filename << std::endl;
                continue;
            }

            double timestamp = static_cast<double>(frameIdx) / 30.0;  // Simulated 30 FPS
            Eigen::Matrix4d pose_cw;
            cv::cv2eigen(w2cam, pose_cw);

            gsFrames.push_back(FrameCache::Frame(rgb, depth, pose_cw, timestamp));

            ++frameIdx;
        }
    }

    // Caching frame logic based on vectors of saved frames
    for (int i = 0; i < std::max(pclFrames.size(), gsFrames.size()); ++i) {
        if (!pclFile.empty() && !gsFolder.empty()) {
            FrameCache::Frame pclFrame = pclFrames[i];
            FrameCache::Frame gsFrame = gsFrames[i];

            if (!cache.saveFrame(gsFrame.rgb, pclFrame.depth, pclFrame.pose_cw, pclFrame.timestamp)) {
                std::cerr << "\nFailed to save frame " << i << std::endl;
                return false;
            }
        }
        else if(!pclFile.empty()) {
            FrameCache::Frame pclFrame = pclFrames[i];

            if (!cache.saveFrame(pclFrame.rgb, pclFrame.depth, pclFrame.pose_cw, pclFrame.timestamp)) {
                std::cerr << "\nFailed to save frame " << i << std::endl;
                return false;
            }
        }
        else {
            FrameCache::Frame gsFrame = gsFrames[i];

            if (!cache.saveFrame(gsFrame.rgb, gsFrame.depth, gsFrame.pose_cw, gsFrame.timestamp)) {
                std::cerr << "\nFailed to save frame " << i << std::endl;
                return false;
            }
        }
    }

    if (!cache.finalizeWriting()) {
        std::cerr << "Failed to finalize frame cache" << std::endl;
        return false;
    }

    cv::destroyAllWindows();
    return true;
}

bool buildMap(const std::filesystem::path& cacheDir,
              const std::filesystem::path& cloudFile,
              const std::filesystem::path& calibFile,
              const CameraCalibration& calib,
              FrameCache& cache,
              bool useNeural,
              [[maybe_unused]] const std::string& vocPath,
              [[maybe_unused]] const std::string& configTemplate,
              [[maybe_unused]] const std::string& features,
              [[maybe_unused]] const std::string& liftfeatModel) {

    std::cout << "\n=== Building Map ===" << std::endl;
    std::string renderTag = useNeural ? "neural" : "projected";
    std::string cloudStem = cloudFile.stem().string();
    std::string map_tag = cloudStem + "_" + renderTag;
#ifdef STELLA_AVAILABLE
    // Keep LiftFeat maps from overwriting ORB maps built from the same cache
    if (features == "liftfeat") {
        map_tag += "_liftfeat";
    }
#endif

    if (!cache.initForReading()) {
        std::cerr << "Failed to load frame cache. Run 'render' mode first." << std::endl;
        return false;
    }

    // Create backend
    std::unique_ptr<SlamBackend> backend;

#ifdef STELLA_AVAILABLE
    std::cout << "Initializing Stella VSLAM backend (features: " << features << ")..." << std::endl;
    try {
        if (features == "liftfeat") {
            backend = std::make_unique<StellaBackend>(calib, StellaBackend::FeatureType::LiftFeat, liftfeatModel);
        } else {
            backend = std::make_unique<StellaBackend>(calib);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
#endif

#ifdef ORBSLAM3_AVAILABLE
    std::cout << "Initializing ORB-SLAM3 backend..." << std::endl;

    std::filesystem::path orbConfig = cacheDir / "orbslam3_config.yaml";
    if (!createORBSLAM3Config(configTemplate, calibFile, orbConfig)) {
        std::cerr << "Failed to create ORB-SLAM3 config file" << std::endl;
        return false;
    }

    backend = std::make_unique<OrbSlam3Backend>(vocPath, orbConfig.string());
#endif

    if (!backend) {
        std::cerr << "No SLAM backend available!" << std::endl;
        return false;
    }

    std::cout << "Backend: " << backend->getName() << std::endl;
    std::cout << "Processing " << cache.getFrameCount() << " frames..." << std::endl;

    auto mapBuildStart = std::chrono::high_resolution_clock::now();

    int successCount = 0;
    for (int i = 0; i < cache.getFrameCount(); i++) {
        FrameCache::Frame frame;
        if (!cache.loadFrame(i, frame)) {
            std::cerr << "Failed to load frame " << i << std::endl;
            continue;
        }

        std::cout << "Frame " << (i + 1) << "/" << cache.getFrameCount() << "\r" << std::flush;

        bool success = backend->addKeyframe(frame.rgb, frame.depth, frame.timestamp, frame.pose_cw);

        if (success) {
            successCount++;
        }
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Finalizing map..." << std::endl;
    std::cout << "Total frames: " << cache.getFrameCount() << std::endl;
    std::cout << "Successful keyframes: " << successCount << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Construct descriptive map name: {cloud}_{rendering}
    std::string mapName = "map";
    if (!map_tag.empty()) {
        mapName = map_tag;
    }

    backend->finalize(cacheDir, mapName);

    // Write metadata sidecar (extension depends on backend)
    std::string meta_ext;
#ifdef STELLA_AVAILABLE
    meta_ext = "_stella.meta";
#endif
#ifdef ORBSLAM3_AVAILABLE
    meta_ext = "_orbslam3.meta";
#endif
    if (!meta_ext.empty()) {
        std::ofstream meta(cacheDir / (mapName + meta_ext));

        writeTrajectoryHeader(meta, "prebuild_map", backend->getName(),
                              cloudStem, "", renderTag);
        auto mapBuildEnd = std::chrono::high_resolution_clock::now();
        auto mapBuildMs = std::chrono::duration_cast<std::chrono::milliseconds>(mapBuildEnd - mapBuildStart).count();
#ifdef STELLA_AVAILABLE
        meta << "# features: " << features << "\n";
#endif
        meta << "# num_keyframes: " << backend->getKeyframeCount() << "\n";
        meta << "# num_landmarks: " << backend->getLandmarkCount() << "\n";
        meta << "# map_construction_time_ms: " << mapBuildMs << "\n";
        meta.close();
    }

    std::cout << "  Final keyframes: " << backend->getKeyframeCount() << std::endl;
    std::cout << "  Final landmarks: " << backend->getLandmarkCount() << std::endl;
    std::cout << "\nMap building complete!" << std::endl;
    std::cout << "Output directory: " << cacheDir << std::endl;

    return true;
}

int main(int argc, char** argv) {
    setenv("OPENCV_IO_ENABLE_OPENEXR", "1", 1);

    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "render" && mode != "build" && mode != "full") {
        std::cerr << "Invalid mode: " << mode << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::string calibFile = argv[2];

    std::string vocPath;
    std::string configTemplate;
    std::string posesFile;
    std::string pclFile;
    std::string gsFolder;
    int lod = 0;
    std::string gsRepo    = "/home/pex/Werk/gaussian-splatting/";
    std::string condaEnv  = "gaussian_splatting";
    std::string alignmentFile;
    bool useNeural = true;
    std::string features = "orb";
    std::string liftfeatModel = std::string(PREBUILD_CONFIGS_DIR) + "/liftfeat_model/liftfeat_core.onnx";
    int argIdx = 3;

    while (argIdx < argc && argv[argIdx][0] == '-') {
        std::string arg = argv[argIdx];
        if (arg == "--pcl" && argIdx + 1 < argc) {
            pclFile = argv[++argIdx];
        } else if (arg == "--gs" && argIdx + 1 < argc) {
            gsFolder = argv[++argIdx];
        } else if (arg == "--lod" && argIdx + 1 < argc) {
            lod = std::stoi(argv[++argIdx]);
        } else if (arg == "--poses" && argIdx + 1 < argc) {
            posesFile = argv[++argIdx];
        } else if (arg == "--gs-repo" && argIdx + 1 < argc) {
            gsRepo = argv[++argIdx];
        } else if (arg == "--conda-env" && argIdx + 1 < argc) {
            condaEnv = argv[++argIdx];
        } else if (arg == "--alignment" && argIdx + 1 < argc) {
            alignmentFile = argv[++argIdx];
        } else if (arg == "--no-neural") {
            useNeural = false;
        } else if (arg == "--voc" && argIdx + 1 < argc) {
            vocPath = argv[++argIdx];
        } else if (arg == "--config" && argIdx + 1 < argc) {
            configTemplate = argv[++argIdx];
        } else if (arg == "--features" && argIdx + 1 < argc) {
            features = argv[++argIdx];
            std::transform(features.begin(), features.end(), features.begin(), ::tolower);
        } else if (arg == "--liftfeat-model" && argIdx + 1 < argc) {
            liftfeatModel = argv[++argIdx];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        ++argIdx;
    }

    // Validation
    if ((mode == "render" || mode == "full") && pclFile.empty() && gsFolder.empty()) {
        std::cerr << "Error: render/full mode requires at least --pcl or --gs\n";
        printUsage(argv[0]);
        return 1;
    }
    if (!alignmentFile.empty() && (pclFile.empty() || gsFolder.empty())) {
        std::cerr << "Error: --alignment requires both --pcl and --gs\n";
        printUsage(argv[0]);
        return 1;
    }
    if (!useNeural && pclFile.empty()) {
        std::cerr << "Error: --no-neural requires --pcl\n";
        printUsage(argv[0]);
        return 1;
    }
    if (features != "orb" && features != "liftfeat") {
        std::cerr << "Error: --features must be 'orb' or 'liftfeat'\n";
        printUsage(argv[0]);
        return 1;
    }
#ifdef STELLA_AVAILABLE
    if (features == "liftfeat" && (mode == "build" || mode == "full") && !std::filesystem::exists(liftfeatModel)) {
        std::cerr << "Error: LiftFeat model not found: " << liftfeatModel << "\n";
        return 1;
    }
#else
    if (features != "orb") {
        std::cout << "Note: --features " << features << " is ignored, the ORB-SLAM3 backend only supports ORB features." << std::endl;
        features = "orb";
    }
#endif

    // Determine cache folder based on --pcl / --gs
    std::filesystem::path cloudFile;
    std::filesystem::path cacheDir;

    if (!pclFile.empty()) {
        // --pcl gegeven (alleen of samen met --gs): gebruik de folder van de pcl
        cloudFile = std::filesystem::path(pclFile);
        cacheDir  = cloudFile.parent_path() / ".cache";
    } else {
        // enkel --gs gegeven: gebruik de folder van de gaussian splat
        cacheDir = std::filesystem::path(gsFolder) / ".cache";
    }

    std::cout << "Caching dir: "  << cacheDir << std::endl;

#ifdef ORBSLAM3_AVAILABLE
    if ((mode == "build" || mode == "full") && (vocPath.empty() || configTemplate.empty())) {
        std::cerr << "ORB-SLAM3 backend requires --voc and --config options.\n";
        printUsage(argv[0]);
        return 1;
    }
#endif

    std::filesystem::create_directories(cacheDir);

    CameraCalibration calib;
    if (!calib.loadCalibration(calibFile)) {
        std::cerr << "Failed to load camera calibration from: " << calibFile << std::endl;
        return 1;
    }

    calib = getAlignedCalibration(calib, 16);

    std::string mapTag;
    if (!pclFile.empty() && !gsFolder.empty()) {
        mapTag = cloudFile.stem().string() + "_gs_pcl" + (useNeural ? "_neural" : "_projected");
    } else if (!pclFile.empty()) {
        mapTag = cloudFile.stem().string() + "_pcl" + (useNeural ? "_neural" : "_projected");
    } else {
        mapTag = std::filesystem::path(gsFolder).stem().string() + "_gs";
    }
    FrameCache cache(cacheDir, mapTag);

    if (mode == "render") {
        return renderFrames(pclFile, gsFolder, lod, cacheDir, calib, cache, posesFile, alignmentFile, condaEnv, gsRepo, useNeural) ? 0 : 1;
    }
    else if (mode == "build") {
        return buildMap(cacheDir, cloudFile, calibFile, calib, cache, useNeural, vocPath, configTemplate, features, liftfeatModel) ? 0 : 1;
    }
    else if (mode == "full") {
        if (!renderFrames(pclFile, gsFolder, lod, cacheDir, calib, cache, posesFile, alignmentFile, condaEnv, gsRepo, useNeural)) {
            return 1;
        }
        FrameCache readCache(cacheDir, mapTag);
        return buildMap(cacheDir, cloudFile, calibFile, calib, readCache, useNeural, vocPath, configTemplate, features, liftfeatModel) ? 0 : 1;
    }

    return 0;
}
