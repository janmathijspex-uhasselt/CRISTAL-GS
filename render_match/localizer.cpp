#include "localizer.h"

#include "feature/xfeattracker.h"
#include "map/xfeatmap.h"
#include "map/cubemapgenerator.h"
#include "RTRenderer/cloudreader.h"
#include "common/pose_generator.h"
#include "common/pose_io.h"
#include "common/trajectory_io.h"
#include <iomanip>
#include <chrono>
#include <sstream>
#include <fstream>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Localizer::Localizer(const CameraCalibration& calibration,
                     const std::filesystem::path& caching_dir,
                     bool useNeural)
    : calibration(calibration), state(TrackingState::INITIALIZING),
      frame_nr{-1}, useNeural_(useNeural), caching_dir_(caching_dir)
{
    std::shared_ptr<XFeat::XFDetector> detector =
        std::make_shared<XFeat::XFDetector>(2500, 0.2);
    tracker = std::make_shared<XFeatTracker>(detector);
    reloc_map = std::make_shared<XFeatMap>(detector, calibration);
    image_size = cv::Size(calibration.getWidth(), calibration.getHeight());
}

void Localizer::setCloudStem(const std::string& stem)
{
    cloudStem_ = stem;
    mapTag_ = stem + "_" + (useNeural_ ? "neural" : "projected");
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

std::filesystem::path Localizer::getMapBinPath() const
{
    return caching_dir_ / (mapTag_ + ".bin");
}

std::filesystem::path Localizer::getFrameCacheDir() const
{
    return caching_dir_ / ("cubemap_frames_" + mapTag_);
}

// ---------------------------------------------------------------------------
// Depth binary I/O  (int32 width, int32 height, raw float32 pixels)
// ---------------------------------------------------------------------------

void Localizer::saveDepthBin(const std::filesystem::path& path,
                             const cv::Mat& depth)
{
    std::ofstream ofs(path, std::ios::binary);
    int32_t w = depth.cols, h = depth.rows;
    ofs.write(reinterpret_cast<const char*>(&w), sizeof(w));
    ofs.write(reinterpret_cast<const char*>(&h), sizeof(h));
    ofs.write(reinterpret_cast<const char*>(depth.data),
              static_cast<std::streamsize>(w) * h * sizeof(float));
}

cv::Mat Localizer::loadDepthBin(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    int32_t w, h;
    ifs.read(reinterpret_cast<char*>(&w), sizeof(w));
    ifs.read(reinterpret_cast<char*>(&h), sizeof(h));
    cv::Mat depth(h, w, CV_32F);
    ifs.read(reinterpret_cast<char*>(depth.data),
             static_cast<std::streamsize>(w) * h * sizeof(float));
    return depth;
}

// ---------------------------------------------------------------------------
// Phase: renderToCache
// ---------------------------------------------------------------------------

void Localizer::renderToCache(const std::filesystem::path& cloudPath,
                              const std::string& posesFile)
{
    std::cout << "\n=== Rendering Cubemap Frames to Cache ===" << std::endl;
    std::cout << "Loading point cloud: " << cloudPath << std::endl;

    auto grid = CloudReader::loadCloud(cloudPath, caching_dir_);

    std::string cubemapEngine = useNeural_
        ? "trt_1440x1440.ts" : std::string();
    auto projector1440 = std::make_shared<ProjectCloud>(grid, cubemapEngine);

    std::vector<cv::Point3f> scannerPositions;
    if (!posesFile.empty()) {
        auto loadedPoses = loadPoses(posesFile);
        for (const auto& w2c : loadedPoses) {
            cv::Matx44d c2w = w2c.inv();
            scannerPositions.emplace_back(c2w(0,3), c2w(1,3), c2w(2,3));
        }
        std::cout << "Loaded " << scannerPositions.size()
                  << " positions from poses file" << std::endl;
    } else {
        scannerPositions = PoseGenerator::getPositions(grid);
    }
    std::cout << scannerPositions.size()
              << " virtual scanner positions found" << std::endl;
    if (scannerPositions.empty()) {
        std::cerr << "No positions — aborting render." << std::endl;
        return;
    }

    // Cubemap calibration (same parameters CubemapGenerator uses internally)
    const cv::Size cubemapSize{1440, 1440};
    CameraCalibration cubemapCalib;
    cubemapCalib.setIntrinsicsMatrix(cv::Matx33d(
        cubemapSize.width / 2.0, 0, cubemapSize.width / 2.0,
        0, cubemapSize.height / 2.0, cubemapSize.height / 2.0,
        0, 0, 1));
    cubemapCalib.setWidth(cubemapSize.width);
    cubemapCalib.setHeight(cubemapSize.height);

    struct Face { cv::Vec3d dir, up; };
    const std::vector<Face> faces = {
        {{ 1, 0, 0}, {0, 0,-1}},
        {{-1, 0, 0}, {0, 0,-1}},
        {{ 0, 1, 0}, {0, 0,-1}},
        {{ 0,-1, 0}, {0, 0,-1}},
    };

    auto frameCacheDir = getFrameCacheDir();
    std::filesystem::create_directories(frameCacheDir);

    std::ofstream posesOut(frameCacheDir / "poses.txt");
    int frameIdx = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Matx44d> w2cams;
    for (size_t pi = 0; pi < scannerPositions.size(); ++pi) {
        cv::Vec3d pos(scannerPositions[pi].x,
                      scannerPositions[pi].y,
                      scannerPositions[pi].z);

        for (const auto& face : faces) {
            // lookAt (same maths as CubemapGenerator::lookAt)
            cv::Vec3d z = face.dir / cv::norm(face.dir);
            cv::Vec3d x = face.up.cross(z); x /= cv::norm(x);
            cv::Vec3d y = z.cross(x);

            cv::Matx44d w2cam = cv::Matx44d::eye();
            w2cam(0,0) = x[0]; w2cam(0,1) = x[1]; w2cam(0,2) = x[2]; w2cam(0,3) = -x.dot(pos);
            w2cam(1,0) = y[0]; w2cam(1,1) = y[1]; w2cam(1,2) = y[2]; w2cam(1,3) = -y.dot(pos);
            w2cam(2,0) = z[0]; w2cam(2,1) = z[1]; w2cam(2,2) = z[2]; w2cam(2,3) = -z.dot(pos);

            cv::Mat color(cubemapSize, CV_8UC3);
            cv::Mat depth(cubemapSize, CV_32F);
            if (useNeural_)
                projector1440->computeFull(cubemapCalib, w2cam, &color, &depth);
            else
                projector1440->computeRGBD(cubemapCalib, w2cam, &color, &depth);

            std::ostringstream tag;
            tag << std::setw(6) << std::setfill('0') << frameIdx;
            cv::imwrite((frameCacheDir / ("frame_" + tag.str() + ".png")).string(), color);
            saveDepthBin(frameCacheDir / ("depth_" + tag.str() + ".bin"), depth);

            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    posesOut << std::setprecision(15) << w2cam(r, c)
                             << ((r == 3 && c == 3) ? "\n" : " ");

            w2cams.push_back(w2cam);
            ++frameIdx;
        }
        std::cout << "Rendered position " << (pi + 1) << "/"
                  << scannerPositions.size() << " (" << frameIdx
                  << " frames)\r" << std::flush;
    }
    posesOut.close();

    if(posesFile.empty())
    {
        // Save generated poses for future reuse
        savePoses(caching_dir_ / "generated_poses_rm.txt", w2cams);
    }

    {
        std::ofstream meta(frameCacheDir / "meta.txt");
        meta << "cloud: "          << cloudStem_ << "\n";
        meta << "rendering: "      << (useNeural_ ? "neural" : "projected") << "\n";
        meta << "num_frames: "     << frameIdx << "\n";
        meta << "cubemap_width: "  << cubemapSize.width << "\n";
        meta << "cubemap_height: " << cubemapSize.height << "\n";
        meta.close();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\nRendered " << frameIdx << " cubemap frames in "
              << ms << " ms" << std::endl;
    std::cout << "Frame cache: " << frameCacheDir << std::endl;
}

// ---------------------------------------------------------------------------
// Phase: buildMapFromCache
// ---------------------------------------------------------------------------

void Localizer::buildMapFromCache()
{
    auto frameCacheDir = getFrameCacheDir();
    std::cout << "\n=== Building Feature Map from Cache ===" << std::endl;
    std::cout << "Frame cache: " << frameCacheDir << std::endl;

    if (!std::filesystem::exists(frameCacheDir / "meta.txt")) {
        std::cerr << "Frame cache not found at " << frameCacheDir
                  << ".  Run 'render' mode first." << std::endl;
        return;
    }

    int numFrames = 0;
    int cubemapW = 1440, cubemapH = 1440;
    {
        std::ifstream meta(frameCacheDir / "meta.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_frames: ", 0) == 0)
                numFrames = std::stoi(line.substr(12));
            else if (line.rfind("cubemap_width: ", 0) == 0)
                cubemapW = std::stoi(line.substr(15));
            else if (line.rfind("cubemap_height: ", 0) == 0)
                cubemapH = std::stoi(line.substr(16));
        }
    }
    if (numFrames <= 0) {
        std::cerr << "No frames in cache." << std::endl;
        return;
    }

    std::vector<cv::Matx44d> poses;
    {
        std::ifstream posesIn(frameCacheDir / "poses.txt");
        for (int i = 0; i < numFrames; ++i) {
            cv::Matx44d m;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    posesIn >> m(r, c);
            poses.push_back(m);
        }
    }

    CameraCalibration cubemapCalib;
    cubemapCalib.setIntrinsicsMatrix(cv::Matx33d(
        cubemapW / 2.0, 0, cubemapW / 2.0,
        0, cubemapH / 2.0, cubemapH / 2.0,
        0, 0, 1));
    cubemapCalib.setWidth(cubemapW);
    cubemapCalib.setHeight(cubemapH);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numFrames; ++i) {
        std::ostringstream tag;
        tag << std::setw(6) << std::setfill('0') << i;

        cv::Mat color = cv::imread(
            (frameCacheDir / ("frame_" + tag.str() + ".png")).string());
        cv::Mat depth = loadDepthBin(
            frameCacheDir / ("depth_" + tag.str() + ".bin"));

        if (color.empty() || depth.empty()) {
            std::cerr << "Failed to load cached frame " << i << std::endl;
            continue;
        }

        reloc_map->add_frame(color, depth, poses[i], cubemapCalib);
        std::cout << "Building map: frame " << (i + 1) << "/"
                  << numFrames << "\r" << std::flush;
    }

    reloc_map->save_keyframes(getMapBinPath().string());

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    {
        std::ofstream meta(getMapBinPath().replace_extension().string() + "_rm.meta");
        writeTrajectoryHeader(meta, "render_match", "xfeat",
                              cloudStem_, "",
                              useNeural_ ? "neural" : "projected");
        meta << "# num_keyframes: " << reloc_map->getKeyframeCount() << "\n";
        meta << "# map_construction_time_ms: " << ms << "\n";
        meta.close();
    }

    std::cout << "\nFeature map built from " << numFrames
              << " cached frames in " << ms << " ms" << std::endl;
    std::cout << "Map saved to: " << getMapBinPath() << std::endl;
}

// ---------------------------------------------------------------------------
// Phase: buildMap  (render + build in one pass, no intermediate cache)
// ---------------------------------------------------------------------------

void Localizer::buildMap(const std::filesystem::path& cloudPath,
                         const std::string& posesFile)
{
    std::cout << "\n=== Building Feature Map (single pass) ===" << std::endl;
    std::cout << "Loading point cloud: " << cloudPath << std::endl;

    auto grid = CloudReader::loadCloud(cloudPath, caching_dir_);

    std::string cubemapEngine = useNeural_
        ? "trt_1440x1440.ts" : std::string();
    auto projector1440 = std::make_shared<ProjectCloud>(grid, cubemapEngine);
    CubemapGenerator c(projector1440, useNeural_);

    std::vector<cv::Point3f> scannerPositions;
    if (!posesFile.empty()) {
        auto loadedPoses = loadPoses(posesFile);
        for (const auto& w2c : loadedPoses) {
            cv::Matx44d c2w = w2c.inv();
            scannerPositions.emplace_back(c2w(0,3), c2w(1,3), c2w(2,3));
        }
        std::cout << "Loaded " << scannerPositions.size()
                  << " positions from poses file" << std::endl;
    } else {
        scannerPositions = PoseGenerator::getPositions(grid);
    }
    std::cout << scannerPositions.size()
              << " virtual scanner positions found" << std::endl;
    if (scannerPositions.empty()) return;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& position : scannerPositions) {
        cv::Vec3d positionVec(position.x, position.y, position.z);
        c.renderCubemapToMap(positionVec, reloc_map);
    }
    reloc_map->save_keyframes(getMapBinPath().string());

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    {
        std::ofstream meta(getMapBinPath().replace_extension().string() + "_rm.meta");
        writeTrajectoryHeader(meta, "render_match", "xfeat",
                              cloudStem_, "",
                              useNeural_ ? "neural" : "projected");
        meta << "# num_keyframes: " << reloc_map->getKeyframeCount() << "\n";
        meta << "# map_construction_time_ms: " << ms << "\n";
        meta.close();
    }

    std::cout << "Feature map built in " << ms << " ms" << std::endl;
    std::cout << "Map saved to: " << getMapBinPath() << std::endl;
}

// ---------------------------------------------------------------------------
// Phase: loadMap
// ---------------------------------------------------------------------------

bool Localizer::loadMap()
{
    auto mapPath = getMapBinPath();
    if (!std::filesystem::exists(mapPath)) {
        return false;
    }
    std::cout << "Loading feature map: " << mapPath << std::endl;
    reloc_map->load_keyframes(mapPath.string());
    return true;
}

// ---------------------------------------------------------------------------
// Phase: initProjector
// ---------------------------------------------------------------------------

void Localizer::initProjector(const std::filesystem::path& cloudPath)
{
    std::cout << "Initialising per-frame projector..." << std::endl;
    auto grid = CloudReader::loadCloud(cloudPath, caching_dir_);

    std::string frameEngine = useNeural_
        ? "trt_" + std::to_string(calibration.getWidth()) + "x"
          + std::to_string(calibration.getHeight()) + ".ts"
        : std::string();
    projector = std::make_shared<ProjectCloud>(grid, frameEngine);
}

void Localizer::process_frame(double timestamp, cv::Mat &frame_cam)
{

    ++frame_nr;
    if(state != TrackingState::TRACKING)
    {
        bool relocalized = reloc_map->find_pose(frame_cam, w2cam);
        if(!relocalized)
        {
            std::cout << "\nFailed to relocalize, try again next frame" << std::endl;
            return;
        }
        else {
            std::cout << "\n---- Relocalize succeeded" << std::endl;
        }
    }

    cv::Mat frame_pcd(image_size, CV_8UC3);
    cv::Mat depth_pcd(image_size, CV_32F);
    int rendered;
    if (useNeural_)
        rendered = projector->computeFull(calibration, w2cam, &frame_pcd, &depth_pcd);
    else
        rendered = projector->computeRGBD(calibration, w2cam, &frame_pcd, &depth_pcd);

    if(rendered <= 0)
    {
        state = TrackingState::RELOCALIZING;
        return;
    }

    tracker->update_keyframe(frame_pcd, depth_pcd, calibration, w2cam);

    if(state == TrackingState::TRACKING)
    {
        Utils::visualize_side_by_side(frame_pcd, last_frame_cam, "Side-by-side");
    }

    int nrMatches = tracker->find_pose(frame_cam, calibration, w2cam);

    if(nrMatches > 0)
    {
        trajectory.push_back(TrajectoryFrame(frame_nr, w2cam, nrMatches, timestamp)); // saving for trajectory file
        if(state != TrackingState::TRACKING)
        {
            state = TrackingState::TRACKING;
        }

    }
    else
    {
        std::cout << "Failed to track position: relocalizing" << std::endl;
        state = TrackingState::RELOCALIZING;
        return;
    }

    last_frame_cam = frame_cam;
}

void Localizer::set_w2c(cv::Matx44d w2c)
{
    w2cam = w2c;
    state = TrackingState::TRACKING;
}

void Localizer::start()
{
}

void Localizer::stop()
{
}

Localizer::~Localizer() {
    stop();
}

void Localizer::save_trajectory(const std::filesystem::path &file_path, const std::string& format,
                                 const std::string& point_cloud, const std::string& video,
                                 const std::string& rendering_mode)
{
    if (!std::filesystem::exists(file_path.parent_path())) {
        if (!std::filesystem::create_directories(file_path.parent_path())) {
            std::cerr << "Failed to create directory: " << file_path.parent_path() << std::endl;
            exit(1);
        }
    }

    std::ofstream ofs(file_path);
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    writeTrajectoryHeader(ofs, "render_match", "xfeat", point_cloud, video, rendering_mode);

    if(format == std::string("COLMAP"))
    {
        int i = 0;
        for (const auto& frame : trajectory)
        {
            auto q = frame.getQuaternion();
            auto t = frame.getPosition();

            ofs << i++ << " "
               << std::setprecision(10)
               << q.w << " " << q.x << " " << q.y << " " << q.z << " "
               << t(0) << " " << t(1) << " " << t(2) << " "
               << 1 << " " << "frame_" << std::setw(6) << std::setfill('0') << frame.getFrameNr() << ".jpg" << "\n\n";

        }
    }
    else if(format == std::string("TUM"))
    {
        for(const auto& frame:trajectory)
        {
            cv::Vec3d t = frame.getPosition();
            cv::Quatd q = frame.getQuaternion();
            ofs << std::setprecision(15) << frame.getTimestamp()  << std::setprecision(9) << " " << t[0] << " " << t[1] << " " << t[2] << " " << q.x << " " << q.y << " " << q.z << " " << q.w << std::endl;
        }
    }
    else
    {
        std::cerr << "Unsupported format to write the trajectory" << std::endl;
    }
    ofs.close();
}
