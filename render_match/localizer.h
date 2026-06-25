#ifndef LOCALIZER_H
#define LOCALIZER_H

#include "feature/featuretracker.h"
#include "RTRenderer/CameraCalibration.h"
#include "trackingstate.h"
#include "project_cloud.h"
#include <filesystem>
#include "map/map.h"
#include "data/trajectoryframe.h"

class Localizer
{
public:
    /**
     * @brief Construct a Localizer with lightweight initialisation.
     *
     * Call setCloudStem() to configure cache / map paths, then use the
     * phase methods (renderToCache, buildMapFromCache, buildMap, loadMap,
     * initProjector) to set up the pipeline as needed.
     */
    Localizer(const CameraCalibration& calibration,
              const std::filesystem::path& caching_dir,
              bool useNeural = true);

    /// Set the point-cloud stem name (drives cache / map file naming).
    void setCloudStem(const std::string& stem);

    // ---- Phase methods ------------------------------------------------

    /** Render cubemap frames from the point cloud and write them to a
     *  frame cache on disk. */
    void renderToCache(const std::filesystem::path& cloudPath,
                       const std::string& posesFile = "");

    /** Build the XFeat feature map from previously cached cubemap frames
     *  and save the result as a .bin file.
     *  Requires renderToCache() to have been run first. */
    void buildMapFromCache();

    /** Render cubemaps and build the feature map in a single pass
     *  (no intermediate frame cache).  Equivalent to renderToCache()
     *  followed by buildMapFromCache(), but more efficient. */
    void buildMap(const std::filesystem::path& cloudPath,
                  const std::string& posesFile = "");

    /** Load a pre-built feature map (.bin).
     *  @return true if the map was found and loaded successfully. */
    bool loadMap();

    /** Create the per-frame renderer / projector needed by process_frame().
     *  Must be called before the localisation loop. */
    void initProjector(const std::filesystem::path& cloudPath);

    // ---- Runtime ------------------------------------------------------

    void process_frame(double timestamp, cv::Mat& frame_cam);

    void set_w2c(cv::Matx44d w2c);
    inline cv::Matx44d get_w2c()
    {
        return w2cam;
    }

    inline TrackingState get_state()
    {
        return state;
    }

    void start();
    void stop();
    ~Localizer();

    void save_trajectory(const std::filesystem::path& file_path, const std::string& format,
                         const std::string& point_cloud = "", const std::string& video = "",
                         const std::string& rendering_mode = "");

    /// Number of keyframes in the loaded/built feature map.
    int getKeyframeCount() const { return reloc_map ? reloc_map->getKeyframeCount() : 0; }

    /// Number of frames processed so far (incremented by process_frame).
    int getProcessedFrames() const { return frame_nr + 1; }

    /// Number of successfully tracked frames.
    int getTrackedFrames() const { return static_cast<int>(trajectory.size()); }

    // ---- Path helpers -------------------------------------------------

    /** Path to the feature-map .bin for the current configuration. */
    std::filesystem::path getMapBinPath() const;

    /** Path to the frame-cache directory for the current configuration. */
    std::filesystem::path getFrameCacheDir() const;

private:
    std::shared_ptr<FeatureTracker> tracker;
    std::shared_ptr<Map> reloc_map;

    int frame_nr;

    std::vector<TrajectoryFrame> trajectory;

    TrackingState state;
    cv::Matx44d w2cam;

    cv::Mat last_frame_cam;

    CameraCalibration calibration;
    cv::Size image_size;
    std::shared_ptr<ProjectCloud> projector;
    bool useNeural_;

    std::filesystem::path caching_dir_;
    std::string cloudStem_;
    std::string mapTag_;

    static void saveDepthBin(const std::filesystem::path& path, const cv::Mat& depth);
    static cv::Mat loadDepthBin(const std::filesystem::path& path);
};

#endif // LOCALIZER_H
