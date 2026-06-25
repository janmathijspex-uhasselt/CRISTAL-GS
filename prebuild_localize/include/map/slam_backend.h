#ifndef SLAM_BACKEND_H
#define SLAM_BACKEND_H

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <memory>
#include <string>

/**
 * @brief Abstract interface for SLAM map building backends.
 * 
 * This interface allows different SLAM implementations (ORB-SLAM3, Stella VSLAM, etc.)
 * to be used interchangeably for building maps from rendered point cloud frames.
 */
class SlamBackend {
public:
    virtual ~SlamBackend() = default;

    /**
     * @brief Add a keyframe to the map with the given RGB image, depth map, and pose.
     * 
     * @param rgb The RGB image (CV_8UC3)
     * @param depth The depth map (CV_32F, values in meters)
     * @param timestamp Frame timestamp in seconds
     * @param pose_cw Camera-to-world transformation matrix (4x4)
     * @return true if keyframe was successfully added, false if skipped
     */
    virtual bool addKeyframe(const cv::Mat& rgb, const cv::Mat& depth, 
                            double timestamp, const Eigen::Matrix4d& pose_cw) = 0;

    /**
     * @brief Finalize the map and save to the specified output path.
     * 
     * @param output_path Directory where the map should be saved
     * @param mapName Base name for the map file (without extension)
     */
    virtual void finalize(const std::filesystem::path& output_path, const std::string& mapName = "map") = 0;

    /**
     * @brief Get the number of keyframes in the map.
     */
    virtual int getKeyframeCount() const = 0;

    /**
     * @brief Get the number of landmarks/map points in the map.
     */
    virtual int getLandmarkCount() const = 0;

    /**
     * @brief Get the ID of the last added keyframe (-1 if none).
     */
    virtual int getLastKeyframeId() const = 0;

    /**
     * @brief Get the name/identifier of this backend.
     */
    virtual std::string getName() const = 0;
};

/**
 * @brief Factory function type for creating SLAM backends.
 */
using SlamBackendFactory = std::unique_ptr<SlamBackend>(*)();

#endif // SLAM_BACKEND_H
