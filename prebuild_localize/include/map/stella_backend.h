#ifndef STELLA_BACKEND_H
#define STELLA_BACKEND_H

#include "slam_backend.h"
#include "DeviceFactory/CameraCalibration.h"

#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/landmark.h>
#include <stella_vslam/feature/orb_extractor.h>
#include <stella_vslam/feature/orb_params.h>
#include <stella_vslam/camera/base.h>
#include <stella_vslam/camera/perspective.h>
#include <stella_vslam/data/map_database.h>

#include <opencv2/opencv.hpp>

/**
 * @brief Stella VSLAM backend for map building.
 */
class StellaBackend : public SlamBackend {
public:
    enum class FeatureType { ORB, LiftFeat };

    /**
     * @brief Construct a Stella VSLAM backend.
     * 
     * @param calib Camera calibration parameters
     * @param features Feature type used for keypoints/descriptors
     * @param liftfeat_model Path to the LiftFeat ONNX model (only used for FeatureType::LiftFeat)
     * @throws std::runtime_error if the LiftFeat model cannot be loaded
     */
    explicit StellaBackend(const CameraCalibration& calib,
                           FeatureType features = FeatureType::ORB,
                           const std::string& liftfeat_model = "");
    ~StellaBackend() override;

    bool addKeyframe(const cv::Mat& rgb, const cv::Mat& depth, 
                    double timestamp, const Eigen::Matrix4d& pose_cw) override;
    
    void finalize(const std::filesystem::path& output_path, const std::string& mapName = "map") override;
    
    int getKeyframeCount() const override;
    int getLandmarkCount() const override;
    int getLastKeyframeId() const override;
    std::string getName() const override { return "stella_vslam"; }

private:
    std::shared_ptr<stella_vslam::data::keyframe> createLidarFrame(
        const cv::Mat& rgb, const cv::Mat& depth, double timestamp, 
        const cv::Mat& mask, const Eigen::Matrix4d& pose_cw);
    
    void addKeyframeInternal(std::shared_ptr<stella_vslam::data::keyframe> kf);
    void createLandmarksForKeyframe(std::shared_ptr<stella_vslam::data::keyframe> kf);
    void reprojectAndMerge(std::shared_ptr<stella_vslam::data::keyframe> kf);
    void mergeLandmarks(std::shared_ptr<stella_vslam::data::landmark> keep,
                       std::shared_ptr<stella_vslam::data::landmark> merge);
    void deduplicateSingleObservationLandmarks(float voxel_size);
    void pruneUnobservedKeypoints();
    void buildSpanningTree();

    stella_vslam::camera::base* cam_;
    stella_vslam::feature::orb_params* orb_params_;
    stella_vslam::feature::orb_extractor* extractor_;
    std::shared_ptr<stella_vslam::data::map_database> map_db_;
    
    int last_keyframe_id_ = -1;
};

#endif // STELLA_BACKEND_H
