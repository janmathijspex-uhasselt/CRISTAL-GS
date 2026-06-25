#ifndef ORBSLAM3_BACKEND_H
#define ORBSLAM3_BACKEND_H

#include "slam_backend.h"
#include "DeviceFactory/CameraCalibration.h"

#include <opencv2/core/core.hpp>
#include <string>

#include "Atlas.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Pinhole.h"
#include "Settings.h"
#include "ImuTypes.h"

/**
 * @brief ORB-SLAM3 backend for map building.
 */
class OrbSlam3Backend : public SlamBackend {
public:
    enum eSensor { MONOCULAR, STEREO, RGBD };

    /**
     * @brief Construct an ORB-SLAM3 backend.
     * 
     * @param vocFile Path to the ORB vocabulary file
     * @param settingsFile Path to the ORB-SLAM3 settings YAML file  
     * @param sensor Sensor type (default: RGBD)
     */
    OrbSlam3Backend(const std::string& vocFile, const std::string& settingsFile, 
                   eSensor sensor = RGBD);
    ~OrbSlam3Backend() override;

    bool addKeyframe(const cv::Mat& rgb, const cv::Mat& depth, 
                    double timestamp, const Eigen::Matrix4d& pose_cw) override;
    
    void finalize(const std::filesystem::path& output_path, const std::string& mapName = "map") override;
    
    int getKeyframeCount() const override;
    int getLandmarkCount() const override;
    int getLastKeyframeId() const override;
    std::string getName() const override { return "orbslam3"; }

private:
    void initializeFromSettings();
    ORB_SLAM3::KeyFrame* createKeyFrame(ORB_SLAM3::Frame& frame);
    void mergeLandmarks(ORB_SLAM3::MapPoint* keep, ORB_SLAM3::MapPoint* merge);
    void reprojectAndMerge(ORB_SLAM3::KeyFrame* kf);
    void mergeMapPoints();
    void pruneUnobservedKeypoints();
    void buildSpanningTree();

    eSensor sensor_;
    
    ORB_SLAM3::ORBVocabulary* orb_vocabulary_;
    ORB_SLAM3::KeyFrameDatabase* keyframe_db_;
    ORB_SLAM3::Atlas* atlas_;
    ORB_SLAM3::ORBextractor* orb_extractor_left_;
    ORB_SLAM3::ORBextractor* orb_extractor_right_;
    ORB_SLAM3::GeometricCamera* camera_;
    ORB_SLAM3::Settings* settings_;
    ORB_SLAM3::KeyFrame* last_keyframe_;
    
    cv::Mat K_;
    Eigen::Matrix3f K_eigen_;
    cv::Mat dist_coef_;
    float bf_;
    float b_;
    float th_depth_;
    float depth_map_factor_;
    float image_scale_;
    bool rgb_;
    
    std::vector<ORB_SLAM3::MapPoint*> all_map_points_;
    std::string vocabulary_file_path_;
};

#endif // ORBSLAM3_BACKEND_H
