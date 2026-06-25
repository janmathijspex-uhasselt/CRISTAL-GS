#include "map/orbslam3_backend.h"
#include "ORBmatcher.h"
#include "Converter.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <openssl/md5.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <stack>

using namespace ORB_SLAM3;
using namespace std;

namespace {
constexpr int kTextFile = 0;
constexpr int kBinaryFile = 1;

std::string calculateCheckSum(const std::string& filename, int type) {
    std::string checksum;
    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if (type == kBinaryFile)
        flags = std::ios::in | std::ios::binary;

    std::ifstream f(filename.c_str(), flags);
    if (!f.is_open()) {
        std::cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << std::endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init(&md5Context);
    while (int count = f.readsome(buffer, sizeof(buffer))) {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();
    MD5_Final(c, &md5Context);

    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        char aux[10];
        sprintf(aux, "%02x", c[i]);
        checksum += aux;
    }

    return checksum;
}
}

OrbSlam3Backend::OrbSlam3Backend(const std::string& vocFile, const std::string& settingsFile, eSensor sensor)
    : sensor_(sensor), orb_vocabulary_(nullptr), keyframe_db_(nullptr), atlas_(nullptr),
      orb_extractor_left_(nullptr), orb_extractor_right_(nullptr),
      camera_(nullptr), 
      bf_(0), b_(0), th_depth_(0), depth_map_factor_(1.0f), image_scale_(1.0f), rgb_(true),
      settings_(nullptr), last_keyframe_(nullptr), vocabulary_file_path_(vocFile) {
    
    cout << "Initializing ORB-SLAM3 Backend..." << endl;

    cout << "Loading ORB Vocabulary from " << vocFile << endl;
    orb_vocabulary_ = new ORBVocabulary();
    bool bVocLoad = orb_vocabulary_->loadFromTextFile(vocFile);
    if (!bVocLoad) {
        cerr << "Failed to open vocabulary file at: " << vocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl;

    keyframe_db_ = new KeyFrameDatabase(*orb_vocabulary_);
    atlas_ = new Atlas(0);
    
    atlas_->SetKeyFrameDababase(keyframe_db_);
    atlas_->SetORBVocabulary(orb_vocabulary_);

    eSensor orbSensor;
    switch (sensor_) {
        case MONOCULAR: orbSensor = MONOCULAR; break;
        case STEREO: orbSensor = STEREO; break;
        case RGBD: orbSensor = RGBD; break;
        default: orbSensor = RGBD;
    }
    
    settings_ = new Settings(settingsFile, orbSensor);
    cout << (*settings_) << endl;

    initializeFromSettings();

    cout << "ORB-SLAM3 Backend initialized successfully!" << endl;
}

OrbSlam3Backend::~OrbSlam3Backend() {
    if (orb_vocabulary_) delete orb_vocabulary_;
    if (keyframe_db_) delete keyframe_db_;
    if (atlas_) delete atlas_;
    if (orb_extractor_left_) delete orb_extractor_left_;
    if (orb_extractor_right_) delete orb_extractor_right_;
    if (settings_) delete settings_;
}

void OrbSlam3Backend::initializeFromSettings() {
    GeometricCamera* pOriginalCamera = settings_->camera1();
    camera_ = atlas_->AddCamera(pOriginalCamera);

    dist_coef_ = cv::Mat::zeros(1, 5, CV_32F);

    K_ = cv::Mat::eye(3, 3, CV_32F);
    K_.at<float>(0, 0) = camera_->getParameter(0);
    K_.at<float>(1, 1) = camera_->getParameter(1);
    K_.at<float>(0, 2) = camera_->getParameter(2);
    K_.at<float>(1, 2) = camera_->getParameter(3);

    K_eigen_.setIdentity();
    K_eigen_(0, 0) = camera_->getParameter(0);
    K_eigen_(1, 1) = camera_->getParameter(1);
    K_eigen_(0, 2) = camera_->getParameter(2);
    K_eigen_(1, 2) = camera_->getParameter(3);

    rgb_ = settings_->rgb();

    depth_map_factor_ = settings_->depthMapFactor();
    if (fabs(depth_map_factor_) < 1e-5)
        depth_map_factor_ = 1.0f;
    else
        depth_map_factor_ = 1.0f / depth_map_factor_;

    if (sensor_ == STEREO || sensor_ == RGBD) {
        bf_ = settings_->bf();
        th_depth_ = settings_->b() * settings_->thDepth();
    }

    int nFeatures = settings_->nFeatures();
    int nLevels = settings_->nLevels();
    float fScaleFactor = settings_->scaleFactor();
    int fIniThFAST = (int)settings_->initThFAST();
    int fMinThFAST = (int)settings_->minThFAST();

    orb_extractor_left_ = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor_ == STEREO)
        orb_extractor_right_ = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl << "ORB-SLAM3 Backend Configuration: " << endl;
    cout << "- Camera: " << (settings_->cameraType() == Settings::PinHole ? "Pinhole" : "Fisheye") << endl;
    cout << "- ORB Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Color Order: " << (rgb_ ? "RGB" : "BGR") << endl;
}

bool OrbSlam3Backend::addKeyframe(const cv::Mat& rgb, const cv::Mat& depth,
                                  double timestamp, const Eigen::Matrix4d& pose_cw) {
    cv::Mat imGray = rgb;
    cv::Mat imDepth = depth;

    if (rgb.channels() == 3) {
        if (rgb_)
            cv::cvtColor(rgb, imGray, cv::COLOR_RGB2GRAY);
        else
            cv::cvtColor(rgb, imGray, cv::COLOR_BGR2GRAY);
    }

    if ((fabs(depth_map_factor_ - 1.0f) > 1e-5) || imDepth.type() != CV_32F) {
        imDepth.convertTo(imDepth, CV_32F, depth_map_factor_);
    }

    Frame frame(imGray, imDepth, timestamp, orb_extractor_left_, orb_vocabulary_,
                K_, dist_coef_, bf_, th_depth_, camera_);

    frame.ComputeBoW();

    int nPointsWithDepth = 0;
    for (int i = 0; i < frame.N; i++) {
        if (frame.mvDepth[i] > 0)
            nPointsWithDepth++;
    }
    if (nPointsWithDepth < 40) {
        std::cout << "Not enough points with depth (" << nPointsWithDepth << "), skipping frame." << std::endl;
        return false;
    }

    KeyFrame* pKF = createKeyFrame(frame);
    pKF->SetPose(Sophus::SE3f(pose_cw.cast<float>()));

    Sophus::SE3f Tcw = pKF->GetPose();
    Sophus::SE3f Twc = Tcw.inverse();
    Map* pMap = atlas_->GetCurrentMap();

    for (int i = 0; i < frame.N; i++) {
        float d = frame.mvDepth[i];
        if (d <= 0) continue;

        cv::KeyPoint kp = frame.mvKeysUn[i];
        Eigen::Vector3f ray = camera_->unprojectEig(cv::Point2f(kp.pt.x, kp.pt.y));
        ray.normalize();
        Eigen::Vector3f p3Dc = ray * (d / ray.z());
        Eigen::Vector3f p3Dw = Twc * p3Dc;

        MapPoint* pMP = new MapPoint(p3Dw, pKF, pMap);
        pKF->AddMapPoint(pMP, i);
        pMP->AddObservation(pKF, i);
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        pMap->AddMapPoint(pMP);
        all_map_points_.push_back(pMP);
    }

    last_keyframe_ = pKF;
    return true;
}

KeyFrame* OrbSlam3Backend::createKeyFrame(Frame& frame) {
    Map* pMap = atlas_->GetCurrentMap();
    KeyFrame* pKF = new KeyFrame(frame, pMap, keyframe_db_);
    pKF->ComputeBoW();
    keyframe_db_->add(pKF);
    atlas_->AddKeyFrame(pKF);
    return pKF;
}

void OrbSlam3Backend::mergeLandmarks(MapPoint* pMPKeep, MapPoint* pMPMerge) {
    if (pMPKeep == pMPMerge) return;

    for (auto& [kf, idxT] : pMPMerge->GetObservations()) {
        int idx = std::get<0>(idxT);
        kf->EraseMapPointMatch(idx);
        pMPMerge->EraseObservation(kf);
        if (std::get<0>(pMPKeep->GetIndexInKeyFrame(kf)) < 0) {
            kf->AddMapPoint(pMPKeep, idx);
            pMPKeep->AddObservation(kf, idx);
        }
    }
    pMPMerge->SetBadFlag();
    pMPKeep->ComputeDistinctiveDescriptors();
    pMPKeep->UpdateNormalAndDepth();
}

void OrbSlam3Backend::reprojectAndMerge(KeyFrame* pKF) {
    const int img_width = settings_->newImSize().width;
    const int img_height = settings_->newImSize().height;
    const int descriptor_threshold = ORBmatcher::TH_LOW;
    
    Sophus::SE3f Tcw = pKF->GetPose();
    
    std::unordered_map<int, std::vector<MapPoint*>> pixelLandmarkMap;

    for (MapPoint* pMP : all_map_points_) {
        if (!pMP || pMP->isBad()) continue;
        if (std::get<0>(pMP->GetIndexInKeyFrame(pKF)) >= 0) continue;
        
        Eigen::Vector3f pos_w = pMP->GetWorldPos();
        Eigen::Vector3f pos_c = Tcw * pos_w;
        
        if (pos_c(2) <= 0) continue;
        
        cv::Point2f uv = camera_->project(cv::Point3f(pos_c(0), pos_c(1), pos_c(2)));
        
        int ix = static_cast<int>(std::round(uv.x));
        int iy = static_cast<int>(std::round(uv.y));
        
        if (ix < 0 || ix >= img_width || iy < 0 || iy >= img_height) continue;
        
        int key = iy * img_width + ix;
        pixelLandmarkMap[key].push_back(pMP);
    }

    std::unordered_map<int, std::pair<MapPoint*, int>> matches;
    
    const std::vector<std::pair<int, int>> offsets = {
        {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1},
        {1,1}, {-1,-1}, {1,-1}, {-1,1},
    };

    for (int i = 0; i < pKF->N; i++) {
        if (pKF->mvDepth[i] <= 0) continue;

        cv::KeyPoint kp = pKF->mvKeys[i];
        int x = static_cast<int>(std::round(kp.pt.x));
        int y = static_cast<int>(std::round(kp.pt.y));
               
        for (const auto& [dx, dy] : offsets) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx < 0 || nx >= img_width || ny < 0 || ny >= img_height) continue;
            
            int key = ny * img_width + nx;
            auto it = pixelLandmarkMap.find(key);
            if (it == pixelLandmarkMap.end()) continue;

            for (MapPoint* pMP : it->second) {
                if (!pMP || pMP->isBad()) continue;

                int dist = cv::norm(pMP->GetDescriptor(), pKF->mDescriptors.row(i), cv::NORM_HAMMING);
                if (dist < descriptor_threshold) {
                    if (!matches.count(i) || dist < matches[i].second) {
                        matches[i] = {pMP, dist};
                    }
                }
            }
        }
    }
      
    int mergeCount = 0;
    for (const auto& [idx, pair] : matches) {
        auto curr = pair.first;
        auto prev = pKF->GetMapPoint(idx);

        if (curr->isBad()) continue;

        if (curr && prev && prev != curr) {
            mergeLandmarks(prev, curr);
            ++mergeCount;
        }
    }
}

void OrbSlam3Backend::mergeMapPoints() {
    std::vector<KeyFrame*> vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();
    
    int i = 0;
    for (KeyFrame* pKF : vpKFs) {
        reprojectAndMerge(pKF);
        std::cout << "Frame " << ++i << "/" << vpKFs.size() << "\r" << std::flush;
    }
    
    std::vector<MapPoint*> vpRemainingMapPoints;
    for (MapPoint* pMP : all_map_points_) {
        if (pMP && !pMP->isBad()) {
            vpRemainingMapPoints.push_back(pMP);
        } else if (pMP) {
            delete pMP;
        }
    }
    
    all_map_points_ = vpRemainingMapPoints;
}

// Helper to access KeyFrame protected members for keypoint pruning.
// mGrid and mvpMapPoints are protected but must be updated when
// compacting keypoint arrays to maintain serialization consistency.
namespace {
    struct KFAccessor : ORB_SLAM3::KeyFrame {
        KFAccessor() = delete;
        using ORB_SLAM3::KeyFrame::mvpMapPoints;
        using ORB_SLAM3::KeyFrame::mGrid;
    };
}

void OrbSlam3Backend::pruneUnobservedKeypoints() {
    auto vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();
    
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        
        const int orig_k = pKF->N;
        std::vector<int> keep_indices;
        keep_indices.reserve(orig_k);
        
        // Collect indices of keypoints that have valid landmarks
        for (int i = 0; i < orig_k; ++i) {
            if (pKF->GetMapPoint(i) != nullptr) {
                keep_indices.push_back(i);
            }
        }
        
        // Skip if all keypoints are observed
        if (keep_indices.size() == static_cast<size_t>(orig_k)) continue;
        
        const int new_k = static_cast<int>(keep_indices.size());
        
        // Create new keyframe data structures with only observed keypoints
        std::vector<cv::KeyPoint> new_keys;
        std::vector<cv::KeyPoint> new_keys_un;
        cv::Mat new_descriptors(new_k, pKF->mDescriptors.cols, pKF->mDescriptors.type());
        std::vector<float> new_depth;
        std::vector<float> new_uright;
        std::vector<MapPoint*> new_map_points;
        
        new_keys.reserve(new_k);
        new_keys_un.reserve(new_k);
        new_depth.reserve(new_k);
        new_uright.reserve(new_k);
        new_map_points.reserve(new_k);
        
        for (size_t ni = 0; ni < keep_indices.size(); ++ni) {
            int old = keep_indices[ni];
            new_keys.push_back(pKF->mvKeys[old]);
            new_keys_un.push_back(pKF->mvKeysUn[old]);
            pKF->mDescriptors.row(old).copyTo(new_descriptors.row(static_cast<int>(ni)));
            new_depth.push_back(pKF->mvDepth[old]);
            new_uright.push_back(pKF->mvuRight[old]);
            new_map_points.push_back(pKF->GetMapPoint(old));
        }
        
        // Erase all old map point observations (before resizing arrays)
        for (int i = 0; i < orig_k; ++i) {
            MapPoint* pMP = pKF->GetMapPoint(i);
            if (pMP && !pMP->isBad()) {
                pMP->EraseObservation(pKF);
            }
            pKF->EraseMapPointMatch(i);
        }
        
        // Update const public data members via const_cast
        const_cast<std::vector<cv::KeyPoint>&>(pKF->mvKeys) = new_keys;
        const_cast<std::vector<cv::KeyPoint>&>(pKF->mvKeysUn) = new_keys_un;
        const_cast<cv::Mat&>(pKF->mDescriptors) = new_descriptors;
        const_cast<std::vector<float>&>(pKF->mvDepth) = new_depth;
        const_cast<std::vector<float>&>(pKF->mvuRight) = new_uright;
        const_cast<int&>(pKF->N) = new_k;
        
        // Resize protected mvpMapPoints to match new N
        auto& mvpMP = reinterpret_cast<KFAccessor*>(pKF)->mvpMapPoints;
        mvpMP.clear();
        mvpMP.resize(new_k, nullptr);
        
        // Add back observations with new indices
        for (size_t ni = 0; ni < new_map_points.size(); ++ni) {
            MapPoint* pMP = new_map_points[ni];
            if (pMP && !pMP->isBad()) {
                pKF->AddMapPoint(pMP, static_cast<int>(ni));
                pMP->AddObservation(pKF, static_cast<int>(ni));
            }
        }
        
        // Rebuild mGrid with new keypoint indices
        auto& grid = reinterpret_cast<KFAccessor*>(pKF)->mGrid;
        for (int ci = 0; ci < pKF->mnGridCols; ci++)
            for (int ri = 0; ri < pKF->mnGridRows; ri++)
                grid[ci][ri].clear();
        
        for (int i = 0; i < new_k; i++) {
            const cv::KeyPoint& kp = pKF->mvKeysUn[i];
            int nGridPosX = static_cast<int>(round((kp.pt.x - pKF->mnMinX) * pKF->mfGridElementWidthInv));
            int nGridPosY = static_cast<int>(round((kp.pt.y - pKF->mnMinY) * pKF->mfGridElementHeightInv));
            if (nGridPosX >= 0 && nGridPosX < pKF->mnGridCols &&
                nGridPosY >= 0 && nGridPosY < pKF->mnGridRows) {
                grid[nGridPosX][nGridPosY].push_back(i);
            }
        }
    }
}

// Graph utilities for spanning tree
namespace {
    using Graph = std::map<unsigned long, std::map<unsigned long, int>>;

    void dfs(unsigned long node, const Graph& graph, std::set<unsigned long>& visited, 
             std::vector<unsigned long>& component) {
        std::stack<unsigned long> s;
        s.push(node);

        while (!s.empty()) {
            unsigned long current = s.top();
            s.pop();

            if (visited.count(current)) continue;
            visited.insert(current);
            component.push_back(current);

            auto it = graph.find(current);
            if (it != graph.end()) {
                for (const auto& [neighbor, weight] : it->second) {
                    if (!visited.count(neighbor)) {
                        s.push(neighbor);
                    }
                }
            }
        }
    }

    std::vector<unsigned long> findLargestConnectedComponent(const Graph& graph) {
        std::set<unsigned long> visited;
        std::vector<unsigned long> largestComponent;

        for (const auto& [node, _] : graph) {
            if (!visited.count(node)) {
                std::vector<unsigned long> component;
                dfs(node, graph, visited, component);
                if (component.size() > largestComponent.size()) {
                    largestComponent = std::move(component);
                }
            }
        }

        return largestComponent;
    }

    Graph extractSubgraph(const Graph& graph, const std::vector<unsigned long>& nodes) {
        Graph subgraph;
        std::set<unsigned long> nodeSet(nodes.begin(), nodes.end());

        for (unsigned long node : nodes) {
            auto it = graph.find(node);
            if (it != graph.end()) {
                for (const auto& [neighbor, weight] : it->second) {
                    if (nodeSet.count(neighbor)) {
                        subgraph[node][neighbor] = weight;
                    }
                }
            }
        }

        return subgraph;
    }
}

void OrbSlam3Backend::buildSpanningTree() {
    auto vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();
    
    Graph graph;
    std::map<unsigned long, int> connection_strength;

    for (MapPoint* pMP : all_map_points_) {
        if (!pMP || pMP->isBad()) continue;
        
        const std::map<KeyFrame*, std::tuple<int, int>>& observations = pMP->GetObservations();
        std::vector<KeyFrame*> obs_kfs;
        for (const auto& [pKF, _] : observations) {
            if (pKF && !pKF->isBad()) {
                obs_kfs.push_back(pKF);
            }
        }
        
        for (size_t i = 0; i < obs_kfs.size(); ++i) {
            for (size_t j = i + 1; j < obs_kfs.size(); ++j) {
                unsigned long id1 = obs_kfs[i]->mnId;
                unsigned long id2 = obs_kfs[j]->mnId;
                graph[id1][id2]++;
                graph[id2][id1]++;
                connection_strength[id1]++;
                connection_strength[id2]++;
            }
        }
    }

    // Remove keyframes not in graph
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        if (graph.find(pKF->mnId) == graph.end()) {
            
            // Remove all map points observed by this keyframe
            const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
            for (size_t i = 0; i < vpMapPoints.size(); ++i) {
                MapPoint* pMP = vpMapPoints[i];
                if (pMP && !pMP->isBad()) {
                    pMP->EraseObservation(pKF);
                }
            }
            
            pKF->SetBadFlag();
        }
    }

    // Find keyframe with most connections as root
    KeyFrame* pRootKF = nullptr;
    int most_connections = -1;
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        unsigned long id = pKF->mnId;
        if (connection_strength[id] > most_connections) {
            pRootKF = pKF;
            most_connections = connection_strength[id];
        }
    }

    // Find largest connected component
    std::vector<unsigned long> largestComponent = findLargestConnectedComponent(graph);
    graph = extractSubgraph(graph, largestComponent);

    // Remove keyframes not in largest connected component
    std::set<unsigned long> component_set(largestComponent.begin(), largestComponent.end());
    int removed_count = 0;
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        if (component_set.find(pKF->mnId) == component_set.end()) {
            
            // Remove all map points observed by this keyframe
            const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
            for (size_t i = 0; i < vpMapPoints.size(); ++i) {
                MapPoint* pMP = vpMapPoints[i];
                if (pMP && !pMP->isBad()) {
                    pMP->EraseObservation(pKF);
                }
            }
            
            pKF->SetBadFlag();
            removed_count++;
        }
    }

    // Build spanning tree using BFS
    if (pRootKF) {
        std::set<KeyFrame*> visited;
        std::queue<KeyFrame*> bfsQueue;
        std::map<unsigned long, KeyFrame*> id_to_kf;
        
        // Create ID to KeyFrame mapping for quick lookup
        vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();
        for (KeyFrame* pKF : vpKFs) {
            if (pKF && !pKF->isBad()) {
                id_to_kf[pKF->mnId] = pKF;
            }
        }
        
        bfsQueue.push(pRootKF);
        visited.insert(pRootKF);
        
        while (!bfsQueue.empty()) {
            KeyFrame* pCurrKF = bfsQueue.front();
            bfsQueue.pop();
            
            // Get neighbors from graph and sort by weight
            unsigned long curr_id = pCurrKF->mnId;
            std::vector<std::pair<int, KeyFrame*>> vWeightedNeighbors;
            
            if (graph.find(curr_id) != graph.end()) {
                for (const auto& [neighbor_id, weight] : graph[curr_id]) {
                    auto it = id_to_kf.find(neighbor_id);
                    if (it != id_to_kf.end()) {
                        KeyFrame* pNeighborKF = it->second;
                        if (visited.find(pNeighborKF) == visited.end()) {
                            vWeightedNeighbors.emplace_back(weight, pNeighborKF);
                        }
                    }
                }
            }
            
            std::sort(vWeightedNeighbors.begin(), vWeightedNeighbors.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            
            for (const auto& [weight, pNeighborKF] : vWeightedNeighbors) {
                pNeighborKF->ChangeParent(pCurrKF);
                visited.insert(pNeighborKF);
                bfsQueue.push(pNeighborKF);
            }
        }

        std::cout << "Spanning tree built. Visited " << visited.size() 
                  << " keyframes." << std::endl;
    }
}

int OrbSlam3Backend::getKeyframeCount() const {
    return atlas_->GetCurrentMap()->GetAllKeyFrames().size();
}

int OrbSlam3Backend::getLandmarkCount() const {
    return all_map_points_.size();
}

int OrbSlam3Backend::getLastKeyframeId() const {
    return last_keyframe_ ? last_keyframe_->mnId : -1;
}

void OrbSlam3Backend::finalize(const std::filesystem::path& output_path, const std::string& mapName) {
    auto vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();
    
    std::cout << "\n=== ORB-SLAM3: BEFORE Landmark Merging ===" << std::endl;
    std::cout << "KeyFrames: " << vpKFs.size() << std::endl;
    std::cout << "Landmarks: " << all_map_points_.size() << std::endl;

    mergeMapPoints();
    
    vpKFs = atlas_->GetCurrentMap()->GetAllKeyFrames();

    int removedCount = 0;
    for (MapPoint* pMP : all_map_points_) {
        if (!pMP || pMP->isBad()) continue;

        if (pMP->Observations() <= 2) {
            pMP->SetBadFlag();
            removedCount++;
        }
    }

    std::vector<MapPoint*> remaining;
    remaining.reserve(all_map_points_.size());
    for (MapPoint* pMP : all_map_points_) {
        if (pMP && !pMP->isBad())
            remaining.push_back(pMP);
    }
    all_map_points_.swap(remaining);

    std::cout << "\n=== ORB-SLAM3: AFTER Landmark Merging ===" << std::endl;
    std::cout << "KeyFrames: " << atlas_->GetCurrentMap()->GetAllKeyFrames().size() << std::endl;
    std::cout << "Landmarks: " << all_map_points_.size() << std::endl;
    
    // Prune unobserved keypoints
    pruneUnobservedKeypoints();
    
    // Recompute MapPoint descriptors and normals after pruning changed observation indices
    for (MapPoint* pMP : all_map_points_) {
        if (pMP && !pMP->isBad()) {
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
        }
    }
    
    // Update connections 
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        pKF->mBowVec.clear();
        pKF->mFeatVec.clear();
        pKF->ComputeBoW();
        pKF->SetFirstConnection(false);
        pKF->UpdateConnections();
        std::cout << "Num Keypoints: " << pKF->N << ", Observed MapPoints: " << pKF->GetMapPoints().size() << std::endl;
    }

    buildSpanningTree();

    // Save atlas
    std::filesystem::path atlas_file = output_path / (mapName + ".osa");
    
    atlas_->PreSave();

    std::string vocabChecksum = "blabla";
    std::size_t found = vocabulary_file_path_.find_last_of("/\\");
    std::string vocabName = (found == std::string::npos) ? vocabulary_file_path_ : vocabulary_file_path_.substr(found + 1);

    std::remove(atlas_file.c_str());
    std::ofstream ofs(atlas_file, std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << vocabName;
    oa << vocabChecksum;
    oa << atlas_;
    
    std::cout << "ORB-SLAM3 map saved to: " << atlas_file << std::endl;
}
