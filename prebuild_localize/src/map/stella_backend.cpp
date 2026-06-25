#include "map/stella_backend.h"

#include <stella_vslam/data/common.h>
#include <stella_vslam/data/frame.h>
#include <stella_vslam/data/frame_observation.h>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <unordered_set>
#include <map>
#include <queue>
#include <set>
#include <stack>

#include <nlohmann/json.hpp>

using namespace stella_vslam;

StellaBackend::StellaBackend(const CameraCalibration& calib) {
    orb_params_ = new feature::orb_params("default");
    extractor_ = new feature::orb_extractor(orb_params_, 800);
    cam_ = new camera::perspective(
        "vid", camera::setup_type_t::RGBD, camera::color_order_t::BGR,
        calib.getWidth(), calib.getHeight(), 30,
        calib.getFocalLengthX(), calib.getFocalLengthY(),
        calib.getPrincipalPointX(), calib.getPrincipalPointY(),
        0, 0, 0, 0, 0
    );
    map_db_ = std::make_shared<data::map_database>(10);
}

StellaBackend::~StellaBackend() {
    delete extractor_;
    delete orb_params_;
    delete cam_;
}

bool StellaBackend::addKeyframe(const cv::Mat& rgb, const cv::Mat& depth,
                                double timestamp, const Eigen::Matrix4d& pose_cw) {
    cv::Mat mask = depth > 0;
    auto kf = createLidarFrame(rgb, depth, timestamp, mask, pose_cw);
    
    if (kf) {
        addKeyframeInternal(kf);
        last_keyframe_id_ = kf->id_;
        return true;
    }
    return false;
}

std::shared_ptr<data::keyframe> StellaBackend::createLidarFrame(
    const cv::Mat& rgb, const cv::Mat& depth, double timestamp,
    const cv::Mat& mask, const Eigen::Matrix4d& pose_cw) {
    
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

    data::frame_observation frm_obs;
    std::vector<cv::KeyPoint> keypts;
    extractor_->extract(gray, mask, keypts, frm_obs.descriptors_);

    if (keypts.size() < 30) return nullptr;

    frm_obs.undist_keypts_ = keypts;
    frm_obs.stereo_x_right_.resize(keypts.size(), -1);
    frm_obs.depths_.resize(keypts.size(), -1);

    for (size_t i = 0; i < keypts.size(); ++i) {
        float x = keypts[i].pt.x;
        float y = keypts[i].pt.y;
        float d = depth.at<float>(static_cast<int>(y), static_cast<int>(x));
        if (d <= 0) continue;
        frm_obs.depths_[i] = d;
        frm_obs.stereo_x_right_[i] = keypts[i].pt.x;
    }

    int kf_id = map_db_->get_all_keyframes().size();
    data::frame frame{static_cast<unsigned int>(kf_id), timestamp, cam_, orb_params_, frm_obs, {}};
    frame.set_pose_cw(pose_cw);

    return data::keyframe::make_keyframe(kf_id, frame);
}

void StellaBackend::addKeyframeInternal(std::shared_ptr<data::keyframe> kf) {
    createLandmarksForKeyframe(kf);
    map_db_->add_keyframe(kf);
}

void StellaBackend::createLandmarksForKeyframe(std::shared_ptr<data::keyframe> kf) {
    auto& frame = kf->frm_obs_;
    int lm_size = map_db_->get_all_landmarks().size();
    
    for (size_t i = 0; i < frame.undist_keypts_.size(); ++i) {
        if (frame.depths_[i] <= 0) continue;
        
        Vec3_t dir = cam_->convert_point_to_bearing(frame.undist_keypts_[i].pt);
        Vec3_t pos_c = dir * (frame.depths_[i] / dir.z());
        Vec4_t pos_c_homo;
        pos_c_homo << pos_c, 1.0;
        Vec4_t pos_w_homo = kf->get_pose_wc() * pos_c_homo;
        Vec3_t pos_w = pos_w_homo.head<3>() / pos_w_homo(3);
        
        auto lm = std::make_shared<data::landmark>(lm_size + i, pos_w, kf);
        kf->add_landmark(lm, i);
        lm->add_observation(kf, i);
        lm->compute_descriptor();
        map_db_->add_landmark(lm);
    }
}

void StellaBackend::mergeLandmarks(std::shared_ptr<data::landmark> keep,
                                   std::shared_ptr<data::landmark> merge) {
    if (keep == merge) return;
    
    for (auto& [kf_weak, idx] : merge->get_observations()) {
        if (auto kf = kf_weak.lock()) {
            kf->erase_landmark_with_index(idx);
            merge->erase_observation(map_db_.get(), kf);
            if (!keep->is_observed_in_keyframe(kf)) {
                kf->add_landmark(keep, idx);
                keep->add_observation(kf, idx);
            }
        }
    }
    keep->compute_descriptor();
}

void StellaBackend::reprojectAndMerge(std::shared_ptr<data::keyframe> kf) {
    const auto& frame = kf->frm_obs_;
    const auto& keypoints = frame.undist_keypts_;
    const auto& descriptors = frame.descriptors_;
    const int img_width = cam_->cols_;
    const int img_height = cam_->rows_;
    const int descriptor_threshold = 50;

    const Mat44_t& pose_w2c = kf->get_pose_cw();
    const Mat33_t& rot_cw = kf->get_rot_cw();
    const Vec3_t& trans_cw = kf->get_trans_cw();

    std::unordered_map<int, std::vector<std::shared_ptr<data::landmark>>> pixel_landmark_map;

    for (const auto& lm : map_db_->get_all_landmarks()) {
        if (lm->is_observed_in_keyframe(kf)) continue;

        const Vec3_t& pos_w = lm->get_pos_in_world();
        Vec4_t pos_w_homo(pos_w(0), pos_w(1), pos_w(2), 1.0);
        Vec4_t pos_c_homo = pose_w2c * pos_w_homo;

        if (pos_c_homo(2) <= 0) continue;

        Vec2_t uv;
        float right;
        if (!cam_->reproject_to_image(rot_cw, trans_cw, pos_w, uv, right)) continue;

        int x = static_cast<int>(std::round(uv(0)));
        int y = static_cast<int>(std::round(uv(1)));
        if (x < 0 || x >= img_width || y < 0 || y >= img_height) continue;

        int key = y * img_width + x;
        pixel_landmark_map[key].emplace_back(lm);
    }

    std::unordered_map<int, std::pair<std::shared_ptr<data::landmark>, int>> matches;

    const std::vector<std::pair<int, int>> offsets = {
        {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1},
        {1,1}, {-1,-1}, {1,-1}, {-1,1}
    };

    for (size_t i = 0; i < keypoints.size(); ++i) {
        int x = static_cast<int>(std::round(keypoints[i].pt.x));
        int y = static_cast<int>(std::round(keypoints[i].pt.y));

        for (const auto& [dx, dy] : offsets) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= img_width || ny < 0 || ny >= img_height) continue;

            int key = ny * img_width + nx;
            auto it = pixel_landmark_map.find(key);
            if (it == pixel_landmark_map.end()) continue;

            for (const auto& lm : it->second) {
                int dist = cv::norm(lm->get_descriptor(), descriptors.row((int)i), cv::NORM_HAMMING);
                if (dist < descriptor_threshold) {
                    if (!matches.count(i) || dist < matches[i].second) {
                        matches[i] = {lm, dist};
                    }
                }
            }
        }
    }

    int mergeCount = 0;
    for (const auto& [idx, pair] : matches) {
        auto curr = pair.first;
        auto prev = kf->get_landmark(idx);
        if (prev != curr) {
            mergeLandmarks(prev, curr);
            ++mergeCount;
        }
    }
}

// Graph utilities for spanning tree
using Graph = std::map<int, std::map<int, int>>;

static void dfs(int node, const Graph& graph, std::set<int>& visited, std::vector<int>& component) {
    std::stack<int> s;
    s.push(node);

    while (!s.empty()) {
        int current = s.top();
        s.pop();

        if (visited.count(current)) continue;
        visited.insert(current);
        component.push_back(current);

        for (const auto& [neighbor, weight] : graph.at(current)) {
            if (!visited.count(neighbor)) {
                s.push(neighbor);
            }
        }
    }
}

static std::vector<int> findLargestConnectedComponent(const Graph& graph) {
    std::set<int> visited;
    std::vector<int> largestComponent;

    for (const auto& [node, _] : graph) {
        if (!visited.count(node)) {
            std::vector<int> component;
            dfs(node, graph, visited, component);
            if (component.size() > largestComponent.size()) {
                largestComponent = std::move(component);
            }
        }
    }

    return largestComponent;
}

static Graph extractSubgraph(const Graph& graph, const std::vector<int>& nodes) {
    Graph subgraph;
    std::set<int> nodeSet(nodes.begin(), nodes.end());

    for (int node : nodes) {
        for (const auto& [neighbor, weight] : graph.at(node)) {
            if (nodeSet.count(neighbor)) {
                subgraph[node][neighbor] = weight;
            }
        }
    }

    return subgraph;
}

void StellaBackend::buildSpanningTree() {
    std::map<int, std::map<int, int>> graph;
    std::map<int, int> connection_strength;

    for (const auto& lm : map_db_->get_all_landmarks()) {
        const auto& obs = lm->get_observations();
        for (auto it1 = obs.begin(); it1 != obs.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != obs.end(); ++it2) {
                if (auto kf1 = it1->first.lock()) {
                    if (auto kf2 = it2->first.lock()) {
                        int id1 = kf1->id_;
                        int id2 = kf2->id_;
                        graph[id1][id2]++;
                        graph[id2][id1]++;
                        connection_strength[id1]++;
                        connection_strength[id2]++;
                    }
                }
            }
        }
    }

    for (auto& kf : map_db_->get_all_keyframes()) {
        if (graph.find(kf->id_) == graph.end()) {
            for (auto& lm : kf->get_valid_landmarks()) {
                lm->erase_observation(map_db_.get(), kf);
            }
            map_db_->erase_keyframe(kf);
        }
    }

    int max_id = -1;
    int most_connections = -1;
    for (auto& kf : map_db_->get_all_keyframes()) {
        int id = kf->id_;
        if (connection_strength[id] > most_connections) {
            max_id = id;
            most_connections = connection_strength[id];
        }
    }

    std::vector<int> largestComponent = findLargestConnectedComponent(graph);
    graph = extractSubgraph(graph, largestComponent);

    for (auto& kf : map_db_->get_all_keyframes()) {
        if (graph.find(kf->id_) == graph.end()) {
            for (auto& lm : kf->get_valid_landmarks()) {
                lm->erase_observation(map_db_.get(), kf);
            }
            map_db_->erase_keyframe(kf);
        }
    }

    std::map<int, int> parent;
    std::map<int, std::vector<int>> children;
    std::set<int> visited;
    std::queue<int> q;

    int root_id = max_id;
    q.push(root_id);
    visited.insert(root_id);
    parent[root_id] = -1;

    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        std::vector<std::pair<int, int>> neighbors;
        for (auto& [nei, w] : graph[curr]) {
            if (!visited.count(nei)) neighbors.emplace_back(nei, w);
        }
        std::sort(neighbors.begin(), neighbors.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& [nei, _] : neighbors) {
            parent[nei] = curr;
            children[curr].push_back(nei);
            visited.insert(nei);
            q.push(nei);
        }
    }

    for (auto& [child, par] : parent) {
        if (par == -1) continue;
        map_db_->get_keyframe(child)->graph_node_->set_spanning_parent(map_db_->get_keyframe(par));
    }

    for (auto& [par, kids] : children) {
        for (int kid : kids)
            map_db_->get_keyframe(par)->graph_node_->add_spanning_child(map_db_->get_keyframe(kid));
    }
}

void StellaBackend::pruneUnobservedKeypoints() {
    for (auto& kf : map_db_->get_all_keyframes()) {
        auto& frm = kf->frm_obs_;
        const int orig_k = static_cast<int>(frm.undist_keypts_.size());
        std::vector<int> keep_indices;
        keep_indices.reserve(orig_k);

        for (int i = 0; i < orig_k; ++i) {
            if (kf->get_landmark(i) != nullptr) {
                keep_indices.push_back(i);
            }
        }

        if (keep_indices.size() == static_cast<size_t>(orig_k)) continue;

        std::vector<int> old2new(orig_k, -1);
        for (size_t ni = 0; ni < keep_indices.size(); ++ni) {
            old2new[keep_indices[ni]] = static_cast<int>(ni);
        }

        std::vector<std::shared_ptr<data::landmark>> landmarks_by_new_index(keep_indices.size());
        for (int old_idx = 0; old_idx < orig_k; ++old_idx) {
            auto lm = kf->get_landmark(old_idx);
            if (lm) {
                int new_idx = old2new[old_idx];
                if (new_idx >= 0) {
                    landmarks_by_new_index[new_idx] = lm;
                }
                kf->erase_landmark_with_index(old_idx);
                lm->erase_observation(map_db_.get(), kf);
            }
        }

        std::vector<cv::KeyPoint> new_kps;
        new_kps.reserve(keep_indices.size());
        cv::Mat new_descriptors;
        if (!frm.descriptors_.empty()) 
            new_descriptors = cv::Mat(static_cast<int>(keep_indices.size()), frm.descriptors_.cols, frm.descriptors_.type());
        std::vector<float> new_depths;
        std::vector<float> new_stereo_x_right;

        for (size_t ni = 0; ni < keep_indices.size(); ++ni) {
            int old = keep_indices[ni];
            new_kps.push_back(frm.undist_keypts_[old]);
            if (!frm.descriptors_.empty()) 
                frm.descriptors_.row(old).copyTo(new_descriptors.row(static_cast<int>(ni)));
            if (!frm.depths_.empty()) 
                new_depths.push_back(frm.depths_[old]);
            if (!frm.stereo_x_right_.empty()) 
                new_stereo_x_right.push_back(frm.stereo_x_right_[old]);
        }

        frm.undist_keypts_.swap(new_kps);
        frm.descriptors_ = new_descriptors;
        frm.depths_.swap(new_depths);
        frm.stereo_x_right_.swap(new_stereo_x_right);

        for (size_t ni = 0; ni < landmarks_by_new_index.size(); ++ni) {
            auto lm = landmarks_by_new_index[ni];
            if (lm) {
                kf->add_landmark(lm, static_cast<int>(ni));
                lm->add_observation(kf, static_cast<int>(ni));
                lm->compute_descriptor();
            }
        }
    }
}

int StellaBackend::getKeyframeCount() const {
    return map_db_->get_all_keyframes().size();
}

int StellaBackend::getLandmarkCount() const {
    return map_db_->get_all_landmarks().size();
}

int StellaBackend::getLastKeyframeId() const {
    return last_keyframe_id_;
}

void StellaBackend::finalize(const std::filesystem::path& output_path, const std::string& mapName) {
    std::cout << "\n=== Stella VSLAM: BEFORE Landmark Merging ===" << std::endl;
    std::cout << "KeyFrames: " << map_db_->get_num_keyframes() << std::endl;
    std::cout << "Landmarks: " << map_db_->get_num_landmarks() << std::endl;

    int i = 0;
    int num_keyframes = map_db_->get_all_keyframes().size();
    for (auto& kf : map_db_->get_all_keyframes()) {
        reprojectAndMerge(kf);
        std::cout << "Frame " << (++i) << "/" << num_keyframes << "\r" << std::flush;
    }

    std::cout << "\n=== Stella VSLAM: AFTER Landmark Merging ===" << std::endl;
    std::cout << "KeyFrames: " << map_db_->get_num_keyframes() << std::endl;
    std::cout << "Landmarks: " << map_db_->get_num_landmarks() << std::endl;

    // Remove single-observation landmarks
    for (auto& lm : map_db_->get_all_landmarks()) {
        if (lm->get_observations().size() <= 1) {
            for (auto& [kf_weak, idx] : lm->get_observations()) {
                if (auto kf = kf_weak.lock()) {
                    kf->erase_landmark_with_index(idx);
                    lm->erase_observation(map_db_.get(), kf);
                }
            }
        }
    }

    pruneUnobservedKeypoints();
    buildSpanningTree();

    std::cout << "Landmarks after filtering: " << map_db_->get_num_landmarks() << std::endl;

    // Save to msgpack
    nlohmann::json json;
    json["cameras"][cam_->name_] = cam_->to_json();
    json["orb_params"][orb_params_->name_] = orb_params_->to_json();

    for (auto& kf : map_db_->get_all_keyframes())
        json["keyframes"][std::to_string(kf->id_)] = kf->to_json();

    for (auto& lm : map_db_->get_all_landmarks())
        json["landmarks"][std::to_string(lm->id_)] = lm->to_json();

    json["keyframe_next_id"] = map_db_->get_all_keyframes().size();
    json["landmark_next_id"] = map_db_->get_all_landmarks().size();

    std::filesystem::path output_file = output_path / (mapName + ".msg");
    std::ofstream ofs(output_file, std::ios::out | std::ios::binary);
    if (ofs.is_open()) {
        const auto msgpack = nlohmann::json::to_msgpack(json);
        ofs.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size() * sizeof(uint8_t));
        ofs.close();
        std::cout << "Stella VSLAM map saved to: " << output_file << std::endl;
    }
}
