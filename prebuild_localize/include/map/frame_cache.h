#ifndef FRAME_CACHE_H
#define FRAME_CACHE_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <fstream>

/**
 * @brief Stores and loads rendered frames (RGB + depth) for reproducible map building.
 * 
 * This allows rendering frames once and then running multiple SLAM backends
 * on the exact same data.
 */
class FrameCache {
public:
    struct Frame {
        cv::Mat rgb;        // CV_8UC3
        cv::Mat depth;      // CV_32F (meters)
        Eigen::Matrix4d pose_cw;
        double timestamp;
    };

    /**
     * @brief Initialize frame cache with a directory path.
     */
    explicit FrameCache(const std::filesystem::path& cache_dir, const std::string& mapTag)
        : cache_dir_(cache_dir / ("rendered_frames_" + mapTag)) {}

    /**
     * @brief Check if a frame cache exists and has frames.
     */
    bool exists() const {
        return std::filesystem::exists(cache_dir_ / "metadata.txt") &&
               std::filesystem::exists(cache_dir_ / "rgb") &&
               std::filesystem::exists(cache_dir_ / "depth");
    }

    /**
     * @brief Get the number of cached frames.
     */
    int frameCount() const {
        if (!exists()) return 0;
        std::ifstream meta(cache_dir_ / "metadata.txt");
        int count = 0;
        meta >> count;
        return count;
    }

    /**
     * @brief Initialize cache for writing frames.
     */
    bool initForWriting() {
        try {
            std::filesystem::create_directories(cache_dir_ / "rgb");
            std::filesystem::create_directories(cache_dir_ / "depth");
            frame_count_ = 0;
            poses_.clear();
            timestamps_.clear();
            return true;
        } catch (...) {
            return false;
        }
    }

    /**
     * @brief Save a rendered frame to the cache.
     */
    bool saveFrame(const cv::Mat& rgb, const cv::Mat& depth, 
                   const Eigen::Matrix4d& pose_cw, double timestamp) {
        std::string idx_str = std::to_string(frame_count_);
        
        // Save RGB as PNG (lossless)
        std::string rgb_path = (cache_dir_ / "rgb" / (idx_str + ".png")).string();
        if (!cv::imwrite(rgb_path, rgb)) {
            std::cerr << "Failed to save RGB frame: " << rgb_path << std::endl;
            return false;
        }

        // Save depth as binary float (preserves precision)
        std::string depth_path = (cache_dir_ / "depth" / (idx_str + ".bin")).string();
        std::ofstream depth_file(depth_path, std::ios::binary);
        if (!depth_file.is_open()) {
            std::cerr << "Failed to save depth frame: " << depth_path << std::endl;
            return false;
        }
        int rows = depth.rows, cols = depth.cols;
        depth_file.write(reinterpret_cast<char*>(&rows), sizeof(int));
        depth_file.write(reinterpret_cast<char*>(&cols), sizeof(int));
        depth_file.write(reinterpret_cast<const char*>(depth.data), rows * cols * sizeof(float));
        depth_file.close();

        poses_.push_back(pose_cw);
        timestamps_.push_back(timestamp);
        frame_count_++;
        return true;
    }

    bool saveFrame(Frame frame) {
        auto rgb = frame.rgb;
        auto depth = frame.depth;
        auto pose_cw = frame.pose_cw;
        auto timestamp = frame.timestamp;

        std::string idx_str = std::to_string(frame_count_);

        // Save RGB as PNG (lossless)
        std::string rgb_path = (cache_dir_ / "rgb" / (idx_str + ".png")).string();
        if (!cv::imwrite(rgb_path, rgb)) {
            std::cerr << "Failed to save RGB frame: " << rgb_path << std::endl;
            return false;
        }

        // Save depth as binary float (preserves precision)
        std::string depth_path = (cache_dir_ / "depth" / (idx_str + ".bin")).string();
        std::ofstream depth_file(depth_path, std::ios::binary);
        if (!depth_file.is_open()) {
            std::cerr << "Failed to save depth frame: " << depth_path << std::endl;
            return false;
        }
        int rows = depth.rows, cols = depth.cols;
        depth_file.write(reinterpret_cast<char*>(&rows), sizeof(int));
        depth_file.write(reinterpret_cast<char*>(&cols), sizeof(int));
        depth_file.write(reinterpret_cast<const char*>(depth.data), rows * cols * sizeof(float));
        depth_file.close();

        poses_.push_back(pose_cw);
        timestamps_.push_back(timestamp);
        frame_count_++;
        return true;
    }

    /**
     * @brief Finalize writing and save metadata.
     */
    bool finalizeWriting() {
        std::ofstream meta(cache_dir_ / "metadata.txt");
        if (!meta.is_open()) return false;
        
        meta << frame_count_ << "\n";
        meta << std::fixed << std::setprecision(15);
        
        for (int i = 0; i < frame_count_; i++) {
            meta << timestamps_[i] << "\n";
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    meta << poses_[i](r, c);
                    if (c < 3) meta << " ";
                }
                meta << "\n";
            }
        }
        meta.close();
        
        std::cout << "Frame cache saved: " << frame_count_ << " frames to " << cache_dir_ << std::endl;
        return true;
    }

    /**
     * @brief Load metadata for reading frames.
     */
    bool initForReading() {
        std::ifstream meta(cache_dir_ / "metadata.txt");
        if (!meta.is_open()) {
            std::cerr << "Failed to open metadata file" << std::endl;
            return false;
        }

        meta >> frame_count_;
        poses_.resize(frame_count_);
        timestamps_.resize(frame_count_);

        for (int i = 0; i < frame_count_; i++) {
            meta >> timestamps_[i];
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    meta >> poses_[i](r, c);
                }
            }
        }
        meta.close();
        
        std::cout << "Frame cache loaded: " << frame_count_ << " frames from " << cache_dir_ << std::endl;
        return true;
    }

    /**
     * @brief Load a specific frame from the cache.
     */
    bool loadFrame(int index, Frame& frame) const {
        if (index < 0 || index >= frame_count_) return false;

        std::string idx_str = std::to_string(index);

        // Load RGB
        std::string rgb_path = (cache_dir_ / "rgb" / (idx_str + ".png")).string();
        frame.rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
        if (frame.rgb.empty()) {
            std::cerr << "Failed to load RGB frame: " << rgb_path << std::endl;
            return false;
        }

        // Load depth
        std::string depth_path = (cache_dir_ / "depth" / (idx_str + ".bin")).string();
        std::ifstream depth_file(depth_path, std::ios::binary);
        if (!depth_file.is_open()) {
            std::cerr << "Failed to load depth frame: " << depth_path << std::endl;
            return false;
        }
        int rows, cols;
        depth_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        depth_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        frame.depth = cv::Mat(rows, cols, CV_32F);
        depth_file.read(reinterpret_cast<char*>(frame.depth.data), rows * cols * sizeof(float));
        depth_file.close();

        frame.pose_cw = poses_[index];
        frame.timestamp = timestamps_[index];
        return true;
    }

    int getFrameCount() const { return frame_count_; }
    const std::filesystem::path& getCacheDir() const { return cache_dir_; }

private:
    std::filesystem::path cache_dir_;
    int frame_count_ = 0;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<double> timestamps_;
};

#endif // FRAME_CACHE_H
