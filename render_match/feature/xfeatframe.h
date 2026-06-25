#ifndef XFEATFRAME_H
#define XFEATFRAME_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "RTRenderer/CameraCalibration.h"

class XfeatFrame
{
public:
    cv::Matx44d w2cam;
    CameraCalibration calibration;
    std::unordered_map<std::string, torch::Tensor> features;
    cv::Mat features_3d;
    cv::Mat render;

    XfeatFrame(){};
    XfeatFrame(cv::Matx44d w2cam, std::unordered_map<std::string, torch::Tensor> features, CameraCalibration calibration, cv::Mat features_3d, cv::Mat render): w2cam(w2cam), features(features), calibration(calibration), features_3d(features_3d), render{render}{}
    XfeatFrame(cv::Matx44d w2cam, std::unordered_map<std::string, torch::Tensor> features, CameraCalibration calibration, cv::Mat features_3d): w2cam(w2cam), features(features), calibration(calibration), features_3d(features_3d){}
    XfeatFrame(cv::Matx44d w2cam, std::unordered_map<std::string, torch::Tensor> features, CameraCalibration calibration): w2cam(w2cam), features(features), calibration(calibration){}

    void save(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(&w2cam), sizeof(cv::Matx44d));

        calibration.save(out);

        size_t map_size = features.size();
        out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
        for (const auto& [key, tensor] : features) {
            size_t key_len = key.size();
            out.write(reinterpret_cast<const char*>(&key_len), sizeof(size_t));
            out.write(key.data(), key_len);

            saveTensor(out, tensor);
        }

        int rows = features_3d.rows;
        int cols = features_3d.cols;
        int type = features_3d.type();
        bool empty = features_3d.empty();

        out.write(reinterpret_cast<const char*>(&empty), sizeof(bool));
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char*>(&type), sizeof(int));

        if (!empty) {
            size_t data_size = features_3d.total() * features_3d.elemSize();
            out.write(reinterpret_cast<const char*>(features_3d.data), data_size);
        }
    }

    void load(std::istream& in) {
        in.read(reinterpret_cast<char*>(&w2cam), sizeof(cv::Matx44d));

        calibration.load(in);

        size_t map_size;
        in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        features.clear();
        for (size_t i = 0; i < map_size; ++i) {
            size_t key_len;
            in.read(reinterpret_cast<char*>(&key_len), sizeof(size_t));
            std::string key(key_len, '\0');
            in.read(&key[0], key_len);

            torch::Tensor tensor = loadTensor(in);
            features[key] = tensor;
        }

        bool empty;
        int rows, cols, type;
        in.read(reinterpret_cast<char*>(&empty), sizeof(bool));
        in.read(reinterpret_cast<char*>(&rows), sizeof(int));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int));
        in.read(reinterpret_cast<char*>(&type), sizeof(int));

        if (!empty) {
            features_3d.create(rows, cols, type);
            size_t data_size = features_3d.total() * features_3d.elemSize();
            in.read(reinterpret_cast<char*>(features_3d.data), data_size);
        } else {
            features_3d.release();
        }
    }

private:
    torch::Tensor loadTensor(std::istream& in) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

        std::string tensor_data(size, '\0');
        in.read(&tensor_data[0], size);

        std::istringstream tensor_buf(tensor_data, std::ios::binary);
        torch::Tensor tensor;
        torch::load(tensor, tensor_buf);  // now it works
        return tensor;
    }

    void saveTensor(std::ostream& out, const torch::Tensor& tensor) const {
        std::ostringstream tensor_buf(std::ios::binary);
        torch::save(tensor, tensor_buf);  // serialize to buffer
        std::string tensor_data = tensor_buf.str();

        size_t size = tensor_data.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        out.write(tensor_data.data(), size);
    }
};

#endif // XFEATFRAME_H
