#include "perceptron.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace omniperception
{
Perceptron::Perceptron(
  const std::vector<io::USBCamera *> & cameras, const std::string & config_path, bool debug)
: config_path_(config_path),
  queue_full_count_(0),
  detection_queue_(10, [this] { queue_full_count_.fetch_add(1, std::memory_order_relaxed); }),
  rr_index_(0),
  decider_(config_path),
  stop_flag_(false),
  debug_(debug)
{
  cameras_ = cameras;
  cameras_.erase(std::remove(cameras_.begin(), cameras_.end(), nullptr), cameras_.end());

  if (cameras_.empty()) {
    tools::logger()->error("No cameras available! System cannot start.");
    throw std::runtime_error("No cameras available");
  }

  // 读取配置文件，为感知相机创建单独的配置
  auto yaml = YAML::LoadFile(config_path);
  std::string perception_device = "CPU";  // 默认使用CPU
  if (yaml["perception_device"]) {
    perception_device = yaml["perception_device"].as<std::string>();
  }

  tools::logger()->info("Perception cameras using device: {}", perception_device);

  // 创建临时配置文件，强制感知相机使用指定设备
  std::string temp_config_path = "/tmp/perception_config.yaml";
  yaml["device"] = perception_device;
  std::ofstream temp_config(temp_config_path);
  temp_config << yaml;
  temp_config.close();

  // 初始化 YOLO，使用感知相机专用配置
  yolo_ = std::make_shared<auto_aim::YOLO>(temp_config_path, false);

  std::this_thread::sleep_for(std::chrono::seconds(2));

  tools::logger()->info("Perceptron camera order (RR):");
  for (size_t i = 0; i < cameras_.size(); ++i) {
    tools::logger()->info("  [{}] {}", i, cameras_[i]->device_name);
  }

  tools::logger()->info("Perceptron debug image output: {}", debug_ ? "ON" : "OFF");

  // 创建推理线程（单线程、轮询推理）
  inference_thread_ = std::thread([this] { inference_loop(); });

  tools::logger()->info("Perceptron initialized.");
}

Perceptron::~Perceptron()
{
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_flag_ = true;
  }
  condition_.notify_all();

  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }
  tools::logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  while (!detection_queue_.empty()) {
    detection_queue_.pop(temp);
    result.push_back(std::move(temp));
  }

  return result;
}

void Perceptron::inference_loop()
{
  if (cameras_.empty()) {
    tools::logger()->error("No camera pointers available! Cannot run inference.");
    return;
  }

  tools::logger()->info("Inference loop started (round-robin), camera count: {}", cameras_.size());

  constexpr int read_timeout_ms = 2;
  uint64_t last_logged_queue_full_count = 0;
  auto last_queue_log_time = std::chrono::steady_clock::now();

  try {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_flag_) break;
      }

      auto * cam = cameras_[rr_index_];
      rr_index_ = (rr_index_ + 1) % cameras_.size();

      if (!cam) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      cv::Mat img;
      std::chrono::steady_clock::time_point ts;
      bool got_frame = cam->read(img, ts, read_timeout_ms);
      if (!got_frame || img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      auto armors = yolo_->detect(img);

      DetectionResult dr;
      dr.armors = std::move(armors);
      dr.timestamp = ts;
      dr.camera_name = cam->device_name;

      if (!dr.armors.empty()) {
        auto delta_angle = decider_.delta_angle(dr.armors, dr.camera_name);
        dr.delta_yaw = delta_angle[0] / 57.3;
        dr.delta_pitch = delta_angle[1] / 57.3;
      } else {
        dr.delta_yaw = 0;
        dr.delta_pitch = 0;
      }

      if (debug_) {
        dr.img = img.clone();
        if (!dr.armors.empty()) {
          for (const auto & armor : dr.armors) {
            tools::draw_points(dr.img, armor.points, {0, 255, 0});
          }
        }

        cv::putText(
          dr.img, dr.camera_name, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
          cv::Scalar(0, 255, 255), 2);
      }

      detection_queue_.push(dr);

      if (debug_) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_queue_log_time >= std::chrono::seconds(1)) {
          auto total_full_count = queue_full_count_.load(std::memory_order_relaxed);
          auto delta_full_count = total_full_count - last_logged_queue_full_count;
          if (delta_full_count > 0) {
            tools::logger()->warn(
              "Perceptron detection queue full {} times in last 1s (total {}), dropping oldest frame "
              "to keep latest",
              delta_full_count, total_full_count);
          }
          last_logged_queue_full_count = total_full_count;
          last_queue_log_time = now;
        }
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Exception in inference_loop: {}", e.what());
  }
}

}  // namespace omniperception
