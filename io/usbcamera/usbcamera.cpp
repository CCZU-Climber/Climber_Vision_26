#include "usbcamera.hpp"

#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace io
{
USBCamera::USBCamera(const std::string & open_name, const std::string & config_path, bool flip)
: open_name_(open_name), flip_(flip), quit_(false), ok_(false), queue_(1), open_count_(0)
{
  auto yaml = tools::load(config_path);
  image_width_ = tools::read<double>(yaml, "image_width");
  image_height_ = tools::read<double>(yaml, "image_height");
  // usb_exposure_ = tools::read<double>(yaml, "usb_exposure");
  usb_frame_rate_ = tools::read<double>(yaml, "usb_frame_rate");
  usb_gamma_ = tools::read<double>(yaml, "usb_gamma");
  usb_gain_ = tools::read<double>(yaml, "usb_gain");
  yaw_offset_deg_ = 0.0;
  device_name = open_name_;

  // 从配置文件中读取相机偏角映射，用于识别不同方位的相机
  if (yaml["camera_name_map"] && yaml["camera_name_map"][open_name_]) {
    const auto node = yaml["camera_name_map"][open_name_];
    try {
      if (node.IsScalar()) {
        yaw_offset_deg_ = node.as<double>();
      } else if (node.IsMap() && node["yaw"]) {
        yaw_offset_deg_ = node["yaw"].as<double>();
      }
      tools::logger()->info(
        "Camera {} yaw offset configured to {:.2f} deg", open_name_, yaw_offset_deg_);
    } catch (const std::exception & e) {
      tools::logger()->warn("Failed to parse yaw offset for {}: {}", open_name_, e.what());
    }
  }
  tools::logger()->info(
    "Camera {} frame flip configured to {}", open_name_, flip_ ? "true" : "false");

  try_open();

  // 守护线程：相机掉线后持续自动重连
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(100ms);

      if (ok_) continue;

      if (capture_thread_.joinable()) capture_thread_.join();

      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        close();
      }

      try_open();
      if (!ok_ && open_count_ % 50 == 0) {
        tools::logger()->warn(
          "Camera {} still offline after {} reopen attempts", this->device_name, open_count_);
      }
    }
  }};
}

USBCamera::~USBCamera()
{
  quit_ = true;
  {
    std::lock_guard<std::mutex> lock(cap_mutex_);
    close();
  }
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
  tools::logger()->info("USBCamera destructed.");
}

cv::Mat USBCamera::read()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  if (!cap_.isOpened()) {
    tools::logger()->warn("Failed to read {} USB camera", this->device_name);
    return cv::Mat();
  }
  cap_ >> img_;
  return img_;
}

void USBCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

bool USBCamera::read(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, int timeout_ms)
{
  CameraData data;
  if (!queue_.pop_for(data, std::chrono::milliseconds(timeout_ms))) {
    return false;
  }

  img = data.img;
  timestamp = data.timestamp;
  return !img.empty();
}

void USBCamera::open()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  std::string true_device_name = "/dev/" + open_name_;
  cap_.open(true_device_name, cv::CAP_V4L);
  if (!cap_.isOpened()) {
    tools::logger()->warn("Failed to open USB camera {}", true_device_name);
    return;
  }

  // 直接使用设备名作为相机标识（camera_front/camera_right等）
  device_name = open_name_;

  cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_.set(cv::CAP_PROP_FPS, usb_frame_rate_);
  cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
  cap_.set(cv::CAP_PROP_GAMMA, usb_gamma_);
  cap_.set(cv::CAP_PROP_GAIN, usb_gain_);

  tools::logger()->info("Camera {} opened (device: {})", device_name, open_name_);

  // 为所有相机设置相同的参数
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
  // cap_.set(cv::CAP_PROP_EXPOSURE, usb_exposure_);

  tools::logger()->info("USBCamera {} opened successfully", device_name);
  tools::logger()->info("  FPS: {}", cap_.get(cv::CAP_PROP_FPS));

  // 取图线程
  capture_thread_ = std::thread{[this] {
    ok_ = true;
    std::this_thread::sleep_for(50ms);
    tools::logger()->info("[{} USB camera] capture thread started", this->device_name);
    while (!quit_) {
      std::this_thread::sleep_for(1ms);

      cv::Mat img;
      bool success;
      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        if (!cap_.isOpened()) {
          break;
        }
        success = cap_.read(img);
      }

      if (!success) {
        tools::logger()->warn("Failed to read frame, exiting capture thread");
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      if (flip_) {
        cv::flip(img, img, -1);
      }
      queue_.push({img, timestamp});
    }
    ok_ = false;
  }};
}

void USBCamera::try_open()
{
  try {
    open_count_++;
    open();
  } catch (const std::exception & e) {
    tools::logger()->warn("{}", e.what());
  }
}

void USBCamera::close()
{
  if (cap_.isOpened()) {
    cap_.release();
    tools::logger()->info("USB camera released.");
  }
}

}  // namespace io
