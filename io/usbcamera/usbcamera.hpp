#ifndef IO__USBCamera_HPP
#define IO__USBCamera_HPP

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace io
{
// USB相机类，支持通过名字和配置文件初始化，后台线程采集图像
class USBCamera
{
public:
  USBCamera(const std::string & open_name, const std::string & config_path, bool flip = false);
  ~USBCamera();

  cv::Mat read();
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  bool read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, int timeout_ms);

  bool is_initialized() const { return ok_; }
  double yaw_offset_deg() const { return yaw_offset_deg_; }
  std::string device_name;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::mutex cap_mutex_;
  cv::VideoCapture cap_;
  cv::Mat img_;
  std::string open_name_;
  double usb_exposure_, usb_frame_rate_;
  int open_count_;
  double image_width_, image_height_;
  double usb_gamma_, usb_gain_;
  double yaw_offset_deg_;
  bool flip_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData> queue_;

  void try_open();
  void open();
  void close();
};

}  // namespace io

#endif
