#ifndef TOOLS__FRAME_SOURCE_HPP
#define TOOLS__FRAME_SOURCE_HPP

#include <Eigen/Geometry>

#include <chrono>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "io/camera.hpp"
#include "tools/recorder.hpp"
#include "tools/replay_reader.hpp"

namespace tools
{
struct FramePacket
{
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

class FrameSource
{
public:
  using QuaternionProvider =
    std::function<Eigen::Quaterniond(const std::chrono::steady_clock::time_point &)>;

  struct Options
  {
    std::string config_path;
    std::string video_path;
    std::string txt_path;
    double replay_speed = 1.0;
    QuaternionProvider live_q_provider;
    bool record_live = true;
  };

  explicit FrameSource(Options options);

  bool replay_mode() const;
  int replay_delay_ms() const;
  bool read(FramePacket & packet);

private:
  bool replay_mode_;
  int replay_delay_ms_ = 1;
  size_t replay_frame_index_ = 0;
  bool warned_missing_replay_pose_ = false;
  QuaternionProvider live_q_provider_;
  cv::VideoCapture cap_;
  std::unique_ptr<io::Camera> camera_;
  std::unique_ptr<tools::Recorder> recorder_;
  std::vector<tools::ReplayFrame> replay_frames_;
};
}  // namespace tools

#endif  // TOOLS__FRAME_SOURCE_HPP
