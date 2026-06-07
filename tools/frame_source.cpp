#include "frame_source.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "tools/logger.hpp"
#include "tools/replay_utils.hpp"

namespace tools
{
FrameSource::FrameSource(Options options)
: replay_mode_(!options.video_path.empty()), live_q_provider_(std::move(options.live_q_provider))
{
  if (replay_mode_) {
    cap_.open(options.video_path);
    if (!cap_.isOpened()) {
      throw std::runtime_error("无法打开视频文件: " + options.video_path);
    }

    if (options.txt_path.empty()) options.txt_path = tools::autodetect_replay_txt(options.video_path);
    if (options.txt_path.empty()) {
      throw std::runtime_error("回放模式需要配套txt姿态文件");
    }

    replay_frames_ = tools::load_replay_frames(options.txt_path);
    if (replay_frames_.empty()) {
      throw std::runtime_error("回放数据为空或无法解析: " + options.txt_path);
    }

    const double fps = cap_.get(cv::CAP_PROP_FPS);
    replay_delay_ms_ = tools::replay_delay_ms(fps, options.replay_speed);
    tools::logger()->info(
      "加载回放数据: {} 帧, {:.1f}fps x{:.1f} = delay={}ms", replay_frames_.size(), fps,
      options.replay_speed, replay_delay_ms_);
    return;
  }

  camera_ = std::make_unique<io::Camera>(options.config_path);
  if (options.record_live) recorder_ = std::make_unique<tools::Recorder>();
}

bool FrameSource::replay_mode() const { return replay_mode_; }

int FrameSource::replay_delay_ms() const { return replay_delay_ms_; }

bool FrameSource::read(FramePacket & packet)
{
  packet = FramePacket{};

  if (replay_mode_) {
    cap_ >> packet.image;
    if (packet.image.empty()) return false;

    if (replay_frame_index_ < replay_frames_.size()) {
      packet.q = replay_frames_[replay_frame_index_].q;
      packet.timestamp = tools::replay_timestamp(replay_frames_, replay_frame_index_);
    } else {
      if (!warned_missing_replay_pose_) {
        tools::logger()->warn(
          "视频帧数超过txt姿态帧数，从第{}帧开始退回到当前时间戳", replay_frame_index_);
        warned_missing_replay_pose_ = true;
      }
      packet.timestamp = std::chrono::steady_clock::now();
    }

    replay_frame_index_++;
    return true;
  }

  camera_->read(packet.image, packet.timestamp);
  if (packet.image.empty()) return false;
  if (live_q_provider_) packet.q = live_q_provider_(packet.timestamp);
  if (recorder_) recorder_->record(packet.image, packet.q, packet.timestamp);
  return true;
}
}  // namespace tools
