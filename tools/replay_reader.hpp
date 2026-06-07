#ifndef TOOLS__REPLAY_READER_HPP
#define TOOLS__REPLAY_READER_HPP

#include <Eigen/Geometry>

#include <chrono>
#include <string>
#include <vector>

namespace tools
{
struct ReplayFrame
{
  double time_offset = 0.0;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

std::vector<ReplayFrame> load_replay_frames(const std::string & txt_path);

std::chrono::steady_clock::time_point replay_timestamp(
  const std::vector<ReplayFrame> & frames, size_t frame_index);
}  // namespace tools

#endif  // TOOLS__REPLAY_READER_HPP
