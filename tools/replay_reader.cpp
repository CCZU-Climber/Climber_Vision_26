#include "replay_reader.hpp"

#include <fstream>
#include <sstream>

namespace tools
{
std::vector<ReplayFrame> load_replay_frames(const std::string & txt_path)
{
  std::vector<ReplayFrame> frames;
  std::ifstream file(txt_path);
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::istringstream stream(line);
    double t = 0.0;
    double qw = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    if (!(stream >> t >> qw >> qx >> qy >> qz)) continue;  // recorder: t qw qx qy qz

    frames.push_back({t, Eigen::Quaterniond(qw, qx, qy, qz)});
  }
  return frames;
}

std::chrono::steady_clock::time_point replay_timestamp(
  const std::vector<ReplayFrame> & frames, size_t frame_index)
{
  if (frames.empty() || frame_index >= frames.size()) {
    return std::chrono::steady_clock::time_point{};
  }

  const auto elapsed_seconds = frames[frame_index].time_offset - frames.front().time_offset;
  return std::chrono::steady_clock::time_point(
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(elapsed_seconds)));
}
}  // namespace tools
