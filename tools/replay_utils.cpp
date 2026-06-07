#include "replay_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace tools
{
std::string autodetect_replay_txt(const std::string & video_path)
{
  std::string txt_path = video_path;
  for (const auto * ext : {".avi", ".mp4", ".AVI", ".MP4"}) {
    const auto pos = txt_path.rfind(ext);
    if (pos != std::string::npos) {
      txt_path.replace(pos, std::string(ext).size(), ".txt");
      break;
    }
  }

  if (txt_path == video_path) return {};
  std::ifstream file(txt_path);
  return file.good() ? txt_path : std::string{};
}

int replay_delay_ms(double fps, double speed)
{
  if (fps <= 0.0 || fps > 120.0 || !std::isfinite(fps)) fps = 30.0;
  int delay = static_cast<int>(1000.0 / fps + 0.5);
  if (speed <= 0.0) return 1;
  if (speed != 1.0) delay = std::max(1, static_cast<int>(delay / speed));
  return delay;
}
}  // namespace tools
