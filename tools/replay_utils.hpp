#ifndef TOOLS__REPLAY_UTILS_HPP
#define TOOLS__REPLAY_UTILS_HPP

#include <string>

namespace tools
{
std::string autodetect_replay_txt(const std::string & video_path);
int replay_delay_ms(double fps, double speed);
}  // namespace tools

#endif  // TOOLS__REPLAY_UTILS_HPP
