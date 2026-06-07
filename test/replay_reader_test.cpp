#include <cassert>
#include <chrono>
#include <fstream>

#include "tools/replay_reader.hpp"

int main()
{
  const char * path = "/tmp/replay_reader_test.txt";
  {
    std::ofstream out(path);
    out << "10.0 0 0 0 1\n";
    out << "bad line\n";
    out << "10.5 0.1 0.2 0.3 0.9\n";
  }

  const auto frames = tools::load_replay_frames(path);
  assert(frames.size() == 2);
  assert(frames[0].time_offset == 10.0);
  assert(frames[1].q.w() == 0.9);
  assert(frames[1].q.x() == 0.1);

  const auto t0 = tools::replay_timestamp(frames, 0);
  const auto t1 = tools::replay_timestamp(frames, 1);
  const auto dt = std::chrono::duration<double>(t1 - t0).count();
  assert(dt > 0.499 && dt < 0.501);
  assert(tools::replay_timestamp({}, 0) == std::chrono::steady_clock::time_point{});
  assert(tools::replay_timestamp(frames, 9) == std::chrono::steady_clock::time_point{});
  return 0;
}
