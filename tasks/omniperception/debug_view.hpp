#ifndef OMNIPERCEPTION__DEBUG_VIEW_HPP
#define OMNIPERCEPTION__DEBUG_VIEW_HPP

#include <cstddef>
#include <list>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"
#include "io/command.hpp"
#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{

class AimOmniDebugView
{
public:
  AimOmniDebugView(bool enabled, std::vector<std::string> camera_names);

  void update_perception(const std::vector<DetectionResult> & detection_queue);

  void render_main(
    const cv::Mat & img, const std::list<auto_aim::Armor> & armors, const std::string & tracker_state,
    const io::Command & command, const std::string & fps_text);

  void render_perception(const std::string & fps_text);

  bool should_exit() const;

private:
  struct PerceptionFrame
  {
    cv::Mat img;
    std::size_t armors_count{0};
    double delta_yaw_rad{0.0};
    double delta_pitch_rad{0.0};
  };

  bool enabled_;
  std::vector<std::string> camera_names_;
  std::map<std::string, PerceptionFrame> latest_frames_;
};

}  // namespace omniperception

#endif
