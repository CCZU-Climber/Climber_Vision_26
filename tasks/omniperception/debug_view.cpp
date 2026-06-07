#include "debug_view.hpp"

#include <algorithm>

#include <fmt/format.h>

#include <utility>

#include "tasks/auto_aim/armor.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
namespace
{
constexpr double kRadToDeg = 57.29577951308232;
constexpr int kPanelPadding = 12;
constexpr int kLineHeight = 30;
}  // namespace

AimOmniDebugView::AimOmniDebugView(bool enabled, std::vector<std::string> camera_names)
: enabled_(enabled), camera_names_(std::move(camera_names))
{
}

void AimOmniDebugView::update_perception(const std::vector<DetectionResult> & detection_queue)
{
  if (!enabled_) return;

  for (const auto & res : detection_queue) {
    if (res.camera_name.empty() || res.img.empty()) continue;

    latest_frames_[res.camera_name] = {
      res.img.clone(), res.armors.size(), res.delta_yaw, res.delta_pitch};
  }
}

void AimOmniDebugView::render_main(
  const cv::Mat & img, const std::list<auto_aim::Armor> & armors, const std::string & tracker_state,
  const io::Command & command, const std::string & fps_text)
{
  if (!enabled_ || img.empty()) return;

  cv::Mat display_img = img.clone();

  for (const auto & armor : armors) {
    tools::draw_points(display_img, armor.points, {0, 0, 255});
    cv::Point center(armor.center_norm.x * display_img.cols, armor.center_norm.y * display_img.rows);
    cv::circle(display_img, center, 8, cv::Scalar(0, 0, 255), 2);

    std::string info = fmt::format(
      "{} {} {}", auto_aim::ARMOR_NAMES[armor.name], auto_aim::ARMOR_TYPES[armor.type],
      auto_aim::COLORS[armor.color]);

    cv::Point info_pos(
      std::clamp(center.x + 10, 10, std::max(10, display_img.cols - 360)),
      std::clamp(center.y - 8, 24, std::max(24, display_img.rows - 12)));
    tools::draw_text(display_img, info, info_pos, {0, 255, 255}, 0.7, 2);
  }

  std::string state_text = fmt::format("State: {}", tracker_state);
  std::string cmd_text = fmt::format(
    "Ctrl:{} Shoot:{} Y:{:.2f} P:{:.2f}", command.control ? "ON" : "OFF",
    command.shoot ? "ON" : "OFF", command.yaw * kRadToDeg, command.pitch * kRadToDeg);

  int panel_w = std::min(std::max(420, display_img.cols / 2), std::max(420, display_img.cols - 24));
  int panel_h = kPanelPadding * 2 + kLineHeight * 3;
  int panel_x = std::max(12, display_img.cols - panel_w - 12);
  int panel_y = 12;
  cv::Rect panel(panel_x, panel_y, panel_w, panel_h);
  cv::rectangle(display_img, panel, cv::Scalar(20, 20, 20), cv::FILLED);
  cv::rectangle(display_img, panel, cv::Scalar(80, 80, 80), 2);

  int text_x = panel_x + kPanelPadding;
  int text_y = panel_y + kPanelPadding + 12;
  tools::draw_text(display_img, fps_text, {text_x, text_y}, {0, 255, 0}, 0.8, 2);
  text_y += kLineHeight;
  tools::draw_text(display_img, state_text, {text_x, text_y}, {255, 255, 0}, 0.8, 2);
  text_y += kLineHeight;
  tools::draw_text(display_img, cmd_text, {text_x, text_y}, {255, 0, 255}, 0.75, 2);

  cv::resize(display_img, display_img, {}, 0.5, 0.5);
  cv::imshow("Main Camera", display_img);
}

void AimOmniDebugView::render_perception(const std::string & fps_text)
{
  if (!enabled_) return;

  for (const auto & camera_name : camera_names_) {
    auto it = latest_frames_.find(camera_name);
    cv::Mat display_img;

    bool has_frame = (it != latest_frames_.end() && !it->second.img.empty());
    std::size_t armors_count = has_frame ? it->second.armors_count : 0;
    double delta_yaw_rad = has_frame ? it->second.delta_yaw_rad : 0.0;
    double delta_pitch_rad = has_frame ? it->second.delta_pitch_rad : 0.0;

    if (has_frame) {
      display_img = it->second.img.clone();
    } else {
      display_img = cv::Mat::zeros(720, 1280, CV_8UC3);
    }

    std::string armors_text = fmt::format("Armors: {}", armors_count);
    double display_delta_yaw_rad = tools::limit_rad(delta_yaw_rad);
    std::string delta_text = armors_count > 0
                               ? fmt::format(
                                   "Delta Y:{:.2f}deg P:{:.2f}deg",
                                   display_delta_yaw_rad * kRadToDeg,
                                   delta_pitch_rad * kRadToDeg)
                               : "Delta: N/A";
    std::string status_text = !has_frame
                                ? "Waiting for frame"
                                : armors_count > 0 ? "Target: YES" : "Target: NO";

    int panel_w = std::min(std::max(440, display_img.cols / 2), std::max(440, display_img.cols - 24));
    int panel_h = kPanelPadding * 2 + kLineHeight * 5;
    int panel_x = 12;
    int panel_y = std::max(12, display_img.rows - panel_h - 12);
    cv::Rect panel(panel_x, panel_y, panel_w, panel_h);
    cv::rectangle(display_img, panel, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(display_img, panel, cv::Scalar(80, 80, 80), 2);

    int text_x = panel_x + kPanelPadding;
    int text_y = panel_y + kPanelPadding + 12;
    tools::draw_text(display_img, fps_text, {text_x, text_y}, {0, 255, 0}, 0.8, 2);
    text_y += kLineHeight;
    tools::draw_text(
      display_img, fmt::format("Camera: {}", camera_name), {text_x, text_y}, {255, 255, 0}, 0.8,
      2);
    text_y += kLineHeight;
    tools::draw_text(display_img, armors_text, {text_x, text_y}, {0, 255, 255}, 0.8, 2);
    text_y += kLineHeight;
    tools::draw_text(display_img, delta_text, {text_x, text_y}, {255, 200, 0}, 0.8, 2);
    text_y += kLineHeight;

    cv::Scalar status_color = armors_count > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    tools::draw_text(display_img, status_text, {text_x, text_y}, status_color, 0.8, 2);

    cv::resize(display_img, display_img, {}, 0.5, 0.5);
    cv::imshow(camera_name, display_img);
  }
}

bool AimOmniDebugView::should_exit() const
{
  if (!enabled_) return false;
  return cv::waitKey(1) == 'q';
}

}  // namespace omniperception
