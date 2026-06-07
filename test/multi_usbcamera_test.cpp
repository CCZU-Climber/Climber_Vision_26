#include <fmt/core.h>
#include <fmt/format.h>

#include <chrono>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/debug_view.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                     | print help message}"
  "{@config-path   | configs/sentry.yaml | positional yaml config path }"
  "{d display      |                     | show video stream       }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto display = cli.has("display");

  auto yaml = tools::load(config_path);
  omniperception::Decider decider(config_path);

  std::vector<std::string> perception_camera_names;
  if (yaml["camera_name_map"]) {
    for (const auto & node : yaml["camera_name_map"]) {
      perception_camera_names.push_back(node.first.as<std::string>());
    }
  }

  if (perception_camera_names.size() > 3) {
    tools::logger()->warn(
      "camera_name_map has {} entries, only first 3 will be used", perception_camera_names.size());
    perception_camera_names.resize(3);
  }

  std::vector<std::unique_ptr<io::USBCamera>> perception_cameras_owned;
  std::vector<io::USBCamera *> perception_cameras;
  perception_cameras_owned.reserve(perception_camera_names.size());
  perception_cameras.reserve(perception_camera_names.size());

  for (const auto & cam_name : perception_camera_names) {
    try {
      bool flip = false;
      const auto camera_config = yaml["camera_name_map"][cam_name];
      if (camera_config && camera_config.IsMap() && camera_config["flip"]) {
        try {
          flip = camera_config["flip"].as<bool>();
        } catch (const std::exception & e) {
          tools::logger()->warn("Failed to parse flip for {}: {}, using false", cam_name, e.what());
        }
      }

      auto cam = std::make_unique<io::USBCamera>(cam_name, config_path, flip);
      if (!cam->is_initialized()) {
        tools::logger()->warn(
          "Perception camera {} not ready at startup, keep auto-reconnect enabled", cam_name);
      } else {
        tools::logger()->info("Perception camera {} initialized", cam_name);
      }
      perception_cameras.push_back(cam.get());
      perception_cameras_owned.push_back(std::move(cam));
    } catch (const std::exception & e) {
      tools::logger()->error("Failed to create perception camera {}: {}", cam_name, e.what());
    }
  }

  std::unique_ptr<omniperception::Perceptron> perceptron;
  if (!perception_cameras.empty()) {
    try {
      perceptron =
        std::make_unique<omniperception::Perceptron>(perception_cameras, config_path, display);
    } catch (const std::exception & e) {
      tools::logger()->error("Perceptron init failed: {}", e.what());
    }
  } else {
    tools::logger()->warn("No perception cameras configured, omni-perception will be disabled");
  }

  omniperception::AimOmniDebugView debug_view(display, perception_camera_names);

  int frame_count = 0;
  auto last_log_time = std::chrono::steady_clock::now();
  auto last_loop_time = std::chrono::steady_clock::now();

  std::map<std::string, omniperception::DetectionResult> latest_results;

  while (!exiter.exit()) {
    std::vector<omniperception::DetectionResult> raw_results;
    if (perceptron) {
      raw_results = perceptron->get_detection_queue();
    }

    auto display_results = raw_results;
    for (auto & dr : display_results) {
      decider.armor_filter(dr.armors);
      decider.set_priority(dr.armors);
      dr.armors.sort(
        [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
    }

    auto decision_results = display_results;
    decider.sort(decision_results);

    for (const auto & res : display_results) {
      latest_results[res.camera_name] = res;
    }

    if (raw_results.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    frame_count++;

    auto now = std::chrono::steady_clock::now();
    auto time_since_log =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count();

    if (time_since_log >= 1000) {
      last_log_time = now;

      int raw_detection_count = 0;
      for (const auto & res : raw_results) {
        raw_detection_count += static_cast<int>(res.armors.size());
      }

      int decision_detection_count = 0;
      for (const auto & res : decision_results) {
        decision_detection_count += static_cast<int>(res.armors.size());
      }

      tools::logger()->info(
        "=== Frame {}: Raw detections: {}, Decision detections: {} ===", frame_count,
        raw_detection_count, decision_detection_count);

      if (!decision_results.empty() && !decision_results.front().armors.empty()) {
        const auto & final_res = decision_results.front();
        const auto & final_armor = final_res.armors.front();
        tools::logger()->info(
          "  Final target: camera={}, armor={} {}, color={}, yaw={:.2f} deg, pitch={:.2f} deg",
          final_res.camera_name, auto_aim::ARMOR_NAMES[final_armor.name],
          auto_aim::ARMOR_TYPES[final_armor.type], auto_aim::COLORS[final_armor.color],
          tools::limit_rad(final_res.delta_yaw) * 57.3, final_res.delta_pitch * 57.3);
      } else {
        tools::logger()->info("  Final target: No final target");
      }

      if (!latest_results.empty()) {
        tools::logger()->info("  Latest detections by camera (cached):");
        for (const auto & [name, res] : latest_results) {
          tools::logger()->info(
            "    {}: {} detections, Yaw: {:.2f} deg, Pitch: {:.2f} deg", name, res.armors.size(),
            tools::limit_rad(res.delta_yaw) * 57.3, res.delta_pitch * 57.3);
          for (const auto & armor : res.armors) {
            tools::logger()->info(
              "      - Armor: {} {} color={}", auto_aim::ARMOR_NAMES[armor.name],
              auto_aim::ARMOR_TYPES[armor.type], auto_aim::COLORS[armor.color]);
          }
        }
      } else {
        tools::logger()->info("  No detections yet");
      }
    }

    if (display) {
      double fps = 1.0 / tools::delta_time(now, last_loop_time);
      last_loop_time = now;
      std::string fps_text = fmt::format("FPS: {:.1f}", fps);

      debug_view.update_perception(display_results);
      debug_view.render_perception(fps_text);
      if (debug_view.should_exit()) break;
    }
  }

  return 0;
}
