#include <fmt/core.h>
#include <fmt/format.h>

#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/debug_view.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                     | print help message}"
  "{@config-path   | configs/sentry.yaml | positional yaml config path }"
  "{d display      |                     | show debug windows       }";

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

  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  (void)aimer;
  (void)shooter;

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

  cv::Mat img;
  steady_clock::time_point timestamp;

  int frame_count = 0;
  auto last_frame_time = steady_clock::now();

  while (!exiter.exit()) {
    auto loop_start = steady_clock::now();

    camera.read(img, timestamp);
    if (img.empty()) {
      tools::logger()->warn("Empty image from camera");
      continue;
    }

    auto armors = yolo.detect(img);

    decider.get_invincible_armor({});
    decider.armor_filter(armors);
    decider.set_priority(armors);

    std::vector<omniperception::DetectionResult> raw_detection_queue;
    std::vector<omniperception::DetectionResult> decision_queue;
    if (perceptron) {
      raw_detection_queue = perceptron->get_detection_queue();
      decision_queue = raw_detection_queue;
      decider.sort(decision_queue);
    }

    auto targets = tracker.track(armors, timestamp);
    (void)targets;

    io::Command command{false, false, 0, 0};
    if (tracker.state() == "switching") {
      if (!decision_queue.empty()) {
        const auto & switch_target = decision_queue.front();
        command.control = !switch_target.armors.empty();
        command.shoot = false;
        command.pitch = tools::limit_rad(switch_target.delta_pitch);
        command.yaw = tools::limit_rad(switch_target.delta_yaw);
      }
    } else if (tracker.state() == "lost") {
      command = decider.decide(decision_queue);
      command.yaw = tools::limit_rad(command.yaw);
      command.pitch = tools::limit_rad(command.pitch);
    }

    double fps = 1.0 / tools::delta_time(loop_start, last_frame_time);
    last_frame_time = loop_start;
    std::string fps_text = fmt::format("FPS: {:.1f}", fps);

    if (command.control) {
      tools::logger()->info(
        "Command: Ctrl:{} Shoot:{} Yaw:{:.2f} Pitch:{:.2f}", command.control ? "ON" : "OFF",
        command.shoot ? "ON" : "OFF", command.yaw * 57.3, command.pitch * 57.3);
    }


    if (display) {
      frame_count++;
      debug_view.update_perception(raw_detection_queue);
      debug_view.render_main(img, armors, tracker.state(), command, fps_text);
      debug_view.render_perception(fps_text);
      if (debug_view.should_exit()) break;
    }
  }

  return 0;
}
