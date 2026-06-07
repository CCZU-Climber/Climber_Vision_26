#include <fmt/core.h>
#include <fmt/format.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "tasks/omniperception/armor_selector.hpp"
#include "tasks/omniperception/debug_view.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "tools/plotter.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"

using namespace std::chrono_literals;
using auto_aim::Plan;

const std::string keys =
  "{help h usage ? |      | print help message }"
  "{@config-path   | configs/sentry.yaml | yaml config path}"
  "{d display      |      | show debug windows}";

int main(int argc, char * argv[])
{
  // 解析运行参数并加载 yaml 配置。
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto display = cli.has("display");
  auto yaml = tools::load(config_path);
  auto omni_memory_duration_ms = yaml["omni_memory_duration_ms"].as<int64_t>();

  // 自瞄与全向感知核心模块。
  tools::Exiter exiter;
  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Planner planner(config_path);
  tools::Plotter plotter;

  omniperception::Decider decider(config_path);
  omniperception::ArmorSelector armor_selector(config_path);

  std::vector<std::string> perception_camera_names;
  if (yaml["camera_name_map"]) {
    for (const auto & node : yaml["camera_name_map"]) {
      perception_camera_names.push_back(node.first.as<std::string>());
    }
  }

  // 当前全向感知链路最多使用 3 路相机。
  if (perception_camera_names.size() > 3) {
    tools::logger()->warn(
      "camera_name_map has {} entries, only first 3 will be used", perception_camera_names.size());
    perception_camera_names.resize(3);
  }

  std::vector<std::unique_ptr<io::USBCamera>> perception_cameras_owned;
  std::vector<io::USBCamera *> perception_cameras;
  perception_cameras_owned.reserve(perception_camera_names.size());
  perception_cameras.reserve(perception_camera_names.size());

  // 动态初始化感知相机：
  // 1) 按配置逐路尝试创建，任意一路失败不会中断整体流程（异常隔离）。
  // 2) `perception_cameras_owned` 持有对象生命周期，避免裸指针悬空。
  // 3) `perception_cameras` 仅向后续感知模块提供非拥有指针视图，便于接口复用。
  // 4) 即使某些相机启动时未就绪，也保留对象并依赖相机内部自动重连机制恢复。
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
      // 初始化状态仅用于启动诊断；未就绪不代表永久不可用。
      if (!cam->is_initialized()) {
        tools::logger()->warn(
          "Perception camera {} not ready at startup, keep auto-reconnect enabled", cam_name);
      } else {
        tools::logger()->info("Perception camera {} initialized", cam_name);
      }
      // 先记录非拥有指针，再转移所有权；两者最终指向同一对象实例。
      perception_cameras.push_back(cam.get());
      perception_cameras_owned.push_back(std::move(cam));
    } catch (const std::exception & e) {
      // 单路创建失败仅记日志并跳过，保证其余相机仍可参与全向感知。
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
  std::list<auto_aim::Armor> armors;
  std::list<auto_aim::Target> targets;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);
  tools::ThreadSafeQueue<std::optional<io::Command>, true> omni_command_queue(1);
  omni_command_queue.push(std::nullopt);

  auto last_omni_time = std::chrono::steady_clock::now();
  io::Command last_omni_command{false, false, 0, 0};

  std::atomic<bool> quit = false;
  // 控制线程：优先消费全向指令，否则使用自瞄规划结果，再下发云台。
  auto plan_thread = std::thread([&]() {
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto gs = gimbal.state();
      auto omni_opt = omni_command_queue.front();

      Plan plan;
      if (omni_opt.has_value()) {
        // 有全向指令时优先使用全向指令。
        auto & cmd = omni_opt.value();
        plan.control = cmd.control;
        plan.fire = cmd.shoot;
        plan.yaw = static_cast<float>(tools::limit_rad(cmd.yaw + gs.yaw));
        plan.pitch = static_cast<float>(cmd.pitch);
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
      } else {
        auto target_opt = target_queue.front();
        if (target_opt.has_value()) {
          // 否则回退到常规自瞄规划。
          plan = planner.plan(target_opt.value(), gs.bullet_speed);
        } else {
          plan = Plan{};
        }
      }

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;
      if (fired) {
        tools::logger()->info(
          "[FIRE] bullet_count={} plan_fire={} omni={}", gs.bullet_count, plan.fire,
          omni_opt.has_value());
      }


      nlohmann::json data;


      data["gimbal_yaw"] = tools::limit_rad(gs.yaw)*57.3;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = tools::limit_rad(gs.pitch)*57.3;
      data["gimbal_pitch_vel"] = gs.pitch_vel;
      data["bullet_speed"] = gs.bullet_speed;
      data["predict_time"] = plan.predict_time * 1e3;
      // tools::logger()->info("[debug] bullet_speed={:.3f}", gs.bullet_speed);


      data["target_yaw"] = plan.target_yaw*57.3;
      data["target_pitch"] = plan.target_pitch*57.3;

      data["plan_yaw"] = plan.yaw*57.3;
      data["plan_yaw_vel"] = plan.yaw_vel;
      // data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch*57.3;
      data["plan_pitch_vel"] = plan.pitch_vel;
      // data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;
      
      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  auto last_frame_time = std::chrono::steady_clock::now();

  // 主感知循环：检测 -> 选择控制来源（全向/自瞄）-> 发布指令。
  while (!exiter.exit()) {
    auto loop_start = std::chrono::steady_clock::now();

    camera.read(img, t);
    if (img.empty()) break;

    q = gimbal.q(t - 1ms);

    solver.set_R_gimbal2world(q);
    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    (void)gimbal_pos;

    armors = yolo.detect(img);

    armor_selector.set_invincible_enemy_ids({});
    armor_selector.armor_filter(armors, omniperception::ArmorFilterScope::MainAim);
    armor_selector.set_priority(armors);

    std::vector<omniperception::DetectionResult> raw_detection_queue;
    std::vector<omniperception::DetectionResult> decision_queue;
    if (perceptron) {
      raw_detection_queue = perceptron->get_detection_queue();
      decision_queue = raw_detection_queue;
      decider.sort(decision_queue);
    }

    targets = tracker.track(armors, t);

    io::Command debug_command{false, false, 0, 0};
    // 仲裁规则：
    // 1) 跟踪丢失：使用全向感知决策。
    // 2) 跟踪存在目标：使用自瞄规划。
    // 3) 其他情况：在短时记忆窗口内复用上一次全向指令。
    if (tracker.state() == "lost" && !decision_queue.empty()) {
      auto cmd = decider.decide(decision_queue);
      cmd.control = true;
      cmd.shoot = false;
      omni_command_queue.push(cmd);
      target_queue.push(std::nullopt);
      last_omni_command = cmd;
      last_omni_time = std::chrono::steady_clock::now();
      debug_command = cmd;
    } else if (!targets.empty()) {
      target_queue.push(targets.front());
      omni_command_queue.push(std::nullopt);
      last_omni_command = {false, false, 0, 0};
    } else {
      auto now = std::chrono::steady_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_omni_time).count();
      if (duration < omni_memory_duration_ms && last_omni_command.control) {
        omni_command_queue.push(last_omni_command);
        debug_command = last_omni_command;
      } else {
        omni_command_queue.push(std::nullopt);
      }
      target_queue.push(std::nullopt);
    }

    double fps = 1.0 / tools::delta_time(loop_start, last_frame_time);
    last_frame_time = loop_start;
    std::string fps_text = fmt::format("FPS: {:.1f}", fps);

    if (display) {
      debug_view.update_perception(raw_detection_queue);
      debug_view.render_main(img, armors, tracker.state(), debug_command, fps_text);
      debug_view.render_perception(fps_text);
      if (debug_view.should_exit()) break;
    }
  }

  // 退出前停止控制线程，并将云台指令安全归零。
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
