#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_big_group.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto target_params = auto_buff::load_target_params(config_path);
  auto_buff::SmallTarget buff_small_target(target_params);
  auto_buff::BigTargetGroup buff_big_group(target_params);
  auto_buff::BigTargetSelector buff_big_selector(config_path);
  auto_buff::Aimer buff_aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      if (last_mode == io::GimbalMode::BIG_BUFF && mode != io::GimbalMode::BIG_BUFF) {
        buff_big_selector.reset_cycle();
      }
      last_mode = mode.load();
    }

    camera.read(img, t);
    auto q = gimbal.q(t);
    auto gs = gimbal.state();
    recorder.record(img, q, t);
    solver.set_R_gimbal2world(q);

    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto armors = yolo.detect(img);
      auto targets = tracker.track(armors, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    } else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      buff_solver.set_R_gimbal2world(q);

      auto_aim::Plan buff_plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        auto observations = buff_detector.detect(img, auto_buff::SMALL);
        buff_solver.solve(observations);
        std::optional<auto_buff::PowerRune> power_rune =
          observations.empty() ? std::nullopt : std::optional<auto_buff::PowerRune>(observations.front());

        buff_small_target.get_target(power_rune, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else {
        auto observations = buff_detector.detect(img, auto_buff::BIG);
        buff_solver.solve(observations);
        buff_big_group.update(observations, t);

        auto selected_target = buff_big_selector.select_target(
          buff_big_group, buff_aimer, t, gs.bullet_speed, gs.yaw, gs.pitch,
          std::chrono::steady_clock::now());
        if (selected_target.has_value()) {
          auto target_copy = selected_target.value();
          buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
          if (buff_plan.fire) {
            buff_big_selector.on_fire(std::chrono::steady_clock::now());
          }
        }
      }

      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

    } else {
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    }

    // FPS 计算和显示
    static auto last_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    double fps = 1.0 / std::chrono::duration<double>(current_time - last_time).count();
    last_time = current_time;
    tools::draw_text(img, fmt::format("FPS: {:.1f}", fps), {10, 30}, {0, 255, 255});
    // tools::logger()->info("{:.2f} fps", fps);

    cv::resize(img, img, {}, 0.5, 0.5); // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
