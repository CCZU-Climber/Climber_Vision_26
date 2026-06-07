#include <chrono>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                  | 输出命令行参数说明}"
  "{@config-path   | configs/uav.yaml | yaml配置文件路径 }";

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

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  auto t0 = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    camera.read(img, t);
    q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    auto targets = tracker.track(armors, t);

    for (const auto & armor : armors) {
      std::string label = fmt::format("{} {:.0f}%",
          auto_aim::ARMOR_NAMES[armor.name], armor.confidence * 100);
      cv::rectangle(img, armor.box, {0, 255, 0}, 2);
      tools::draw_text(img, label,
          {armor.box.x, armor.box.y - 5}, {0, 255, 0});
    }

    //     if (!targets.empty()) {
    //       const auto & target = targets.front();

    //       std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
    //       for (const Eigen::Vector4d & xyza : armor_xyza_list) {
    //         auto image_points =
    //           solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    //         for (size_t i = 0; i < image_points.size(); i++) {
    //           cv::line(img, image_points[i], image_points[(i + 1) % image_points.size()], {0, 0, 255}, 2);
    //         }
    //       }

    recorder.record(img, q, t);

    /// 自瞄
    io::Command command{};
    // if (mode == io::Mode::auto_aim || mode == io::Mode::outpost) {
      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      command = aimer.aim(targets, t, cboard.bullet_speed, q);
      command.shoot = shooter.shoot(command, aimer, targets, ypr);

      if (!targets.empty()) {
        const auto x = targets.front().ekf_x();
        command.horizon_distance = std::sqrt(x[0] * x[0] + x[2] * x[2]);
      }

      cboard.send(command);
    // }

    /// PlotJuggler 调试曲线
    {
      Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0);

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["bullet_speed"] = cboard.bullet_speed;
      data["mode"] = static_cast<int>(mode);

      data["imu_yaw"] = ypr[0];
      data["imu_pitch"] = ypr[1];
      data["imu_roll"] = ypr[2];

      data["armor_count"] = static_cast<int>(armors.size());
      data["target_count"] = static_cast<int>(targets.size());

      // if (mode == io::Mode::auto_aim || mode == io::Mode::outpost) {
        data["cmd_yaw"] = command.yaw;
        data["cmd_pitch"] = command.pitch;
        data["cmd_control"] = command.control ? 1 : 0;
        data["cmd_shoot"] = command.shoot ? 1 : 0;
        data["horizon_distance"] = command.horizon_distance;
      // }

      if (!targets.empty()) {
        const auto & target = targets.front();
        const auto x = target.ekf_x();

        // data["target_x"] = x[0];
        // data["target_vx"] = x[1];
        // data["target_y"] = x[2];
        // data["target_vy"] = x[3];
        // data["target_z"] = x[4];
        // data["target_vz"] = x[5];
        data["w"] = x[7];
        data["v"] = std::sqrt(x[1] * x[1] + x[3] * x[3]);
        data["distance"] = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);

        // data["aim_x"] = aimer.debug_aim_point.xyza[0];
        // data["aim_y"] = aimer.debug_aim_point.xyza[1];
        // data["aim_z"] = aimer.debug_aim_point.xyza[2];
        // data["aim_yaw"] = aimer.debug_aim_point.xyza[3];
      } else {
        data["w"] = 0.0;
        data["v"] = 0.0;
        data["distance"] = 0.0;
      }

      plotter.plot(data);
    }

    // FPS 计算和显示
    static auto last_time = std::chrono::steady_clock::now();
                auto current_time = std::chrono::steady_clock::now();
    double fps = 1.0 / std::chrono::duration<double>(current_time - last_time).count();
    last_time = current_time;
    tools::draw_text(img, fmt::format("FPS: {:.1f}", fps), {10, 30}, {0, 255, 255});
    // tools::logger()->info("{:.2f} fps", fps);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}