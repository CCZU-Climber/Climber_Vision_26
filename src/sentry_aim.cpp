#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/omniperception/armor_selector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;
using auto_aim::Plan;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Detector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  omniperception::ArmorSelector armor_selector(config_path);

  // 从配置读取 t_camera2gimbal 便于后续计算到相机的距离
  auto yaml_node = YAML::LoadFile(config_path);
  auto t_cam2gim = yaml_node["t_camera2gimbal"].as<std::vector<double>>();
  Eigen::Vector3d t_camera2gimbal(t_cam2gim[0], t_cam2gim[1], t_cam2gim[2]);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target_opt = target_queue.front();
      auto gs = gimbal.state();
      
      Plan plan;
      if (target_opt.has_value()) {
        plan = planner.plan(target_opt.value(), gs.bullet_speed);
      } else {
        // 没有目标时返回空计划
        plan = Plan{};
      }

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;
      if (fired) {
        tools::logger()->info(
          "[FIRE] bullet_count={} plan_fire={} target_valid={}", gs.bullet_count, plan.fire,
          target_opt.has_value());
      }

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

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

      // if (target_opt.has_value()) {
      //   data["target_z"] = target_opt->ekf_x()[4];   //z
      //   data["target_vz"] = target_opt->ekf_x()[5];  //vz
      // }

      // if (target_opt.has_value()) {
      //   const auto x = target_opt->ekf_x();
      //   data["w"] = x[7];
      //   data["v"] = std::sqrt(x[1] * x[1] + x[3] * x[3]);
      //   data["distance"] = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
      // } else {
      //   data["w"] = 0.0;
      //   data["v"] = 0.0;
      //   data["distance"] = 0.0;
      // }

      // 世界坐标系下实际选定击打的装甲板到云台中心的距离
      double armor_dist = std::sqrt(
        std::pow(planner.debug_xyza[0], 2) +
        std::pow(planner.debug_xyza[1], 2) +
        std::pow(planner.debug_xyza[2], 2));
      data["distance"] = armor_dist;

      // 计算同一块装甲板到相机的真实物理距离
      Eigen::Vector3d target_world = planner.debug_xyza.head(3);
      // solver 已经实时更新了当前的云台位姿, 借此变换回云台系
      Eigen::Vector3d target_gimbal = solver.R_gimbal2world().transpose() * target_world;
      // 减去相机的平移向量得到在相机系下的坐标，求模长即为真实相距
      data["distance_cam"] = (target_gimbal - t_camera2gimbal).norm();

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    camera.read(img, t);
    auto q = gimbal.q(t);

    // recorder.record(img, q, t);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    // auto armors = detector.detect(img);
    armor_selector.armor_filter(armors, omniperception::ArmorFilterScope::MainAim);
    armor_selector.set_priority(armors);
    auto targets = tracker.track(armors, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    if (!targets.empty()) {
      auto target = targets.front();
      // if (target.is_switch()) {
      //   tools::logger()->info("[SWITCH] id={} w={:.3f}", target.last_id, target.ekf_x()[7]);
      // }
      // static int track_log_count = 0;
      // if (++track_log_count % 120 == 0) {
      //   const auto x = target.ekf_x();
      //   tools::logger()->info(
      //     "[TRACK] id={} switch={} w={:.3f} r={:.3f} l={:.3f}", target.last_id, target.is_switch(),
      //     x[7], x[8], x[9]);
      // }

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸


    
    // FPS 计算和显示
    static auto last_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    double fps = 1.0 / std::chrono::duration<double>(current_time - last_time).count();
    last_time = current_time;
    tools::draw_text(img, fmt::format("FPS: {:.1f}", fps), {10, 30}, {0, 255, 255});

    cv::imshow("reprojection", img);
    recorder.record(img, q, t);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
