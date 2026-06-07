#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
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
#include "tools/frame_source.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{@config-path   | configs/test_buff.yaml   | yaml配置文件路径}"
  "{video v        |                          | 视频文件路径，提供则进入回放模式}"
  "{txt t          |                          | 时间戳+四元数txt文件路径（与video配套）}"
  "{mode m         | small                    | 回放模式下的buff大小：small/big/auto}"
  "{speed          | 1.0                      | 回放倍速，1.0=原速，2.0=两倍速}";

using namespace std::chrono_literals;

bool finite_vec3(const Eigen::Vector3d & v)
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

const cv::Scalar kDbgPurple(190, 0, 190);
const cv::Scalar kDbgGreen(0, 220, 0);
const cv::Scalar kDbgYellow(0, 255, 255);
const cv::Scalar kDbgWhite(255, 255, 255);
const cv::Scalar kDbgGray(180, 180, 180);
constexpr double kBigFaceAngle = 2.0 * CV_PI / 5.0;

io::GimbalState replay_gimbal_state_from_q(
  const Eigen::Quaterniond & q, double bullet_speed,
  const std::optional<std::pair<Eigen::Vector2d, std::chrono::steady_clock::time_point>> & last_pose,
  const std::chrono::steady_clock::time_point & timestamp)
{
  Eigen::Quaterniond q_norm = q;
  if (!q_norm.coeffs().allFinite() || q_norm.norm() < 1e-6) {
    q_norm = Eigen::Quaterniond::Identity();
  } else {
    q_norm.normalize();
  }

  const Eigen::Matrix3d R = q_norm.toRotationMatrix();
  const double yaw = tools::limit_rad(std::atan2(R(1, 0), R(0, 0)));
  const double pitch = tools::limit_rad(std::asin(std::clamp(-R(2, 0), -1.0, 1.0)));

  double yaw_vel = 0.0;
  double pitch_vel = 0.0;
  if (last_pose.has_value()) {
    const double dt = tools::delta_time(timestamp, last_pose->second);
    if (dt > 1e-4) {
      yaw_vel = tools::limit_rad(yaw - last_pose->first[0]) / dt;
      pitch_vel = (pitch - last_pose->first[1]) / dt;
    }
  }

  return {
    static_cast<float>(yaw),
    static_cast<float>(yaw_vel),
    static_cast<float>(pitch),
    static_cast<float>(pitch_vel),
    static_cast<float>(bullet_speed),
    0};
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return -1;
  }

  auto config_path = cli.get<std::string>(0);
  auto video_path = cli.get<std::string>("video");
  auto txt_path = cli.get<std::string>("txt");
  auto mode_str = cli.get<std::string>("mode");
  bool replay_mode = !video_path.empty();
  double replay_speed = cli.get<double>("speed");
  if (replay_speed <= 0.0) replay_speed = 1.0;

  if (replay_mode && mode_str != "small" && mode_str != "big" && mode_str != "auto") {
    tools::logger()->error("无效的mode参数: {}，仅支持 small/big/auto", mode_str);
    return -1;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  // ---- 实时模式：初始化硬件 ----
  std::unique_ptr<io::Gimbal> gimbal;
  std::unique_ptr<std::thread> plan_thread;
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::unique_ptr<auto_aim::YOLO> yolo;
  std::unique_ptr<auto_aim::Solver> solver;
  std::unique_ptr<auto_aim::Tracker> tracker;
  std::unique_ptr<auto_aim::Planner> planner;

  uint16_t last_bullet_count = 0;
  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  if (!replay_mode) {
    gimbal = std::make_unique<io::Gimbal>(config_path);

    yolo = std::make_unique<auto_aim::YOLO>(config_path, true);
    solver = std::make_unique<auto_aim::Solver>(config_path);
    tracker = std::make_unique<auto_aim::Tracker>(config_path, *solver);
    planner = std::make_unique<auto_aim::Planner>(config_path);

    plan_thread = std::make_unique<std::thread>([&]() {
      while (!quit) {
        if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
          auto target = target_queue.front();
          auto gs = gimbal->state();
          auto plan = planner->plan(target, gs.bullet_speed);
          gimbal->send(plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                       plan.pitch, plan.pitch_vel, plan.pitch_acc);
          std::this_thread::sleep_for(10ms);
        } else {
          std::this_thread::sleep_for(200ms);
        }
      }
    });
  } else {
    // 回放模式：设置固定模式
    if (mode_str == "small") mode = io::GimbalMode::SMALL_BUFF;
    else if (mode_str == "big") mode = io::GimbalMode::BIG_BUFF;
    else mode = io::GimbalMode::AUTO_AIM;
  }

  tools::FrameSource::Options frame_options;
  frame_options.config_path = config_path;
  frame_options.video_path = video_path;
  frame_options.txt_path = txt_path;
  frame_options.replay_speed = replay_speed;
  if (!replay_mode) {
    frame_options.live_q_provider = [&](const std::chrono::steady_clock::time_point & timestamp) {
      return gimbal->q(timestamp);
    };
  }
  tools::FrameSource frame_source(std::move(frame_options));
  const int replay_delay_ms = frame_source.replay_delay_ms();

  // ---- 通用组件 ----
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto target_params = auto_buff::load_target_params(config_path);
  auto_buff::SmallTarget buff_small_target(target_params);
  auto_buff::BigTargetGroup buff_big_group(target_params);
  auto_buff::BigTargetSelector buff_big_selector(config_path);
  auto_buff::Aimer buff_aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::optional<std::chrono::steady_clock::time_point> motion_first_t;
  std::ofstream motion_csv;
  if (const char * csv_path = std::getenv("BUFF_MOTION_CSV"); csv_path != nullptr && csv_path[0] != '\0') {
    motion_csv.open(csv_path);
    motion_csv << std::fixed << std::setprecision(9);
    motion_csv
      << "frame,t,slot,blade_id,selected,obs_valid,obs_roll,obs_phase,obs_slot_angle,obs_image_phase,"
      << "ekf_roll,ekf_phase,ekf_spd,pred_phase_010,pred_phase_020,"
      << "pred_phase_040,pred_phase_050,n_det,fire,control\n";
  }

  cv::namedWindow("实时调试", cv::WINDOW_NORMAL);
  auto t0 = std::chrono::steady_clock::now();

  int frame_count = 0;

  bool has_trad = false;
  std::optional<std::pair<Eigen::Vector2d, std::chrono::steady_clock::time_point>> replay_last_pose;

  while (!exiter.exit()) {
    // ---- 模式获取 ----
    if (!replay_mode) {
      mode = gimbal->mode();
      if (last_mode != mode) {
        tools::logger()->info("Switch to {}", gimbal->str(mode));
        if (last_mode == io::GimbalMode::BIG_BUFF && mode != io::GimbalMode::BIG_BUFF) {
          buff_big_selector.reset_cycle();
        }
        last_mode = mode.load();
      }
    }

    // ---- 图像获取 ----
    tools::FramePacket frame;
    if (!frame_source.read(frame)) {
      if (replay_mode) {
        tools::logger()->info("视频播放完毕");
        break;
      }
      std::this_thread::sleep_for(10ms);
      continue;
    }
    img = frame.image;
    t = frame.timestamp;
    if (!motion_first_t.has_value()) motion_first_t = t;
    const auto q = frame.q;

    frame_count++;

    // ---- 处理 ----
    nlohmann::json debug_data;
    std::vector<auto_buff::PowerRune> buff_observations;
    int buff_selected_id = -1;
    std::optional<auto_buff::BigTarget> debug_big_target = std::nullopt;
    std::optional<auto_buff::PowerRune> debug_power_rune = std::nullopt;
    cv::Point2f debug_aim_pixel{-1, -1};
    cv::Point2f debug_aim_pred_pixel{-1, -1};
    cv::Point2f debug_ballistic_pixel{-1, -1};
    bool control_active = false;
    bool shoot_active = false;

    double current_yaw = 0.0;
    double current_pitch = 0.0;
    double bullet_speed = 24.0;
    io::GimbalState gimbal_state = {0, 0, 0, 0, static_cast<float>(bullet_speed), 0};

    if (!replay_mode) {
      gimbal_state = gimbal->state();
      current_yaw = gimbal_state.yaw;
      current_pitch = gimbal_state.pitch;
      bullet_speed = gimbal_state.bullet_speed;
      solver->set_R_gimbal2world(q);
    } else {
      gimbal_state = replay_gimbal_state_from_q(q, bullet_speed, replay_last_pose, t);
      replay_last_pose = {Eigen::Vector2d(gimbal_state.yaw, gimbal_state.pitch), t};
      current_yaw = gimbal_state.yaw;
      current_pitch = gimbal_state.pitch;
      bullet_speed = gimbal_state.bullet_speed;
    }

    io::GimbalMode cur_mode = mode.load();
    const auto fire_control_now = replay_mode ? t : std::chrono::steady_clock::now();

    if (!replay_mode && cur_mode == io::GimbalMode::AUTO_AIM) {
      auto armors = yolo->detect(img);
      auto targets = tracker->track(armors, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    } else if (cur_mode == io::GimbalMode::SMALL_BUFF || cur_mode == io::GimbalMode::BIG_BUFF) {
      buff_solver.set_R_gimbal2world(q);

      auto_aim::Plan buff_plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      if (cur_mode == io::GimbalMode::SMALL_BUFF) {
        buff_observations = buff_detector.detect(img, auto_buff::SMALL);
        has_trad = false;
        if (!buff_observations.empty()) {
          has_trad = true;
        }
        buff_solver.solve(buff_observations);
        std::optional<auto_buff::PowerRune> power_rune =
          buff_observations.empty() ? std::nullopt
                                    : std::optional<auto_buff::PowerRune>(buff_observations.front());
        debug_power_rune = power_rune;

        buff_small_target.get_target(power_rune, t);

        auto &target_ref = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_ref, t,
          {static_cast<float>(current_yaw), 0.0f, static_cast<float>(current_pitch), 0.0f, static_cast<float>(bullet_speed), 0}, true,
          &fire_control_now);
      } else {
        buff_observations = buff_detector.detect(img, auto_buff::BIG);
        buff_solver.solve(buff_observations);
        if (!buff_observations.empty()) {
          debug_power_rune = buff_observations.front();
        }
        buff_big_group.update(buff_observations, t);

        auto selected_target = buff_big_selector.select_target(
          buff_big_group, buff_aimer, t, bullet_speed, current_yaw, current_pitch, fire_control_now);
        buff_selected_id = buff_big_selector.selected_id();
        if (selected_target.has_value()) {
          debug_big_target = selected_target;
          auto & target_ref = selected_target.value();
          buff_plan = buff_aimer.mpc_aim(target_ref, t,
            {static_cast<float>(current_yaw), 0.0f, static_cast<float>(current_pitch), 0.0f, static_cast<float>(bullet_speed), 0}, true,
            &fire_control_now);
          if (buff_plan.fire) {
            buff_big_selector.on_fire(fire_control_now);
          }
        }
      }

      if (!replay_mode) {
        gimbal->send(buff_plan.control, buff_plan.fire, buff_plan.yaw,
                     buff_plan.yaw_vel, buff_plan.yaw_acc,
                     buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
      }

      control_active = buff_plan.control;
      shoot_active = buff_plan.fire;
      const bool fired = !replay_mode && gimbal_state.bullet_count > last_bullet_count;
      if (!replay_mode) last_bullet_count = gimbal_state.bullet_count;

      debug_data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      debug_data["bullet_speed"] = bullet_speed;
      debug_data["control"] = buff_plan.control ? 1 : 0;
      // debug_data["fc_detect_now_gap"] = tools::delta_time(fire_control_now, t);
      debug_data["gimbal_yaw"] = gimbal_state.yaw;
      debug_data["gimbal_yaw_vel"] = gimbal_state.yaw_vel;
      debug_data["gimbal_pitch"] = gimbal_state.pitch;
      debug_data["gimbal_pitch_vel"] = gimbal_state.pitch_vel;
      debug_data["target_yaw"] = buff_plan.target_yaw;
      debug_data["target_pitch"] = buff_plan.target_pitch;
      debug_data["plan_yaw"] = buff_plan.yaw;
      debug_data["plan_yaw_vel"] = buff_plan.yaw_vel;
      debug_data["plan_yaw_acc"] = buff_plan.yaw_acc;
      debug_data["plan_pitch"] = buff_plan.pitch;
      debug_data["plan_pitch_vel"] = buff_plan.pitch_vel;
      debug_data["plan_pitch_acc"] = buff_plan.pitch_acc;
      debug_data["plan_gimbal_yaw_error"] = tools::limit_rad(buff_plan.yaw - gimbal_state.yaw);
      debug_data["plan_gimbal_pitch_error"] = buff_plan.pitch - gimbal_state.pitch;
      debug_data["fire"] = buff_plan.fire ? 1 : 0;
      debug_data["fired"] = fired ? 1 : 0;

      if (motion_csv.is_open() && cur_mode == io::GimbalMode::BIG_BUFF) {
        const double t_rel = motion_first_t.has_value() ? tools::delta_time(t, motion_first_t.value()) : 0.0;
        for (int slot = 0; slot < auto_buff::BigTargetGroup::kSlotCount; ++slot) {
          auto target = buff_big_group.get_target_copy(slot);
          if (!target.has_value() || target->is_unsolve()) continue;
          const auto ekf_x = target->ekf_x();
          if (ekf_x.size() <= 6 || !std::isfinite(ekf_x[5])) continue;

          bool obs_valid = false;
          double obs_roll = 0.0;
          double obs_phase = 0.0;
          double obs_slot_angle = 0.0;
          double obs_image_phase = 0.0;
          for (const auto & obs : buff_observations) {
            if (obs.slot_id != slot) continue;
            obs_valid = true;
            obs_roll = obs.ypr_in_world[2];
            obs_phase = obs_roll + slot * kBigFaceAngle;
            obs_slot_angle = obs.slot_angle;
            obs_image_phase = obs_slot_angle - slot * kBigFaceAngle;
            break;
          }

          auto predict_phase = [&](double dt) {
            auto pred = target.value();
            pred.predict(dt);
            const auto pred_x = pred.ekf_x();
            return pred_x[5] + slot * kBigFaceAngle;
          };

          motion_csv
            << frame_count << ',' << t_rel << ',' << slot << ',' << target->blade_id() << ','
            << (slot == buff_selected_id ? 1 : 0) << ',' << (obs_valid ? 1 : 0) << ','
            << obs_roll << ',' << obs_phase << ',' << obs_slot_angle << ',' << obs_image_phase << ','
            << ekf_x[5] << ',' << (ekf_x[5] + slot * kBigFaceAngle) << ','
            << ekf_x[6] << ','
            << predict_phase(0.10) << ',' << predict_phase(0.20) << ','
            << predict_phase(0.40) << ',' << predict_phase(0.50) << ','
            << buff_observations.size() << ',' << (buff_plan.fire ? 1 : 0) << ','
            << (buff_plan.control ? 1 : 0) << '\n';
        }
      }

      auto_buff::Target * debug_target = nullptr;
      if (cur_mode == io::GimbalMode::SMALL_BUFF && !buff_small_target.is_unsolve()) {
        debug_target = &buff_small_target;
      } else if (debug_big_target.has_value() && !debug_big_target->is_unsolve()) {
        debug_target = &debug_big_target.value();
      }

      Eigen::Vector3d buff_aim_world = Eigen::Vector3d::Zero();
      double buff_aim_d = 0.0;
      bool has_buff_aim_world = false;
      if (debug_target != nullptr) {
        buff_aim_world = debug_target->point_buff2world(
          Eigen::Vector3d(0.0, 0.0, debug_target->params_.target_radius));
        buff_aim_d = std::hypot(buff_aim_world.x(), buff_aim_world.y());
        debug_data["buff_aim_d"] = buff_aim_d;
        debug_data["buff_aim_z"] = buff_aim_world.z();
        debug_data["buff_aim_geo_pitch"] = std::atan2(buff_aim_world.z(), buff_aim_d);
        has_buff_aim_world = true;
        // 当前帧aim像素（不预测）
        debug_aim_pixel = buff_solver.world2pixel(buff_aim_world);
        // 预测aim: save → predict → point_buff2world → restore
        debug_target->save_ekf_state();
        debug_target->predict(0.1);
        Eigen::Vector3d pred_aim_world = debug_target->point_buff2world(
          Eigen::Vector3d(0.0, 0.0, debug_target->params_.target_radius));
        debug_target->restore_ekf_state();
        debug_data["buff_aim_pred_z"] = pred_aim_world.z();
        debug_aim_pred_pixel = buff_solver.world2pixel(pred_aim_world);
        // 弹道补偿点: 在预测aim上方加子弹下坠高度，枪管实际指向这里
        double fly_t = buff_aim_d / bullet_speed;
        double gravity_drop = 0.5 * 9.81 * fly_t * fly_t;
        Eigen::Vector3d ballistic_world = pred_aim_world + Eigen::Vector3d(0, 0, gravity_drop);
        debug_ballistic_pixel = buff_solver.world2pixel(ballistic_world);
      }

      if (debug_power_rune.has_value()) {
        const auto & obs_target = debug_power_rune->target_xyz_in_world;
        const double obs_target_d = std::hypot(obs_target.x(), obs_target.y());
        const double pnp_origin_d = std::hypot(
          debug_power_rune->xyz_in_world.x(), debug_power_rune->xyz_in_world.y());
        const double trad_r_d = std::hypot(
          debug_power_rune->trad_xyz_in_world.x(), debug_power_rune->trad_xyz_in_world.y());
        const cv::Point2f pnp_r_delta = debug_power_rune->pnp_r_center - debug_power_rune->r_center;
        const auto & trad_r_gimbal = debug_power_rune->trad_xyz_in_gimbal;
        const auto & trad_r_world = debug_power_rune->trad_xyz_in_world;
        const Eigen::Matrix3d R_gimbal2world = buff_solver.R_gimbal2world();
        debug_data["buff_obs_target_d"] = obs_target_d;
        // debug_data["buff_obs_target_camera_z"] = debug_power_rune->target_xyz_in_camera.z();
        // debug_data["buff_obs_target_gimbal_x"] = debug_power_rune->target_xyz_in_gimbal.x();
        // debug_data["buff_obs_target_gimbal_y"] = debug_power_rune->target_xyz_in_gimbal.y();
        // debug_data["buff_obs_target_gimbal_z"] = debug_power_rune->target_xyz_in_gimbal.z();
        debug_data["buff_obs_target_z"] = debug_power_rune->target_xyz_in_world.z();
        debug_data["buff_pnp_origin_d"] = pnp_origin_d;
        debug_data["buff_pnp_origin_z"] = debug_power_rune->xyz_in_world.z();
        // debug_data["buff_pnp_r_delta_x"] = pnp_r_delta.x;
        // debug_data["buff_pnp_r_delta_y"] = pnp_r_delta.y;
        // debug_data["buff_pnp_r_error"] = cv::norm(pnp_r_delta);
        debug_data["buff_trad_r_d"] = trad_r_d;
        // debug_data["buff_trad_r_camera_z"] = debug_power_rune->trad_xyz_in_camera.z();
        // debug_data["buff_trad_r_gimbal_x"] = trad_r_gimbal.x();
        // debug_data["buff_trad_r_gimbal_y"] = trad_r_gimbal.y();
        // debug_data["buff_trad_r_gimbal_z"] = trad_r_gimbal.z();
        debug_data["buff_trad_r_world_x"] = trad_r_world.x();
        debug_data["buff_trad_r_world_y"] = trad_r_world.y();
        debug_data["buff_trad_r_z"] = trad_r_world.z();
        // debug_data["buff_trad_r_world_z_from_gimbal_x"] = R_gimbal2world(2, 0) * trad_r_gimbal.x();
        // debug_data["buff_trad_r_world_z_from_gimbal_y"] = R_gimbal2world(2, 1) * trad_r_gimbal.y();
        // debug_data["buff_trad_r_world_z_from_gimbal_z"] = R_gimbal2world(2, 2) * trad_r_gimbal.z();
        debug_data["buff_gimbal2world_z_x"] = R_gimbal2world(2, 0);
        debug_data["buff_gimbal2world_z_y"] = R_gimbal2world(2, 1);
        debug_data["buff_gimbal2world_z_z"] = R_gimbal2world(2, 2);
        // if (has_buff_aim_world) {
        //   debug_data["buff_aim_minus_obs_target_d"] = buff_aim_d - obs_target_d;
        //   debug_data["buff_aim_minus_obs_target_z"] =
        //     buff_aim_world.z() - debug_power_rune->target_xyz_in_world.z();
        // }
      }

    } else if (!replay_mode) {
      gimbal->send(false, false, 0, 0, 0, 0, 0, 0);
    }

    // ---- 可视化 ----
    if (cur_mode == io::GimbalMode::SMALL_BUFF || cur_mode == io::GimbalMode::BIG_BUFF) {
      if (!buff_observations.empty()) {
        for (size_t i = 0; i < buff_observations.size(); ++i) {
          const auto & buff_data = buff_observations[i];
          const bool is_selected_obs =
            cur_mode == io::GimbalMode::BIG_BUFF
              ? buff_data.slot_id == buff_selected_id
              : static_cast<int>(i) == buff_selected_id;
          cv::Scalar color = is_selected_obs ? kDbgWhite : kDbgGreen;
          tools::draw_point(img, buff_data.r_center, kDbgGray, 4);
          tools::draw_point(img, buff_data.target().center, color, 5);
        }
      } else {
        cv::putText(img, "No Buff Detected", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, kDbgGray, 2);
      }

      cv::putText(img, control_active ? "CONTROL: YES" : "CONTROL: NO",
                  cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  control_active ? kDbgGreen : kDbgGray, 2);
      cv::putText(img, shoot_active ? "SHOOT: YES" : "SHOOT: NO",
                  cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  shoot_active ? kDbgGreen : kDbgGray, 2);
      const double shown_yaw = replay_mode ? current_yaw : gimbal_state.yaw;
      const double shown_pitch = replay_mode ? current_pitch : gimbal_state.pitch;
      cv::putText(img, fmt::format("PLAN_YAW: {:.2f} PLAN_PITCH: {:.2f}",
                    shown_yaw * 57.3, shown_pitch * 57.3),
                  cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.8, kDbgYellow, 2);
      if (cur_mode == io::GimbalMode::SMALL_BUFF && !buff_small_target.is_unsolve()) {
        cv::putText(img, fmt::format("BUF_PITCH:{:.2f} BUF_YAW:{:.2f} ROLL:{:.1f}",
                      buff_small_target.buff_pitch() * 57.3, buff_small_target.buff_yaw() * 57.3,
                      buff_small_target.get_roll() * 57.3),
                    cv::Point(20, 210), cv::FONT_HERSHEY_SIMPLEX, 0.8, kDbgGray, 1);
      }


      // 小符专用调试绘制
      if (cur_mode == io::GimbalMode::SMALL_BUFF) {
        if (!buff_observations.empty()) {
          auto & obs = buff_observations.front();
          cv::circle(img, obs.model_r_center, 4, kDbgGreen, -1);
          cv::putText(img, "MODEL_R", obs.model_r_center + cv::Point2f(8, 8),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, kDbgGreen, 1);

          // PNP_R: 由solver填入的PnP几何中心 (OBJECT_POINTS[6] 重投影)
          cv::circle(img, obs.pnp_r_center, 7, kDbgYellow, -1);
          cv::putText(img, "PNP_R", obs.pnp_r_center + cv::Point2f(8, -8),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, kDbgYellow, 1);
          if (has_trad) {
            cv::circle(img, obs.r_center, 6, kDbgWhite, 1);
            cv::putText(img, "TRAD_R", obs.r_center + cv::Point2f(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, kDbgWhite, 1);
          }
        }

        // EKF反投影: 把EKF估计的pose重投影到图上，验证EKF是否跟踪准确
        if (!buff_small_target.is_unsolve()) {
          auto ekf_x = buff_small_target.ekf_x();
          Eigen::Vector3d xyz_w = buff_small_target.center_filtered();
          if (finite_vec3(xyz_w) && std::isfinite(ekf_x[5])) {
            auto ekf_corners = buff_solver.reproject_buff(xyz_w,
              Eigen::Vector3d(buff_small_target.buff_yaw(), buff_small_target.buff_pitch(), ekf_x[5]));
            for (const auto & pt : ekf_corners)
              cv::circle(img, pt, 3, kDbgPurple, -1);
          }
        }

        // 弹道预测点: predict 0.1s后的aim（白色十字）
        if (debug_aim_pred_pixel.x > 0 && debug_aim_pred_pixel.y > 0) {
          cv::drawMarker(img, debug_aim_pred_pixel, kDbgWhite, cv::MARKER_TILTED_CROSS, 16, 2);
          cv::putText(img, "AIM", debug_aim_pred_pixel + cv::Point2f(10, -10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, kDbgWhite, 1);
        }
        // 弹道补偿点: 枪管实际指向（含重力补偿，白色菱形）
        if (debug_ballistic_pixel.x > 0 && debug_ballistic_pixel.y > 0) {
          cv::drawMarker(img, debug_ballistic_pixel, kDbgWhite, cv::MARKER_DIAMOND, 16, 2);
          cv::putText(img, "BAL", debug_ballistic_pixel + cv::Point2f(10, -10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, kDbgWhite, 1);
        }

        {
          const cv::Point org(20, 260);
          const int dy = 22;
          double pd = !buff_observations.empty() ? buff_observations.front().ypd_in_world[2] : -1.0;
          double ed = buff_small_target.is_unsolve() ? -1.0 : buff_small_target.center_filtered().norm();
          cv::putText(img, fmt::format("DIST PnP:{:.2f}m  EKF:{:.2f}m", pd, ed),
                      org, cv::FONT_HERSHEY_SIMPLEX, 0.5, kDbgGray, 1);

          if (!buff_small_target.is_unsolve()) {
            auto ex = buff_small_target.ekf_x();
            cv::putText(img, fmt::format("spd:{:.2f}rad/s roll:{:.1f}deg",
                        ex[6], ex[5] * 180.0 / CV_PI),
                        org + cv::Point(0, dy), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        kDbgGray, 1);
          }
        }
      }

      if (cur_mode == io::GimbalMode::BIG_BUFF && debug_big_target.has_value()) {
        auto & big_target = debug_big_target.value();
        if (!big_target.is_unsolve()) {
          auto ekf_x = big_target.ekf_x();
          Eigen::Vector3d xyz_w = tools::ypd2xyz(Eigen::Vector3d(ekf_x[0], ekf_x[2], ekf_x[3]));
          if (finite_vec3(xyz_w) && std::isfinite(ekf_x[4]) && std::isfinite(ekf_x[5])) {
            auto ekf_corners = buff_solver.reproject_buff(
              xyz_w, Eigen::Vector3d(ekf_x[4], big_target.buff_pitch(), ekf_x[5]));
            for (const auto & pt : ekf_corners)
              cv::circle(img, pt, 3, kDbgPurple, -1);
          }

          cv::putText(img, fmt::format("SEL_SLOT:{} BID:{} ROLL:{:.1f}deg",
                        buff_selected_id, big_target.blade_id(), ekf_x[5] * 57.3),
                      cv::Point(20, 210), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      kDbgYellow, 2);
        }
      }

      // 像素级对比: 5槽位EKF靶心 + 检测靶心 + 选中靶心
      if (cur_mode == io::GimbalMode::BIG_BUFF) {
        int y_off = 0;

        struct PxInfo { cv::Point2f ekf_px; int blade_id; double roll_deg; double spd; bool valid; };
        std::array<PxInfo, auto_buff::BigTargetGroup::kSlotCount> ekf_info = {};

        // 5个槽位候选的靶心像素
        for (int i = 0; i < auto_buff::BigTargetGroup::kSlotCount; ++i) {
          auto tgt = buff_big_group.get_target_copy(i);
          if (!tgt.has_value() || tgt->is_unsolve()) continue;
          auto ekf_x = tgt->ekf_x();
          if (ekf_x.size() < 6) continue;
          // BigTarget stores physical blade roll in ekf_x[5]; blade_index is only unwrap metadata.
          double blade_roll = ekf_x[5];
          Eigen::Vector3d ekf_xyz = tools::ypd2xyz(Eigen::Vector3d(ekf_x[0], ekf_x[2], ekf_x[3]));
          cv::Point2f ekf_px;
          if (finite_vec3(ekf_xyz) && std::isfinite(blade_roll)) {
            auto corner_pts = buff_solver.reproject_buff(
                ekf_xyz, Eigen::Vector3d(ekf_x[4], 0.0, blade_roll));
            // 靶心是第5个点 (object_points_[4], index 4)
            if (corner_pts.size() > 4) ekf_px = corner_pts[4];
          }
          ekf_info[i] = {ekf_px, tgt->blade_id(), ekf_x[5]*57.3, ekf_x[6], ekf_px.x > 0};
          if (ekf_px.x > 0) {
            cv::drawMarker(img, ekf_px, kDbgPurple, cv::MARKER_CROSS, 20, 2);
            cv::putText(img, fmt::format("EKF S{} BID{}", i, tgt->blade_id()),
                        ekf_px + cv::Point2f(12, 12 + y_off),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, kDbgPurple, 2);
          }
          // EKF各叶片角
          cv::putText(img, fmt::format("EKF S{} BID{} roll={:.1f} spd={:.2f}",
                        i, tgt->blade_id(), ekf_x[5]*57.3, ekf_x[6]),
                      cv::Point(20, 250 + i*18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      kDbgPurple, 2);
        }

        // 检测靶心像素 (每路观测)
        for (size_t i = 0; i < buff_observations.size(); ++i) {
          const auto & obs = buff_observations[i];
          cv::Point2f c = obs.target().center;
          const int det_blade_id = buff_big_group.blade_id_for_slot(obs.slot_id);
          cv::Scalar det_color = (obs.slot_id == buff_selected_id) ? kDbgWhite : kDbgGreen;
          cv::drawMarker(img, c, det_color, cv::MARKER_TILTED_CROSS, 16, 2);
          cv::putText(img, fmt::format(
                        "DET{} S{} BID{} roll={:.1f}",
                        i, obs.slot_id, det_blade_id, obs.ypr_in_world[2]*57.3),
                      c + cv::Point2f(-10, -15),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, det_color, 2);
        }

        // 选中指示
        if (buff_selected_id >= 0) {
          for (const auto & obs : buff_observations) {
            if (obs.slot_id != buff_selected_id) continue;
            auto sel_center = obs.target().center;
            const int sel_blade_id = buff_big_group.blade_id_for_slot(obs.slot_id);
            cv::drawMarker(img, sel_center, kDbgWhite, cv::MARKER_STAR, 24, 2);
            cv::putText(img, fmt::format("SEL BID{}", sel_blade_id), sel_center + cv::Point2f(-30, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, kDbgWhite, 2);
          }
        }

        // 每30帧打印像素对比到控制台
        if (frame_count % 30 == 1) {
          tools::logger()->warn(
            "PXCMP f={} | SEL={} nDet={}", frame_count, buff_selected_id, buff_observations.size());
          for (int i = 0; i < auto_buff::BigTargetGroup::kSlotCount; ++i) {
            const auto & e = ekf_info[i];
            if (!e.valid) continue;
            tools::logger()->warn(
              "  EKF{}: px=({:.0f},{:.0f}) blade_id={} roll={:.1f}deg spd={:.2f}",
              i, e.ekf_px.x, e.ekf_px.y, e.blade_id, e.roll_deg, e.spd);
          }
          for (size_t i = 0; i < buff_observations.size(); ++i) {
            auto tc = buff_observations[i].target().center;
            const int det_blade_id = buff_big_group.blade_id_for_slot(buff_observations[i].slot_id);
            tools::logger()->warn(
              "  DET{}: center=({:.0f},{:.0f}) slot={} blade_id={} roll={:.1f}deg",
              i, tc.x, tc.y, buff_observations[i].slot_id, det_blade_id,
              buff_observations[i].ypr_in_world[2]*57.3);
          }
        }
      }
    }

    std::string mode_label = replay_mode ? mode_str : gimbal->str(cur_mode);
    cv::putText(img, fmt::format("MODE: {}  frame:{}", mode_label, frame_count),
                cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, kDbgGreen, 2);

    plotter.plot(debug_data);
    // cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("实时调试", img);

    const int key = cv::waitKey(replay_mode ? replay_delay_ms : 1);
    if (key == 113 || key == 27) {
      break;
    } else if (key == 32) {
      cv::putText(img, "Paused", cv::Point(img.cols / 2 - 50, img.rows / 2),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, kDbgGray, 3);
      cv::imshow("实时调试", img);
      while (true) {
        const int pause_key = cv::waitKey(30);
        if (pause_key == 32) break;
        else if (pause_key == 113 || pause_key == 27) {
          quit = true;
          goto exit_loop;
        }
      }
    }
  }


exit_loop:
  quit = true;
  if (!replay_mode && plan_thread && plan_thread->joinable()) plan_thread->join();
  if (!replay_mode && gimbal) gimbal->send(false, false, 0, 0, 0, 0, 0, 0);
  return 0;
}
