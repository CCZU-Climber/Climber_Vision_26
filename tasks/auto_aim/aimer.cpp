#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/air_resist_trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  outpost_pitch_offset_ =
	    yaml["outpost_pitch_offset"] ? yaml["outpost_pitch_offset"].as<double>() / 57.3 : 0.0;
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  air_resistance_k_ =
    yaml["air_resistance_k"] ? yaml["air_resistance_k"].as<double>() : 0.0;
  R_gimbal2imubody_ =
    Eigen::Map<const Eigen::Matrix3d>(yaml["R_gimbal2imubody"].as<std::vector<double>>().data());
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  const Eigen::Quaterniond & q, bool to_now)
{
  (void)q;
  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front();

  const double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (bullet_speed < 14) bullet_speed = 23;

  auto future = timestamp;
  if (to_now) {
    const double dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(static_cast<int>(dt * 1e6));
    target.predict(future);
  } else {
    const double dt = 0.005 + delay_time;
    future += std::chrono::microseconds(static_cast<int>(dt * 1e6));
    target.predict(future);
  }

  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  const Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  const auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::AirResistTrajectory trajectory0(bullet_speed, d0, xyz0[2], air_resistance_k_);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  double prev_fly_time = trajectory0.fly_time;
  tools::AirResistTrajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);

  for (int iter = 0; iter < 10; ++iter) {
    const auto predict_time =
      future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    const Eigen::Vector3d xyz = aim_point.xyza.head(3);
    const double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::AirResistTrajectory(bullet_speed, d, xyz.z(), air_resistance_k_);

    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) break;
    prev_fly_time = current_traj.fly_time;
  }

  const Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  const double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);
  if (target.name == ArmorName::outpost) pitch -= outpost_pitch_offset_;

  return {true, false, yaw, pitch};
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  const Eigen::VectorXd ekf_x = target.ekf_x();
  const std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  const int armor_num = static_cast<int>(armor_xyza_list.size());

  if (armor_xyza_list.empty()) return {false, Eigen::Vector4d::Zero()};

  // 前哨站: 直接选水平距离最近的装甲板（对齐 Planner 逻辑）
  if (target.name == ArmorName::outpost) {
    double min_dist = std::numeric_limits<double>::max();
    Eigen::Vector4d best_xyza = armor_xyza_list[0];
    for (const auto & xyza : armor_xyza_list) {
      double dist = xyza.head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        best_xyza = xyza;
      }
    }
    return {true, best_xyza};
  }

  if (!target.jumped) {
    if (target.name == ArmorName::outpost && target.has_primary_armor_xyza()) {
      return {true, target.primary_armor_xyza()};
    }
    // 旋转目标（|w|>=1.0 rad/s）：选离中心 yaw 最近的板，避免打到正在转出视野的板
    if (std::abs(ekf_x[7]) >= 1.0) {
      const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
      int best_i = 0;
      double best_delta = std::numeric_limits<double>::max();
      for (int i = 0; i < armor_num; i++) {
        const auto delta = std::abs(tools::limit_rad(armor_xyza_list[i][3] - center_yaw));
        if (delta < best_delta) {
          best_delta = delta;
          best_i = i;
        }
      }
      return {true, armor_xyza_list[best_i]};
    }
    return {true, armor_xyza_list[0]};
  }

  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  std::vector<double> delta_angle_list;
  delta_angle_list.reserve(armor_num);
  for (int i = 0; i < armor_num; i++) {
    const auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    if (id_list.size() > 1) {
      const int id0 = id_list[0];
      const int id1 = id_list[1];
      if (lock_id_ != id0 && lock_id_ != id1) {
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
      }
      return {true, armor_xyza_list[lock_id_]};
    }

    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  double coming_angle = comming_angle_;
  double leaving_angle = leaving_angle_;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  }

  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
