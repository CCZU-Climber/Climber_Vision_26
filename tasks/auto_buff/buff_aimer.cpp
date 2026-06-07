#include "buff_aimer.hpp"

#include "tools/air_resist_trajectory.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{
Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();
  if (yaml["air_resistance_k"])
    air_resistance_k_ = yaml["air_resistance_k"].as<double>();

  last_fire_t_ = std::chrono::steady_clock::now();
}

io::Command Aimer::aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp,
  double bullet_speed, bool to_now,
  const std::chrono::steady_clock::time_point * now_override)
{
  io::Command command = {false, false, 0, 0};
  command.yaw = last_yaw_;
  command.pitch = -last_pitch_;
  if (target.is_unsolve()) return command;

  if (bullet_speed < 10 || bullet_speed > 28) bullet_speed = 24;

  auto now = (now_override != nullptr) ? *now_override : std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw = 0.0;
  double pitch = 0.0;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    command.yaw = yaw;
    command.pitch = -pitch;
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      // 切板时保持连续控制，只抑制开火，避免电控出现control中断。
      command.control = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;
  }

  if (!command.control) return command;

  if (switch_fanblade_) {
    command.shoot = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  // 短时预测帧：强制禁止射击，但保留 control 角度
  if (target.is_predicted()) {
    command.shoot = false;
  }

  return command;
}

auto_aim::Plan Aimer::mpc_aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now,
  const std::chrono::steady_clock::time_point * now_override)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  plan.yaw = static_cast<float>(last_yaw_);
  plan.pitch = static_cast<float>(-last_pitch_);
  plan.target_yaw = plan.yaw;
  plan.target_pitch = plan.pitch;
  if (target.is_unsolve()) {
    first_in_aimer_ = true;
    return plan;
  }

  double bullet_speed = gs.bullet_speed;
  if (bullet_speed < 10 || bullet_speed > 28) bullet_speed = 24;

  auto now = (now_override != nullptr) ? *now_override : std::chrono::steady_clock::now();

  // 考虑detector所消耗的时间，此外假设aimer的用时可忽略不计
  auto detect_now_gap = tools::delta_time(now, timestamp);
  // 如果 to_now 为 true，则根据当前时间和时间戳预测目标位置
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw = 0.0;
  double pitch = 0.0;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    plan.yaw = yaw;
    plan.pitch = -pitch;
    plan.target_yaw = plan.yaw;
    plan.target_pitch = plan.pitch;
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      // 切板时保持连续控制，只抑制开火，避免电控出现control中断。
      plan.control = true;
      first_in_aimer_ = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;

    if (plan.control) {
      if (first_in_aimer_) {
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
        first_in_aimer_ = false;
      } else {
        auto dt = predict_time_;
        double last_yaw_mpc = 0.0;
        double last_pitch_mpc = 0.0;
        get_send_angle(target, predict_time_ * -1, bullet_speed, to_now, last_yaw_mpc, last_pitch_mpc);
        plan.yaw_vel = tools::limit_rad(yaw - last_yaw_mpc) / (2 * dt);
        plan.yaw_acc = (tools::limit_rad(yaw - gs.yaw) - tools::limit_rad(gs.yaw - last_yaw_mpc)) /
                       std::pow(dt, 2);

        plan.pitch_vel = tools::limit_rad(-pitch + last_pitch_mpc) / (2 * dt);
        plan.pitch_acc = (-pitch - gs.pitch - (gs.pitch + last_pitch_mpc)) / std::pow(dt, 2);
        // plan.yaw_vel = tools::limit_min_max(plan.yaw_vel, -6.28, 6.28);
        // plan.yaw_acc = tools::limit_min_max(plan.yaw_acc, -50, 50);
        // plan.pitch_vel = tools::limit_min_max(plan.pitch_vel, -6.28, 6.28);
        // plan.pitch_acc = tools::limit_min_max(plan.pitch_acc, -100, 100);
      }
    }
  }

  if (!plan.control) {
    first_in_aimer_ = true;
    return plan;
  }

  if (switch_fanblade_) {
    plan.fire = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  // 短时预测帧：强制禁止射击，但保留 control 角度
  if (target.is_predicted()) {
    plan.fire = false;
  }

  return plan;
}

bool Aimer::preview(
  Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
  double & yaw, double & pitch, bool to_now,
  const std::chrono::steady_clock::time_point * now_override) const
{
  if (target.is_unsolve()) return false;
  if (bullet_speed < 10 || bullet_speed > 28) bullet_speed = 24;

  auto now = (now_override != nullptr) ? *now_override : std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  if (!calc_send_angle(target, future, bullet_speed, yaw, pitch)) return false;

  // preview is used by BigTargetSelector against gimbal pitch, so return command semantics.
  pitch = -pitch;
  return true;
}

bool Aimer::calc_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  double & yaw, double & pitch) const
{
  // 根据给定预测时间先走一次目标状态预测
  target.predict(predict_time);

  // 计算目标点的空间坐标
  const Eigen::Vector3d aim_point_in_buff(0.0, 0.0, target.params_.target_radius);
  auto aim_in_world = target.point_buff2world(aim_point_in_buff);
  double d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  double h = aim_in_world[2];

  auto make_traj = [&](double bs, double dist, double height) {
    return tools::AirResistTrajectory(bs, dist, height, air_resistance_k_);
  };

  // 创建弹道对象
  auto trajectory0 = make_traj(bullet_speed, d, h);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: bs={:.1f} d={:.2f} h={:.2f}", bullet_speed, d, h);
    return false;
  }

  // 根据第一个弹道飞行时间再次预测目标位置
  target.predict(trajectory0.fly_time);

  // 计算新的目标点的空间坐标
  aim_in_world = target.point_buff2world(aim_point_in_buff);
  d = fsqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  h = aim_in_world[2];
  auto trajectory1 = make_traj(bullet_speed, d, h);
  if (trajectory1.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory1: bs={:.1f} d={:.2f} h={:.2f}", bullet_speed, d, h);
    return false;
  }

  // 计算时间误差
  auto time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.03) {
    tools::logger()->debug("[Aimer] Large time error: {:.3f} bs={:.1f} d={:.2f}h={:.2f}", time_error, bullet_speed, d, h);
    return false;
  }

  // 计算偏航角和俯仰角，并返回命中结果
  yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
  pitch = trajectory1.pitch + pitch_offset_;
  return true;
}

bool Aimer::get_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  const bool to_now, double & yaw, double & pitch)
{
  (void)to_now;

  // calc_send_angle 会多次调用 target.predict()；这些预测仅用于弹道迭代，
  // 不能污染主EKF状态，否则会在下一帧表现为相位漂移。
  target.save_ekf_state();

  bool ok = calc_send_angle(target, predict_time, bullet_speed, yaw, pitch);
  if (ok) angle = target.get_roll();

  target.restore_ekf_state();

  return ok;
}

}  // namespace auto_buff
