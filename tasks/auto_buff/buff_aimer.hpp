#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true,
    const std::chrono::steady_clock::time_point * now_override = nullptr);

  // for mpc
  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true,
    const std::chrono::steady_clock::time_point * now_override = nullptr);

  bool preview(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    double & yaw, double & pitch, bool to_now = true,
    const std::chrono::steady_clock::time_point * now_override = nullptr) const;

  double angle;
  double t_gap = 0;

private:
  SmallTarget target_;
  double yaw_offset_;
  double pitch_offset_;

  double fire_gap_time_;
  double predict_time_;

  int mistake_count_ = 0;
  bool switch_fanblade_ = false;

  double last_yaw_ = 0;
  double last_pitch_ = 0;

  bool first_in_aimer_ = true;

  std::chrono::steady_clock::time_point last_fire_t_;

  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch);

  double air_resistance_k_ = 0.0;  // 线性阻力系数

  bool calc_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    double & yaw, double & pitch) const;
};
}  // namespace auto_buff
#endif
