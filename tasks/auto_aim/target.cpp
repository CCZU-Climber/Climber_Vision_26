#include "target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr int kOutpostSlots = 3;
constexpr double kOutpostYawSigma = 0.20;
constexpr double kOutpostPitchSigma = 0.20;
constexpr double kOutpostDistanceSigma = 0.35;
constexpr double kOutpostAngleSigma = 0.35;
constexpr double kOutpostZSigma = 0.08;
constexpr double kOutpostYawGate = 1.0;
constexpr double kOutpostPitchGate = 0.9;
constexpr double kOutpostDistanceGate = 2.0;
constexpr double kOutpostAngleGate = 1.0;
constexpr double kOutpostZGate = 0.35;
constexpr double kOutpostReverseDirectionPenalty = 2.5;
constexpr double kOutpostMaxPairCost = 18.0;
constexpr double kOutpostFastReanchorCost = 8.0;
constexpr int kOutpostFastReanchorFrames = 2;

struct OutpostAssignment
{
  bool valid = false;
  int obs_count = 0;
  double total_cost = std::numeric_limits<double>::infinity();
  std::array<int, kOutpostSlots> obs_to_slot{{-1, -1, -1}};
  std::array<double, kOutpostSlots> obs_cost{{0.0, 0.0, 0.0}};
};

struct OutpostPrimaryDecision
{
  int slot = -1;
  int rule = 0;
};

double outpost_match_cost(const Armor & armor, const Eigen::Vector4d & slot_xyza)
{
  const Eigen::Vector3d slot_ypd = tools::xyz2ypd(slot_xyza.head(3));
  const double yaw_error = std::abs(tools::limit_rad(armor.ypd_in_world[0] - slot_ypd[0]));
  const double pitch_error = std::abs(tools::limit_rad(armor.ypd_in_world[1] - slot_ypd[1]));
  const double distance_error = std::abs(armor.ypd_in_world[2] - slot_ypd[2]);
  const double angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - slot_xyza[3]));
  const double z_error = std::abs(armor.xyz_in_world[2] - slot_xyza[2]);

  if (yaw_error > kOutpostYawGate || pitch_error > kOutpostPitchGate ||
      distance_error > kOutpostDistanceGate || angle_error > kOutpostAngleGate ||
      z_error > kOutpostZGate)
  {
    return std::numeric_limits<double>::infinity();
  }

  double cost = yaw_error / kOutpostYawSigma + pitch_error / kOutpostPitchSigma +
                distance_error / kOutpostDistanceSigma + angle_error / kOutpostAngleSigma +
                z_error / kOutpostZSigma;

  return cost;
}

OutpostAssignment find_best_outpost_assignment(
  const std::vector<Armor> & armors, const std::vector<Eigen::Vector4d> & slot_xyza,
  double angular_velocity, int primary_slot, bool has_primary_slot)
{
  OutpostAssignment result;
  result.obs_count = std::min(static_cast<int>(armors.size()), kOutpostSlots);
  if (result.obs_count <= 0 || static_cast<int>(slot_xyza.size()) < kOutpostSlots) return result;

  std::array<std::array<double, kOutpostSlots>, kOutpostSlots> cost_matrix{};
  for (int i = 0; i < result.obs_count; i++)
  {
    for (int slot = 0; slot < kOutpostSlots; slot++)
    {
      double cost = outpost_match_cost(armors[i], slot_xyza[slot]);
      if (has_primary_slot && std::abs(angular_velocity) > 0.4 && slot != primary_slot)
      {
        if (angular_velocity > 0)
        {
          const int forward_steps = (slot - primary_slot + kOutpostSlots) % kOutpostSlots;
          if (forward_steps == 2) cost += kOutpostReverseDirectionPenalty;
        }
        else
        {
          const int backward_steps = (primary_slot - slot + kOutpostSlots) % kOutpostSlots;
          if (backward_steps == 2) cost += kOutpostReverseDirectionPenalty;
        }
      }
      if (cost > kOutpostMaxPairCost) cost = std::numeric_limits<double>::infinity();
      cost_matrix[i][slot] = cost;
    }
  }

  std::array<int, kOutpostSlots> current_slots{{-1, -1, -1}};
  std::array<double, kOutpostSlots> current_cost{{0.0, 0.0, 0.0}};

  std::function<void(int, int, double)> dfs = [&](int obs_idx, int used_slots, double total_cost)
  {
    if (obs_idx >= result.obs_count)
    {
      if (total_cost < result.total_cost)
      {
        result.valid = true;
        result.total_cost = total_cost;
        result.obs_to_slot = current_slots;
        result.obs_cost = current_cost;
      }
      return;
    }

    for (int slot = 0; slot < kOutpostSlots; slot++)
    {
      if ((used_slots & (1 << slot)) != 0) continue;
      const double pair_cost = cost_matrix[obs_idx][slot];
      if (!std::isfinite(pair_cost)) continue;
      const double next_total = total_cost + pair_cost;
      if (next_total >= result.total_cost) continue;

      current_slots[obs_idx] = slot;
      current_cost[obs_idx] = pair_cost;
      dfs(obs_idx + 1, used_slots | (1 << slot), next_total);
      current_slots[obs_idx] = -1;
      current_cost[obs_idx] = 0.0;
    }
  };

  dfs(0, 0, 0.0);
  return result;
}

int find_observation_index_for_slot(const OutpostAssignment & assignment, int slot)
{
  for (int i = 0; i < assignment.obs_count; i++)
  {
    if (assignment.obs_to_slot[i] == slot) return i;
  }
  return -1;
}

int find_min_cost_observation_index(const OutpostAssignment & assignment)
{
  int best_obs = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < assignment.obs_count; i++)
  {
    if (std::isfinite(assignment.obs_cost[i]) && assignment.obs_cost[i] < best_cost)
    {
      best_cost = assignment.obs_cost[i];
      best_obs = i;
    }
  }
  return best_obs;
}

int find_min_cost_slot(const OutpostAssignment & assignment)
{
  const int best_obs = find_min_cost_observation_index(assignment);
  return best_obs >= 0 ? assignment.obs_to_slot[best_obs] : -1;
}

OutpostPrimaryDecision choose_outpost_primary_slot(
  const OutpostAssignment & assignment, double angular_velocity, bool has_primary_slot,
  int current_primary_slot)
{
  OutpostPrimaryDecision decision;

  // Rule 1: initialize primary only when observation count is at least 2.
  if (!has_primary_slot)
  {
    if (assignment.obs_count < 2)
    {
      decision.rule = 0; // no_history_obs_lt2
      return decision;
    }

    decision.slot = find_min_cost_slot(assignment);
    if (decision.slot >= 0)
    {
      decision.rule = 1; // init_min_cost
    }
    else
    {
      decision.rule = 7; // init_failed
    }
    return decision;
  }

  const int current_slot = (current_primary_slot % kOutpostSlots + kOutpostSlots) % kOutpostSlots;
  decision.slot = current_slot;

  // Rule 2: with history and single observation, never switch.
  if (assignment.obs_count < 2)
  {
    decision.rule = 2; // keep_prev_obs_lt2
    return decision;
  }

  // Rule 3: keep old primary when it is still observed.
  if (find_observation_index_for_slot(assignment, current_slot) >= 0)
  {
    decision.rule = 3; // keep_prev_visible
    return decision;
  }

  // Rule 4: if angular velocity direction is uncertain, keep old primary.
  if (std::abs(angular_velocity) <= 0.4)
  {
    decision.rule = 4; // keep_prev_low_w
    return decision;
  }

  // Rule 5: only allow one-step switch along angular velocity direction.
  const int preferred_slot =
    (current_slot + (angular_velocity > 0 ? 1 : -1) + kOutpostSlots) % kOutpostSlots;
  if (find_observation_index_for_slot(assignment, preferred_slot) >= 0)
  {
    decision.slot = preferred_slot;
    decision.rule = 5; // switch_one_step
    return decision;
  }

  decision.rule = 6; // keep_prev_step_not_visible
  return decision;
}
} // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  priority(armor.priority),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(false),
  t_(t)
{
  const auto r = radius;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const auto center_x = xyz[0] + r * std::cos(ypr[0]);
  const auto center_y = xyz[1] + r * std::sin(ypr[0]);
  const auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  if (name == ArmorName::outpost && armor_num_ == 3)
  {
    // For outpost: x4 is middle plate height baseline, x9/x10 are low/high offsets.
    x0[9] = -0.10;
    x0[10] = 0.10;
  }
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  const auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

/**
 * @brief 执行 EKF 预测步骤，根据时间差推算目标的新状态
 * @param dt 距离上一次更新的经过时间
 */
void Target::predict(double dt)
{
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  double v1 = 100;
  double v2 = 200;
  if (name == ArmorName::outpost)
  {
    v1 = 10;
    v2 = 100;
  }

  const auto a = dt * dt * dt * dt / 4;
  const auto b = dt * dt * dt / 2;
  const auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // Outpost angular velocity clamp after convergence.
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
  {
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  if (name == ArmorName::outpost && armor_num_ == 3)
  {
    update_outpost(std::vector<Armor>{armor});
    return;
  }

  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  const int candidate_count = std::min(3, static_cast<int>(xyza_i_list.size()));
  for (int i = 0; i < candidate_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                             std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update_outpost(const std::vector<Armor> & armors)
{
  if (name != ArmorName::outpost || armor_num_ != 3)
  {
    if (!armors.empty()) update(armors.front());
    return;
  }

  ekf_.data["outpost_assignment_valid"] = 0.0;
  ekf_.data["outpost_fast_reanchor_cost"] = std::numeric_limits<double>::quiet_NaN();
  ekf_.data["outpost_fast_reanchor_streak"] = static_cast<double>(outpost_mismatch_streak_);
  ekf_.data["outpost_fast_reanchor_triggered"] = 0.0;

  if (armors.empty())
  {
    outpost_mismatch_streak_ = 0;
    ekf_.data["outpost_fast_reanchor_streak"] = 0.0;
    return;
  }

  const int obs_count = std::min(static_cast<int>(armors.size()), kOutpostSlots);
  ekf_.data["outpost_obs_count"] = static_cast<double>(obs_count);
  std::vector<Armor> observations;
  observations.reserve(obs_count);
  for (int i = 0; i < obs_count; i++)
  {
    observations.push_back(armors[i]);
  }

  const std::vector<Eigen::Vector4d> slots_xyza = armor_xyza_list();
  const auto assignment = find_best_outpost_assignment(
    observations, slots_xyza, ekf_.x[7], outpost_primary_slot_, has_outpost_primary_slot_);

  ekf_.data["outpost_assignment_cost"] = assignment.total_cost;
  ekf_.data["outpost_fast_reanchor_cost"] = assignment.total_cost;
  if (!assignment.valid)
  {
    outpost_mismatch_streak_ = 0;
    ekf_.data["outpost_fast_reanchor_streak"] = 0.0;
    // tools::logger()->debug(
    //   "[OutpostAssoc] invalid obs={} cost={:.3f} w={:.3f}", obs_count, assignment.total_cost,
    //   ekf_.x[7]);
    return;
  }

  const int observation_dim = obs_count * 4;
  const int measurement_dim = observation_dim;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(measurement_dim);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(measurement_dim, ekf_.x.size());
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
  std::array<int, kOutpostSlots> segment_slots{{0, 0, 0}};

  for (int i = 0; i < obs_count; i++)
  {
    const Armor & armor = observations[i];
    const int slot = assignment.obs_to_slot[i];
    segment_slots[i] = slot;

    const int row = 4 * i;
    z.segment<4>(row) = Eigen::Vector4d{
      armor.ypd_in_world[0], armor.ypd_in_world[1], armor.ypd_in_world[2], armor.ypr_in_world[0]};
    H.block(row, 0, 4, ekf_.x.size()) = h_jacobian(ekf_.x, slot);

    const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
    Eigen::VectorXd R_dig{{4e-3,
                           4e-3,
                           std::log(std::abs(delta_angle) + 1) + 1,
                           std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
    R.block(row, row, 4, 4) = R_dig.asDiagonal();
  }

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd hx = Eigen::VectorXd::Zero(measurement_dim);
    for (int i = 0; i < obs_count; i++)
    {
      const int slot = segment_slots[i];
      const int row = 4 * i;
      const Eigen::Vector3d xyz = h_armor_xyz(x, slot);
      const Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
      const auto angle = tools::limit_rad(x[6] + slot * 2 * CV_PI / armor_num_);
      hx.segment<4>(row) = Eigen::Vector4d{ypd[0], ypd[1], ypd[2], angle};
    }
    return hx;
  };

  auto z_subtract = [observation_dim](
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    const int segment_count = observation_dim / 4;
    for (int i = 0; i < segment_count; i++)
    {
      const int row = 4 * i;
      c[row + 0] = tools::limit_rad(c[row + 0]);
      c[row + 1] = tools::limit_rad(c[row + 1]);
      c[row + 3] = tools::limit_rad(c[row + 3]);
    }
    return c;
  };

  const Eigen::VectorXd residual_before = z_subtract(z, h(ekf_.x));
  std::array<double, kOutpostSlots> z_residuals{
    {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
     std::numeric_limits<double>::quiet_NaN()}};
  std::array<double, kOutpostSlots> distance_residuals{
    {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
     std::numeric_limits<double>::quiet_NaN()}};
  for (int i = 0; i < obs_count; i++)
  {
    const int slot = segment_slots[i];
    const Eigen::Vector3d predicted_xyz = h_armor_xyz(ekf_.x, slot);
    z_residuals[i] = observations[i].xyz_in_world[2] - predicted_xyz[2];
    distance_residuals[i] = residual_before[4 * i + 2];
  }

  const int old_primary_slot = has_outpost_primary_slot_ ? outpost_primary_slot_ : -1;
  const bool primary_visible =
    has_outpost_primary_slot_ &&
    find_observation_index_for_slot(assignment, outpost_primary_slot_) >= 0;
  const bool fast_reanchor_candidate =
    has_outpost_primary_slot_ && obs_count >= 2 && !primary_visible &&
    std::isfinite(assignment.total_cost) && assignment.total_cost >= kOutpostFastReanchorCost;
  if (fast_reanchor_candidate)
  {
    outpost_mismatch_streak_ =
      std::min(outpost_mismatch_streak_ + 1, kOutpostFastReanchorFrames);
  }
  else
  {
    outpost_mismatch_streak_ = 0;
  }
  int fast_reanchor_streak_for_debug = outpost_mismatch_streak_;
  bool fast_reanchor_triggered = false;

  auto primary_decision = choose_outpost_primary_slot(
    assignment, ekf_.x[7], has_outpost_primary_slot_, outpost_primary_slot_);
  if (fast_reanchor_candidate && outpost_mismatch_streak_ >= kOutpostFastReanchorFrames)
  {
    const int reanchor_slot = find_min_cost_slot(assignment);
    if (reanchor_slot >= 0)
    {
      primary_decision.slot = reanchor_slot;
      primary_decision.rule = 8; // fast_reanchor_min_cost
      outpost_mismatch_streak_ = 0;
      fast_reanchor_triggered = true;
    }
  }

  ekf_.update(z, H, R, h, z_subtract);

  if (primary_decision.slot >= 0)
  {
    if (has_outpost_primary_slot_ && primary_decision.slot != outpost_primary_slot_)
    {
      is_switch_ = true;
      switch_count_++;
      jumped = true;
    }
    else
    {
      is_switch_ = false;
    }
    outpost_primary_slot_ = primary_decision.slot;
    has_outpost_primary_slot_ = true;
    last_id = primary_decision.slot;
  }
  else
  {
    is_switch_ = false;
  }

  update_count_++;

  ekf_.data["outpost_assignment_valid"] = 1.0;
  ekf_.data["outpost_primary_rule"] = static_cast<double>(primary_decision.rule);
  ekf_.data["outpost_primary_slot"] =
    has_outpost_primary_slot_ ? static_cast<double>(outpost_primary_slot_) : -1.0;
  ekf_.data["outpost_primary_slot_prev"] = static_cast<double>(old_primary_slot);
  ekf_.data["outpost_assignment_slot0"] = static_cast<double>(assignment.obs_to_slot[0]);
  ekf_.data["outpost_assignment_slot1"] = static_cast<double>(assignment.obs_to_slot[1]);
  ekf_.data["outpost_assignment_slot2"] = static_cast<double>(assignment.obs_to_slot[2]);
  ekf_.data["outpost_z_residual0"] = z_residuals[0];
  ekf_.data["outpost_z_residual1"] = z_residuals[1];
  ekf_.data["outpost_z_residual2"] = z_residuals[2];
  ekf_.data["outpost_distance_residual0"] = distance_residuals[0];
  ekf_.data["outpost_distance_residual1"] = distance_residuals[1];
  ekf_.data["outpost_distance_residual2"] = distance_residuals[2];
  ekf_.data["outpost_fast_reanchor_streak"] = static_cast<double>(fast_reanchor_streak_for_debug);
  ekf_.data["outpost_fast_reanchor_triggered"] = fast_reanchor_triggered ? 1.0 : 0.0;

  // tools::logger()->debug(
  //   "[OutpostAssoc] obs={} slot=[{},{},{}] cost={:.3f} z=[{:.3f},{:.3f},{:.3f}] "
  //   "d=[{:.3f},{:.3f},{:.3f}] primary={}=>{} rule={} switch={} jumped={} "
  //   "fast=[{}/{}] w={:.3f}",
  //   obs_count, assignment.obs_to_slot[0], assignment.obs_to_slot[1], assignment.obs_to_slot[2],
  //   assignment.total_cost, z_residuals[0], z_residuals[1], z_residuals[2],
  //   distance_residuals[0], distance_residuals[1], distance_residuals[2], old_primary_slot,
  //   has_outpost_primary_slot_ ? outpost_primary_slot_ : -1, primary_decision.rule,
  //   is_switch_ ? 1 : 0, jumped ? 1 : 0, fast_reanchor_streak_for_debug,
  //   fast_reanchor_triggered ? 1 : 0, ekf_.x[7]);
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{{4e-3,
                         4e-3,
                         std::log(std::abs(delta_angle) + 1) + 1,
                         std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

bool Target::is_switch() const { return is_switch_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return xyza_list;
}

bool Target::has_primary_armor_xyza() const
{
  return name == ArmorName::outpost && armor_num_ == 3 && has_outpost_primary_slot_;
}

Eigen::Vector4d Target::primary_armor_xyza() const
{
  const auto xyza_list = armor_xyza_list();
  if (xyza_list.empty()) return Eigen::Vector4d::Zero();
  if (!has_primary_armor_xyza()) return xyza_list[0];

  const int max_index = static_cast<int>(xyza_list.size()) - 1;
  const int slot = std::clamp(outpost_primary_slot_, 0, max_index);
  return xyza_list[slot];
}

bool Target::diverged() const
{
  if (name == ArmorName::outpost && armor_num_ == 3)
  {
    const bool r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    if (r_ok) return false;
    tools::logger()->debug("[Target] outpost r={:.3f}", ekf_.x[8]);
    return true;
  }

  const bool r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const bool l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }
  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);

  if (name == ArmorName::outpost && armor_num_ == 3)
  {
    auto armor_x = x[0] - x[8] * std::cos(angle);
    auto armor_y = x[2] - x[8] * std::sin(angle);
    auto armor_z = x[4];
    if (id == 0)
    {
      armor_z += x[9];
    }
    else if (id == 2)
    {
      armor_z += x[10];
    }
    return {armor_x, armor_y, armor_z};
  }

  const bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto r = use_l_h ? x[8] + x[9] : x[8];
  const auto armor_x = x[0] - r * std::cos(angle);
  const auto armor_y = x[2] - r * std::sin(angle);
  const auto armor_z = use_l_h ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);

  if (name == ArmorName::outpost && armor_num_ == 3)
  {
    const auto r = x[8];
    const auto dx_da = r * std::sin(angle);
    const auto dy_da = -r * std::cos(angle);
    const auto dx_dr = -std::cos(angle);
    const auto dy_dr = -std::sin(angle);
    const auto dz_dx9 = (id == 0) ? 1.0 : 0.0;
    const auto dz_dx10 = (id == 2) ? 1.0 : 0.0;

    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr,      0,       0},
      {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr,      0,       0},
      {0, 0, 0, 0, 1, 0,     0, 0,     0, dz_dx9, dz_dx10},
      {0, 0, 0, 0, 0, 0,     1, 0,     0,      0,       0}
    };
    // clang-format on

    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    // clang-format off
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {                0,                 0,                 0, 1}
    };
    // clang-format on

    return H_armor_ypda * H_armor_xyza;
  }

  const bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto r = use_l_h ? x[8] + x[9] : x[8];
  const auto dx_da = r * std::sin(angle);
  const auto dy_da = -r * std::cos(angle);
  const auto dx_dr = -std::cos(angle);
  const auto dy_dr = -std::sin(angle);
  const auto dx_dl = use_l_h ? -std::cos(angle) : 0.0;
  const auto dy_dl = use_l_h ? -std::sin(angle) : 0.0;
  const auto dz_dh = use_l_h ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

} // namespace auto_aim
