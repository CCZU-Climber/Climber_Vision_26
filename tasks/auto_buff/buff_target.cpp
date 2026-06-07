#include "buff_target.hpp"
#include "tools/yaml.hpp"

#include <Eigen/Geometry>
#include <cstdio>
#include <cstring>

namespace auto_buff
{
namespace
{
Eigen::Vector3d small_r_ypd_observation(const PowerRune & p)
{
  if (p.trad_xyz_in_world.allFinite() && p.trad_xyz_in_world.norm() > 1e-6) {
    return tools::xyz2ypd(p.trad_xyz_in_world);
  }
  return p.ypd_in_world;
}

}  // namespace

///voter
Voter::Voter() : clockwise_(0) {}
void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return;
  double delta = tools::limit_rad(angle_now - angle_last);
  if (std::abs(delta) > CV_PI / 4) return;
  if (delta < 0) clockwise_--;
  else clockwise_++;
}
int Voter::clockwise() { return clockwise_ > 0 ? 1 : (clockwise_ < 0 ? -1 : 0); }

/// Target

Target::Target(const TargetParams & params)
  : params_(params), first_in_(true), unsolvable_(true), center_initialized_(false) {};

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_) return Eigen::Vector3d(0, 0, 0);
  if (center_initialized_) {
    Eigen::Matrix3d R = R_anchor_buff2world_;
    if (!R.allFinite()) {
      return point_buff2world_from_state(point_in_buff, ekf_.x);
    } else if (use_rotation_anchor_) {
      // 能量机关面板物理上垂直，强制 pitch=0。
      // 直接从EKF状态取yaw/roll建矩阵，避免从含pitch的R_buff2world做Euler分解
      // 再置零再重建时yaw/roll被pitch耦合污染。对齐SP: rotation_matrix(yaw, 0.0, roll)
      R = tools::rotation_matrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5]));
    }
    return R * point_in_buff + center_filtered_;
  }
  return point_buff2world_from_state(point_in_buff, ekf_.x);
}

Eigen::Vector3d Target::point_buff2world_from_state(
  const Eigen::Vector3d & point_in_buff, const Eigen::VectorXd & x) const
{
  if (x.size() < 6) return Eigen::Vector3d::Zero();
  const double pitch = use_observed_pitch_ ? buff_pitch_ : 0.0;
  Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(x[4], pitch, x[5]));
  auto R_yaw = x[0];
  auto R_pitch = x[2];
  auto R_dis = x[3];
  return R_buff2world * point_in_buff + Eigen::Vector3d(
    R_dis * std::cos(R_pitch) * std::cos(R_yaw),
    R_dis * std::cos(R_pitch) * std::sin(R_yaw),
    R_dis * std::sin(R_pitch));
}

bool Target::is_unsolve() const { return unsolvable_; }
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

void Target::save_ekf_state()
{
  saved_ekf_x_ = ekf_.x;
  saved_ekf_P_ = ekf_.P;
}
void Target::restore_ekf_state()
{
  ekf_.x = saved_ekf_x_;
  ekf_.P = saved_ekf_P_;
}

void Target::set_pose_anchor(const PowerRune & p, double roll_anchor)
{
  Eigen::Vector3d center = p.xyz_in_world;
  if (!center.allFinite() || center.norm() < 1e-6) center = p.trad_xyz_in_world;
  if (!center.allFinite() || center.norm() < 1e-6) {
    center_initialized_ = false;
    use_rotation_anchor_ = false;
    return;
  }

  center_filtered_ = center;
  R_anchor_buff2world_ = p.R_buff2world.allFinite()
                            ? p.R_buff2world
                            : tools::rotation_matrix(p.ypr_in_world);
  roll_anchor_ = roll_anchor;
  center_initialized_ = true;
  use_rotation_anchor_ = true;
}

Eigen::Vector3d Target::center_filtered() const
{
  if (center_initialized_) return center_filtered_;
  if (unsolvable_ || ekf_.x.size() < 4) return Eigen::Vector3d::Zero();
  return Eigen::Vector3d(
    ekf_.x[3] * std::cos(ekf_.x[2]) * std::cos(ekf_.x[0]),
    ekf_.x[3] * std::cos(ekf_.x[2]) * std::sin(ekf_.x[0]),
    ekf_.x[3] * std::sin(ekf_.x[2]));
}

/// SmallTarget — SP 7-state EKF baseline

SmallTarget::SmallTarget(const TargetParams & params) : Target(params)
{
  use_observed_pitch_ = true;
}

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, p.value());
    first_in_ = false;
  }

  if (lost_cn > params_.max_lost_frames) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 丢失buff");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  unsolvable_ = false;
  update(time_gap, p.value());
  lost_cn = 0;
  buff_yaw_ = ekf_.x[4];
  buff_pitch_ = p.value().ypr_in_world[1];

  if (
    std::abs(ekf_.x[6]) > params_.small_w + params_.divergence_eps ||
    std::abs(ekf_.x[6]) < params_.small_w - params_.divergence_eps) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 小符角度发散spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
    first_in_ = true;
    return;
  }
}

void SmallTarget::predict(double dt)
{
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  auto v1 = params_.ekf_q_v1;
  auto v_roll = params_.prediction_roll_q;
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Q_ << a * v1, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, a * v_roll, b * v_roll,
           0.0,    0.0, 0.0, 0.0, 0.0, b * v_roll, c * v_roll;

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5]);
    return x_prior;
  };
  ekf_.predict(A_, Q_, f);
}

void SmallTarget::init(double nowtime, const PowerRune & p)
{
  lasttime_ = nowtime;
  center_initialized_ = false;

  x0_.resize(7);
  P0_.resize(7, 7);
  A_.resize(7, 7);
  Q_.resize(7, 7);
  H_.resize(7, 7);
  R_.resize(7, 7);

  const Eigen::Vector3d R_ypd = small_r_ypd_observation(p);
  int direction = voter.clockwise();
  if (direction == 0) direction = 1;
  x0_ << R_ypd[0], 0.0, R_ypd[1], R_ypd[2],
         p.ypr_in_world[0], p.ypr_in_world[2],
         params_.small_w * direction;

  P0_ << 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-2;

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    return c;
  };
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void SmallTarget::update(double nowtime, const PowerRune & p)
{
  const Eigen::Vector3d R_ypd = small_r_ypd_observation(p);
  const Eigen::VectorXd & ypr = p.ypr_in_world;

  if (std::abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        break;
      }
    }
  }

  voter.vote(ekf_.x[5], ypr[2]);
  if (std::abs(voter.raw_count()) >= 5 && voter.clockwise() * ekf_.x[6] < 0) ekf_.x[6] *= -1;

  predict(nowtime - lasttime_);

  Eigen::MatrixXd H1{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
  };

  Eigen::MatrixXd R1{
    {params_.ekf_r1_yaw, 0.0, 0.0, 0.0},
    {0.0, params_.ekf_r1_pitch, 0.0, 0.0},
    {0.0, 0.0, params_.ekf_r1_dis, 0.0},
    {0.0, 0.0, 0.0, params_.ekf_r1_roll}
  };

  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], ypr[2]}};
  ekf_.update(z1, H1, R1, z_subtract1);
  ekf_.x[4] = ypr[0];
  ekf_.x[5] = ypr[2];
  set_pose_anchor(p, ekf_.x[5]);

  lasttime_ = nowtime;
}

Eigen::MatrixXd SmallTarget::h_jacobian() const
{
  Eigen::MatrixXd H0{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
  };

  Eigen::VectorXd R_ypd{{ekf_.x[0], ekf_.x[2], ekf_.x[3]}};
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);
  Eigen::MatrixXd H1{
    {H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0},
    {H_ypd2xyz(1, 0), H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0},
    {H_ypd2xyz(2, 0), H_ypd2xyz(2, 1), H_ypd2xyz(2, 2), 0.0, 0.0},
    {            0.0,             0.0,             0.0, 1.0, 0.0},
    {            0.0,             0.0,             0.0, 0.0, 1.0}
  };

  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  double cos_roll = std::cos(roll);
  double sin_roll = std::sin(roll);
  const double r = params_.target_radius;
  Eigen::MatrixXd H2{
    {1.0, 0.0, 0.0, r * cos_yaw * sin_roll,  r * sin_yaw * cos_roll},
    {0.0, 1.0, 0.0, r * sin_yaw * sin_roll, -r * cos_yaw * cos_roll},
    {0.0, 0.0, 1.0,                    0.0,         -r * sin_roll}
  };

  Eigen::VectorXd B_xyz =
    point_buff2world_from_state(Eigen::Vector3d(0.0, 0.0, r), ekf_.x);
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);
  return H3 * H2 * H1 * H0;
}

/// BigTarget (unchanged from HEAD, kept verbatim)
BigTarget::BigTarget(const TargetParams & params) : Target(params), spd_fitter_(100, 0.5, 1.884, 2.000)
{
  use_observed_pitch_ = true;
}

void BigTarget::set_blade_index(int idx)
{
  if (idx < 0) {
    blade_index_ = -1;
    phase_initialized_ = false;
    return;
  }

  if (idx >= 5) idx %= 5;
  if (blade_index_ >= 0 && idx != blade_index_ && ekf_.x.size() > 5 && phase_initialized_) {
    const double phase = ekf_.x[5] + blade_index_ * kFaceAngle;
    ekf_.x[5] = phase - idx * kFaceAngle;
    unwrapped_phase_ = ekf_.x[5];
    last_canonical_ = tools::limit_rad(phase);
    roll_anchor_ = ekf_.x[5];
  }
  blade_index_ = idx;
}

std::optional<BigTarget> BigTarget::copy_for_slot(int slot_id) const
{
  if (slot_id < 0 || slot_id >= 5) return std::nullopt;
  if (is_unsolve() || ekf_.x.size() <= 5 || blade_index_ < 0) return std::nullopt;

  BigTarget copy = *this;
  const double phase = ekf_.x[5] + blade_index_ * kFaceAngle;
  copy.ekf_.x[5] = phase - slot_id * kFaceAngle;
  copy.blade_index_ = slot_id;
  copy.unwrapped_phase_ = copy.ekf_.x[5];
  copy.last_canonical_ = tools::limit_rad(phase);
  copy.phase_initialized_ = true;
  copy.roll_anchor_ = copy.ekf_.x[5];
  return copy;
}

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  static std::chrono::steady_clock::time_point start_timestamp = timestamp;

  if (p.has_value()) {
    auto time_gap = tools::delta_time(timestamp, start_timestamp);
    lost_count_ = 0;
    is_predicted_ = false;

    if (first_in_) {
      unsolvable_ = true;
      init(time_gap, p.value());
      first_in_ = false;
    }

    unsolvable_ = false;
    buff_yaw_ = p.value().ypr_in_world[0];
    buff_pitch_ = p.value().ypr_in_world[1];
    if (!std::isfinite(buff_yaw_)) buff_yaw_ = 0.0;
    if (!std::isfinite(buff_pitch_)) buff_pitch_ = 0.0;
    update(time_gap, p.value());

    if (ekf_.x[7] > 1.045 * 1.5 || ekf_.x[7] < 0.78 / 1.5 ||
        ekf_.x[8] > 2.0 * 1.5 || ekf_.x[8] < 1.884 / 1.5) {
      tools::logger()->debug("[Target] 大符角度发散a: {:.2f}b:{:.2f}", ekf_.x[7], ekf_.x[8]);
      first_in_ = true;
      return;
    }
    return;
  }

  if (first_in_ || lost_count_ >= params_.max_lost_frames) {
    if (lost_count_ >= params_.max_lost_frames) {
      if (lost_count_ < params_.max_lost_frames + 3)
        tools::logger()->debug("[Target] 丢失buff (lost_count={})", lost_count_);
      first_in_ = true;
      is_predicted_ = false;
    }
    unsolvable_ = true;
    lost_count_ = std::min(lost_count_ + 1, params_.max_lost_frames + 10);
    return;
  }

  {
    auto time_gap = tools::delta_time(timestamp, start_timestamp);
    double dt = time_gap - lasttime_;
    if (dt > 0 && dt < 1.0) {
      predict(dt);
      lasttime_ = time_gap;
    }
    lost_count_++;
    unsolvable_ = false;
    is_predicted_ = true;
    if (lost_count_ % 3 == 1)
      tools::logger()->debug("[Target] 短时预测 lost={}/{} angle={:.1f}deg dist={:.2f}m",
        lost_count_, params_.max_lost_frames, ekf_.x[5] * 180 / CV_PI, ekf_.x[3]);
  }
}

void BigTarget::predict(double dt)
{
  double spd = fit_spd_;
  double a = ekf_.x[7], w = ekf_.x[8], fi = ekf_.x[9];
  double t = lasttime_ + dt;
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, voter.clockwise() * dt, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sin(w*t+fi)-1, t*a*cos(w*t+fi), a*cos(w*t+fi),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  auto v1 = params_.ekf_q_v1_big;
  auto a1 = dt*dt*dt*dt/4, b1 = dt*dt*dt/2, c1 = dt*dt;
  Q_ << a1*v1, b1*v1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        b1*v1, c1*v1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, params_.ekf_q_spd_big, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = x;
    x_prior[0] = tools::limit_rad(x_prior[0] + dt * x_prior[1]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5] + voter.clockwise() *
      (-a/w*cos(w*t+fi) + a/w*cos(w*lasttime_+fi) + (2.09-a)*dt));
    x_prior[6] = a*sin(w*t+fi) + 2.09 - a;
    return x_prior;
  };
  ekf_.predict(A_, Q_, f);
}

void BigTarget::init(double nowtime, const PowerRune & p)
{
  lasttime_ = nowtime;
  unsolvable_ = true;

  // Blade index + phase unwrapping init (参考RMCV LargeLsmModel)
  {
    double r = tools::limit_rad(p.ypr_in_world[2]);
    int idx = static_cast<int>(std::round(r / kFaceAngle)) % 5;
    if (idx < 0) idx += 5;
    if (blade_index_ < 0) blade_index_ = idx;
    // canonical only drives unwrap continuity; ekf_.x[5] keeps the physical blade roll.
    // PnP roll follows the opposite order from image-plane slot ids: phase = roll + slot*72deg.
    double canonical = tools::limit_rad(r + blade_index_ * kFaceAngle);
    unwrapped_phase_ = canonical - blade_index_ * kFaceAngle;
    last_canonical_ = canonical;
    phase_initialized_ = true;
  }
  x0_.resize(10); P0_.resize(10,10); A_.resize(10,10);
  Q_.resize(10,10); H_.resize(7,10); R_.resize(7,7);
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], unwrapped_phase_, 1.1775, 0.9125, 1.942, 0.0;
  P0_ << 10.0,0,0,0,0,0,0,0,0,0, 0,10.0,0,0,0,0,0,0,0,0, 0,0,10.0,0,0,0,0,0,0,0,
         0,0,0,10.0,0,0,0,0,0,0, 0,0,0,0,10.0,0,0,0,0,0, 0,0,0,0,0,10.0,0,0,0,0,
         0,0,0,0,0,0,100.0,0,0,0, 0,0,0,0,0,0,0,10.0,0,0, 0,0,0,0,0,0,0,0,10.0,0,
         0,0,0,0,0,0,0,0,0,400.0;
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]); c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]); c[5] = tools::limit_rad(c[5]);
    c[9] = tools::limit_rad(c[9]);
    return c;
  };
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void BigTarget::update(double nowtime, const PowerRune & p)
{
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;

  // 相位归一化+解缠 (参考RMCV LargeLsmModel::feed):
  //   1. 归一化: canonical = limit_rad(ypr[2] + blade_index * 72deg)
  //   2. 解缠:   physical_roll += limit_rad(canonical - last_canonical)
  // ekf_.x[5] must remain physical roll because projection/aiming use it as pose roll.
  double obs_roll = ypr[2];
  if (blade_index_ >= 0 && phase_initialized_) {
    double canonical = tools::limit_rad(ypr[2] + blade_index_ * kFaceAngle);
    unwrapped_phase_ += tools::limit_rad(canonical - last_canonical_);
    last_canonical_ = canonical;
    obs_roll = unwrapped_phase_;
    static int unwrap_log = 0;
    if (++unwrap_log <= 3)
      tools::logger()->warn("[BigTarget] unwrap: PnP={:.1f} b{} canonical={:.1f} physical={:.1f}deg",
                            ypr[2] * 180 / CV_PI, blade_index_,
                            canonical * 180 / CV_PI, unwrapped_phase_ * 180 / CV_PI);
  }

  if (abs(obs_roll - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double ac = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(ac - obs_roll) < CV_PI / 5) { ekf_.x[5] += i * 2 * CV_PI / 5; break; }
    }
  }
  voter.vote(ekf_.x[5], obs_roll);
  auto anglelast = ekf_.x[5];
  predict(nowtime - lasttime_);

  const double dyaw = std::abs(tools::limit_rad(R_ypd[0] - ekf_.x[0]));
  const double dpitch = std::abs(tools::limit_rad(R_ypd[1] - ekf_.x[2]));
  const double ddist = std::abs(R_ypd[2] - ekf_.x[3]);
  if (dyaw > 0.55 || dpitch > 0.55 || ddist > 2.5) {
    tools::logger()->debug("[Target] reset by outlier obs dyaw={:.2f} dpitch={:.2f} ddist={:.2f}",
      dyaw, dpitch, ddist);
    unsolvable_ = true; first_in_ = true; return;
  }

  Eigen::MatrixXd H1{
    {1.0,0,0,0,0,0,0,0,0,0}, {0,0,1.0,0,0,0,0,0,0,0},
    {0,0,0,1.0,0,0,0,0,0,0}, {0,0,0,0,0,1.0,0,0,0,0}};
  Eigen::MatrixXd R1{
    {params_.ekf_r1_yaw,0,0,0}, {0,params_.ekf_r1_pitch,0,0},
    {0,0,params_.ekf_r1_dis,0}, {0,0,0,params_.ekf_r1_roll}};

  auto zs1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]); c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };
  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], obs_roll}};
  ekf_.update(z1, H1, R1, zs1);

  Eigen::MatrixXd H2 = h_jacobian();
  Eigen::MatrixXd R2{
    {params_.ekf_r2_yaw,0,0}, {0,params_.ekf_r2_pitch,0}, {0,0,params_.ekf_r2_dis}};

  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
    Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
    Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
    Eigen::VectorXd B_xyz =
      point_buff2world_from_state(Eigen::Vector3d(0.0, 0.0, params_.target_radius), x);
    Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
    return B_ypd;
  };

  auto zs2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]); c[1] = tools::limit_rad(c[1]);
    return c;
  };

  Eigen::VectorXd z2{{B_ypd[0], B_ypd[1], B_ypd[2]}};
  ekf_.update(z2, H2, R2, h2, zs2);

  if (ekf_.x[6] < 2.1 && ekf_.x[6] >= 0) spd_fitter_.add_data(nowtime, ekf_.x[6]);
  spd_fitter_.fit();
  fit_spd_ = spd_fitter_.sine_function(
    nowtime, spd_fitter_.best_result_.A, spd_fitter_.best_result_.omega,
    spd_fitter_.best_result_.phi, spd_fitter_.best_result_.C);
  spd = voter.clockwise() * tools::limit_rad(ekf_.x[5] - anglelast) / (nowtime - lasttime_);
  spd = fit_spd_;
  if (std::abs(spd) > 4) spd = 0;

  // 大符当前帧位姿使用观测值，EKF/正弦拟合只负责未来角度增量。
  // 否则 x[0]/x[2]/x[3]/x[4]/x[5] 会被历史状态拖滞，当前帧重投影明显偏离 PnP。
  ekf_.x[0] = R_ypd[0];
  ekf_.x[2] = R_ypd[1];
  ekf_.x[3] = R_ypd[2];
  ekf_.x[4] = ypr[0];
  ekf_.x[5] = obs_roll;
  set_pose_anchor(p, ekf_.x[5]);

  lasttime_ = nowtime;
  unsolvable_ = false;
}

Eigen::MatrixXd BigTarget::h_jacobian() const
{
  Eigen::MatrixXd H0{
    {1,0,0,0,0,0,0,0,0,0}, {0,0,1,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0}, {0,0,0,0,1,0,0,0,0,0}, {0,0,0,0,0,1,0,0,0,0}};
  Eigen::VectorXd R_ypd{{ekf_.x[0], ekf_.x[2], ekf_.x[3]}};
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);
  Eigen::MatrixXd H1{
    {H_ypd2xyz(0,0), H_ypd2xyz(0,1), H_ypd2xyz(0,2), 0, 0},
    {H_ypd2xyz(1,0), H_ypd2xyz(1,1), H_ypd2xyz(1,2), 0, 0},
    {H_ypd2xyz(2,0), H_ypd2xyz(2,1), H_ypd2xyz(2,2), 0, 0},
    {0,0,0,1,0}, {0,0,0,0,1}};
  double yaw = ekf_.x[4], roll = ekf_.x[5];
  double cy = cos(yaw), sy = sin(yaw), cr = cos(roll), sr = sin(roll);
  const double r = params_.target_radius;
  Eigen::MatrixXd H2{
    {1,0,0, r*cy*sr,  r*sy*cr},
    {0,1,0, r*sy*sr, -r*cy*cr},
    {0,0,1, 0,       -r*sr}};
  Eigen::VectorXd B_xyz =
    point_buff2world_from_state(Eigen::Vector3d(0.0, 0.0, r), ekf_.x);
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);
  return H3 * H2 * H1 * H0;
}

TargetParams load_target_params(const std::string & config_path)
{
  // ── EKF / 跟踪参数硬编码（实车验证最优，不再从 yaml 读取） ──
  TargetParams p;
  p.max_lost_frames = 20;        // 无识别时短时预测最大帧数
  p.lost_count_cap = 20;         // lost_count_ 封顶值
  p.lost_log_suppress = 3;       // 超阈值后只在前N帧打印丢失日志
  p.small_w = 1.047;             // 小符标称角速度 rad/s (≈60°/s)
  try {
    auto yaml = tools::load(config_path);
    if (yaml["buff_r_to_target"]) p.target_radius = yaml["buff_r_to_target"].as<double>();
  } catch (...) {
    tools::logger()->warn("[TargetParams] failed to read buff_r_to_target, keep default 0.700m");
  }
  p.converge_delay = 5;          // 收敛前置免发散判断的帧数
  p.divergence_eps = 0.524;      // 小符速度发散容差 rad/s (≈30°/s)
  p.max_roll_residual = 0.55;    // roll观测残差异常阈值 rad

  // 小符 & 大符 R1 测量噪声 — 直接观测 (R_yaw, R_pitch, R_dis, roll)
  //   ↑值 = EKF更平滑但滞后  ↓值 = 更灵敏但毛刺多
  p.ekf_r1_yaw = 0.05;           // R中心方位角测量噪声 rad
  p.ekf_r1_pitch = 0.05;         // R中心俯仰角测量噪声 rad
  p.ekf_r1_dis = 1.0;            // R中心距离测量噪声 m
  p.ekf_r1_roll = 0.004;         // 面板自转角度测量噪声 rad

  // 小符 & 大符 R2 测量噪声 — blade位置约束 (B_yaw, B_pitch, B_dis)
  p.ekf_r2_yaw = 0.1;            // blade方位角测量噪声 rad
  p.ekf_r2_pitch = 0.1;          // blade俯仰角测量噪声 rad
  p.ekf_r2_dis = 1.0;            // blade距离测量噪声 m

  // 小符过程噪声
  p.ekf_q_v1 = 0.001;            // yaw角加速度方差 rad²/s²
  p.prediction_roll_q = 0.005;   // roll预测过程噪声

  // 大符过程噪声
  p.ekf_q_v1_big = 0.9;          // 大符 yaw角加速度方差
  p.ekf_q_roll_big = 0.09;       // 大符 roll过程噪声
  p.ekf_q_spd_big = 0.5;         // 大符 spd过程噪声

  p.max_prediction_variance = 0.3; // 预测方差上限

  // 原 yaml 加载逻辑（注释保留供回溯）：
  // try {
  //   auto yaml = tools::load(config_path);
  //   if (!yaml["buff_target"]) return p;
  //   auto bt = yaml["buff_target"];
  //   if (bt["max_lost_frames"]) p.max_lost_frames = bt["max_lost_frames"].as<int>();
  //   ...
  // } catch (...) { ... }
  //
  //  调参速查:
  //   波形毛刺多 → 翻倍 R1_yaw / R1_pitch (0.05 → 0.1 → 0.2)
  //   跟踪滞后  → 减半 R1_yaw / R1_pitch
  //   距离跳动  → 翻倍 R1_dis / R2_dis
  //   切扇叶跳变→ 翻倍 R1_roll
  //   速度发散频繁 → 放宽 divergence_eps

  return p;
}

}  // namespace auto_buff
