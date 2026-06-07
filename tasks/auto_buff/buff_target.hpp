#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/ransac_sine_fitter.hpp"

namespace auto_buff
{

/// 可调参数结构体（从YAML读取）
struct TargetParams
{
  // 默认值同步自 load_target_params() 硬编码（实车验证最优）
  int max_lost_frames = 20;
  int lost_count_cap = 20;
  int lost_log_suppress = 3;
  double small_w = 1.047;
  double target_radius = 0.700;
  int converge_delay = 5;
  double divergence_eps = 0.524;
  double max_roll_residual = 0.55;
  double ekf_r1_yaw = 0.05;
  double ekf_r1_pitch = 0.05;
  double ekf_r1_dis = 1.0;
  double ekf_r1_roll = 0.004;
  double ekf_r2_yaw = 0.1;
  double ekf_r2_pitch = 0.1;
  double ekf_r2_dis = 1.0;
  double ekf_q_v1 = 0.001;
  double prediction_roll_q = 0.005;
  double max_prediction_variance = 0.3;
  double ekf_q_v1_big = 0.9;
  double ekf_q_roll_big = 0.09;
  double ekf_q_spd_big = 0.5;
};

class Voter
{
public:
  Voter();
  void vote(const double angle_last, const double angle_now);
  int clockwise();        // 1=CCW, -1=CW, 0=unknown
  void reset() { clockwise_ = 0; }
  bool known() const { return clockwise_ != 0; }
  int raw_count() const { return clockwise_; }

private:
  int clockwise_;
};

/// Target 基类

class Target
{
public:
  Target(const TargetParams & params = TargetParams());

  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;

  virtual void predict(double dt) = 0;

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  bool is_unsolve() const;
  bool is_predicted() const { return is_predicted_; }

  void save_ekf_state();
  void restore_ekf_state();

  Eigen::VectorXd ekf_x() const;
  virtual double get_roll() const = 0;

  double buff_yaw() const
  {
    return center_initialized_ || ekf_.x.size() <= 4 ? buff_yaw_ : ekf_.x[4];
  }
  double buff_pitch() const { return buff_pitch_; }
  Eigen::Vector3d center_filtered() const;

  // Observation state machine counters
  int accepted_obs() const { return accepted_obs_; }
  int rejected_obs() const { return fanblade_jump_reject_ + roll_outlier_reject_ + nonfinite_reject_; }
  int tracking_loss_events() const { return tracking_loss_events_; }
  int fanblade_jump_reject_count() const { return fanblade_jump_reject_; }
  int roll_outlier_reject_count() const { return roll_outlier_reject_; }
  int nonfinite_reject_count() const { return nonfinite_reject_; }

  // Roll trace (debug diagnostic)
  bool trace_enabled_ = false;
  FILE * trace_file_ = nullptr;
  int trace_frame_idx_ = 0;

  double spd = 0;

  TargetParams params_;

protected:
  virtual void init(double nowtime, const PowerRune & p) = 0;
  virtual void update(double nowtime, const PowerRune & p) = 0;
  Eigen::Vector3d point_buff2world_from_state(
    const Eigen::Vector3d & point_in_buff, const Eigen::VectorXd & x) const;
  void set_pose_anchor(const PowerRune & p, double roll_anchor);

  double buff_yaw_ = 0.0;
  double buff_pitch_ = 0.0;
  bool use_observed_pitch_ = false;
  Eigen::Vector3d center_filtered_ = Eigen::Vector3d::Zero();
  bool center_initialized_ = false;
  Eigen::Matrix3d R_anchor_buff2world_ = Eigen::Matrix3d::Identity();
  double roll_anchor_ = 0.0;
  bool use_rotation_anchor_ = false;

  Eigen::VectorXd x0_;
  Eigen::MatrixXd P0_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd H_;
  Eigen::MatrixXd R_;
  tools::ExtendedKalmanFilter ekf_;
  double lasttime_ = 0;
  Voter voter;
  bool first_in_;
  bool unsolvable_;
  bool is_predicted_ = false;
  int lost_count_ = 0;
  int accepted_obs_ = 0;
  int fanblade_jump_reject_ = 0;
  int roll_outlier_reject_ = 0;
  int nonfinite_reject_ = 0;
  int tracking_loss_events_ = 0;

  Eigen::VectorXd saved_ekf_x_;
  Eigen::MatrixXd saved_ekf_P_;

  // Trace state (per-frame)
  const char * trace_gate_ = "no_obs";
  int trace_fanblade_shift_i_ = 0;
  int trace_shift_frame_voted_ = 0;
};

/// SmallTarget子类 — SP 7-state EKF baseline

class SmallTarget : public Target
{
public:
  SmallTarget(const TargetParams & params = TargetParams());

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;
  void predict(double dt) override;
  double get_roll() const override { return ekf_.x[5]; }

private:
  void init(double nowtime, const PowerRune & p) override;
  void update(double nowtime, const PowerRune & p) override;
  Eigen::MatrixXd h_jacobian() const;
};

/// BigTarget子类 (unchanged from HEAD)

class BigTarget : public Target
{
public:
  BigTarget(const TargetParams & params = TargetParams());

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;
  void predict(double dt) override;
  double get_roll() const override { return ekf_.x[5]; }

private:
  void init(double nowtime, const PowerRune & p) override;
  void update(double nowtime, const PowerRune & p) override;

  Eigen::MatrixXd h_jacobian() const;

public:
  int blade_index() const { return blade_index_; }
  void set_blade_index(int idx);
  int blade_id() const { return blade_id_; }
  void set_blade_id(int id) { blade_id_ = id; }
  std::optional<BigTarget> copy_for_slot(int slot_id) const;

private:
  int blade_index_ = -1;
  int blade_id_ = -1;
  static constexpr double kFaceAngle = 2.0 * CV_PI / 5.0;

  // 相位解缠: 归一化 + 累积, 参考RMCV LargeLsmModel::feed()
  double unwrapped_phase_ = 0.0;
  double last_canonical_ = 0.0;
  bool phase_initialized_ = false;

  tools::RansacSineFitter spd_fitter_;
  double fit_spd_;
};

TargetParams load_target_params(const std::string & config_path);

}  // namespace auto_buff
#endif
