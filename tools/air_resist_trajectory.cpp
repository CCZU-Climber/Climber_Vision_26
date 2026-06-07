#include "air_resist_trajectory.hpp"

#include <ceres/ceres.h>

#include <cmath>

namespace tools
{
constexpr double kGravity = 9.7833;

// ============================================================================
// Ceres 残差函数: 线性阻力 2D 弹道
//
// 物理方程 (世界系, z 向上):
//   dvx/dt = -k * vx
//   dvz/dt = -g - k * vz
//
// 解析解 (来自 RMCV / rm.cv.fans):
//   (k*v0*sin(θ) + g) * k*w / (k²*v0*cos(θ)) + g*log(1 - k*w/(v0*cos(θ))) / k² = h
// ============================================================================
class LinearDragResidual2D
{
public:
  LinearDragResidual2D(double w, double h, double v0, double g, double k)
    : w_(w), h_(h), v0_(v0), g_(g), k_(k) {}

  template <typename T>
  bool operator()(const T * const pitch, T * residual) const
  {
    T sin_a = ceres::sin(pitch[0]);
    T cos_a = ceres::cos(pitch[0]);

    if (ceres::abs(cos_a) < T(1e-6)) {
      residual[0] = T(1e6);
      return true;
    }

    T k = T(k_);
    T g = T(g_);
    T v0 = T(v0_);
    T w = T(w_);
    T h = T(h_);

    T ratio = k * w / (v0 * cos_a);
    if (ratio >= T(0.99)) {
      residual[0] = T(1e6);
      return true;
    }

    T term1 = (k * v0 * sin_a + g) * k * w / (k * k * v0 * cos_a);
    T term2 = g * ceres::log(T(1.0) - ratio) / (k * k);
    residual[0] = term1 + term2 - h;
    return true;
  }

private:
  double w_, h_, v0_, g_, k_;
};

AirResistTrajectory::AirResistTrajectory(double v0, double d, double h, double k)
{
  if (d < 0.1 || v0 < 5.0) {
    unsolvable = true;
    return;
  }

  yaw = 0.0;       // auto_buff 只关心 pitch，yaw 上层自己算
  double horiz_d = d;

  // 初始猜测：真空弹道
  double a = kGravity * horiz_d * horiz_d / (2.0 * v0 * v0);
  double b = -horiz_d;
  double c = a + h;
  double delta = b * b - 4.0 * a * c;

  double pitch_guess;
  if (delta >= 0 && std::abs(a) > 1e-9) {
    double tan_1 = (-b + std::sqrt(delta)) / (2.0 * a);
    double tan_2 = (-b - std::sqrt(delta)) / (2.0 * a);
    // 选低弧：飞行时间更短的那个
    double p1 = std::atan(tan_1), p2 = std::atan(tan_2);
    double t1 = horiz_d / (v0 * std::cos(p1));
    double t2 = horiz_d / (v0 * std::cos(p2));
    pitch_guess = (t1 < t2) ? p1 : p2;
  } else {
    pitch_guess = std::atan2(h, horiz_d);
  }

  pitch_guess = std::clamp(pitch_guess, -M_PI / 3.0, M_PI / 3.0);

  if (k < 1e-6) {
    // 真空：直接用闭式解
    pitch = pitch_guess;
    fly_time = horiz_d / (v0 * std::cos(pitch));
    unsolvable = false;
    return;
  }

  // Ceres 优化
  double pitch_opt = pitch_guess;
  ceres::Problem problem;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<LinearDragResidual2D, 1, 1>(
      new LinearDragResidual2D(horiz_d, h, v0, kGravity, k)),
    nullptr, &pitch_opt);

  problem.SetParameterLowerBound(&pitch_opt, 0, -M_PI / 3.0);
  problem.SetParameterUpperBound(&pitch_opt, 0, M_PI / 3.0);

  ceres::Solver::Options options;
  options.max_num_iterations = 50;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (!summary.IsSolutionUsable()) {
    unsolvable = true;
    return;
  }

  pitch = pitch_opt;
  double cos_p = std::cos(pitch);
  double ratio = k * horiz_d / (v0 * cos_p);
  if (ratio >= 0.99) ratio = 0.99;
  fly_time = -std::log(1.0 - ratio) / k;
  unsolvable = false;
}

}  // namespace tools
