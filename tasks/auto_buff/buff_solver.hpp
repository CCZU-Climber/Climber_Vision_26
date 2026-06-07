#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <optional>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"
namespace auto_buff
{
const double THETA = 2.0 * CV_PI / 5.0;

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // PnP模式："4"=靶面4点IPPE；"4i"=靶面4点IPPE+ITERATIVE；"5"=加入R中心精化。
  void set_pnp_mode(const std::string & m) { pnp_mode_ = m; }
  std::string pnp_mode() const { return pnp_mode_; }

  void solve(std::optional<PowerRune> & ps) const;

  void solve(std::vector<PowerRune> & ps) const;

  cv::Point2f point_buff2pixel(cv::Point3f x);

  // 将世界系点投影到像素坐标（调试可视化用）
  cv::Point2f world2pixel(const Eigen::Vector3d & xyz_in_world) const;

  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Vector3d & ypr_in_world) const;

  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

private:
  bool solve_one(PowerRune & p) const;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;
  double reprojection_error_threshold_px_ = 30.0;
  std::string pnp_mode_ = "4i";  // 硬编码靶面4点IPPE+ITERATIVE；4/5代码保留注释供回溯

  mutable cv::Vec3d rvec_, tvec_;

  // 能量机关buff坐标系下的3D参考点，单位：m。
  // 坐标约定：x为靶面法线方向；y为靶面左右/切向；z为从R中心指向靶心的径向。
  // 当前模型直接输出R标中心，和SP原版“灯臂下端点再外推R中心”的关键点语义不同。
  std::vector<cv::Point3f> object_points_ = {
    cv::Point3f(0, 0, 700e-3 + 115e-3), // [0] 靶面上边中点：径向外侧
    cv::Point3f(0, 150e-3, 700e-3),     // [1] 靶面左边中点：切向左侧
    cv::Point3f(0, 0, 700e-3 - 115e-3), // [2] 靶面下边中点：径向内侧
    cv::Point3f(0, -115e-3, 700e-3),    // [3] 靶面右边中点：切向右侧
    cv::Point3f(0, 0, 700e-3),          // [4] 靶心/扇叶靶面中心
    cv::Point3f(0, 0, 0),               // [5] SP遗留R参考点；当前模型不用它做PnP
    cv::Point3f(0, 0, 0)};              // [6] 当前模型R中心；5PnP将p.r_center对应到这里
  // 调试用
  // std::vector<std::vector<cv::Point3f>> OBJECT_POINTS = {
  //   {cv::Point3f(0, 160e-3, 858.5e-3), cv::Point3f(0, -160e-3, 858.5e-3),
  //    cv::Point3f(0, -186e-3, 541.5e-3), cv::Point3f(0, 186e-3, 541.5e-3),
  //    cv::Point3f(0, 0, 700e-3)},
  //   {},
  //   {},
  //   {},
  //   {}};  // 单位：米

  // TODO
  // 函数：生成绕x轴旋转的旋转矩阵
  cv::Matx33f rotation_matrix(double angle) const;

  // 函数：旋转点并填充到 OBJECT_POINTS 中
  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);
};
}  // namespace auto_buff
#endif
