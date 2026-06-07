#include "buff_solver.hpp"
#include "tools/logger.hpp"

#include <Eigen/Geometry>
#include <tuple>

namespace auto_buff
{
cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  pnp_mode_ = yaml["pnp_mode"] ? yaml["pnp_mode"].as<std::string>() : "4i";
  if (pnp_mode_ != "4" && pnp_mode_ != "4i" && pnp_mode_ != "5") {
    tools::logger()->warn("[Buff_Solver] invalid pnp_mode '{}', fallback to 4i", pnp_mode_);
    pnp_mode_ = "4i";
  }

  reprojection_error_threshold_px_ =
    yaml["reprojection_error_threshold_px"] ? yaml["reprojection_error_threshold_px"].as<double>()
                                           : 30.0;

  const double r_to_target =
    yaml["buff_r_to_target"] ? yaml["buff_r_to_target"].as<double>() : 0.700;
  const double target_top =
    yaml["buff_target_top"] ? yaml["buff_target_top"].as<double>() : 0.115;
  const double target_bottom =
    yaml["buff_target_bottom"] ? yaml["buff_target_bottom"].as<double>() : 0.115;
  const double target_left =
    yaml["buff_target_left"] ? yaml["buff_target_left"].as<double>() : 0.150;
  const double target_right =
    yaml["buff_target_right"] ? yaml["buff_target_right"].as<double>() : 0.115;
  const double r_center_z =
    yaml["buff_r_center_z"] ? yaml["buff_r_center_z"].as<double>() : 0.0;

  object_points_ = {
    cv::Point3f(0, 0, r_to_target + target_top),
    cv::Point3f(0, target_left, r_to_target),
    cv::Point3f(0, 0, r_to_target - target_bottom),
    cv::Point3f(0, -target_right, r_to_target),
    cv::Point3f(0, 0, r_to_target),
    cv::Point3f(0, 0, r_center_z),
    cv::Point3f(0, 0, r_center_z)};
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value()) return;
  if (!solve_one(ps.value())) ps.reset();
}

void Solver::solve(std::vector<PowerRune> & ps) const
{
  std::vector<PowerRune> valid;
  valid.reserve(ps.size());
  for (auto & p : ps) {
    if (solve_one(p)) valid.emplace_back(std::move(p));
  }
  ps = std::move(valid);
}

bool Solver::solve_one(PowerRune & p) const
{
  // compute_rotated_points(object_points_);
  // std::vector<cv::Point2f> image_points;
  // std::vector<cv::Point3f> object_points;
  // int i = 0;
  // for (auto & fanblade : p.fanblades) {
  //   if (fanblade.type != _unlight) {
  //     image_points.insert(image_points.end(), fanblade.points.begin(), fanblade.points.end());
  //     image_points.emplace_back(fanblade.center);
  //     object_points.insert(object_points.end(), object_points_[i].begin(), object_points_[i].end());
  //   }
  //   ++i;
  // }
  // image_points.emplace_back(p.r_center);  //r_center
  // object_points.emplace_back(cv::Point3f(0, 0, 0));
  // image_points.emplace_back(p.target().center);

  if (p.fanblades.empty()) {
    tools::logger()->debug("[Buff_Solver] reject obs by empty fanblades");
    return false;
  }

  // 三种PnP模式共用同一套输出接口，后面的target/aimer无需感知模式差异：
  //   4  : 只用靶面4点IPPE解姿态，R标不参与solvePnP。
  //   4i : 先4点IPPE，再用同样4个靶面点ITERATIVE精化。
  //   5  : 在4点IPPE基础上加入R标精化；当前模型的p.r_center对应object_points_[6]。
  std::vector<cv::Point2f> image_points(p.target().points.begin(), p.target().points.begin() + 4);
  std::vector<cv::Point3f> object_points{
    object_points_[0], object_points_[1], object_points_[2], object_points_[3]};

  // 第一步固定先跑4点IPPE，保证平面PnP有稳定初值和正深度解。
  bool ok = cv::solvePnP(
    object_points, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_, false,
    cv::SOLVEPNP_IPPE);
  if (!ok) {
    tools::logger()->debug("[Buff_Solver] reject obs by IPPE failure");
    return false;
  }

  if (pnp_mode_ == "4i") {
    // 4i: IPPE初值后靶面4点ITERATIVE精化，不引入R标共面干扰
    ok = cv::solvePnP(
      object_points, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_, true,
      cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
      tools::logger()->debug("[Buff_Solver] reject obs by 4pt ITERATIVE failure");
      return false;
    }
  } else if (pnp_mode_ == "5") {
    // 5PnP只额外加入当前模型直出的R中心；不要把它接到SP的灯臂下端点语义上。
    image_points.emplace_back(p.r_center);
    object_points.emplace_back(object_points_[6]);
    ok = cv::solvePnP(
      object_points, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_, true,
      cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
      tools::logger()->debug("[Buff_Solver] reject obs by ITERATIVE failure");
      return false;
    }
  }

  std::vector<cv::Point2f> reproj_r_points;
  cv::projectPoints(
    std::vector<cv::Point3f>{object_points_[6]}, rvec_, tvec_, camera_matrix_, distort_coeffs_,
    reproj_r_points);
  const double r_err = cv::norm(reproj_r_points.front() - p.r_center);
  // R标重投影误差仅5模式有意义，4/4i不依赖R标故不检查
  // if (pnp_mode_ == "5" && r_err > reprojection_error_threshold_px_) {
  //   tools::logger()->debug(
  //     "[Buff_Solver] R reproj err high {:.2f}px > {:.2f}px (keep obs)", r_err,
  //     reprojection_error_threshold_px_);
  // }
  // PNP几何中心存入独立字段；r_center保留检测器TRAD_R不被覆写
  p.pnp_r_center = reproj_r_points.front();

  // get R_buff2camera t_buff2camera
  Eigen::Vector3d t_buff2camera;
  cv::cv2eigen(tvec_, t_buff2camera);
  cv::Mat rmat;
  cv::Rodrigues(rvec_, rmat);
  Eigen::Matrix3d R_buff2camera;
  cv::cv2eigen(rmat, R_buff2camera);

  Eigen::Vector3d blade_xyz_in_buff{
    object_points_[4].x, object_points_[4].y, object_points_[4].z};

  // buff -> camera
  Eigen::Vector3d xyz_in_camera = t_buff2camera;
  Eigen::Vector3d blade_xyz_in_camera = R_buff2camera * blade_xyz_in_buff + t_buff2camera;

  // camera -> gimbal
  Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
  Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  Eigen::Vector3d blade_xyz_in_gimbal = R_camera2gimbal_ * blade_xyz_in_camera + t_camera2gimbal_;

  /// gimbal -> world
  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.R_buff2world = R_buff2world;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);


  // TRAD_R & target center world position: ray-plane intersection (PnP pose + pixel)
  // Buff panel in x=0 plane (all object_points_ have x=0), normal = buff x-axis -> col(0) of R_buff2camera
  {
    cv::Mat R_mat;
    cv::Rodrigues(rvec_, R_mat);
    auto ray_plane_intersect = [&](cv::Point2f px) -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> {
      cv::Vec3d n_c(R_mat.at<double>(0,0), R_mat.at<double>(1,0), R_mat.at<double>(2,0));
      cv::Vec3d t_c(tvec_[0], tvec_[1], tvec_[2]);
      double fx = camera_matrix_.at<double>(0,0), fy = camera_matrix_.at<double>(1,1);
      double cx = camera_matrix_.at<double>(0,2), cy = camera_matrix_.at<double>(1,2);
      cv::Vec3d d_c((px.x - cx) / fx, (px.y - cy) / fy, 1.0);
      double nd = n_c.dot(d_c);
      if (std::abs(nd) > 1e-6) {
        double depth = n_c.dot(t_c) / nd;
        if (depth > 0.1 && depth < 20.0) {
          cv::Vec3d P_c = depth * d_c;
          Eigen::Vector3d tc_cam(P_c[0], P_c[1], P_c[2]);
          Eigen::Vector3d tc_gimbal = R_camera2gimbal_ * tc_cam + t_camera2gimbal_;
          Eigen::Vector3d tc_world = R_gimbal2world_ * tc_gimbal;
          return {tc_cam, tc_gimbal, tc_world};
        }
      }
      // fallback: PnP origin
      return {xyz_in_camera, xyz_in_gimbal, R_gimbal2world_ * xyz_in_gimbal};
    };
    std::tie(p.trad_xyz_in_camera, p.trad_xyz_in_gimbal, p.trad_xyz_in_world) =
      ray_plane_intersect(p.r_center);
    std::tie(p.target_xyz_in_camera, p.target_xyz_in_gimbal, p.target_xyz_in_world) =
      ray_plane_intersect(p.target().center);
    p.target_ypd_in_world = tools::xyz2ypd(p.target_xyz_in_world);
  }

  p.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
  return true;
}

cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  // buff坐标系(单位:m)到像素坐标系
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

cv::Point2f Solver::world2pixel(const Eigen::Vector3d & xyz_in_world) const
{
  Eigen::Vector3d p_gimbal = R_gimbal2world_.transpose() * xyz_in_world;
  Eigen::Vector3d p_cam = R_camera2gimbal_.transpose() * (p_gimbal - t_camera2gimbal_);
  if (p_cam.z() < 1e-3) return { -1, -1 };
  std::vector<cv::Point3d> pc{{p_cam.x(), p_cam.y(), p_cam.z()}};
  std::vector<cv::Point2d> img;
  cv::Mat rvec_zero = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec_zero = cv::Mat::zeros(3, 1, CV_64F);
  cv::projectPoints(pc, rvec_zero, tvec_zero, camera_matrix_, distort_coeffs_, img);
  return img.back();
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, const Eigen::Vector3d & ypr_in_world) const
{
  const auto R_buff2world = tools::rotation_matrix(ypr_in_world);
  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  const Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  const Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points_, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  return reproject_buff(xyz_in_world, Eigen::Vector3d(yaw, 0.0, row));
}
}  // namespace auto_buff
