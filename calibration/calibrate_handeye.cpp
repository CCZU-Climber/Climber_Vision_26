/**
 * calibrate_handeye.cpp
 * 实时手眼标定（RobotWorldHandEye 方法，同步解算标定板世界位姿）
 *
 * Use Gimbal protocol for IMU quaternion input.
 *
 * 操作：
 *   s / 空格  ——  手动捕获（需检测到标定板且图像清晰）
 *   d         ——  删除最后一帧
 *   c         ——  清空所有帧
 *   r         ——  立即重新标定
 *   q / Esc   ——  输出 YAML 并退出
 *
 * 自动捕获（--auto）：
 *   图像稳定（角点位移 < --stable-thresh px）且云台角度变化 ≥ --min-angle 度时自动捕获。
 */

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{@config-path   | configs/calibration.yaml | yaml配置文件路径 }"
  "{min-samples n  |          15              | 开始标定所需最少样本数}"
  "{reproj-thresh  |          1.5             | 重投影误差离群点阈值（像素）}"
  "{auto           |                          | 启用自动捕获模式}"
  "{min-angle a    |          5.0             | 自动捕获：最小云台角度变化（度）}"
  "{stable-thresh  |          1.0             | 自动捕获：角点稳定阈值（像素）}"
  "{sharp-thresh  s|          20              | 清晰度阈值（拉普拉斯方差），低于此值认为模糊}";

// ─────────────────── 工具函数 ───────────────────

// 棋盘格角点坐标（中心化，x=0 平面，适配 RobotWorldHandEye）
std::vector<cv::Point3f> board_corners_rw(const cv::Size & sz, float sq_mm)
{
  std::vector<cv::Point3f> pts;
  for (int r = 0; r < sz.height; r++)
    for (int c = 0; c < sz.width; c++)
      pts.push_back({
        0.f,
        (-c + 0.5f * sz.width)  * sq_mm,
        (-r + 0.5f * sz.height) * sq_mm});
  return pts;
}

double sharpness(const cv::Mat & img)
{
  cv::Mat gray, lap;
  if (img.channels() == 3)
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  else
    gray = img;
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar m, s;
  cv::meanStdDev(lap, m, s);
  return s[0] * s[0];
}

double reproj_rms(
  const std::vector<cv::Point3f> & obj, const std::vector<cv::Point2f> & img,
  const cv::Mat & rvec, const cv::Mat & tvec,
  const cv::Matx33d & K, const cv::Mat & D)
{
  std::vector<cv::Point2f> proj;
  cv::projectPoints(obj, rvec, tvec, K, D, proj);
  double e = 0;
  for (size_t i = 0; i < img.size(); i++) e += cv::norm(img[i] - proj[i]);
  return e / img.size();
}

double corner_motion(
  const std::vector<cv::Point2f> & a, const std::vector<cv::Point2f> & b)
{
  if (a.size() != b.size() || a.empty()) return 1e9;
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) s += cv::norm(a[i] - b[i]);
  return s / a.size();
}

// ─────────────────── 标定数据结构 ───────────────────

struct Frame {
  cv::Mat R_world2gimbal;   // 3×3
  cv::Mat t_world2gimbal;   // 3×1 零向量
  cv::Mat rvec;             // PnP 旋转向量（board→camera）
  cv::Mat tvec;             // PnP 平移向量（单位 mm）
  double  reproj_err;
  Eigen::Vector3d ypr_deg;
};

// ─────────────────── RobotWorldHandEye 标定 ───────────────────

bool do_calibrate(
  const std::vector<Frame> & frames,
  cv::Mat & R_camera2gimbal, cv::Mat & t_camera2gimbal,
  cv::Mat & R_board2world,   cv::Mat & t_board2world)
{
  if (frames.size() < 4) return false;

  std::vector<cv::Mat> Rg, Tg, Rc, Tc;
  for (auto & f : frames) {
    Rg.push_back(f.R_world2gimbal);
    Tg.push_back(f.t_world2gimbal);
    Rc.push_back(f.rvec);
    Tc.push_back(f.tvec);
  }

  // calibrateRobotWorldHandEye 解：
  //   R_world2board, t_world2board  （标定板在 IMU 世界坐标系中的固定位姿）
  //   R_gimbal2camera, t_gimbal2camera  （我们要求的手眼变换）
  cv::Mat R_w2b, t_w2b, R_g2c, t_g2c;
  try {
    cv::calibrateRobotWorldHandEye(
      Rc, Tc,   // R/t: board→camera（solvePnP 输出）
      Rg, Tg,   // R/t: world→gimbal
      R_w2b, t_w2b,
      R_g2c, t_g2c,
      cv::CALIB_ROBOT_WORLD_HAND_EYE_SHAH);
  } catch (const cv::Exception & e) {
    tools::logger()->error("calibrateRobotWorldHandEye 异常: {}", e.what());
    return false;
  }

  // 求逆：gimbal→camera → camera→gimbal
  R_camera2gimbal = R_g2c.t();
  t_camera2gimbal = -R_camera2gimbal * t_g2c;

  // 求逆：world→board → board→world
  R_board2world = R_w2b.t();
  t_board2world = -R_board2world * t_w2b;

  return true;
}

// ─────────────────── 输出 YAML ───────────────────

void print_yaml(
  const std::vector<double> & R_g2imubody_data,
  const cv::Mat & R_c2g,    const cv::Mat & t_c2g,
  const cv::Mat & R_b2w,    const cv::Mat & t_b2w,
  const Eigen::Vector3d & camera_ypr)
{
  // 标定板到世界原点水平距离（诊断用）
  double bx = t_b2w.at<double>(0);
  double by = t_b2w.at<double>(1);
  double dist = std::sqrt(bx * bx + by * by);

  Eigen::Matrix3d R_b2w_e;
  cv::cv2eigen(R_b2w, R_b2w_e);
  Eigen::Vector3d board_ypr = tools::eulers(R_b2w_e, 2, 1, 0) * 57.3;

  YAML::Emitter out;
  std::vector<double> Rv(R_c2g.begin<double>(), R_c2g.end<double>());
  std::vector<double> tv(t_c2g.begin<double>(), t_c2g.end<double>());

  out << YAML::BeginMap;
  out << YAML::Key << "R_gimbal2imubody" << YAML::Value << YAML::Flow << R_g2imubody_data;
  out << YAML::Newline << YAML::Newline;
  out << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree",
    camera_ypr[0], camera_ypr[1], camera_ypr[2]));
  out << YAML::Newline;
  out << YAML::Comment(fmt::format(
    "标定板到世界原点水平距离: {:.2f} m", dist));
  out << YAML::Newline;
  out << YAML::Comment(fmt::format(
    "标定板同竖直摆放时的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree",
    board_ypr[0], board_ypr[1], board_ypr[2]));
  out << YAML::Key << "R_camera2gimbal" << YAML::Value << YAML::Flow << Rv;
  out << YAML::Key << "t_camera2gimbal" << YAML::Value << YAML::Flow << tv;
  out << YAML::Newline;
  out << YAML::EndMap;

  fmt::print("\n========== 手眼标定结果 ==========\n{}\n==================================\n",
             out.c_str());
}

// ─────────────────── 主函数 ───────────────────

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) { cli.printMessage(); return 0; }

  auto config_path   = cli.get<std::string>(0);
  auto min_samples   = cli.get<int>("min-samples");
  auto reproj_thresh = cli.get<double>("reproj-thresh");
  bool auto_mode     = cli.has("auto");
  double min_angle   = cli.get<double>("min-angle");
  double stable_thr  = cli.get<double>("stable-thresh");
  double sharp_thresh = cli.get<double>("sharp-thresh");

  // ---- 加载配置 ----
  auto yaml_cfg = YAML::LoadFile(config_path);
  auto pattern_cols        = yaml_cfg["pattern_cols"].as<int>();
  auto pattern_rows        = yaml_cfg["pattern_rows"].as<int>();
  auto center_distance_mm  = yaml_cfg["center_distance_mm"].as<float>();
  auto pattern_type        = yaml_cfg["pattern_type"].as<std::string>();
  auto R_g2imubody_data    = yaml_cfg["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data  = yaml_cfg["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml_cfg["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_g2imubody_data.data());
  cv::Matx33d K(camera_matrix_data.data());
  cv::Mat D(distort_coeffs_data);
  auto obj_pts = board_corners_rw(pattern_size, center_distance_mm);

  const cv::Size subpix_win(11, 11);
  const cv::TermCriteria subpix_crit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.01);

  // ---- 初始化通信协议（IMU 四元数源） ----
  std::function<Eigen::Quaterniond(std::chrono::steady_clock::time_point)> get_imu;
  std::shared_ptr<io::Gimbal> gimbal_ptr;

  gimbal_ptr = std::make_shared<io::Gimbal>(config_path);
  get_imu = [&](auto t) { return gimbal_ptr->q(t); };
  tools::logger()->info("Protocol: Gimbal");

  // ---- 初始化相机 ----
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  tools::logger()->info(
    "Hand-eye calibration pattern:{} {}x{} protocol:gimbal auto:{}",
    pattern_type, pattern_cols, pattern_rows, auto_mode ? "ON" : "OFF");
  tools::logger()->info(
    "[s/空格]手动捕获  [d]删末帧  [c]清空  [r]重标定  [q/Esc]输出并退出");

  // ---- 状态变量 ----
  std::vector<Frame> frames;
  cv::Mat R_c2g, t_c2g, R_b2w, t_b2w;
  bool calib_valid = false;
  std::vector<cv::Point2f> prev_corners;
  Eigen::Vector3d last_ypr = Eigen::Vector3d::Zero();
  int frame_idx = 0;

  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = get_imu(timestamp);

    // ---- 计算云台旋转矩阵 ----
    Eigen::Matrix3d R_imubody2world = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2world * R_gimbal2imubody;
    Eigen::Matrix3d R_world2gimbal = R_gimbal2world.transpose();
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;

    std::vector<cv::Point2f> corners;
    bool detected = false;
    double sharp = 0;
    bool blurry = true;
    double motion = 1e9;
    bool do_detect = (frame_idx % 2 == 0);

    if (do_detect) {
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      if (pattern_type == "circles") {
        detected = cv::findCirclesGrid(img, pattern_size, corners, cv::CALIB_CB_SYMMETRIC_GRID);
      } else {
        detected = cv::findChessboardCorners(
          gray, pattern_size, corners,
          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        if (detected)
          cv::cornerSubPix(gray, corners, subpix_win, {-1, -1}, subpix_crit);
      }
      sharp = sharpness(img);
      blurry = sharp < sharp_thresh;
      motion = detected ? corner_motion(prev_corners, corners) : 1e9;
      if (detected) prev_corners = corners;
    }
    bool stable = detected && !blurry && (motion < stable_thr);
    frame_idx++;

    // ---- 自动捕获判断 ----
    bool do_capture = false;
    if (auto_mode && stable) {
      double angle_diff = (ypr - last_ypr).norm();
      if (angle_diff >= min_angle) do_capture = true;
    }

    // ---- 绘制 UI (底部，不遮挡画面) ----
    const int ui_h = 90;
    cv::Mat display;
    cv::vconcat(img, cv::Mat(ui_h, img.cols, CV_8UC3, cv::Scalar(20, 20, 20)), display);
    if (detected)
      cv::drawChessboardCorners(display, pattern_size, corners, true);

    // IMU 姿态
    int base_y = img.rows + 22;
    tools::draw_text(display, fmt::format("YAW  {:+7.2f}", ypr[0]),   {10, base_y}, {100, 220, 100}, 0.8);
    tools::draw_text(display, fmt::format("PITCH{:+7.2f}", ypr[1]),   {220, base_y}, {100, 220, 100}, 0.8);
    tools::draw_text(display, fmt::format("ROLL {:+7.2f}", ypr[2]),   {450, base_y}, {100, 220, 100}, 0.8);

    // 检测状态
    std::string det_str;
    cv::Scalar  det_col;
    if (!detected) { det_str = "NO BOARD"; det_col = {0, 80, 255}; }
    else if (blurry) { det_str = fmt::format("BLURRY({:.0f})", sharp); det_col = {0,0,255}; }
    else if (!stable) { det_str = fmt::format("MOVING({:.1f})", motion); det_col = {0,200,255}; }
    else { det_str = fmt::format("STABLE({:.0f})", sharp); det_col = {0,255,0}; }
    tools::draw_text(display, det_str, {750, base_y}, det_col, 0.8);

    // 帧数 + 标定状态
    tools::draw_text(display,
      fmt::format("frames:{:2d}/{:2d}  {}", frames.size(), min_samples, calib_valid ? "CALIB OK" : "waiting..."),
      {10, base_y + 28},
      calib_valid ? cv::Scalar{0, 255, 128} : cv::Scalar{180, 180, 0}, 0.8);

    // 当前标定结果
    if (calib_valid) {
      Eigen::Matrix3d Re;
      cv::cv2eigen(R_c2g, Re);
      Eigen::Matrix3d R_gimbal2ideal{{0,-1,0},{0,0,-1},{1,0,0}};
      auto dypr = tools::eulers(R_gimbal2ideal * Re, 1, 0, 2) * 57.3;
      tools::draw_text(display,
        fmt::format("cam_yaw:{:.1f} pitch:{:.1f} roll:{:.1f}", dypr[0], dypr[1], dypr[2]),
        {10, base_y + 56}, {255, 200, 50}, 0.7);
    }

    tools::draw_text(display, "[s]cap [d]del [c]clr [r]calib [q]quit", {10, base_y + 80}, {130, 130, 130}, 0.7);

    cv::Mat small;
    cv::resize(display, small, {}, 0.5, 0.5);
    cv::imshow("Hand-Eye Calibration", small);
    int key = cv::waitKey(1);

    // ---- 键盘响应 ----
    if (key == 'q' || key == 27) break;

    if (key == 'd' && !frames.empty()) {
      frames.pop_back();
      calib_valid = false;
      tools::logger()->info("已删除最后一帧，剩余 {}", frames.size());
      continue;
    }
    if (key == 'c') {
      frames.clear();
      calib_valid = false;
      tools::logger()->info("已清空所有帧");
      continue;
    }
    if (key == 'r') {
      calib_valid = do_calibrate(frames, R_c2g, t_c2g, R_b2w, t_b2w);
      tools::logger()->info(calib_valid ? "重标定完成" : "标定失败（帧数不足或数据质量差）");
      continue;
    }

    bool manual = (key == 's' || key == ' ');
    if (!manual && !do_capture) continue;

    // ---- 捕获帧 ----
    if (!detected) { tools::logger()->warn("未检测到标定板，跳过"); continue; }
    if (blurry)    { tools::logger()->warn("图像模糊 (sharp={:.0f})，跳过", sharp); continue; }

    // PnP 求解
    cv::Mat rvec, tvec;
    cv::solvePnP(obj_pts, corners, K, D, rvec, tvec, false, cv::SOLVEPNP_UPNP);
    double err = reproj_rms(obj_pts, corners, rvec, tvec, K, D);
    if (err > reproj_thresh) {
      tools::logger()->warn("重投影误差过大 ({:.2f}px)，跳过", err);
      //continue;
    }

    // 构建 Frame
    Frame f;
    cv::eigen2cv(R_world2gimbal, f.R_world2gimbal);
    f.t_world2gimbal = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    f.rvec       = rvec;
    f.tvec       = tvec;
    f.reproj_err = err;
    f.ypr_deg    = ypr;
    frames.push_back(f);
    last_ypr = ypr;

    tools::logger()->info("捕获第 {:2d} 帧  reproj={:.2f}px", frames.size(), err);

    // 达到最小帧数后自动更新标定
    if ((int)frames.size() >= min_samples && frames.size() % 3 == 0) {
      calib_valid = do_calibrate(frames, R_c2g, t_c2g, R_b2w, t_b2w);
      if (calib_valid)
        tools::logger()->info("在线标定已更新（{} 帧）", frames.size());
    }
  }

  // ---- 退出时输出结果 ----
  if (!calib_valid && (int)frames.size() >= 4) {
    calib_valid = do_calibrate(frames, R_c2g, t_c2g, R_b2w, t_b2w);
  }

  if (!calib_valid) {
    tools::logger()->warn("标定未完成（有效帧 {} < 最小 {} 帧）", frames.size(), min_samples);
    return 1;
  }

  // t 单位 mm → m
  cv::Mat t_c2g_m = t_c2g / 1e3;
  cv::Mat t_b2w_m = t_b2w / 1e3;

  // 计算相机偏角
  Eigen::Matrix3d R_c2g_e;
  cv::cv2eigen(R_c2g, R_c2g_e);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  Eigen::Vector3d camera_ypr = tools::eulers(R_gimbal2ideal * R_c2g_e, 1, 0, 2) * 57.3;

  print_yaml(R_g2imubody_data, R_c2g, t_c2g_m, R_b2w, t_b2w_m, camera_ypr);
  return 0;
}
