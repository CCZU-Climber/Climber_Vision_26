#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "io/usbcamera/usbcamera.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{@config-path   | configs/calibration.yaml      | yaml配置文件路径（包含USB相机配置）}";

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});
  return centers_3d;
}
// 输出标定结果为YAML格式
void print_yaml(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double error)
{
  YAML::Emitter result;
  std::vector<double> camera_matrix_data(
    camera_matrix.begin<double>(), camera_matrix.end<double>());
  std::vector<double> distort_coeffs_data(
    distort_coeffs.begin<double>(), distort_coeffs.end<double>());

  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
  result << YAML::Key << "camera_matrix";
  result << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs";
  result << YAML::Value << YAML::Flow << distort_coeffs_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

std::pair<double, double> compute_fov_deg(const cv::Mat & camera_matrix, const cv::Size & img_size)
{
  double fx = camera_matrix.at<double>(0, 0);
  double fy = camera_matrix.at<double>(1, 1);

  if (fx <= 0 || fy <= 0 || img_size.width <= 0 || img_size.height <= 0) return {0.0, 0.0};

  double fov_h = 2.0 * std::atan(img_size.width / (2.0 * fx)) * 180.0 / CV_PI;
  double fov_v = 2.0 * std::atan(img_size.height / (2.0 * fy)) * 180.0 / CV_PI;
  return {fov_h, fov_v};
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  // 读取YAML配置，列举可用的USB相机
  auto yaml = YAML::LoadFile(config_path);
  auto camera_name_map = yaml["camera_name_map"];
  
  if (!camera_name_map) {
    tools::logger()->error("camera_name_map not found in {}", config_path);
    return -1;
  }

  std::vector<std::string> camera_names;
  for (auto it = camera_name_map.begin(); it != camera_name_map.end(); ++it) {
    camera_names.push_back(it->first.as<std::string>());
  }

  if (camera_names.empty()) {
    tools::logger()->error("No cameras configured in camera_name_map");
    return -1;
  }

  auto format_camera_label = [](const YAML::Node & node) {
    try {
      if (node.IsScalar()) return fmt::format("{:.2f} deg", node.as<double>());
      if (node.IsMap() && node["yaw"]) return fmt::format("{:.2f} deg", node["yaw"].as<double>());
    } catch (...) {
      return std::string("invalid");
    }
    return std::string("unknown");
  };

  // 列举相机供用户选择（显示序号 + 名字）
  tools::logger()->info("Available USB cameras:");
  for (size_t i = 0; i < camera_names.size(); i++) {
    auto cn = format_camera_label(camera_name_map[camera_names[i]]);
    fmt::print("  [{}] {} ({})\n", i + 1, camera_names[i], cn);
  }

  // 输入数字索引选择，默认选第一个
  size_t camera_idx = 0;
  fmt::print("Select camera [1-{}] (default 1): ", camera_names.size());
  std::string input;
  std::getline(std::cin, input);
  if (!input.empty()) {
    try {
      camera_idx = std::stoul(input);
      camera_idx = (camera_idx >= 1 && camera_idx <= camera_names.size()) ? (camera_idx - 1) : 0;
    } catch (...) {
      camera_idx = 0;
    }
  }

  auto selected_camera = camera_names[camera_idx];

  auto camera_label = format_camera_label(camera_name_map[selected_camera]);
  tools::logger()->info("Selected camera: {} ({})", selected_camera, camera_label);

  // 初始化相机
  std::unique_ptr<io::USBCamera> camera;
  try {
    camera = std::make_unique<io::USBCamera>(selected_camera, config_path);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    if (!camera->is_initialized()) {
      tools::logger()->error("Camera failed to initialize");
      return -1;
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Camera initialization failed: {}", e.what());
    return -1;
  }

  // 读取标定板参数
  auto pattern_cols = yaml["pattern_cols"].as<int>(11);
  auto pattern_rows = yaml["pattern_rows"].as<int>(8);
  auto center_distance_mm = yaml["center_distance_mm"].as<double>(30);
  auto pattern_type = yaml["pattern_type"].as<std::string>("chessboard");
  cv::Size pattern_size(pattern_cols, pattern_rows);

  tools::logger()->info("Pattern: {}x{}, distance: {}mm, type: {}", 
    pattern_cols, pattern_rows, center_distance_mm, pattern_type);
  tools::logger()->info("Instructions:");
  fmt::print("  'c' or SPACE  - Capture frame for calibration\n");
  fmt::print("  'q' or ESC    - Finish and calibrate\n");
  fmt::print("  's'           - Skip (don't use current frame)\n");

  cv::Size img_size;
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> img_points;
  int frame_count = 0;
  int capture_count = 0;

  while (true) {
    cv::Mat img;
    std::chrono::steady_clock::time_point ts;
    camera->read(img, ts);

    if (img.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    img_size = img.size();
    cv::Mat display = img.clone();

    // Ensure grayscale for detection APIs that require single channel input
    cv::Mat gray;
    if (img.channels() == 3)
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
      cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
      gray = img;

    // 检测棋盘
    std::vector<cv::Point2f> centers_2d;
    bool success = false;
    if (pattern_type == "circles") {
      success = cv::findCirclesGrid(gray, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);
    } else {
      success = cv::findChessboardCorners(gray, pattern_size, centers_2d);
      if (success) {
        cv::cornerSubPix(gray, centers_2d, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
      }
    }

    cv::drawChessboardCorners(display, pattern_size, centers_2d, success);

    // 显示信息
    std::string status = success ? "Pattern FOUND" : "Pattern NOT found";
    cv::Scalar color = success ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    tools::draw_text(display, status, cv::Point(10, 30), color);
    tools::draw_text(display, fmt::format("Captured: {}", capture_count), 
      cv::Point(10, 60), cv::Scalar(255, 255, 0));
    tools::draw_text(display, "Press 'c' to capture, 'q' to finish", 
      cv::Point(10, 90), cv::Scalar(200, 200, 200));

    cv::imshow("USB Camera Calibration", display);
    int key = cv::waitKey(30) & 0xFF;

    if (key == 'q' || key == 27) {  // ESC
      break;
    } else if (key == 'c' || key == ' ') {  // SPACE or 'c'
      if (!success) {
        tools::logger()->warn("Pattern not detected in current frame, skipping");
        continue;
      }
      img_points.push_back(centers_2d);
      obj_points.push_back(centers_3d(pattern_size, center_distance_mm));
      capture_count++;
      tools::logger()->info("Captured frame {} for calibration", capture_count);
    } else if (key == 's') {  // skip
      tools::logger()->info("Skipped current frame");
    }

    frame_count++;
  }

  cv::destroyAllWindows();

  if (img_points.empty()) {
    tools::logger()->error("No valid calibration frames captured");
    return -1;
  }

  tools::logger()->info("Calibrating with {} frames...", capture_count);

  // 相机标定
  cv::Mat camera_matrix, distort_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  auto criteria = cv::TermCriteria(
    cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
  cv::calibrateCamera(
    obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs, 
    cv::CALIB_FIX_K3, criteria);

  // 重投影误差
  double error_sum = 0;
  size_t total_points = 0;
  for (size_t i = 0; i < obj_points.size(); i++) {
    std::vector<cv::Point2f> reprojected_points;
    cv::projectPoints(
      obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);

    total_points += reprojected_points.size();
    for (size_t j = 0; j < reprojected_points.size(); j++)
      error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
  }
  auto error = error_sum / total_points;

  // 输出结果
  tools::logger()->info("Calibration complete. Reprojection error: {:.4f}px", error);
  fmt::print("\n=== Calibration Results for {} ({}) ===\n", selected_camera, camera_label);
  auto [fov_h, fov_v] = compute_fov_deg(camera_matrix, img_size);
  tools::logger()->info("Estimated FOV: {:.2f} deg (H) x {:.2f} deg (V)", fov_h, fov_v);
  fmt::print("Estimated FOV: {:.2f} deg (H) x {:.2f} deg (V)\n", fov_h, fov_v);
  print_yaml(camera_matrix, distort_coeffs, error);

  return 0;
}