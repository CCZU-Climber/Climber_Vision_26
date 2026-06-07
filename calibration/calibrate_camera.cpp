/**
 * calibrate_camera.cpp
 * 实时内参标定：打开相机视频流，检测标定板后按键采集帧，采集完后自动标定输出 YAML。
 *
 * 操作：
 *   s / 空格  ——  捕获当前帧（需检测到标定板且图像清晰）
 *   d         ——  删除最后一帧
 *   q / Esc   ——  标定并输出 YAML 后退出
 */

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{@config-path   | configs/calibration.yaml | yaml配置文件路径 }"
  "{sharp-thresh s |          20              | 清晰度阈值（拉普拉斯方差），低于此值认为模糊}";

// ─────────────────── 工具 ───────────────────

std::vector<cv::Point3f> board_corners(const cv::Size & sz, float sq_mm)
{
  std::vector<cv::Point3f> pts;
  for (int r = 0; r < sz.height; r++)
    for (int c = 0; c < sz.width; c++)
      pts.push_back({c * sq_mm, r * sq_mm, 0.f});
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

void print_yaml(const cv::Mat & K, const cv::Mat & D, double err)
{
  YAML::Emitter out;
  std::vector<double> Kv(K.begin<double>(), K.end<double>());
  std::vector<double> Dv(D.begin<double>(), D.end<double>());
  out << YAML::BeginMap;
  out << YAML::Comment(fmt::format("重投影误差: {:.4f}px", err));
  out << YAML::Key << "camera_matrix" << YAML::Value << YAML::Flow << Kv;
  out << YAML::Key << "distort_coeffs" << YAML::Value << YAML::Flow << Dv;
  out << YAML::Newline << YAML::EndMap;
  fmt::print("\n{}\n", out.c_str());
}

// ─────────────────── 主函数 ───────────────────

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  // ---- 加载标定板参数 ----
  auto yaml_cfg = YAML::LoadFile(config_path);
  auto pattern_cols        = yaml_cfg["pattern_cols"].as<int>();
  auto pattern_rows        = yaml_cfg["pattern_rows"].as<int>();
  auto center_distance_mm  = yaml_cfg["center_distance_mm"].as<float>();
  auto pattern_type        = yaml_cfg["pattern_type"].as<std::string>();
  cv::Size pattern_size(pattern_cols, pattern_rows);
  auto obj_pts_tmpl = board_corners(pattern_size, center_distance_mm);

  const double sharp_thresh = cli.get<double>("sharp-thresh");
  const cv::Size subpix_win(5, 5);  // 改小亚像素窗口，防止较远时抓到相邻角点
  const cv::TermCriteria subpix_crit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.01);

  tools::logger()->info(
    "内参标定  pattern:{} {}x{}  [s/空格]捕获  [d]删末帧  [q/Esc]标定退出",
    pattern_type, pattern_cols, pattern_rows);

  // ---- 打开相机 ----
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point ts;

  std::vector<std::vector<cv::Point3f>> all_obj_pts;
  std::vector<std::vector<cv::Point2f>> all_img_pts;
  cv::Size img_size;
  int frame_idx = 0;

  while (true) {
    camera.read(img, ts);
    img_size = img.size();

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    bool detected = false;
    double sharp = 0;
    bool blurry = true;
    bool do_detect = (frame_idx % 2 == 0);

    if (do_detect) {
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
    }
    frame_idx++;

    // ---- 绘制 UI (底部，不遮挡画面) ----
    const int ui_h = 70;
    cv::Mat display;
    cv::vconcat(img, cv::Mat(ui_h, img.cols, CV_8UC3, cv::Scalar(20, 20, 20)), display);
    if (detected)
      cv::drawChessboardCorners(display, pattern_size, corners, true);

    int base_y = img.rows + 22;
    cv::Scalar det_color = detected ? (blurry ? cv::Scalar{0, 0, 255} : cv::Scalar{0, 255, 0})
                                    : cv::Scalar{0, 100, 255};
    std::string det_str  = detected ? (blurry ? "BLURRY" : "DETECTED") : "NO BOARD";
    tools::draw_text(display, det_str, {10, base_y}, det_color, 0.8);
    tools::draw_text(display, fmt::format("frames:{:2d}  sharp:{:.0f}", all_obj_pts.size(), sharp),
      {200, base_y}, {200, 200, 0}, 0.8);
    tools::draw_text(display, "[s]capture [d]del [q]calibrate&quit", {10, base_y + 35}, {150, 150, 150}, 0.7);

    cv::Mat small;
    cv::resize(display, small, {}, 0.5, 0.5);
    cv::imshow("Intrinsic Calibration", small);
    int key = cv::waitKey(1);

    if (key == 'q' || key == 27) break;

    if (key == 'd') {
      if (!all_obj_pts.empty()) {
        all_obj_pts.pop_back();
        all_img_pts.pop_back();
        tools::logger()->info("已删除最后一帧，剩余 {} 帧", all_obj_pts.size());
      }
      continue;
    }

    if (key != 's' && key != ' ') continue;

    // ---- 保存帧 ----
    if (!detected) { tools::logger()->warn("未检测到标定板，跳过"); continue; }
    if (blurry)    { tools::logger()->warn("图像模糊 (sharp={:.0f})，跳过", sharp); continue; }

    all_obj_pts.push_back(obj_pts_tmpl);
    all_img_pts.push_back(corners);
    tools::logger()->info("捕获第 {} 帧", all_obj_pts.size());
  }

  // ---- 标定 ----
  if (all_obj_pts.size() < 5) {
    tools::logger()->error("有效帧数不足（{} 帧），至少需要 5 帧", all_obj_pts.size());
    return 1;
  }

  tools::logger()->info("正在标定（{} 帧）...", all_obj_pts.size());
  cv::Mat K, D;
  std::vector<cv::Mat> rvecs, tvecs;
  auto crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-6);
  cv::calibrateCamera(
    all_obj_pts, all_img_pts, img_size, K, D, rvecs, tvecs,
    cv::CALIB_FIX_K3,  // 视场角较小，无需 k3
    crit);

  // ---- 重投影误差 ----
  double err_sum = 0;
  size_t total = 0;
  for (size_t i = 0; i < all_obj_pts.size(); i++) {
    std::vector<cv::Point2f> proj;
    cv::projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, D, proj);
    total += proj.size();
    for (size_t j = 0; j < proj.size(); j++)
      err_sum += cv::norm(all_img_pts[i][j] - proj[j]);
  }
  double err = err_sum / total;

  tools::logger()->info("标定完成，重投影误差: {:.4f}px", err);
  print_yaml(K, D, err);
  return 0;
}
