#include "buff_detector.hpp"

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
constexpr int kSlotCount = 5;
constexpr double kFaceAngle = 2.0 * CV_PI / kSlotCount;

void assign_slot(PowerRune & powerrune)
{
  if (powerrune.fanblades.empty()) {
    powerrune.slot_id = -1;
    powerrune.slot_angle = 0.0;
    return;
  }

  const cv::Point2f v = powerrune.target().center - powerrune.r_center;
  if (!std::isfinite(v.x) || !std::isfinite(v.y) || cv::norm(v) < 1e-3) {
    powerrune.slot_id = -1;
    powerrune.slot_angle = 0.0;
    return;
  }

  double angle = std::atan2(-v.y, v.x);
  if (angle < 0.0) angle += 2.0 * CV_PI;
  int slot_id = static_cast<int>(std::round(angle / kFaceAngle)) % kSlotCount;
  if (slot_id < 0) slot_id += kSlotCount;

  powerrune.slot_id = slot_id;
  powerrune.slot_angle = angle;
}
}  // namespace

Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0), MODE_(config)
{
  auto yaml = YAML::LoadFile(config);
  if (yaml["binary_threshold"]) binary_threshold_ = yaml["binary_threshold"].as<int>();
  if (yaml["dilate_kernel_size"]) dilate_kernel_size_ = yaml["dilate_kernel_size"].as<int>();
  if (yaml["dilate_iterations"]) dilate_iterations_ = yaml["dilate_iterations"].as<int>();
  if (yaml["r_center_extrapolate"]) r_center_extrapolate_ = yaml["r_center_extrapolate"].as<double>();
  if (yaml["r_center_mask_ratio"]) r_center_mask_ratio_ = yaml["r_center_mask_ratio"].as<double>();
  if (yaml["r_center_contour_div"]) r_center_contour_div_ = yaml["r_center_contour_div"].as<double>();
  if (yaml["r_center_refine_max_shift"])
    r_center_refine_max_shift_ = yaml["r_center_refine_max_shift"].as<double>();
  if (yaml["dedup_distance"]) dedup_distance_ = yaml["dedup_distance"].as<double>();
  if (yaml["small_assoc_max_jump_px"])
    small_assoc_max_jump_px_ = yaml["small_assoc_max_jump_px"].as<double>();
  if (yaml["lose_max"]) lose_max_ = yaml["lose_max"].as<int>();
}

void Buff_Detector::handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img)
{
  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
  // cv::imshow("gray", gray_img);  // 调试用

  // 进行二值化           :把高于设定阈值变成255，低于变成0
  // 二值化：高于阈值置255，低于阈值置0
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, binary_threshold_, 255, cv::THRESH_BINARY);
  // cv::imshow("binary", binary_img);  // 调试用

  // 膨胀
  // 膨胀，增强连通区域
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_kernel_size_, dilate_kernel_size_));
  cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), dilate_iterations_);
  // cv::imshow("Dilated Image", dilated_img);  // 调试用
}

cv::Point2f Buff_Detector::get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img)
{
  if (fanblades.empty()) {
    /// error
    tools::logger()->debug("[Buff_Detector] 无法计算r_center!");
    return {0, 0};
  }

  // 新模型直出R标中心，直接取kpt[5]均值作为初值
  cv::Point2f r_center_t = {0, 0};
  for (auto & fanblade : fanblades) {
    r_center_t += fanblade.points[5];
  }
  r_center_t /= static_cast<float>(fanblades.size());

  /// 处理图片,mask选出大概范围
  // 在粗估中心附近做mask约束，再找轮廓细化中心
  cv::Mat dilated_img;
  handle_img(bgr_img, dilated_img);
  double radius = cv::norm(fanblades[0].points[2] - fanblades[0].center) * r_center_mask_ratio_;
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U);
  circle(mask, r_center_t, radius, cv::Scalar(255), -1);
  bitwise_and(dilated_img, mask, dilated_img);
  tools::draw_point(bgr_img, r_center_t, {255, 255, 0}, 5);  // 调试用
  // cv::imshow("Dilated Image", dilated_img);                // 调试用

  /// 获取轮廓点,矩阵框筛选  TODO
  std::vector<std::vector<cv::Point>> contours;
  auto r_center = r_center_t;
  cv::findContours(dilated_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  double ratio_1 = INF;
  for (auto & it : contours) {
    auto rotated_rect = cv::minAreaRect(it);
    double ratio = rotated_rect.size.height > rotated_rect.size.width
                     ? rotated_rect.size.height / rotated_rect.size.width
                     : rotated_rect.size.width / rotated_rect.size.height;
    ratio += cv::norm(rotated_rect.center - r_center_t) / (radius / r_center_contour_div_);
    if (ratio < ratio_1) {
      ratio_1 = ratio;
      r_center = rotated_rect.center;
    }
  }
  const cv::Point2f correction = r_center - r_center_t;
  const double correction_norm = cv::norm(correction);
  if (correction_norm > r_center_refine_max_shift_ && correction_norm > 1e-6) {
    r_center = r_center_t + correction * static_cast<float>(r_center_refine_max_shift_ / correction_norm);
  }
  return r_center;
}

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= lose_max_) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
  }
  status_ = TEM_LOSE;
}

std::vector<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img, PowerRune_type rune_type)
{
  /// onnx 模型检测
  // 小符：单候选；大符：多候选
  std::vector<YOLO11_BUFF::Object> results;
  if (rune_type == BIG)
    results = MODE_.get_multicandidateboxes(bgr_img);
  else
    results = MODE_.get_onecandidatebox(bgr_img);

  if (results.empty()) {
    /// 处理未获得的情况
    handle_lose();
    return {};
  }

  // 先用全部候选粗估r中心
  std::vector<FanBlade> all_fanblades;
  all_fanblades.reserve(results.size());
  for (auto & result : results) {
    /// results转扇叶FanBlade
    all_fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }

  // 计算r_center,筛选fanblade
  auto r_center = get_r_center(all_fanblades, bgr_img);

  // 置信度降序，优先保留高质量候选
  std::sort(results.begin(), results.end(), [](const YOLO11_BUFF::Object & a, const YOLO11_BUFF::Object & b) {
    return a.prob > b.prob;
  });


  // 小符关联约束：优先选择与上一帧target中心连续的候选。
  if (rune_type == SMALL && last_powerrune_.has_value() && !results.empty()) {
    const auto last_center = last_powerrune_.value().target().center;
    std::stable_sort(results.begin(), results.end(), [&](const YOLO11_BUFF::Object & a, const YOLO11_BUFF::Object & b) {
      return cv::norm(a.kpt[4] - last_center) < cv::norm(b.kpt[4] - last_center);
    });

    const double best_jump = cv::norm(results.front().kpt[4] - last_center);
    if (best_jump > small_assoc_max_jump_px_) {
      tools::logger()->debug(
        "[Buff_Detector] skip SMALL obs by jump {:.1f}px > {:.1f}px", best_jump,
        small_assoc_max_jump_px_);
      handle_lose();
      return {};
    }
  }

  std::vector<PowerRune> powerrunes;
  std::vector<cv::Point2f> kept_centers;
  const size_t max_candidates = rune_type == BIG ? 2 : 1;

  for (const auto & result : results) {
    // 简单去重：中心太近当作重复框
    bool too_close = false;
    for (const auto & c : kept_centers) {
      if (cv::norm(result.kpt[4] - c) < static_cast<float>(dedup_distance_)) {
        too_close = true;
        break;
      }
    }
    if (too_close) continue;

    std::vector<FanBlade> fanblades;
    /// 生成PowerRune
    fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
    PowerRune powerrune(fanblades, r_center, last_powerrune_);
    if (powerrune.is_unsolve()) continue;
    assign_slot(powerrune);

    kept_centers.emplace_back(result.kpt[4]);
    powerrunes.emplace_back(powerrune);
    if (powerrunes.size() >= max_candidates) break;
  }

  if (powerrunes.empty()) {
    /// handle error
    handle_lose();
    return {};
  }

  status_ = TRACK;
  // tools::logger()->debug("[Buff_Detector] 检测成功 conf={:.2f}", results[0].prob);
  lose_ = 0;
  // 兼容旧状态机：仅记录一个last_powerrune
  last_powerrune_.emplace(powerrunes.front());
  return powerrunes;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  auto power_runes = detect(bgr_img, BIG);
  if (power_runes.empty()) return std::nullopt;
  return power_runes.front();
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  auto power_runes = detect(bgr_img, SMALL);
  if (power_runes.empty()) return std::nullopt;
  return power_runes.front();
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f v)
{
  /// onnx 模型检测
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);

  if (results.empty()) return std::nullopt;

  std::vector<FanBlade> fanblades_t;
  for (auto & result : results) {
    fanblades_t.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));
  }

  // 计算r_center后按偏移筛选一个扇叶（调试模式）
  auto r_center = get_r_center(fanblades_t, bgr_img);
  std::vector<FanBlade> fanblades;
  for (auto & fanblade : fanblades_t) {
    if (cv::norm((fanblade.center - r_center) - v) < 10 || results.size() == 1) {
      fanblades.emplace_back(fanblade);
      break;
    }
  }
  if (fanblades.empty()) return std::nullopt;
  PowerRune powerrune(fanblades, r_center, std::nullopt);
  assign_slot(powerrune);

  std::optional<PowerRune> P;
  P.emplace(powerrune);
  return P;
}

}  // namespace auto_buff
