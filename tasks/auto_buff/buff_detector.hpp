#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
namespace auto_buff
{
class Buff_Detector
{
public:
  Buff_Detector(const std::string & config);

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  std::vector<PowerRune> detect(cv::Mat & bgr_img, PowerRune_type rune_type);

  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  void handle_lose();

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  int binary_threshold_ = 100;     // 二值化阈值
  int dilate_kernel_size_ = 5;       // 膨胀核大小
  int dilate_iterations_ = 1;        // 膨胀迭代次数
  double r_center_extrapolate_ = 1.4; // R标中心外推因子
  double r_center_mask_ratio_ = 0.8;  // mask半径/扇叶半径比例
  double r_center_contour_div_ = 3.0; // 轮廓距离惩罚除数
  double r_center_refine_max_shift_ = 5.0; // 传统细化相对模型R的最大修正量 px
  double dedup_distance_ = 18.0;      // 候选去重距离 px
  double small_assoc_max_jump_px_ = 140.0;  // 小符目标中心跨帧最大跳变
  int lose_max_ = 20;                // 连续丢失阈值
  double lastlen_;
  std::optional<PowerRune> last_powerrune_ = std::nullopt;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
