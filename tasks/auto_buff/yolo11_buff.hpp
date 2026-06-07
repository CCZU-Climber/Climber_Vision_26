#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "tools/logger.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"red", "blue"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
  };

  YOLO11_BUFF(const std::string & config);

  // 使用NMS，用来获取多个框
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 寻找置信度最高的框
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

  // 设置敌方颜色，能量机关自动取反（打我方的）
  void set_enemy_color(const std::string & color);

private:
  double ConfidenceThreshold = 0.2;
  double IouThreshold = 0.2;
  ov::Core core;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  ov::Tensor input_tensor;
  static const int NUM_POINTS = 9;     // raw model output keypoints
  static const int REMAPPED_KPTS = 6;  // remapped to old 6-pt layout for solver compatibility
  static const int NUM_CLASSES = 2;
  static const int KPT_DIM = 3;          // x, y, visibility per keypoint
  static const int KPT_START = 6;        // keypoints start at row 6 (after 4 bbox + 2 class)
  int enemy_class_idx_ = 0;             // 0=red, 1=blue
  bool use_legacy_preprocess_ = false;

  // Remap YOLO11-pose 9-pt to old 6-pt solver layout
  // Raw [0]=R_center, [1,8]=bottom, [2,3]=right, [4,5]=top, [6,7]=left
  // Remapped: [0]=top, [1]=left, [2]=bottom, [3]=right, [4]=blade_center, [5]=R_center
  // Edge midpoints are more inward than original corners — may need tuning later
  std::vector<cv::Point2f> remap_keypoints(const std::vector<cv::Point2f> & raw) const;

  void convert(
    const cv::Mat & input, cv::Mat & output, const bool normalize, const bool exchangeRB) const;

  float fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const;
  float fill_tensor_data_image_legacy(ov::Tensor & input_tensor, const cv::Mat & input_image) const;

  void printInputAndOutputsInfo(const ov::Model & network);

  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif
