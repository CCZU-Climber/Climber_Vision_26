#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <list>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;
  std::chrono::steady_clock::time_point timestamp;
  double delta_yaw;    //rad
  double delta_pitch;  //rad
  std::string camera_name;  // 相机名称 (left, right等)
  cv::Mat img; // 图像数据

  // Assignment operator
  DetectionResult & operator=(const DetectionResult & other)
  {
    if (this != &other) {
      armors = other.armors;
      timestamp = other.timestamp;
      delta_yaw = other.delta_yaw;
      delta_pitch = other.delta_pitch;
      camera_name = other.camera_name;
      img = other.img.clone();
    }
    return *this;
  }
};
}  // namespace omniperception

#endif