#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <iostream>
#include <list>
#include <unordered_map>

#include "detection.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/omniperception/armor_selector.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace omniperception
{
class Decider
{
public:
  Decider(const std::string & config_path);


  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  bool armor_filter(std::list<auto_aim::Armor> & armors);

  void set_priority(std::list<auto_aim::Armor> & armors);
  //对队列中的每一个DetectionResult进行过滤，同时将DetectionResult排序
  void sort(std::vector<DetectionResult> & detection_queue);

  Eigen::Vector4d get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

  void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

  void get_auto_aim_target(
    std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target);

private:
  int img_width_;
  int img_height_;
  double default_fov_h_;
  double default_fov_v_;
  double default_perception_pitch_offset_;
  int count_;

  ArmorSelector armor_selector_;
  auto_aim::YOLO detector_;

  // 相机视角配置结构体
  struct CameraViewConfig
  {
    double yaw_deg{0.0};
    double fov_h_deg{60.0};
    double fov_v_deg{45.0};
    double pitch_offset_deg{0.0};
  };

  std::unordered_map<std::string, CameraViewConfig> camera_views_;
};

}  // namespace omniperception

#endif
