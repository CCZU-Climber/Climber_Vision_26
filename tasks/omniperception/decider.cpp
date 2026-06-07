#include "decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
Decider::Decider(const std::string & config_path)
: count_(0), armor_selector_(config_path), detector_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  img_width_ = yaml["image_width"].as<double>();
  img_height_ = yaml["image_height"].as<double>();
  default_fov_h_ = yaml["default_fov_h"].as<double>();
  default_fov_v_ = yaml["default_fov_v"].as<double>();
  default_perception_pitch_offset_ = 0.0;
  if (yaml["perception_pitch_offset"]) {
    default_perception_pitch_offset_ = yaml["perception_pitch_offset"].as<double>();
  }

  // 读取相机偏角与视场配置，允许标量（仅偏角）或映射（偏角+视场）两种格式
  if (yaml["camera_name_map"]) {
    for (const auto & node : yaml["camera_name_map"]) {
      const auto cam_name = node.first.as<std::string>();
      CameraViewConfig cfg{0.0, default_fov_h_, default_fov_v_, default_perception_pitch_offset_};

      try {
        const auto & value = node.second;
        if (value.IsScalar()) {
          cfg.yaw_deg = value.as<double>();
        } else if (value.IsMap()) {
          if (value["yaw"]) cfg.yaw_deg = value["yaw"].as<double>();
          if (value["fov_h"]) cfg.fov_h_deg = value["fov_h"].as<double>();
          if (value["fov_v"]) cfg.fov_v_deg = value["fov_v"].as<double>();
          if (value["pitch_offset"]) cfg.pitch_offset_deg = value["pitch_offset"].as<double>();
        } else {
          tools::logger()->warn("camera_name_map entry for {} is neither scalar nor map", cam_name);
          continue;
        }

        camera_views_[cam_name] = cfg;
        tools::logger()->info(
          "Camera {} yaw offset {:.2f} deg, fov_h {:.1f}, fov_v {:.1f}, pitch offset {:.2f} deg",
          cam_name, cfg.yaw_deg, cfg.fov_h_deg, cfg.fov_v_deg, cfg.pitch_offset_deg);
      } catch (const std::exception & e) {
        tools::logger()->warn("Failed to parse camera_name_map for {}: {}", cam_name, e.what());
      }
    }
  } else {
    tools::logger()->warn("camera_name_map missing in config, using default yaw 0");
  }
}



io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) {
    return io::Command{false, false, 0, 0};
  }

  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  tools::logger()->info(
    "omniperceptron find {},delta yaw is {:.4f}", auto_aim::ARMOR_NAMES[dr.armors.front().name],
    dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
};

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  Eigen::Vector2d delta_angle = Eigen::Vector2d::Zero();
  if (armors.empty()) return delta_angle;

  CameraViewConfig cfg{0.0, default_fov_h_, default_fov_v_, default_perception_pitch_offset_};
  auto it = camera_views_.find(camera);
  if (it != camera_views_.end()) {
    cfg = it->second;
  } else {
    tools::logger()->warn("Unknown camera {}, using default yaw 0 and fov ({:.1f}, {:.1f})", camera,
      cfg.fov_h_deg, cfg.fov_v_deg);
  }
  //计算偏转角度
  const auto & armor = armors.front();
  delta_angle[0] = cfg.yaw_deg + (cfg.fov_h_deg / 2.0) - armor.center_norm.x * cfg.fov_h_deg;
  delta_angle[1] = armor.center_norm.y * cfg.fov_v_deg - cfg.fov_v_deg / 2.0 + cfg.pitch_offset_deg;
  return delta_angle;
}

bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  return armor_selector_.armor_filter(armors, ArmorFilterScope::OmniPerception);
}

void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  armor_selector_.set_priority(armors);
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return;

  // 对每个 DetectionResult 调用 armor_filter 和 set_priority
  for (auto & dr : detection_queue) {
    armor_filter(dr.armors);
    set_priority(dr.armors);

    // 对每个 DetectionResult 中的 armors 进行排序
    dr.armors.sort(
      [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
  }

  // 移除armors为空的DetectionResult
  detection_queue.erase(
    std::remove_if(
      detection_queue.begin(), detection_queue.end(),
      [](const DetectionResult & dr) { return dr.armors.empty(); }),
    detection_queue.end());

  // 如果过滤后没有有效的检测结果，直接返回
  if (detection_queue.empty()) return;

  // 根据优先级对 DetectionResult 进行排序
  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      return a.armors.front().priority < b.armors.front().priority;
    });
}

Eigen::Vector4d Decider::get_target_info(
  const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets)
{
  if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

  auto target = targets.front();

  for (const auto & armor : armors) {
    if (armor.name == target.name) {
      return Eigen::Vector4d{
        armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
        static_cast<double>(armor.name) + 1};  //避免歧义+1(详见通信协议)
    }
  }

  return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
  armor_selector_.set_invincible_enemy_ids(invincible_enemy_ids);
}

void Decider::get_auto_aim_target(
  std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
  if (auto_aim_target.empty()) return;

  std::vector<auto_aim::ArmorName> auto_aim_targets;

  for (const auto & target : auto_aim_target) {
    if (target <= 0 || static_cast<size_t>(target) > auto_aim::ARMOR_NAMES.size()) {
      tools::logger()->warn("Received invalid auto_aim target value: {}", int(target));
      continue;
    }
    auto_aim_targets.push_back(static_cast<auto_aim::ArmorName>(target - 1));
    tools::logger()->info("nav send auto_aim target is {}", auto_aim::ARMOR_NAMES[target - 1]);
  }

  if (auto_aim_targets.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
           auto_aim_targets.end();
  });
}

}  // namespace omniperception
