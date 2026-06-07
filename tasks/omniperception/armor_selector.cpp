#include "armor_selector.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>

#include "tools/logger.hpp"

namespace omniperception
{
ArmorSelector::ArmorSelector(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  enemy_color_ =
    (yaml["enemy_color"].as<std::string>() == "red") ? auto_aim::Color::red : auto_aim::Color::blue;
  mode_ = yaml["mode"].as<int>();

  // 仅感知相机的全向感知链路支持配置是否过滤前哨站；
  // 主相机前哨站过滤由 Tracker 内部控制（step 2.5 顶板剔除）。
  filter_outpost_omni_ = yaml["filter_outpost_omni"]
                           ? yaml["filter_outpost_omni"].as<bool>()
                           : true;
}

bool ArmorSelector::armor_filter(
  std::list<auto_aim::Armor> & armors, ArmorFilterScope scope) const
{
  if (armors.empty()) return true;

  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 25赛季没有5号装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::five; });

  // 主相机不在此处过滤前哨站（由 Tracker 负责）；仅感知相机可配置过滤
  const bool filter_outpost =
    (scope == ArmorFilterScope::OmniPerception) && filter_outpost_omni_;
  if (filter_outpost) {
    armors.remove_if(
      [&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::outpost; });
  }

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
           invincible_armor_.end();
  });

  return armors.empty();
}

void ArmorSelector::set_priority(std::list<auto_aim::Armor> & armors) const
{
  if (armors.empty()) return;

  const auto & priorities = priority_map();
  for (auto & armor : armors) {
    armor.priority = priorities.at(armor.name);
  }
}

void ArmorSelector::set_invincible_enemy_ids(const std::vector<int8_t> & invincible_enemy_ids)
{
  invincible_armor_.clear();

  if (invincible_enemy_ids.empty()) return;

  for (const auto & id : invincible_enemy_ids) {
    tools::logger()->info("invincible armor id: {}", id);
    invincible_armor_.push_back(auto_aim::ArmorName(id - 1));
  }
}

const ArmorSelector::PriorityMap & ArmorSelector::priority_map() const
{
  return (mode_ == MODE_ONE) ? mode1_ : mode2_;
}

}  // namespace omniperception
