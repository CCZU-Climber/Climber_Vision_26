#ifndef OMNIPERCEPTION__ARMOR_SELECTOR_HPP
#define OMNIPERCEPTION__ARMOR_SELECTOR_HPP

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
enum class ArmorFilterScope
{
  MainAim,
  OmniPerception
};

enum PriorityMode
{
  MODE_ONE = 1,
  MODE_TWO
};

class ArmorSelector
{
public:
  explicit ArmorSelector(const std::string & config_path);

  bool armor_filter(
    std::list<auto_aim::Armor> & armors,
    ArmorFilterScope scope = ArmorFilterScope::MainAim) const;

  void set_priority(std::list<auto_aim::Armor> & armors) const;

  void set_invincible_enemy_ids(const std::vector<int8_t> & invincible_enemy_ids);

private:
  using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

  const PriorityMap & priority_map() const;

  int mode_;
  bool filter_outpost_omni_;
  auto_aim::Color enemy_color_;
  std::vector<auto_aim::ArmorName> invincible_armor_;

  const PriorityMap mode1_ = {
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::forth},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::fifth}};

  const PriorityMap mode2_ = {
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};
};

}  // namespace omniperception

#endif
