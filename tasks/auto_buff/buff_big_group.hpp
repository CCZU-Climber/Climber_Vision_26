#ifndef AUTO_BUFF__BIG_GROUP_HPP
#define AUTO_BUFF__BIG_GROUP_HPP

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "buff_target.hpp"

namespace auto_buff
{
class Aimer;

class BigTargetGroup
{
public:
  static constexpr int kSlotCount = 5;

  explicit BigTargetGroup(const TargetParams & params = TargetParams())
    : model_(params) {}

  void update(
    const std::vector<PowerRune> & observations, std::chrono::steady_clock::time_point & timestamp);

  std::optional<BigTarget> get_target_copy(int id) const;
  int blade_id_for_slot(int id) const;

private:
  struct BladeTrack
  {
    bool active = false;
    int slot_id = -1;
    double roll = 0.0;
    int missed_frames = 0;
  };

  static int slot_id_from_observation(const PowerRune & observation);
  static double slot_angle_from_observation(const PowerRune & observation);
  static double roll_distance(double a, double b);
  void update_blade_tracks();
  void age_blade_tracks();

  BigTarget model_;
  std::array<bool, kSlotCount> visible_slots_ = {false, false, false, false, false};
  std::array<int, kSlotCount> slot_blade_ids_ = {-1, -1, -1, -1, -1};
  std::array<BladeTrack, kSlotCount> blade_tracks_ = {};
  int max_blade_missed_frames_ = 30;
};

class BigTargetSelector
{
public:
  explicit BigTargetSelector(const std::string & config_path);

  std::optional<BigTarget> select_target(
    const BigTargetGroup & group, const Aimer & aimer,
    std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    double current_yaw, double current_pitch, std::chrono::steady_clock::time_point now);

  void on_fire(std::chrono::steady_clock::time_point now);

  void reset_cycle();
  int selected_id() const { return locked_id_; }

private:
  int select(
    const std::array<bool, BigTargetGroup::kSlotCount> & valid,
    const std::array<double, BigTargetGroup::kSlotCount> & yaws,
    const std::array<double, BigTargetGroup::kSlotCount> & pitchs,
    const std::array<int, BigTargetGroup::kSlotCount> & blade_ids, double current_yaw, double current_pitch,
    std::chrono::steady_clock::time_point now);

  void on_fire(int selected_id, std::chrono::steady_clock::time_point now);

  double score(
    double yaw, double pitch, double current_yaw, double current_pitch) const;

  double pitch_weight_ = 0.4;

  int locked_id_ = -1;
  int locked_blade_id_ = -1;
};

}  // namespace auto_buff

#endif
