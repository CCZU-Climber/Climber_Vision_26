#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_big_group.hpp"
#include "tasks/auto_buff/buff_target.hpp"

namespace
{
constexpr double kFaceAngle = 2.0 * CV_PI / 5.0;

void require(bool ok, const char * message)
{
  if (!ok) {
    std::cerr << "FAIL: " << message << std::endl;
    throw std::runtime_error(message);
  }
}

auto_buff::PowerRune make_big_rune(
  const auto_buff::TargetParams & params, double roll, int forced_slot_id = -1)
{
  auto_buff::PowerRune rune;
  int slot_id = forced_slot_id;
  if (slot_id < 0) {
    slot_id = static_cast<int>(std::round(tools::limit_rad(roll) / kFaceAngle) + 5) % 5;
  }
  constexpr float cx = 100.0F;
  constexpr float cy = 100.0F;
  constexpr float radius_px = 60.0F;
  const float angle = static_cast<float>(slot_id * kFaceAngle);

  auto_buff::FanBlade blade;
  blade.type = auto_buff::_target;
  blade.center = cv::Point2f(cx + radius_px * std::cos(angle), cy - radius_px * std::sin(angle));
  rune.r_center = cv::Point2f(cx, cy);
  rune.fanblades.push_back(blade);
  rune.slot_id = slot_id;
  rune.slot_angle = angle;

  rune.ypd_in_world = Eigen::Vector3d(0.0, 0.0, 6.0);
  rune.xyz_in_world = tools::ypd2xyz(rune.ypd_in_world);
  rune.ypr_in_world = Eigen::Vector3d(0.0, 0.0, roll);
  rune.R_buff2world = tools::rotation_matrix(rune.ypr_in_world);

  const Eigen::Vector3d blade_in_buff(0.0, 0.0, params.target_radius);
  rune.blade_xyz_in_world = rune.R_buff2world * blade_in_buff + rune.xyz_in_world;
  rune.blade_ypd_in_world = tools::xyz2ypd(rune.blade_xyz_in_world);
  rune.target_xyz_in_world = rune.blade_xyz_in_world;
  rune.target_ypd_in_world = rune.blade_ypd_in_world;
  return rune;
}

void test_big_target_uses_physical_blade_roll()
{
  auto_buff::TargetParams params;
  auto_buff::BigTarget target(params);
  target.set_blade_index(1);

  auto t = std::chrono::steady_clock::now();
  auto rune = make_big_rune(params, kFaceAngle, 1);
  std::optional<auto_buff::PowerRune> observed = rune;
  target.get_target(observed, t);

  const Eigen::Vector3d aim_in_buff(0.0, 0.0, params.target_radius);
  const Eigen::Vector3d actual = target.point_buff2world(aim_in_buff);
  const Eigen::Vector3d expected = rune.R_buff2world * aim_in_buff + rune.xyz_in_world;
  require((actual - expected).norm() < 1e-6, "BigTarget aim projection lost blade roll offset");
}

void test_big_group_single_observation_exports_observed_slot()
{
  auto_buff::TargetParams params;
  auto_buff::BigTargetGroup group(params);
  auto t = std::chrono::steady_clock::now();

  std::vector<auto_buff::PowerRune> first{make_big_rune(params, -kFaceAngle, 1)};
  group.update(first, t);
  auto target1 = group.get_target_copy(1);
  require(target1.has_value(), "single observation should export observed slot 1");
  require(target1->blade_index() == 1, "slot 1 candidate should keep slot id 1");

  t += std::chrono::milliseconds(20);
  std::vector<auto_buff::PowerRune> second{make_big_rune(params, -2.0 * kFaceAngle, 2)};
  group.update(second, t);
  auto target2 = group.get_target_copy(2);
  require(target2.has_value(), "single observation for a new blade should export slot 2");
  require(target2->blade_index() == 2, "slot 2 candidate should keep slot id 2");
}

void test_big_group_exports_observed_slots_by_slot_id()
{
  auto_buff::TargetParams params;
  auto_buff::BigTargetGroup group(params);
  auto t = std::chrono::steady_clock::now();

  const double phase = 0.08;
  auto slot1_obs = make_big_rune(params, phase - 1.0 * kFaceAngle, 1);
  auto slot2_obs = make_big_rune(params, phase - 2.0 * kFaceAngle, 2);
  std::vector<auto_buff::PowerRune> observations{slot1_obs, slot2_obs};
  group.update(observations, t);

  auto slot1 = group.get_target_copy(1);
  auto slot2 = group.get_target_copy(2);
  require(slot1.has_value(), "group should export observed slot 1");
  require(slot2.has_value(), "group should export observed slot 2");
  require(slot1->blade_index() == 1, "slot 1 copy should keep slot id 1");
  require(slot2->blade_index() == 2, "slot 2 copy should keep slot id 2");

  const Eigen::Vector3d aim_in_buff(0.0, 0.0, params.target_radius);
  const Eigen::Vector3d actual = slot2->point_buff2world(aim_in_buff);
  const Eigen::Vector3d expected = slot2_obs.R_buff2world * aim_in_buff + slot2_obs.xyz_in_world;
  require((actual - expected).norm() < 1e-6, "slot 2 projection should use global phase and slot offset");
}

void test_big_group_keeps_blade_id_when_slot_bucket_changes()
{
  auto_buff::TargetParams params;
  auto_buff::BigTargetGroup group(params);
  auto t = std::chrono::steady_clock::now();

  std::vector<auto_buff::PowerRune> first{make_big_rune(params, -kFaceAngle, 1)};
  group.update(first, t);
  auto slot1 = group.get_target_copy(1);
  require(slot1.has_value(), "first observation should export slot 1");
  const int blade_id = slot1->blade_id();
  require(blade_id >= 0, "first observation should allocate a stable blade id");

  t += std::chrono::milliseconds(20);
  std::vector<auto_buff::PowerRune> second{make_big_rune(params, -kFaceAngle + 0.03, 2)};
  group.update(second, t);
  auto slot2 = group.get_target_copy(2);
  require(slot2.has_value(), "same physical blade should export its new slot");
  require(slot2->blade_index() == 2, "candidate slot id should follow the current bucket");
  require(slot2->blade_id() == blade_id, "physical blade id should stay stable across slot buckets");
}

void test_big_selector_keeps_current_target_until_it_disappears()
{
  auto_buff::TargetParams params;
  auto_buff::BigTargetGroup group(params);
  auto_buff::BigTargetSelector selector("configs/infantry_5.yaml");
  auto_buff::Aimer aimer("configs/infantry_5.yaml");
  auto now = std::chrono::steady_clock::now();

  std::vector<auto_buff::PowerRune> observations{
    make_big_rune(params, -kFaceAngle, 1),
    make_big_rune(params, -2.0 * kFaceAngle, 2)};
  group.update(observations, now);

  auto first_target = selector.select_target(group, aimer, now, 24.0, 0.0, 0.0, now);
  require(first_target.has_value(), "selector should choose an initial valid target");
  const int first = selector.selected_id();
  require(first >= 0, "selector should expose the selected slot id for diagnostics");

  selector.on_fire(now);
  now += std::chrono::milliseconds(100);
  auto after_fire_target = selector.select_target(group, aimer, now, 24.0, CV_PI, 0.0, now);
  require(after_fire_target.has_value(), "selector should keep a target after fire");
  require(selector.selected_id() == first, "selector should keep locking a still-valid target after fire");

  const int remaining_slot = first == 1 ? 2 : 1;
  observations = {make_big_rune(params, -remaining_slot * kFaceAngle, remaining_slot)};
  now += std::chrono::milliseconds(20);
  group.update(observations, now);
  auto after_disappear_target = selector.select_target(group, aimer, now, 24.0, 0.0, 0.0, now);
  require(after_disappear_target.has_value(), "selector should find another target after lock disappears");
  require(selector.selected_id() != first, "selector should switch after the locked target disappears");
}

}  // namespace

int main()
{
  test_big_target_uses_physical_blade_roll();
  test_big_group_single_observation_exports_observed_slot();
  test_big_group_exports_observed_slots_by_slot_id();
  test_big_group_keeps_blade_id_when_slot_bucket_changes();
  test_big_selector_keeps_current_target_until_it_disappears();

  auto_buff::TargetParams params;
  params.max_lost_frames = 3;

  auto_buff::SmallTarget target;
  auto t = std::chrono::steady_clock::now();
  std::optional<auto_buff::PowerRune> none;
  for (int i = 0; i < 4; ++i) {
    target.get_target(none, t);
    t += std::chrono::milliseconds(10);
  }

  auto_buff::PowerRune rune;
  rune.ypd_in_world = Eigen::Vector3d(0.0, 0.0, 6.0);
  rune.ypr_in_world = Eigen::Vector3d(0.0, 0.0, 0.0);
  rune.blade_ypd_in_world = Eigen::Vector3d(0.0, 0.0, 6.7);
  std::optional<auto_buff::PowerRune> observed = rune;
  target.get_target(observed, t);

  return 0;
}
