#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
const cv::Point2f IMG_CENTER(1440 / 2, 1080 / 2);  // TODO: 适配不同分辨率的分发

double center_distance(const Armor & armor) { return cv::norm(armor.center - IMG_CENTER); }
}  // namespace

Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

std::string Tracker::state() const { return state_; }

/**
 * @brief 执行每一帧的核心追踪逻辑
 * @param armors 视觉检测模块传入的视野内所见装甲板集合
 * @param t 当前处理这帧画面的时刻，计算卡尔曼运动推演 dt 的关键参数
 * @param use_enemy_color 是否仅追踪敌方颜色的装甲板
 * @return 跟踪的目标列表。实际目前对于单自瞄任务，一旦锁定某个目标，返回值为含有一个该 Target 对象；如果没有任何锁定目标则返回空。
 */
std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  (void)use_enemy_color;

  const auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 1. 如果帧间隔过长（比如画面卡顿超过 0.1s），为防止卡尔曼预测的错误外推飞得太离谱，主动放弃当前目标的跟踪进入 lost（丢失）状态。
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 2. 根据己方配置自动过滤掉我们不关心的己方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 2.5 过滤前哨站顶部装甲板：顶板倾角约27.5°，侧板约-15°，若顶板模型重投影误差更小则剔除
  armors.remove_if([this](const auto_aim::Armor & a) {
    return a.name == ArmorName::outpost &&
           solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
             solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  });

  // 3. 对识别到的装甲板按到图像中心的欧氏距离进行升序排序，使优先锁定靠近准星或视图中心的装甲板。
  armors.sort([](const Armor & a, const Armor & b) {
    return center_distance(a) < center_distance(b);
  });

  // 4. 按装甲板的语义或兵种优先级排序（例如：英雄 > 步兵 > 哨兵等预设逻辑）。保持稳定的靶标选择。
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  // 5. 根据Tracker目前所处的状态机状态决定采取何种策略。
  if (state_ == "lost") {
    // 丢失状态下重新建图并捕获新目标，一旦设好 found = true
    found = set_target(armors, t);
  } else if (
    state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    // 主相机视野中出现更高优先级目标，立即切换（对齐 sp_vision_25）
    found = set_target(armors, t);
    tools::logger()->debug("[Tracker] Switch target to {}", ARMOR_NAMES[armors.front().name]);
  } else {
    // 若当前持有一个跟踪目标，尝试用新一帧数据更新（收敛）该目标
    found = update_target(armors, t);
  }

  // 6. 根据 found 判定结果和丢失帧数等阈值，流转状态机（检测中 <=> 追踪中 <=> 暂丢 <=> 丢失）。
  state_machine(found);

  // 7. 如果在跟踪期间发现 EKF 发散严重（比如推算的半径变为负数或者无穷大失真），强制舍弃当前目标。
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 8. EKF NIS (Normalized Innovation Squared) 诊断卡尔曼滤波收敛退化，失败过拟合占比超过40%时抛弃重建目标
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  return {target_};
}

/**
 * @brief 更新追踪器状态机
 * @param found 是否在当前帧中成功找到或更新目标
 */
void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;
    state_ = "detecting";
    detect_count_ = 1;
    return;
  }

  if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
    return;
  }

  if (state_ == "tracking") {
    if (found) return;
    temp_lost_count_ = 1;
    state_ = "temp_lost";
    return;
  }

  if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
    return;
  }

  if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost) {
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      } else {
        max_temp_lost_count_ = normal_temp_lost_count_;
      }
      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  return set_target(armor, t);
}

bool Tracker::set_target(Armor & armor, std::chrono::steady_clock::time_point t)
{
  solver_.solve(armor);

  const auto is_balance = (armor.type == ArmorType::big) &&
                          (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                           armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  } else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 1, 100, 1, 0.5, 0.5}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  } else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  } else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  std::vector<Armor> matched_armors;
  matched_armors.reserve(3);

  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    solver_.solve(armor);
    matched_armors.push_back(armor);
    if (target_.name == ArmorName::outpost && matched_armors.size() >= 3) break;
  }

  if (matched_armors.empty()) return false;

  if (target_.name == ArmorName::outpost) {
    target_.update_outpost(matched_armors);
    auto iter = target_.ekf().data.find("outpost_assignment_valid");
    if (iter != target_.ekf().data.end() && iter->second < 0.5) return false;
    return true;
  }

  for (const auto & armor : matched_armors) {
    target_.update(armor);
  }
  return true;
}

}  // namespace auto_aim
