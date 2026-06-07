#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
// #include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
/**
 * @brief 装甲板追踪器类
 * @details 负责维护目标跟踪状态机，利用扩展卡尔曼滤波(EKF)跟踪装甲板，并处理目标的丢失与重新捕获。
 *          通过管理目标的生命周期（检测中、追踪中、暂时丢失、完全丢失），实现对目标的稳定追踪。
 */
class Tracker
{
public:
  /**
   * @brief 构造函数，初始化追踪器
   * @param config_path 配置文件路径，包含颜色、丢失帧数阈值等参数
   * @param solver 负责计算装甲板3D位姿的解算器引用
   */
  Tracker(const std::string & config_path, Solver & solver);

  /**
   * @brief 获取当前追踪器的状态机状态
   * @return 当前状态字符串，如 "lost" (丢失), "detecting" (检测中), "tracking" (追踪中), "temp_lost" (暂时丢失)
   */
  std::string state() const;

  /**
   * @brief 核心追踪函数，输入当前帧识别到的装甲板进行目标匹配与更新
   * @param armors 当前帧检测到的所有装甲板列表
   * @param t 当前帧的时间戳
   * @param use_enemy_color 是否预先过滤掉非敌方颜色的装甲板（默认：true）
   * @return 包含当前正在跟踪的目标列表（通常只返回正在跟踪的那一个目标，若丢失则返回空列表）
   */
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  // std::tuple<omniperception::DetectionResult, std::list<Target>> track(
  //   const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  //   std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;             ///< 位姿解算器引用
  Color enemy_color_;           ///< 敌人颜色，用于剔除自身的装甲板
  int min_detect_count_;        ///< 从"检测中"进入"追踪中"状态所需的最小连续目标存在帧数
  int max_temp_lost_count_;     ///< 允许的最大暂时丢失帧数上限，超过则认为彻底丢失进入"lost"
  int detect_count_;            ///< 当前连续检测到目标的帧数计数器
  int temp_lost_count_;         ///< 当前连续丢失目标的帧数计数器
  int outpost_max_temp_lost_count_; ///< 特供：前哨站专用的最大暂时丢失帧数（防抖机制）
  int normal_temp_lost_count_;  ///< 常规：普通装甲板的最大暂时丢失帧数
  std::string state_, pre_state_; ///< 当前状态与前一帧的状态记录
  Target target_;                 ///< 当前锁定的跟踪目标对象
  std::chrono::steady_clock::time_point last_timestamp_; ///< 上次更新的时间戳，用于计算两帧间的 dt
  ArmorPriority omni_target_priority_; ///< 全向感知的目标优先级

  /**
   * @brief 核心状态机更新：根据当前帧是否匹配到目标，切换跟踪器的内部状态
   * @param found 当前帧是否成功匹配/识别到了正在跟踪的目标，或找到了新目标
   */
  void state_machine(bool found);

  /**
   * @brief 当状态为 lost (丢失) 时调用，尝试从当前视野中的装甲板列表中选择一个最优的新目标去锁定
   * @param armors 视野内的装甲板列表
   * @param t 当前时间戳
   * @return 是否成功锁定了新目标
   */
  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
  bool set_target(Armor & armor, std::chrono::steady_clock::time_point t);

  /**
   * @brief 当状态不为 lost 时调用，使用当前视野中的装甲板测量值，去更新当前的跟踪目标（EKF观测更新）
   * @param armors 视野内筛选后的同分类装甲板列表
   * @param t 当前时间戳
   * @return 是否在当前帧中成功找到并更新了对应的跟踪目标
   */
  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
