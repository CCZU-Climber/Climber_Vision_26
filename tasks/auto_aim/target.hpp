#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

/**
 * @brief 目标类，包装了EKF（扩展卡尔曼滤波）以对特定装甲板目标的 3D 运动状态（位置、速度、偏航角等）进行估计
 * @details 该类维护了装甲板目标的物理属性（如半径、高度）与各类运动状态参数，
 *          能够根据不同兵种的装甲板（包括特殊的前哨站三装甲板模型）适配不同的系统动力学和观测模型。
 */
class Target
{
public:
  ArmorName name;             ///< 目标的名称（如步兵编号 1~5 或是 outpost 前哨站、base 基地等）
  ArmorType armor_type;       ///< 目标的装甲板类型（大装甲板/小装甲板）
  ArmorPriority priority;     ///< 锁定的优先级
  bool jumped;                ///< 标志位：目标在本次更新中是否发生了较大的跳变
  int last_id;                ///< debug用：上一次匹配的装甲板 ID，用于判定跳变/装甲板切换

  Target() = default;
  
  /**
   * @brief 利用真实观测数据（一块装甲板）初次构建目标状态
   * @param armor 检测到的初始装甲板
   * @param t 检测对应的时间戳
   * @param radius 该类型车辆的装甲板旋转半径先验值
   * @param armor_num 该类型车辆含有的装甲板总数（常为 4 个，前哨站为 3 个）
   * @param P0_dig EKF 初始状态协方差矩阵对角线元素（经验预设）
   */
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  
  /**
   * @brief 使用直接注入的状态量来构造一个虚拟的目标（常用于测试/模拟）
   * @param x 目标中心的世界坐标X
   * @param vyaw 目标的自旋偏航角速度
   * @param radius 装甲板旋转半径
   * @param h 装甲板高度的差值 / Z轴偏移量
   */
  Target(double x, double vyaw, double radius, double h);

  /**
   * @brief 调用卡尔曼滤波进行当前时刻的目标状态预测（时间推进）
   * @param t 当前时间戳，通过对比之前保存的时间计算出时间增量 dt
   */
  void predict(std::chrono::steady_clock::time_point t);

  /**
   * @brief 调用卡尔曼滤波进行目标状态预测的基础重载版本
   * @param dt 时间增量
   */
  void predict(double dt);

  /**
   * @brief 利用卡尔曼滤波根据当前的新观测结果（单块装甲板匹配）进行后验融合更新
   * @param armor 匹配成功的装甲板结果
   */
  void update(const Armor & armor);

  /**
   * @brief 专门针对前哨站（Outpost，3面装甲板同轴快速旋转模型）设计的高级关联更新逻辑
   * @details 通过建立二分匹配和代价值计算，处理前哨站旋转中多块装甲板同时可用或产生遮挡的问题
   * @param armors 识别出并在视野内的前哨站装甲板列表
   */
  void update_outpost(const std::vector<Armor> & armors);

  /** @return EKF内的后验估计状态量向量 X */
  Eigen::VectorXd ekf_x() const;
  
  /** @return 当前维护的 EKF 实例的常量引用 */
  const tools::ExtendedKalmanFilter & ekf() const;
  
  /**
   * @brief 获取目标的多个装甲板在世界坐标系下空间位置的三维预期估计信息
   * @return 每个装甲板的 (X, Y, Z, yaw) 包含在 Vector4d 组成的数组中
   */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  
  /** @return 针对前哨站目标，当前是否有一个主要可锁定的优先槽位装甲板 */
  bool has_primary_armor_xyza() const;
  
  /** @return 如果有主要的锁定装甲板，该装甲板预测出来的位姿 (X, Y, Z, yaw) */
  Eigen::Vector4d primary_armor_xyza() const;
  
  /** @return 上述更新阶段，当前锁定的装甲板面是否发生了切换事件（例如旧面转背，新面迎向） */
  bool is_switch() const;

  /**
   * @brief 检查 EKF 是否处于发散状态（例如：半径估算爆炸）
   * @return 若卡尔曼滤波输出不合理，说明目标跟踪已散架，返回 true
   */
  bool diverged() const;

  /**
   * @brief 检查目标滤波状态是否收敛到相对可信和稳定的估计结果
   * @return 收敛时返回 true，便于决策级是否可开火
   */
  bool convergened();

  bool isinit = false;         ///< 是否完成初始化的标识位

  bool checkinit();            ///< 检查系统初始化的状态

private:
  int armor_num_;              ///< 当前判定该目标具备的装甲面数量
  int switch_count_;           ///< 目标装甲板切换面的次数统计
  int update_count_;           ///< EKF更新进行的积累次数统计

  bool is_switch_, is_converged_; ///< 保存装甲板切换和EKF是否收敛的状态位
  bool has_outpost_primary_slot_ = false; ///< 前哨站是否锁定了一个主要受打击面
  int outpost_primary_slot_ = 0;          ///< 前哨站当前要打击的面 (0~2 对应三个装甲面之一)
  int outpost_mismatch_streak_ = 0;       ///< 前哨站追踪面匹配失败的连续帧数累计

  tools::ExtendedKalmanFilter ekf_;       ///< 核心卡尔曼滤波器实例
  std::chrono::steady_clock::time_point t_; ///< 内部记录的最近一次处理的时间戳用于算dt

  /**
   * @brief 传统目标的专用观测更新步骤，使用装甲板极坐标 (yaw, pitch, distance, angle) 来做观测反馈
   * @param armor 最新观测到的单块最优装甲板对象
   * @param id 将该装甲板对应到圆柱模型上的编号ID
   */
  void update_ypda(const Armor & armor, int id); // yaw pitch distance angle

  /**
   * @brief 观测模型辅助函数：根据EKF状态和目标面ID，推算出预测中的物理装甲面世界坐标 (x,y,z)
   */
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  
  /**
   * @brief 获取观测函数的雅可比矩阵 H （即观测关于状态的偏导数矩阵）用于EKF增益计算
   */
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

} // namespace auto_aim

#endif // AUTO_AIM__TARGET_HPP
