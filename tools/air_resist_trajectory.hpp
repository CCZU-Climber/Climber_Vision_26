#ifndef TOOLS__AIR_RESIST_TRAJECTORY_HPP
#define TOOLS__AIR_RESIST_TRAJECTORY_HPP

namespace tools
{
struct AirResistTrajectory
{
  bool unsolvable = false;
  double fly_time = 0.0;  // 飞行时间，单位：s
  double pitch = 0.0;     // 抬头为正，单位：rad
  double yaw = 0.0;       // 偏航角，单位：rad

  // 线性空气阻力弹道
  // v0  子弹初速度，单位：m/s
  // d   目标水平距离（枪口原点+世界轴向），单位：m
  // h   目标竖直高度（枪口原点+世界轴向），单位：m
  // k   线性阻力系数，默认 0（真空）。实测标定值约 0.01~0.03
  //     物理模型: dv/dt = -k*v - g
  AirResistTrajectory(double v0, double d, double h, double k = 0.0);
};

}  // namespace tools

#endif  // TOOLS__AIR_RESIST_TRAJECTORY_HPP
