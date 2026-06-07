#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

class FanBlade
{
public:
  cv::Point2f center;               // 扇页中心
  std::vector<cv::Point2f> points;  // 四个点从左上角开始逆时针
  double angle, width, height;
  FanBlade_type type;  // 类型

  explicit FanBlade() = default;

  // explicit FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;       // TRAD_R: 检测器形态学中心，不被solver覆写
  cv::Point2f model_r_center; // MODEL_R: 模型原始R标中心
  cv::Point2f pnp_r_center;  // PNP_R: PnP重投影几何中心，由solver填入
  std::vector<FanBlade> fanblades;  // 按target开始顺时针
  int slot_id = -1;           // 0-4 slot bucket around the R center, not a physical identity
  double slot_angle = 0.0;    // image-plane angle used to assign slot_id

  int light_num;

  Eigen::Vector3d trad_xyz_in_camera = Eigen::Vector3d::Zero(); // TRAD_R camera coords
  Eigen::Vector3d trad_xyz_in_gimbal = Eigen::Vector3d::Zero(); // TRAD_R gimbal coords
  Eigen::Vector3d trad_xyz_in_world = Eigen::Vector3d::Zero(); // TRAD_R world R center
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Matrix3d R_buff2world = Eigen::Matrix3d::Identity();  // PnP完整姿态
  Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();  // 单位：rad
  Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero();  // 单位：m
  Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero();  // 球坐标系, 单位: m

  Eigen::Vector3d target_xyz_in_camera = Eigen::Vector3d::Zero(); // 靶心相机坐标
  Eigen::Vector3d target_xyz_in_gimbal = Eigen::Vector3d::Zero(); // 靶心云台坐标
  Eigen::Vector3d target_xyz_in_world = Eigen::Vector3d::Zero(); // 靶心世界坐标
  Eigen::Vector3d target_ypd_in_world = Eigen::Vector3d::Zero(); // 靶心球坐标

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[0]; };
  const FanBlade & target() const { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP
