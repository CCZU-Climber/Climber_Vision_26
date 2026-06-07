#ifndef IO__CBOARD_HPP
#define IO__CBOARD_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>

#include "io/command.hpp"
#include "serial/serial.h"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
enum Mode
{
  idle,
  auto_aim,
  small_buff,
  big_buff,
  outpost
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};



// 接收：下位机 -> Vision (IMU, Speed, Mode)
struct __attribute__((packed)) BoardToVision
{
  uint8_t head[2] = {'B', 'V'}; // 帧头
  uint8_t mode;                 // 射击模式：0手动，1自瞄，2开符
  float bullet_speed;           //  1e2
  // float q[4];                   // wxyz顺序，四元数
  float yaw;
  float pitch;
  float roll;
  // uint16_t crc16;
  uint8_t tail[2] = {'E', 'N'}; // 帧尾
};
static_assert(sizeof(BoardToVision) <= 64);

// 发送：Vision -> 下位机 (Command)
struct __attribute__((packed)) VisionToBoard
{
  uint8_t head[2] = {'V', 'B'}; // 帧头
  uint8_t control;      //1：控制
  uint8_t shoot;        //1：射击
  float yaw;          // 单位弧度
  float pitch;
  float horizon_distance;
  // uint16_t crc16;
  uint8_t tail[2] = {'E', 'N'}; // 帧尾
};
static_assert(sizeof(VisionToBoard) <= 64);


class CBoard
{
public:
  double bullet_speed;
  Mode mode;
  // ShootMode shoot_mode;
  // double ft_angle;  //无人机专有

  CBoard(const std::string & config_path);
  ~CBoard();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  void send(Command command);


private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  tools::ThreadSafeQueue<IMUData> queue_{5000}; 
  serial::Serial serial_;
  bool speed_control = false;
  double bullet_speed_config = 0.0;

  std::thread thread_;
  std::atomic<bool> quit_=false;

  IMUData data_ahead_;
  IMUData data_behind_;

  BoardToVision rx_data_;
  VisionToBoard tx_data_;

  void read_thread();
  bool read(uint8_t *buffer, size_t size);
  void reconnect();
};

}  // namespace io

#endif  // IO__CBOARD_HPP