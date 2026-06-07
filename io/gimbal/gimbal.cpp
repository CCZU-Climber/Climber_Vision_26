#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"


namespace io
{
Gimbal::Gimbal(const std::string & config_path)
  : solver_(config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  auto baudrate = tools::read<int>(yaml, "baudrate");
  speed_control_ = tools::read<bool>(yaml, "speed_control");
  bullet_speed_config_ = tools::read<double>(yaml, "bullet_speed");
  auto_fire_ = tools::read<bool>(yaml, "auto_fire");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(baudrate);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout time_out=serial::Timeout::simpleTimeout(20);
    serial_.setTimeout(time_out);
    serial_.open();
    usleep(1000000); // 1s wait
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  if (!auto_fire_ && tx_data_.mode == 2) {
    tx_data_.mode = 1;
  }
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  // tx_data_.crc16 = tools::get_crc16(
  //   reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));
  tx_data_.tail[0] = 'E';
  tx_data_.tail[1] = 'N';
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  fire = fire && auto_fire_;
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  // tx_data_.crc16 = tools::get_crc16(
  //   reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));
  tx_data_.tail[0] = 'E';
  tx_data_.tail[1] = 'N';

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    // tools::logger()->debug(
    //   "[Gimbal] Sent command - MODE: {}, Yaw: {:.2f}, Yaw vel: {:.2f}, Yaw acc: {:.2f}, "
    //   "Pitch: {:.2f}, Pitch vel: {:.2f}, Pitch acc: {:.2f}",
    //   tx_data_.mode, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  constexpr float kBulletSpeedDiffEps = 1e-3F;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (rx_data_.head[0] != 'G' || rx_data_.head[1] != 'V') continue;

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    // if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
    //   tools::logger()->debug("[Gimbal] CRC16 check failed.");
    //   continue;
    // }
    if (rx_data_.tail[0] != 'E' || rx_data_.tail[1] != 'N') continue;


    error_count = 0;
    auto yaw_ = rx_data_.yaw;
    auto pitch_ = rx_data_.pitch;
    auto roll_ = rx_data_.roll;
    // Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    Eigen::Quaterniond q= 
          (Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll_, Eigen::Vector3d::UnitX())).normalized();

    auto x = q.x();
    auto y = q.y();
    auto z = q.z();
    auto w = q.w();
      
    // 四元数有效性检查
    if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w))
    {
      tools::logger()->warn("[Cboard] Invalid q: NaN detected - w={}, x={}, y={}, z={}", w, x, y, z);
    }
    else if (std::abs(x * x + y * y + z * z + w * w - 1) > 1e-2)
    {
      tools::logger()->warn("[Cboard] Invalid q: magnitude check failed - w={}, x={}, y={}, z={}", w, x, y, z);
    }
    else
    {
      queue_.push({{ w, x, y, z}, t});
    }
  
    // Eigen::Quaterniond q_ = q;
    // solver_.set_R_gimbal2world(q_);
    // Eigen::Vector3d euler = solver_.R_gimbal2world().eulerAngles(2, 1, 0);  // yaw-pitch-roll
    std::lock_guard<std::mutex> lock(mutex_);

    if (speed_control_) {
      const float current_bullet_speed = rx_data_.bullet_speed;
      if (
        !has_last_bullet_speed_sample_ ||
        std::fabs(current_bullet_speed - last_bullet_speed_sample_) > kBulletSpeedDiffEps)
      {
        bullet_speed_samples_[bullet_speed_sample_index_] = current_bullet_speed;
        bullet_speed_sample_index_ = (bullet_speed_sample_index_ + 1) % bullet_speed_samples_.size();
        if (bullet_speed_sample_count_ < bullet_speed_samples_.size()) {
          bullet_speed_sample_count_++;
        }
        last_bullet_speed_sample_ = current_bullet_speed;
        has_last_bullet_speed_sample_ = true;
      }

      if (bullet_speed_sample_count_ == bullet_speed_samples_.size()) {
        float sum = 0.0F;
        for (float speed_sample : bullet_speed_samples_) {
          sum += speed_sample;
        }
        state_.bullet_speed = sum / static_cast<float>(bullet_speed_samples_.size());
      } else {
        state_.bullet_speed = current_bullet_speed;
      }
    } else {
      state_.bullet_speed = bullet_speed_config_;
    }

    // state_.yaw = euler[0];
    state_.yaw = tools::limit_rad(rx_data_.yaw);
    state_.yaw_vel = rx_data_.yaw_vel;
    // state_.pitch = euler[1];
    state_.pitch = tools::limit_rad(rx_data_.pitch);
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_count = rx_data_.bullet_count;

    // tools::logger()->debug("[Gimbal] IMU YPR: yaw={:.2f}, pitch={:.2f}, roll={:.2f}", yaw_*57.3, pitch_*57.3, roll_*57.3);

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      bullet_speed_samples_ = {0.0F, 0.0F, 0.0F};
      bullet_speed_sample_count_ = 0;
      bullet_speed_sample_index_ = 0;
      has_last_bullet_speed_sample_ = false;
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io