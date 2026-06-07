#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "io/command.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | Show command line help}"
  "{delta-angle a  |          8          | Delta angle on yaw axis}"
  "{circle      c  |         0.2         | Number of slices in one period}"
  "{signal-mode m  |     triangle_wave   | Signal mode}"
  "{axis        x  |         yaw         | Controlled axis}"
  "{@config-path   | configs/sentry.yaml | Positional yaml config path }";

double yaw_cal(double t)
{
  double A = -6;
  double T = 2;  // s
  return A * std::sin(2 * M_PI * t / T);
}

double pitch_cal(double t)
{
  double A = -6;
  double T = 2;  // s
  return A * std::sin(2 * M_PI * t / T + M_PI / 2) - 18;
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  auto delta_angle = cli.get<double>("delta-angle");
  auto circle = cli.get<double>("circle");
  auto signal_mode = cli.get<std::string>("signal-mode");
  auto axis = cli.get<std::string>("axis");
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  io::Gimbal gimbal(config_path);
  auto_aim::Solver solver(config_path);

  auto init_angle = 0;
  double slice = circle * 100;
  auto dangle = delta_angle / slice;
  double cmd_angle = init_angle;

  int axis_index = axis == "yaw" ? 0 : 1;

  int count = 0;
  int step_dir = 1;
  const double step_limit = 20.0;

  gimbal.send(true, false, 0, 0, 0, 0, 0, 0);
  std::this_thread::sleep_for(5s);

  io::Command command{0};
  io::Command last_command{0};

  double t = 0;
  auto last_t = t;
  double dt = 0.005;

  auto t0 = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    nlohmann::json data;
    auto timestamp = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(1ms);

    // Eigen::Quaterniond q = gimbal.q(timestamp);
    // solver.set_R_gimbal2world(q);

    // Eigen::Vector3d eulers = tools::eulers(q, 2, 1, 0);

    auto gs = gimbal.state();

    if (signal_mode == "triangle_wave") {
      if (count == slice) {
        cmd_angle = init_angle;
        command = {1, 0, 0, 0};
        if (axis_index == 0)
          command.yaw = cmd_angle / 57.3;
        else
          command.pitch = cmd_angle / 57.3;
        count = 0;

      } else {
        cmd_angle += dangle;
        if (axis_index == 0)
          command.yaw = cmd_angle / 57.3;
        else
          command.pitch = cmd_angle / 57.3;
        count++;
      }

      gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
      if (axis_index == 0) {
        data["cmd_yaw"] = command.yaw * 57.3;
        data["last_cmd_yaw"] = last_command.yaw * 57.3;
        data["gimbal_yaw"] = gs.yaw * 57.3;
      } else {
        data["cmd_pitch"] = command.pitch * 57.3;
        data["last_cmd_pitch"] = last_command.pitch * 57.3;
        data["gimbal_pitch"] = gs.pitch * 57.3;
      }
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      last_command = command;
      plotter.plot(data);
      std::this_thread::sleep_for(8ms);
    }

    else if (signal_mode == "step") {
      if (count == 300) {
        cmd_angle += step_dir * delta_angle;
        if (cmd_angle >= init_angle + step_limit || cmd_angle <= init_angle - step_limit) {
          step_dir = -step_dir;
        }
        count = 0;
      }

      count++;

      if (axis_index == 0) {
        command = {1, 0, tools::limit_rad(cmd_angle / 57.3), 0};
        data["cmd_yaw"] = command.yaw * 57.3;
        data["last_cmd_yaw"] = last_command.yaw * 57.3;
        data["gimbal_yaw"] = gs.yaw * 57.3;
      } else {
        command = {1, 0, 0, tools::limit_rad(cmd_angle / 57.3)};
        data["cmd_pitch"] = command.pitch * 57.3;
        data["last_cmd_pitch"] = last_command.pitch * 57.3;
        data["gimbal_pitch"] = gs.pitch * 57.3;
      }

      gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
      last_command = command;
      plotter.plot(data);
      std::this_thread::sleep_for(8ms);
    }

    else if (signal_mode == "circle") {
      command.yaw = yaw_cal(t) / 57.3;
      command.pitch = pitch_cal(t) / 57.3;
      command.control = 1;
      command.shoot = 0;
      t += dt;
      if (t - last_t > 2) {
        t += 2.4;
        last_t = t;
      }
      gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

      data["t"] = t;
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["gimbal_yaw"] = gs.yaw * 57.3;
      data["gimbal_pitch"] = gs.pitch * 57.3;
      plotter.plot(data);
      std::this_thread::sleep_for(9ms);
    }
  }
  return 0;
}
