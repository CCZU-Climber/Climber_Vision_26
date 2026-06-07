#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
// #include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("@config-path");

  tools::Exiter exiter;
  tools::Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // auto_aim::Detector detector(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  auto last_t = std::chrono::steady_clock::now();
  nlohmann::json data;

  while (!exiter.exit()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    camera.read(img, t);
    if (img.empty()) break;

    Eigen::Quaterniond q = gimbal.q(t);

    solver.set_R_gimbal2world(q);

    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // auto armors = detector.detect(img);
    auto armors = yolo.detect(img);

    auto targets = tracker.track(armors, t);

    auto gimbal_state = gimbal.state();
    auto command = aimer.aim(targets, t, gimbal_state.bullet_speed, q);

    shooter.shoot(command, aimer, targets, gimbal_pos);

    auto dt = tools::delta_time(t, last_t);
    last_t = t;

    data["dt"] = dt;
    data["fps"] = 1 / dt;
    plotter.plot(data);

    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_z"] = armor.xyz_in_world[2];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_pitch"] = armor.ypr_in_world[1] * 57.3;
      data["armor_distance"] = armor.ypd_in_world[2];
    }

    if (!targets.empty()) {
      auto target = targets.front();
      tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      Eigen::VectorXd x = target.ekf_x();
      
      // Print EKF state to terminal
      fmt::print("\n========== EKF State ==========\n");
      fmt::print("x[0]={:.3f} x[1]={:.3f} | y[2]={:.3f} y[3]={:.3f} | z[4]={:.3f} z[5]={:.3f}\n",
                 x[0], x[1], x[2], x[3], x[4], x[5]);
      fmt::print("angle[6]={:.3f}({:.1f}deg) w[7]={:.3f}({:.1f}deg/s)\n",
                 x[6], x[6]*57.3, x[7], x[7]*57.3);
      fmt::print("r[8]={:.3f} l[9]={:.3f} h[10]={:.3f}\n", x[8], x[9], x[10]);
      
      // Print key covariance values from P matrix
      auto & ekf = target.ekf();
      auto P = ekf.P;
      fmt::print("P(w)={:.3f} P(angle)={:.3f} | P(r)={:.3f} P(l)={:.3f} P(z)={:.3f}\n",
                 P(7,7), P(6,6), P(8,8), P(9,9), P(4,4));
      fmt::print("P(vx)={:.3f} P(vy)={:.3f} P(vz)={:.3f}\n",
                 P(1,1), P(3,3), P(5,5));
      
      // Print residuals
      fmt::print("residuals: yaw={:.3f} pitch={:.3f} dist={:.3f} angle={:.3f}\n",
                 ekf.data.at("residual_yaw"), ekf.data.at("residual_pitch"),
                 ekf.data.at("residual_distance"), ekf.data.at("residual_angle"));
      fmt::print("match: last_id={} switch={} jumped={}\n",
                 target.last_id, target.is_switch(), target.jumped);
      fmt::print("nis={:.3f} nees={:.3f} | nis_fail={} nees_fail={}\n",
                 ekf.data.at("nis"), ekf.data.at("nees"),
                 ekf.data.at("nis_fail"), ekf.data.at("nees_fail"));
      fmt::print("dt={:.3f}s distance={:.2f}m\n", dt, 
                 std::sqrt(x[0]*x[0] + x[2]*x[2] + x[4]*x[4]));

      data["target_x"] = x[0];
      data["target_vx"] = x[1];
      data["target_y"] = x[2];
      data["target_vy"] = x[3];
      data["target_z"] = x[4];
      data["target_vz"] = x[5];
      data["target_a"] = x[6] * 57.3;
      data["target_w"] = x[7];
      data["target_r"] = x[8];
      data["target_l"] = x[9];
      data["target_h"] = x[10];
      data["last_id"] = target.last_id;
      data["distance"] = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);

      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    data["gimbal_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_pitch"] = gimbal_state.pitch * 57.3;
    data["bullet_speed"] = gimbal_state.bullet_speed;

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
