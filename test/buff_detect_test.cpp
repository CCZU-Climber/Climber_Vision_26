#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/auto_buff/buff_big_group.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/frame_source.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/stats.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{mode m         | small                  | buff模式：small/big (同--m)}"
  "{video v        |                        | 视频文件路径，如果提供则从视频读取帧}"
  "{txt            |                        | 同名txt数据文件路径 (可选)}"
  "{speed          | 1                      | 回放倍速 (1=实时, 0=最快)}"
  "{@config-path   | configs/test_buff.yaml | yaml配置文件的路径}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return -1;
  }

  const std::string config_path = cli.get<std::string>(0);
  const std::string video_path = cli.get<std::string>("video");
  const std::string txt_path = cli.get<std::string>("txt");
  std::string buff_size = cli.get<std::string>("mode");

  if (buff_size != "small" && buff_size != "big") {
    tools::logger()->error("无效的buff大小参数: {}，仅支持 small 或 big", buff_size);
    return -1;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;
  cv::namedWindow("实时能量机关检测", cv::WINDOW_NORMAL);

  tools::FrameSource::Options frame_options;
  frame_options.config_path = config_path;
  frame_options.video_path = video_path;
  frame_options.txt_path = txt_path;
  frame_options.replay_speed = cli.get<double>("speed");
  tools::FrameSource frame_source(std::move(frame_options));
  const bool use_video = frame_source.replay_mode();

  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto target_params = auto_buff::load_target_params(config_path);

  auto_buff::BigTargetGroup big_group(target_params);

  std::unique_ptr<auto_buff::Target> target;
  if (buff_size == "big")
    target = std::make_unique<auto_buff::BigTarget>(target_params);
  else
    target = std::make_unique<auto_buff::SmallTarget>(target_params);

  cv::Mat frame;
  const auto time_base = std::chrono::steady_clock::now();
  int frame_count = 0;
  // ---- 定量指标收集 ----
  std::vector<double> rc_pnp_trad_errs;    // R_center: |PNP_R - TRAD_R| px
  std::vector<double> rc_ekf_pnp_errs;     // R_center: |EKF_R - PNP_R| px
  std::vector<double> rc_ekf_pnp_cont;     // R_center continuous accepted
  std::vector<double> blade_errs;          // Blade: mean corner |EKF - PNP| px (4 corners)
  std::vector<double> blade_errs_cont;     // Blade continuous accepted
  std::vector<double> tc_ekf_model_errs;   // Target_center: |EKF_reproj - model_target_pixel|
  std::vector<double> tc_pnp_model_errs;   // Target_center_obs: |PNP_reproj - model_target_pixel|
  std::vector<double> roll_errs_deg;
  int rc_ekf_pnp_gt200 = 0;
  bool prev_frame_accepted = false;
  tools::logger()->info("实时检测开始！按  q 退出，按  （空格）暂停");

  Eigen::Quaterniond prev_q = Eigen::Quaterniond::Identity();
  cv::Point2f prev_ekf_pixel(-1, -1);
  bool has_prev_quat = false;

  while (!exiter.exit()) {
    tools::FramePacket packet;
    if (!frame_source.read(packet)) {
      if (use_video) {
        tools::logger()->info("视频已播放完毕");
        break;
      }
      tools::logger()->warn("无法读取图像帧，跳过当前循环");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    frame = packet.image;
    auto frame_timestamp = packet.timestamp;

    frame_count++;

    // 实时视觉调试不依赖下位机，q 保持单位姿态；回放模式使用 txt 四元数。
    const auto q_current = packet.q;
    solver.set_R_gimbal2world(q_current);

    const double frame_timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(frame_timestamp - time_base).count();

    nlohmann::json debug_data;
    std::vector<auto_buff::PowerRune> observations;

    const bool is_small = (buff_size != "big");
    if (is_small)
      observations = detector.detect(frame, auto_buff::SMALL);
    else
      observations = detector.detect(frame, auto_buff::BIG);

    solver.solve(observations);

    std::optional<auto_buff::PowerRune> power_rune = std::nullopt;
    std::optional<auto_buff::BigTarget> big_debug_target = std::nullopt;
    auto_buff::Target * debug_target = target.get();

    if (is_small) {
      power_rune =
        observations.empty() ? std::nullopt : std::optional<auto_buff::PowerRune>(observations.front());
      target->get_target(power_rune, frame_timestamp);
    } else {
      big_group.update(observations, frame_timestamp);

      double best_pair_cost = std::numeric_limits<double>::max();
      for (int id = 0; id < auto_buff::BigTargetGroup::kSlotCount; ++id) {
        auto candidate = big_group.get_target_copy(id);
        if (!candidate.has_value()) continue;

        if (observations.empty()) {
          if (!big_debug_target.has_value()) big_debug_target = candidate;
          continue;
        }

        for (const auto & obs : observations) {
          double cost = std::abs(tools::limit_rad(candidate->get_roll() - obs.ypr_in_world[2]));
          for (int si = -5; si <= 5; ++si) {
            cost = std::min(cost, std::abs(tools::limit_rad(
              candidate->get_roll() - (obs.ypr_in_world[2] + si * 2.0 * CV_PI / 5.0))));
          }
          if (cost < best_pair_cost) {
            best_pair_cost = cost;
            big_debug_target = candidate;
            power_rune = obs;
          }
        }
      }

      if (big_debug_target.has_value()) debug_target = &big_debug_target.value();
    }

      // ---- R 中心统一语义说明 ----
      // MODEL_R: 模型原始R标中心 (raw[0] remap到kpt[5])
      // TRAD_R: Buff_Detector::get_r_center() 形态学中心 (r_center, 不被solver覆写)
      // PNP_R:  Solver::solve_one() 中 OBJECT_POINTS[6]=(0,0,0) 的 PnP 重投影 (pnp_r_center)
      // EKF_R:  SmallTarget: center_filtered() + 3-axis reproject; BigTarget: 7-state -> xyz -> reproject

      // 绘制 PnP 角点 & R 中心
      if (power_rune.has_value()) {
        auto pnp_corners = solver.reproject_buff(
          power_rune->xyz_in_world, power_rune->ypr_in_world);
        for (const auto & pt : pnp_corners)
          cv::circle(frame, pt, 4, cv::Scalar(0, 255, 255), 1);
        if (!pnp_corners.empty()) {
          const double flat_r_err = cv::norm(power_rune->r_center - power_rune->pnp_r_center);
          rc_pnp_trad_errs.push_back(flat_r_err);
        }
        cv::circle(frame, power_rune->model_r_center, 4, cv::Scalar(0, 165, 255), -1);
        cv::putText(
          frame, "MODEL_R", power_rune->model_r_center + cv::Point2f(8, 8),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 165, 255), 1);
        cv::circle(frame, power_rune->pnp_r_center, 7, cv::Scalar(0, 255, 255), -1);
        cv::putText(
          frame, "PNP_R", power_rune->pnp_r_center + cv::Point2f(8, -8),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
        cv::circle(frame, power_rune->r_center, 6, cv::Scalar(255, 0, 255), 1);
        cv::putText(
          frame, "TRAD_R", power_rune->r_center + cv::Point2f(8, -8),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 255), 1);
      }

      // 绘制 EKF 角点
      if (!debug_target->is_unsolve()) {
        Eigen::Vector3d ekf_xyz_w;
        double ekf_yaw, ekf_pitch, ekf_roll;
        if (is_small) {
          ekf_xyz_w = debug_target->center_filtered();
          ekf_yaw = debug_target->buff_yaw();
          ekf_pitch = debug_target->buff_pitch();
          ekf_roll = debug_target->get_roll();
        } else {
          auto ekf_x = debug_target->ekf_x();
          Eigen::Vector3d R_ypd(ekf_x[0], ekf_x[2], ekf_x[3]);
          ekf_xyz_w = tools::ypd2xyz(R_ypd);
          ekf_yaw = ekf_x[4];
          ekf_pitch = debug_target->buff_pitch();
          ekf_roll = ekf_x[5];
        }
        auto ekf_corners = solver.reproject_buff(
          ekf_xyz_w, Eigen::Vector3d(ekf_yaw, ekf_pitch, ekf_roll));
        if (!ekf_corners.empty()) {
          prev_q = q_current;
          prev_ekf_pixel = ekf_corners[3];
          has_prev_quat = true;
        }

        if (debug_target->is_predicted()) {
          for (const auto & pt : ekf_corners)
            cv::circle(frame, pt, 3, cv::Scalar(0, 165, 255), -1);

          // --- Prediction-frame debug: A/B/C reprojection comparison ---
          // A: reproject EKF world point with CURRENT gimbal (already done above = ekf_corners)
          // B: reproject SAME world point with PREVIOUS gimbal
          if (has_prev_quat) {
            solver.set_R_gimbal2world(prev_q);
            auto ekf_corners_B = solver.reproject_buff(
              ekf_xyz_w, Eigen::Vector3d(ekf_yaw, ekf_pitch, ekf_roll));
            solver.set_R_gimbal2world(q_current);  // restore
            for (const auto & pt : ekf_corners_B)
              cv::circle(frame, pt, 2, cv::Scalar(0, 255, 255), -1);  // yellow = B
          }
          // C: simple 2D pixel extrapolation (previous pixel + pixel delta)
          if (prev_ekf_pixel.x >= 0 && !ekf_corners.empty()) {
            cv::Point2f c_extrap(
              prev_ekf_pixel.x + (ekf_corners[3].x - prev_ekf_pixel.x),
              prev_ekf_pixel.y + (ekf_corners[3].y - prev_ekf_pixel.y));
            cv::circle(frame, c_extrap, 2, cv::Scalar(255, 255, 0), -1);  // cyan = C
          }

          // Debug log every 60 frames
          if (frame_count % 60 == 0) {
            Eigen::Matrix3d Rblade = tools::rotation_matrix(
              Eigen::Vector3d(ekf_yaw, ekf_pitch, ekf_roll));
            Eigen::Vector3d blade_w = ekf_xyz_w + Rblade * Eigen::Vector3d(0.0, 0.0, 0.7);
            tools::logger()->info("[PRED_DEBUG] frame={} q=({:.4f},{:.4f},{:.4f},{:.4f}) "
              "center_w=({:.3f},{:.3f},{:.3f}) blade_w=({:.3f},{:.3f},{:.3f}) "
              "ekf_px=({:.1f},{:.1f})",
              frame_count, q_current.w(), q_current.x(), q_current.y(), q_current.z(),
              ekf_xyz_w[0], ekf_xyz_w[1], ekf_xyz_w[2],
              blade_w[0], blade_w[1], blade_w[2],
              ekf_corners.empty() ? -1.0 : ekf_corners[3].x,
              ekf_corners.empty() ? -1.0 : ekf_corners[3].y);
          }
        } else {
          for (const auto & pt : ekf_corners)
            cv::circle(frame, pt, 3, cv::Scalar(255, 0, 0), -1);
        }

        // EKF_R vs PNP_R 误差
        if (power_rune.has_value()) {
          auto pnp_corners = solver.reproject_buff(
            power_rune->xyz_in_world, power_rune->ypr_in_world);
          if (!pnp_corners.empty() && !ekf_corners.empty()) {
            // R_center error
            double r_center_err = cv::norm(ekf_corners.back() - pnp_corners.back());
            rc_ekf_pnp_errs.push_back(r_center_err);
            if (r_center_err > 200.0) rc_ekf_pnp_gt200++;
            // Blade corner error (mean of 4 corners)
            double blade_sum = 0;
            for (int ci = 0; ci < 4; ci++)
              blade_sum += cv::norm(ekf_corners[ci] - pnp_corners[ci]);
            double blade_err = blade_sum / 4.0;
            blade_errs.push_back(blade_err);

            // Target center error: EKF_reproj vs model pixel (OBJECT_POINTS[4] = blade center)
            if (pnp_corners.size() > 4 && ekf_corners.size() > 4) {
              double tc_ekf = cv::norm(ekf_corners[4] - power_rune->target().center);
              double tc_pnp = cv::norm(pnp_corners[4] - power_rune->target().center);
              tc_ekf_model_errs.push_back(tc_ekf);
              tc_pnp_model_errs.push_back(tc_pnp);
            }
            {
              const double raw = std::abs(tools::limit_rad(ekf_roll - power_rune->ypr_in_world[2]));
              double best = raw;
              for (int si = -5; si <= 5; ++si) {
                best = std::min(best, std::abs(tools::limit_rad(
                  ekf_roll - (power_rune->ypr_in_world[2] + si * 2.0 * CV_PI / 5.0))));
              }
              roll_errs_deg.push_back(best * 180.0 / CV_PI);
            }

            // Continuous segment tracking
            bool this_accepted = !debug_target->is_predicted() && !debug_target->is_unsolve();
            if (this_accepted && prev_frame_accepted) {
              rc_ekf_pnp_cont.push_back(r_center_err);
              blade_errs_cont.push_back(blade_err);
            }
            prev_frame_accepted = this_accepted;
          }
        }
      }

      // 测距与EKF状态
      {
        const cv::Point org(20, 240);
        const int dy = 22;
        const auto fn = cv::FONT_HERSHEY_SIMPLEX;
        const double fs = 0.50;
        const auto c0 = cv::Scalar(200, 200, 200);
        int r = 0;

        double pd = power_rune.has_value() ? power_rune->ypd_in_world[2] : -1.0;
        double ed = -1.0;
        if (!debug_target->is_unsolve()) {
          ed = is_small ? debug_target->center_filtered().norm() : debug_target->ekf_x()[3];
        }
        cv::putText(frame,
          fmt::format("DIST PnP:{:.2f}m  EKF:{:.2f}m", pd, ed),
          org + cv::Point(0, dy * r++), fn, fs, c0, 1);

        if (!debug_target->is_unsolve()) {
          auto ex = debug_target->ekf_x();
          bool pred = debug_target->is_predicted();
          auto sc = pred ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);

          if (is_small) {
            cv::putText(frame,
              fmt::format("vroll:{:.2f}rad/s roll:{:.1f}deg",
                ex[1], ex[0] * 180.0 / CV_PI),
              org + cv::Point(0, dy * r++), fn, fs, c0, 1);
          } else {
            cv::putText(frame,
              fmt::format("spd:{:.2f}rad/s roll:{:.1f}deg",
                ex[6], ex[5] * 180.0 / CV_PI),
              org + cv::Point(0, dy * r++), fn, fs, c0, 1);
          }

          cv::putText(frame,
            fmt::format("STATUS: {}", pred ? "PRED" : "NORMAL"),
            org + cv::Point(0, dy * r++), fn, fs, sc, 1);
          if (!rc_ekf_pnp_errs.empty())
            cv::putText(frame,
              fmt::format("EKF_RC: {:.1f}px", rc_ekf_pnp_errs.back()),
              org + cv::Point(0, dy * r++), fn, fs,
              rc_ekf_pnp_errs.back() > 200.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1);
          if (!rc_pnp_trad_errs.empty())
            cv::putText(frame,
              fmt::format("RC_ERR: {:.1f}px", rc_pnp_trad_errs.back()),
              org + cv::Point(0, dy * r++), fn, fs, c0, 1);
        }
      }

    debug_data["frame_count"] = frame_count;
    debug_data["timestamp_ms"] = frame_timestamp_ms;
    debug_data["buff"]["status"] = observations.empty() ? "not_detected" : "detected";
    debug_data["buff"]["mode"] = buff_size;

    if (!observations.empty()) {
      for (size_t i = 0; i < observations.size(); ++i) {
        const auto & buff_data = observations[i];
        tools::draw_point(frame, buff_data.r_center, cv::Scalar(255, 0, 255), 4);
        tools::draw_point(frame, buff_data.target().center, cv::Scalar(0, 255, 0), 5);
      }
    } else {
      cv::putText(
        frame, "No Buff Detected", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(0, 0, 255), 2);
    }

    const std::string mode_text = std::string("MODE: ") + buff_size;
    cv::putText(
      frame, mode_text, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0),
      2);

    plotter.plot(debug_data);
    // cv::resize(frame, frame, {}, 0.5, 0.5);
    cv::imshow("实时能量机关检测", frame);

    const int key = cv::waitKey(use_video ? frame_source.replay_delay_ms() : 1);
    if (key == 113 || key == 27) {
      tools::logger()->info("用户按下退出键，实时检测结束");
      break;
    } else if (key == 32) {
      tools::logger()->info("实时检测暂停，再次按空格继续");
      cv::putText(
        frame, "Paused", cv::Point(frame.cols / 2 - 50, frame.rows / 2),
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0), 3);
      cv::resize(frame, frame, {}, 0.5, 0.5);
    cv::imshow("实时能量机关检测", frame);
      while (true) {
        const int pause_key = cv::waitKey(30);
        if (pause_key == 32) {
          tools::logger()->info("实时检测继续");
          break;
        } else if (pause_key == 113 || pause_key == 27) {
          cv::destroyAllWindows();
          tools::logger()->info("实时检测退出");
          return 0;
        }
      }
    }
  }

  cv::destroyAllWindows();

  // ---- 输出定量统计 ----
  tools::logger()->info("========================================");
  tools::logger()->info("  METRICS (mode={}, video={}, frames={})", buff_size,
    video_path.empty() ? "live" : video_path.substr(video_path.find_last_of('/')+1), frame_count);
  tools::logger()->info("========================================");
  tools::logger()->info("--- R_center Error (|EKF_reprojected - PnP_reprojected| at OBJECT_POINTS[6]=(0,0,0)) ---");
  tools::logger()->info("R_center_full:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(rc_ekf_pnp_errs), tools::p95(rc_ekf_pnp_errs), rc_ekf_pnp_errs.size());
  tools::logger()->info("R_center_cont:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(rc_ekf_pnp_cont), tools::p95(rc_ekf_pnp_cont), rc_ekf_pnp_cont.size());
  tools::logger()->info("R_center_full  >200px: {} frames ({:.1f}%)",
    rc_ekf_pnp_gt200, rc_ekf_pnp_errs.empty() ? 0.0 : 100.0*rc_ekf_pnp_gt200/rc_ekf_pnp_errs.size());
  tools::logger()->info("--- R_center |PNP_R - TRAD_R| (Solver PnP vs Detector TRAD) ---");
  tools::logger()->info("R_center_PNP_vs_TRAD:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(rc_pnp_trad_errs), tools::p95(rc_pnp_trad_errs), rc_pnp_trad_errs.size());
  tools::logger()->info("--- Blade Error (mean|EKF_corner_i - PnP_corner_i|, i=0..3) ---");
  tools::logger()->info("Blade_full:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(blade_errs), tools::p95(blade_errs), blade_errs.size());
  tools::logger()->info("Blade_cont:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(blade_errs_cont), tools::p95(blade_errs_cont), blade_errs_cont.size());
  tools::logger()->info("--- Target_center Error (|EKF_reproj_target - model_target_pixel|) ---");
  tools::logger()->info("Target_center_full:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(tc_ekf_model_errs), tools::p95(tc_ekf_model_errs), tc_ekf_model_errs.size());
  tools::logger()->info("--- Target_center_obs_pnp (|PNP_reproj_target - model_target_pixel|) ---");
  tools::logger()->info("Target_center_obs_pnp:  max={:.2f}px  p95={:.2f}px  n={}",
    tools::max_value(tc_pnp_model_errs), tools::p95(tc_pnp_model_errs), tc_pnp_model_errs.size());
  tools::logger()->info("--- Roll phase Error (min modulo 72deg, |EKF_roll - obs_roll|) ---");
  tools::logger()->info("Roll_phase:  max={:.2f}deg  p95={:.2f}deg  n={}",
    tools::max_value(roll_errs_deg), tools::p95(roll_errs_deg), roll_errs_deg.size());
  tools::logger()->info("========================================");
  tools::logger()->info("实时检测正常结束，已释放所有资源");

  return 0;
}
