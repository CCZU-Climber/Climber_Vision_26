#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <deque>
#include <numeric>
#include <algorithm>

#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/detector.hpp"

#include "tools/exiter.hpp"
#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明 }"
    "{@config-path   | configs/test_aim.yaml | yaml配置文件的路径}"
    "{output o       | logs/serial_delay.csv | CSV输出路径}";

using namespace std::chrono_literals;

struct Sample {
    double t;              // seconds since start (use detection end time)
    double vision_x;       // pixel x displacement (centered)
    double vision_y;       // pixel y displacement (centered)
    double imu_yaw_deg;    // yaw in degrees
    double imu_pitch_deg;  // pitch in degrees
};

struct DelayEstimate {
    double yaw_ms;      // delay from vision_x vs yaw
    double pitch_ms;    // delay from vision_y vs pitch
    double yaw_corr;    // correlation coefficient for yaw
    double pitch_corr;  // correlation coefficient for pitch
};

static constexpr double kMinVisionSpanPx   = 40.0; // 水平像素跨度至少 40px
static constexpr double kMinVisionSpanPy   = 30.0; // 垂直像素跨度至少 30px
static constexpr double kMinYawSpanDeg     = 2.0;  // 云台 yaw 跨度至少 2°
static constexpr double kMinPitchSpanDeg   = 1.5;  // 云台 pitch 跨度至少 1.5°
// 互相关最大搜索时长（秒），防止真实延时大于旧的 300ms 范围
static constexpr double kMaxDelaySearchS   = 3.0;  // 3s 上限

// 线性插值到等间隔时间轴，便于稳定互相关
static bool resample_uniform(const std::deque<Sample>& buf, double step_s,
                             std::vector<double>& out_t,
                             std::vector<double>& out_vx,
                             std::vector<double>& out_vy,
                             std::vector<double>& out_yaw,
                             std::vector<double>& out_pitch) {
    if (buf.size() < 2) return false;
    const double t0 = buf.front().t;
    const double t1 = buf.back().t;
    if (t1 - t0 < step_s * 10) return false; // 时长太短

    const int M = static_cast<int>((t1 - t0) / step_s) + 1;
    out_t.resize(M);
    out_vx.resize(M);
    out_vy.resize(M);
    out_yaw.resize(M);
    out_pitch.resize(M);

    size_t j = 0; // 游标
    for (int i = 0; i < M; ++i) {
        const double tg = t0 + i * step_s;
        while (j + 1 < buf.size() && buf[j + 1].t < tg) {
            ++j;
        }
        // 边界保护
        const size_t j1 = std::min(j + 1, buf.size() - 1);
        const double tA = buf[j].t, tB = buf[j1].t;
        const double alpha = (tB - tA) > 1e-6 ? (tg - tA) / (tB - tA) : 0.0;
        const auto lerp = [&](double a, double b) { return a + alpha * (b - a); };
        out_t[i] = tg;
        out_vx[i] = lerp(buf[j].vision_x, buf[j1].vision_x);
        out_vy[i] = lerp(buf[j].vision_y, buf[j1].vision_y);
        out_yaw[i] = lerp(buf[j].imu_yaw_deg, buf[j1].imu_yaw_deg);
        out_pitch[i] = lerp(buf[j].imu_pitch_deg, buf[j1].imu_pitch_deg);
    }
    return true;
}

    // 简单互相关，返回 yaw 与 pitch 两个轴的时间偏移（毫秒）
    static DelayEstimate compute_delay_ms(const std::deque<Sample>& buf) {
    // 使用插值后的等间隔序列，减少丢帧/不均匀采样的影响
    std::vector<double> t, vx, vy, yyaw, ypitch;
    if (!resample_uniform(buf, 0.01, t, vx, vy, yyaw, ypitch)) { // 100 Hz 时间轴
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    const int N = static_cast<int>(vx.size());
    if (N < 60) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    // 零均值，抑制漂移
    const auto mean = [](const std::vector<double>& a){
        return std::accumulate(a.begin(), a.end(), 0.0) / std::max<size_t>(1, a.size());
    };
    const auto span = [](const std::vector<double>& a){
        auto [mn, mx] = std::minmax_element(a.begin(), a.end());
        return (*mx) - (*mn);
    };

    const double mvx = mean(vx), mvy = mean(vy), myaw = mean(yyaw), mpitch = mean(ypitch);
    for (int i = 0; i < N; ++i) {
        vx[i] -= mvx;
        vy[i] -= mvy;
        yyaw[i] = -(yyaw[i] - myaw);       // 像素与 yaw 方向相反
        ypitch[i] = -(ypitch[i] - mpitch); // 像素 y 与 pitch 方向相反（俯仰）
    }

    const double span_vx = span(vx), span_vy = span(vy);
    const double span_yaw = span(yyaw), span_pitch = span(ypitch);

    struct AxisResult {
        double delay_ms;
        double corr;
        bool hit_edge;
    };

    const double dt = t[1] - t[0];
    // 最多搜索 3s 的延时；受限于可用样本数量
    const int max_shift = std::min(static_cast<int>(kMaxDelaySearchS / dt), N / 4);
    auto compute_axis = [&](const std::vector<double>& a, const std::vector<double>& b) -> AxisResult {
        double best_corr = -2.0;  // 支持负相关
        int best_k = 0;
        for (int k = -max_shift; k <= max_shift; ++k) {
            int i0 = std::max(0, -k);
            int i1 = std::min(N, N - k);
            if (i1 - i0 < N / 5) continue;  // 要求更多重叠样本
            double num = 0.0, da2 = 0.0, db2 = 0.0;
            for (int i = i0; i < i1; ++i) {
                const double da = a[i];
                const double db = b[i + k];
                num += da * db;
                da2 += da * da;
                db2 += db * db;
            }
            const double denom = std::sqrt(da2 * db2) + 1e-9;
            const double corr = num / denom;
            if (corr > best_corr) { best_corr = corr; best_k = k; }
        }
        const bool hit_edge = std::abs(best_k) == max_shift;
        return {best_k * dt * 1000.0, best_corr, hit_edge};
    };

    DelayEstimate est{std::numeric_limits<double>::quiet_NaN(), 
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN()};

    if (span_vx >= kMinVisionSpanPx && span_yaw >= kMinYawSpanDeg) {
        const AxisResult r = compute_axis(vx, yyaw);
        // 只有当相关系数 > 0.3 时才认为结果可靠
        if (std::abs(r.corr) > 0.3) {
            est.yaw_ms = r.delay_ms;
            est.yaw_corr = r.corr;
            if (r.hit_edge) {
                tools::logger()->warn("[SerialDelay] yaw delay hit search edge ({:.1f} ms)", r.delay_ms);
            }
        }
    }
    if (span_vy >= kMinVisionSpanPy && span_pitch >= kMinPitchSpanDeg) {
        const AxisResult r = compute_axis(vy, ypitch);
        if (std::abs(r.corr) > 0.3) {
            est.pitch_ms = r.delay_ms;
            est.pitch_corr = r.corr;
            if (r.hit_edge) {
                tools::logger()->warn("[SerialDelay] pitch delay hit search edge ({:.1f} ms)", r.delay_ms);
            }
        }
    }

    return est;
}

int main(int argc, char *argv[])
{
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    auto config_path = cli.get<std::string>(0);
    auto output_path = cli.get<std::string>("output");
    auto yaml = tools::load(config_path);
    auto use_tradition = tools::read<bool>(yaml, "use_traditional");

    if (!cli.check()) {
        cli.printErrors();
        return -1;
    }

    // 确保输出目录存在
    auto parent = std::filesystem::path(output_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream csv(output_path);
    csv << "t_frame_s,t_detect_s,dt_detect_ms,center_x,center_y,armor_count,"
           "imu_w,imu_x,imu_y,imu_z,imu_yaw_deg,imu_pitch_deg" << std::endl;

    tools::logger()->info("[SerialDelay] logging to {}", output_path);

    tools::Exiter exiter;
    auto_aim::YOLO yolo(config_path, false);
    auto_aim::Detector detector(config_path, false);

    io::Camera camera(config_path);
    io::Gimbal gimbal(config_path);

    // 加载 IMU 到云台的坐标转换（参考手眼标定）
    Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();
    try {
        if (yaml["R_gimbal2imubody"]) {
            auto Rdata = yaml["R_gimbal2imubody"].as<std::vector<double>>();
            if (Rdata.size() == 9) {
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> M(Rdata.data());
                R_gimbal2imubody = M;
            } else {
                tools::logger()->warn("[SerialDelay] R_gimbal2imubody size {} invalid, using Identity", Rdata.size());
            }
        } else {
            auto calib = tools::load("configs/calibration.yaml");
            if (calib["R_gimbal2imubody"]) {
                auto Rdata = calib["R_gimbal2imubody"].as<std::vector<double>>();
                if (Rdata.size() == 9) {
                    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> M(Rdata.data());
                    R_gimbal2imubody = M;
                } else {
                    tools::logger()->warn("[SerialDelay] calib R_gimbal2imubody size {} invalid, using Identity", Rdata.size());
                }
            } else {
                tools::logger()->warn("[SerialDelay] R_gimbal2imubody not found, using Identity");
            }
        }
    } catch (...) {
        tools::logger()->warn("[SerialDelay] Failed to load R_gimbal2imubody, using Identity");
    }

    std::chrono::steady_clock::time_point frame_tp;
    std::chrono::steady_clock::time_point start_tp;
    bool start_inited = false;
    int frame_id = 0;
    std::deque<Sample> buffer; // 保存最近的样本
    const int max_buf = 1200;  // 更长窗口，约40s@30fps（增加互相关窗口长度）

    while (!exiter.exit())
    {
        cv::Mat img;
        camera.read(img, frame_tp);
        if (img.empty())
            continue;

        const auto detect_start = std::chrono::steady_clock::now();
        auto armors = use_tradition ? detector.detect(img) : yolo.detect(img);
        const auto detect_end = std::chrono::steady_clock::now();

        if (!start_inited) {
            start_tp = frame_tp; // 用首帧时间作为统一零点，避免负时间
            start_inited = true;
        }

        // 取面积最大的目标中心，便于后续对齐 IMU 序列
        cv::Point2f center(std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN());
        if (!armors.empty()) {
            const auto *best = &armors.front();
            int best_area = best->box.area();
            for (const auto &armor : armors) {
                const int area = armor.box.area();
                if (area > best_area) {
                    best = &armor;
                    best_area = area;
                }
            }
            center = best->center;
        }

        // 对齐时间：IMU 插值也用检测结束时刻，避免视觉和姿态时间基不一致
        const auto q = gimbal.q(detect_end);
        // IMU(机体) -> 云台世界坐标的转换（与手眼标定一致）
        Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
        Eigen::Matrix3d R_gimbal2world = R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
        const Eigen::Vector3d ypr_deg = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;

        const double t_frame_s = std::chrono::duration<double>(frame_tp - start_tp).count();
        const double t_detect_s = std::chrono::duration<double>(detect_end - start_tp).count();
        const double dt_detect_ms = tools::delta_time(detect_end, detect_start) * 1000.0;

        // 记录样本（像素位移以图像中心为零）
        double vision_x = std::numeric_limits<double>::quiet_NaN();
        double vision_y = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(center.x)) {
            vision_x = (center.x - img.cols * 0.5); // 像素偏移（未归一化）
        }
        if (!std::isnan(center.y)) {
            vision_y = (center.y - img.rows * 0.5);
        }
        if (!std::isnan(vision_x) && !std::isnan(vision_y)) {
            // 用检测结束时刻作为视觉样本时间戳，避免视觉序列提前
            buffer.push_back({t_detect_s, vision_x, vision_y, ypr_deg[0], ypr_deg[1]});
            if ((int)buffer.size() > max_buf) buffer.pop_front();
        }

        csv << fmt::format(
            "{:.6f},{:.6f},{:.3f},{:.1f},{:.1f},{},"  // time + detection
            "{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{:.3f}\n",  // imu
            t_frame_s, t_detect_s, dt_detect_ms, center.x, center.y, armors.size(),
            q.w(), q.x(), q.y(), q.z(), ypr_deg[0], ypr_deg[1]);

        // 轻量显示：仅画中心点与帧时间
        cv::Mat view = img.clone();
        if (!std::isnan(center.x)) {
            cv::circle(view, center, 5, {0, 0, 255}, 2);
        }
        // 计算窗口互相关延时（yaw/x 与 pitch/y）
        const DelayEstimate delay_ms = compute_delay_ms(buffer);
        cv::putText(view, fmt::format("algo {:.2f} ms", dt_detect_ms), {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
        if (!std::isnan(delay_ms.yaw_ms)) {
            cv::Scalar color = std::abs(delay_ms.yaw_corr) > 0.5 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::putText(view, fmt::format("yaw delay {:.1f} ms (r:{:.2f})", delay_ms.yaw_ms, delay_ms.yaw_corr), {20, 80},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
        if (!std::isnan(delay_ms.pitch_ms)) {
            cv::Scalar color = std::abs(delay_ms.pitch_corr) > 0.5 ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0);
            cv::putText(view, fmt::format("pitch delay {:.1f} ms (r:{:.2f})", delay_ms.pitch_ms, delay_ms.pitch_corr), {20, 120},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
        cv::putText(view, fmt::format("buf size: {}", buffer.size()), {20, 160},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
        cv::imshow("serial_delay", view);
        const auto key = cv::waitKey(1);
        if (key == 'q')
            break;

        frame_id++;
        if (frame_id % 100 == 0) {
            csv.flush();
            tools::logger()->info("[SerialDelay] logged {} frames", frame_id);
        }
    }

    csv.flush();
    tools::logger()->info("[SerialDelay] finished, total {} frames", frame_id);
    return 0;
}