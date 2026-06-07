#include <fmt/core.h>

#include <chrono>
#include <deque>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "io/gimbal/gimbal.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明 }"
    "{window w       | 1.0  | 滑动窗口时长(秒)，用于计算实时频率}"
    "{@config-path   | configs/sentry.yaml | yaml配置文件的路径}";

int main(int argc, char *argv[])
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

    auto config_path = cli.get<std::string>(0);
    double window_sec = cli.get<double>("window");

    tools::Exiter exiter;
    io::Gimbal gimbal(config_path);

    uint64_t total_count = 0;
    std::deque<std::chrono::steady_clock::time_point> recv_times;

    auto stat_timer = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point prev_time{};
    double min_interval_ms =  1e9;
    double max_interval_ms = -1e9;
    double sum_interval_ms = 0.0;
    uint64_t interval_count = 0;

    tools::logger()->info("[Gimbal Freq Test] 开始测试，滑动窗口 = {:.2f} s", window_sec);

    while (!exiter.exit())
    {
        auto t_now = std::chrono::steady_clock::now();
        Eigen::Quaterniond q = gimbal.q(t_now - 1ms);
        auto recv_time = std::chrono::steady_clock::now();

        ++total_count;
        recv_times.push_back(recv_time);

        auto window_dur = std::chrono::duration<double>(window_sec);
        while (!recv_times.empty() &&
               (recv_time - recv_times.front()) > window_dur)
        {
            recv_times.pop_front();
        }

        if (total_count > 1) {
            double interval_ms = std::chrono::duration<double, std::milli>(recv_time - prev_time).count();
            min_interval_ms = std::min(min_interval_ms, interval_ms);
            max_interval_ms = std::max(max_interval_ms, interval_ms);
            sum_interval_ms += interval_ms;
            ++interval_count;
        }
        prev_time = recv_time;

        double elapsed_stat = std::chrono::duration<double>(recv_time - stat_timer).count();
        if (elapsed_stat >= 1.0)
        {
            double realtime_hz = recv_times.size() / window_sec;

            double avg_interval_ms = (interval_count > 0) ? sum_interval_ms / interval_count : 0.0;
            double avg_hz = (avg_interval_ms > 0) ? 1000.0 / avg_interval_ms : 0.0;

            Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0) * 57.3;

            auto state = gimbal.state();

            tools::logger()->info(
                "[Gimbal FreqTest] 总帧数={:6d} | 实时频率={:6.1f} Hz (窗口{:.1f}s) | "
                "平均={:6.1f} Hz | 帧间隔 min={:.2f} ms avg={:.2f} ms max={:.2f} ms | "
                "yaw={:.2f}° pitch={:.2f}° | bullet_speed={:.1f} m/s",
                total_count,
                realtime_hz,
                window_sec,
                avg_hz,
                min_interval_ms,
                avg_interval_ms,
                max_interval_ms,
                euler[0], euler[1],
                state.bullet_speed);

            stat_timer = recv_time;
        }
    }

    double avg_interval_ms = (interval_count > 0) ? sum_interval_ms / interval_count : 0.0;
    double avg_hz = (avg_interval_ms > 0) ? 1000.0 / avg_interval_ms : 0.0;
    tools::logger()->info(
        "[Gimbal FreqTest] 测试结束。总帧数={} | 平均频率={:.2f} Hz | "
        "帧间隔 min={:.2f} ms avg={:.2f} ms max={:.2f} ms",
        total_count, avg_hz, min_interval_ms, avg_interval_ms, max_interval_ms);

    return 0;
}
