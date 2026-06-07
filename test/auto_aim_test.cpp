#include <fmt/core.h>

#include <chrono>
#include <fstream>
// #include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/omniperception/armor_selector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"
#include "tools/math_tools.hpp"
// #include "tools/plotter.hpp"
#include "io/camera.hpp"


using namespace std::chrono_literals; 

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明 }"
    "{tradition t    | false | 是否使用传统方法识别}"
    "{@config-path   | configs/test_aim.yaml | yaml配置文件的路径}";

int main(int argc, char *argv[])
{
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    
    auto config_path = cli.get<std::string>(0);
    auto yaml = tools::load(config_path);
    auto tradition_detect_ = tools::read<bool>(yaml, "tradition_detect");
    
    if (!cli.check()) {
        cli.printErrors();
        return -1;
    }

    // tools::Plotter plotter;
    tools::Exiter exiter;


    auto_aim::YOLO yolo(config_path, false);  // 第二个参数是debug标志，不是文件名
    auto_aim::Detector detector(config_path, false);
    auto_aim::Solver solver(config_path);
    auto_aim::Tracker tracker(config_path, solver);
    auto_aim::Aimer aimer(config_path);
    omniperception::ArmorSelector armor_selector(config_path);
    // auto_aim::Shooter shooter(config_path);

    cv::Mat img, drawing;
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::chrono::steady_clock::time_point t;

    // Eigen::Quaterniond q;

    // 帧率统计相关变量
    int frame_count = 0;              // 统计到目前为止的帧数
    double sum_dt_30 = 0.0;           // 累积最近 30 帧的整体时间，用于计算平均帧率
    double sum_dt_yolo_30 = 0.0;      // 累积最近 30 帧的 YOLO 时间，用于计算 YOLO 平均帧率
    
    io::Camera camera(config_path);
    while (!exiter.exit())
    {   
        cv::Mat img, drawing;
        std::list<auto_aim::Armor> armors;
        std::list<auto_aim::Target> targets;
        camera.read(img, t);
        if (img.empty())
            break;

        auto last = std::chrono::steady_clock::now();

        // solver.set_R_gimbal2world(q);
        // Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

        if (tradition_detect_) armors = detector.detect(img);
        else armors = yolo.detect(img);

        armor_selector.armor_filter(armors, omniperception::ArmorFilterScope::MainAim);
        armor_selector.set_priority(armors);

        auto yolo = std::chrono::steady_clock::now();
        // cv::Mat draw_img = img.clone();
       
        targets = tracker.track(armors, t);

        auto track = std::chrono::steady_clock::now();

        // cv::resize(img, img, {}, 0.5, 0.5); // 显示时缩小图片尺寸
        cv::namedWindow("reprojection", cv::WINDOW_FREERATIO);
        cv::imshow("reprojection", img);

        auto now = std::chrono::steady_clock::now();
        auto dt = tools::delta_time(now, last);
        auto dt_yolo = tools::delta_time(yolo, last);
        auto dt_track = tools::delta_time(track, yolo);

        // 实时帧率
        // tools::logger()->info("{:.2f} fps, yolo: {:.2f}ms, track: {:.2f}ms", 1 / dt, dt_yolo * 1000, dt_track * 1000);

        // 每 30 帧统计一次平均帧率（整体 + YOLO）
        frame_count++;
        sum_dt_30 += dt;
        sum_dt_yolo_30 += dt_yolo;
        if (frame_count % 30 == 0)
        {
            double avg_dt = sum_dt_30 / 30.0;
            double avg_fps = 1.0 / avg_dt;
            double avg_dt_yolo = sum_dt_yolo_30 / 30.0;
            double avg_fps_yolo = 1.0 / avg_dt_yolo;

            tools::logger()->info("[AVG] last 30 frames: total {:.2f} fps, yolo {:.2f} fps ({:.2f} ms)", 
                                  avg_fps, avg_fps_yolo, avg_dt_yolo * 1000.0);

            // 清零，重新统计下一组 30 帧
            sum_dt_30 = 0.0;
            sum_dt_yolo_30 = 0.0;
        }

        auto key = cv::waitKey(1);
        if (key == 'q')
            break;

    }

    return 0;
}
