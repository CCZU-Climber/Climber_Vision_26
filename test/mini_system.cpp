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
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
// #include "tools/plotter.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"


using namespace std::chrono_literals;  

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明 }"
    "{@config-path   | configs/leg_aim.yaml | yaml配置文件的路径}";
    // "{tradition t    | true | 是否使用传统方法识别}";

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
    auto use_tradition = tools::read<bool>(yaml, "use_traditional");
    
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
    auto_aim::Shooter shooter(config_path);

    cv::Mat img, drawing;
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::chrono::steady_clock::time_point t;

    Eigen::Quaterniond q;

    io::Camera camera(config_path);
    io::Gimbal gimbal(config_path);
    double last_t = -1;

    while (!exiter.exit())
    {
        camera.read(img, t);
        if (img.empty())
            break;
        q = gimbal.q(t - 1ms);

        auto last = std::chrono::steady_clock::now();

        solver.set_R_gimbal2world(q);
        Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

        if (use_tradition) armors = detector.detect(img);
        else armors = yolo.detect(img);
        cv::Mat draw_img = img.clone();
        for (const auto &armor : armors)
        {
            // std::vector<cv::Point> armor_point(4);
            // for (int i = 0; i < 4; i++)
            // {
            //     armor_point[i] = cv::Point(static_cast<int>(armor.points[i].x), static_cast<int>(armor.points[i].y));
            // }
            // cv::polylines(draw_img, std::vector<std::vector<cv::Point>>{armor_point}, true, cv::Scalar(0, 255, 0), 2);

            tools::draw_text(draw_img, fmt::format("ID:{},conf{:.2f},type:{}",
                auto_aim::ARMOR_NAMES[armor.name],
                armor.confidence, 
                auto_aim::ARMOR_TYPES[armor.type]),
                 armor.center);
        }

        targets = tracker.track(armors, t);

        if (!targets.empty())
        {
            auto target = targets.front();
            tools::draw_text(draw_img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

            // 当前帧target更新后
            std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
            for (const Eigen::Vector4d &xyza : armor_xyza_list)
            {
                auto image_points =
                    solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);          
                tools::draw_points(draw_img, image_points, {0, 255, 255});
            }

            // aimer瞄准位置
            auto aim_point = aimer.debug_aim_point;
            Eigen::Vector4d aim_xyza = aim_point.xyza;
            auto image_points =
                solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            if (aim_point.valid)
                tools::draw_points(draw_img, image_points, {0, 0, 255});  // red
            else
                tools::draw_points(draw_img, image_points, {255, 0, 0});  // blue

            auto now = std::chrono::steady_clock::now();

            auto dt = tools::delta_time(now, last);
            tools::logger()->info("{:.2f} fps", 1 / dt);
        }
        solver.draw_armor_info(draw_img, armors.front());
        cv::resize(draw_img, draw_img, {}, 0.5, 0.5); // 显示时缩小图片尺寸
        // cv::putText(draw_img, fmt::format("{}", use_tradition ? "CV" : "YOLO"), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
        cv::imshow("reprojection", draw_img);

        auto key = cv::waitKey(1);
        if (key == 'q')
            break;

        auto command = aimer.aim(targets, t, gimbal.state().bullet_speed, q);
        // 发射逻辑
        command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

        gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
    }

    return 0;
}