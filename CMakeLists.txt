cmake_minimum_required(VERSION 3.16.3)

project(Climber_Vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2026.1.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

link_libraries(${CERES_LIBRARIES})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(sentry src/sentry.cpp)
add_executable(sentry_debug src/sentry_debug.cpp)
add_executable(sentry_aim src/sentry_aim.cpp)
add_executable(infantry src/infantry.cpp)
add_executable(uav src/uav.cpp)
add_executable(standard src/standard.cpp)

target_link_libraries(infantry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(sentry_aim ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools io)
##################calibration################
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_usb_camera calibration/calibrate_usb_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)
add_executable(serial_delay calibration/serial_delay.cpp)

target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(calibrate_usb_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(serial_delay ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
##################tests##################

###auto aim test
add_executable(aim_detect_test test/aim_detect_test.cpp)
add_executable(auto_aim_test test/auto_aim_test.cpp)
add_executable(mini_system test/mini_system.cpp)
add_executable(camera_test test/camera_test.cpp)
add_executable(infantry_mpc_debug test/infantry_mpc_debug.cpp)
add_executable(handeye_test test/handeye_test.cpp)
add_executable(cboard_test test/Cboard_test.cpp)
add_executable(cboard_response_test test/cboard_response_test.cpp)
add_executable(gimbal_response_test test/gimbal_response_test.cpp)
add_executable(gimbal_test test/gimbal_test.cpp)
add_executable(gimbal_ekf_test test/gimbal_ekf_test.cpp)
add_executable(fire_test test/fire_test.cpp)

target_link_libraries(aim_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(mini_system ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(infantry_mpc_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(cboard_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(gimbal_ekf_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
###auto_buff test

add_executable(auto_buff_test test/auto_buff_test.cpp)
add_executable(buff_detect_test test/buff_detect_test.cpp)
add_executable(replay_reader_test test/replay_reader_test.cpp)
add_executable(buff_target_recovery_test test/buff_target_recovery_test.cpp)
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff auto_aim frame_source tools io)
target_link_libraries(buff_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff auto_aim frame_source tools io)
target_link_libraries(replay_reader_test ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(buff_target_recovery_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff auto_aim tools io)


####omniperception test
add_executable(multi_usbcamera_test test/multi_usbcamera_test.cpp)
add_executable(aim_omniperception_test test/aim_omniperception_test.cpp)

target_link_libraries(aim_omniperception_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
##################ROS2相关代码##################
# # 检查 ROS 环境
# find_package(ament_cmake QUIET)
# find_package(rclcpp QUIET)
# find_package(std_msgs QUIET)
# find_package(rosidl_typesupport_cpp QUIET)
# find_package(sp_msgs QUIET)
# # 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
# if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
#     # 编译 sentry 和 publish_test
#     add_executable(sentry src/sentry.cpp)
#     add_executable(sentry_bp src/sentry_bp.cpp)
#     add_executable(sentry_debug src/sentry_debug.cpp)
#     add_executable(sentry_multithread src/sentry_multithread.cpp)
#     add_executable(publish_test tests/publish_test.cpp)
#     add_executable(subscribe_test tests/subscribe_test.cpp)
#     add_executable(topic_loop_test tests/topic_loop_test.cpp)

#     target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
#     target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
#     target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
#     target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
#     target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
#     target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
#     target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
# else()
#     message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
# endif()
