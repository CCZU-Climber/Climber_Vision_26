cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    plotter.cpp
    trajectory.cpp
    air_resist_trajectory.cpp
    recorder.cpp
    replay_reader.cpp
    replay_utils.cpp
    stats.cpp
    logger.cpp
    pid.cpp
    crc.cpp
)

add_library(frame_source STATIC frame_source.cpp)
target_link_libraries(frame_source PUBLIC tools io)
