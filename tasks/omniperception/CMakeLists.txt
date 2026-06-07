cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2026.1.0/runtime/cmake/")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(omniperception OBJECT
    armor_selector.cpp
    decider.cpp
    perceptron.cpp
    debug_view.cpp
)
target_link_libraries(omniperception openvino::runtime)
