cmake_minimum_required(VERSION 3.10)
project(openvino_demo)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(demo src/main.cpp)
target_link_libraries(demo PRIVATE openvino::runtime ${OpenCV_LIBS})

# 添加包含目录
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS} inc)