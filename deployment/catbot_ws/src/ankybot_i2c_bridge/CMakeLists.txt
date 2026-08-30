cmake_minimum_required(VERSION 3.8)
project(ankybot_i2c_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_library(pca9685_lib
  src/i2c_device.cpp
  src/pca9685.cpp
)

add_executable(pca9685_commander_node src/pca9685_commander_node.cpp)
target_link_libraries(pca9685_commander_node pca9685_lib)
ament_target_dependencies(pca9685_commander_node rclcpp sensor_msgs)

add_executable(mega_feedback_reader_node src/mega_feedback_reader_node.cpp)
target_link_libraries(mega_feedback_reader_node pca9685_lib)
ament_target_dependencies(mega_feedback_reader_node rclcpp sensor_msgs)

install(TARGETS
  pca9685_commander_node
  mega_feedback_reader_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
