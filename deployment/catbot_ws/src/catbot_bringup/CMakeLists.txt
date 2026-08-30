cmake_minimum_required(VERSION 3.8)
project(catbot_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/check_i2c_devices.py
  scripts/check_dino_head.py
  scripts/spoof_joint_states.py
  scripts/fake_imu.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
