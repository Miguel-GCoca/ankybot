#!/bin/bash
set -e
source /opt/ros/${ROS_DISTRO:?set ROS_DISTRO or hardcode the distro name here}/setup.bash
source "${CATBOT_WS:?set CATBOT_WS to the colcon workspace root, e.g. ~/catbot_ws}/install/setup.bash"
exec ros2 launch catbot_bringup ankybot_bringup.launch.py no_mega:=True
