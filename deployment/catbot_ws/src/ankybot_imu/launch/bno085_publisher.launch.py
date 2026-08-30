import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ankybot_imu')
    config = os.path.join(pkg_share, 'config', 'bno085.yaml')

    return LaunchDescription([
        Node(
            package='ankybot_imu',
            executable='bno085_publisher',
            name='bno085_publisher',
            output='screen',
            parameters=[config],
        ),
    ])
