import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ankybot_policy_runner')

    config = os.path.join(pkg_share, 'config', 'ankybot_policy.yaml')

    default_policy_path = os.path.join(pkg_share, 'policy', 'policy.onnx')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use Isaac Sim clock (True) or wall clock (False) for hardware'),
        DeclareLaunchArgument(
            'policy_path',
            default_value=default_policy_path,
            description='Absolute path to the exported ONNX policy file'),
        DeclareLaunchArgument(
            'use_height_command',
            default_value='True',
            description='Include height command in observation (True=49-obs, False=48-obs)'),
        Node(
            package='ankybot_policy_runner',
            executable='policy_runner',
            name='policy_runner',
            output='screen',
            parameters=[
                config,
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'policy_path': LaunchConfiguration('policy_path'),
                    'use_height_command': LaunchConfiguration('use_height_command'),
                },
            ],
        ),
    ])
