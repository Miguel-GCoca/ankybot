import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ankybot_policy_runner')

    config = os.path.join(pkg_share, 'config', 'no_sensor_policy.yaml')

    # Exported checkpoint from the no_pos_no_imu/v1 stand-only env_cfg
    # (velocity_commands pinned to zero, no IMU/joint_pos/joint_vel obs
    # terms) - kept at its training log location rather than copied into
    # the package, since the .onnx references its external data file by
    # the fixed name "policy.onnx.data" in the same directory.
    default_policy_path = (
        '/workspace/Ankybot_v3/logs/rsl_rl/no_pos_no_imu/v1/exported/policy.onnx'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use Isaac Sim clock (True) or wall clock (False) for hardware'),
        DeclareLaunchArgument(
            'policy_path',
            default_value=default_policy_path,
            description='Absolute path to the exported ONNX policy file'),
        Node(
            package='ankybot_policy_runner',
            executable='no_sensor_policy',
            name='no_sensor_policy',
            output='screen',
            parameters=[
                config,
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'policy_path': LaunchConfiguration('policy_path'),
                },
            ],
        ),
    ])
