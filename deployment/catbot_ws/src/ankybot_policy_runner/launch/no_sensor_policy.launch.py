import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ankybot_policy_runner')

    config = os.path.join(pkg_share, 'config', 'no_sensor_policy.yaml')

    # No default: the Ankybot-Static-v3 env_cfg (36-obs joint_pos+joint_vel+
    # last_action) has no exported checkpoint yet as of 2026-07-05 (only
    # logs/rsl_rl/v3_static_env/2026-07-04_16-05-03/model_0.pt exists, not
    # yet exported to onnx). Pass policy_path explicitly at launch time.
    # Whatever checkpoint is used, keep the .onnx and its .onnx.data
    # sidecar together in the training log's exported/ dir rather than
    # renaming/copying them - the .onnx references the sidecar by that
    # exact filename.
    default_policy_path = ''

    return LaunchDescription([
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
                    'use_sim_time': False,
                    'policy_path': LaunchConfiguration('policy_path'),
                },
            ],
        ),
    ])
