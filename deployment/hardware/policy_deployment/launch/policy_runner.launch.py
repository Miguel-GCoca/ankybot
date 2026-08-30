import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('ankybot_policy_runner'),
        'config',
        'ankybot_policy.yaml',
    )

    return LaunchDescription([
        Node(
            package='ankybot_policy_runner',
            executable='policy_runner',
            name='policy_runner',
            output='screen',
            parameters=[config, {'use_sim_time':True}],
        ),
    ])
