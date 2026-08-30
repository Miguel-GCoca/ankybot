from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # beta_speech_recognition_node.py has no registered entry point, intentionally not launched here.
    return LaunchDescription([
        DeclareLaunchArgument(
            'no_speech_recognizer',
            default_value='False',
            description='Skip speech_2_txt (e.g. running it on another machine instead) - command_interpreter/dino_head_controller still start and just listen on /speech_to_text'),
        Node(
            package='speech_detection',
            executable='speech_2_txt',
            name='speech_recognizer',
            output='screen',
            condition=UnlessCondition(LaunchConfiguration('no_speech_recognizer')),
        ),
        Node(
            package='speech_detection',
            executable='command_interpreter',
            name='command_interpreter',
            output='screen',
        ),
        Node(
            package='speech_detection',
            executable='dino_head_controller',
            name='dino_head_controller',
            output='screen',
        ),
    ])
