from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # sole I2C actuation path - serial_communication_py was deleted 2026-07-20, nothing else to double-drive the PCA9685 with.
    return LaunchDescription([
        DeclareLaunchArgument(
            'no_mega',
            default_value='False',
            description='Skip mega_feedback_reader_node (Arduino Mega not wired up yet)'),
        DeclareLaunchArgument(
            'mega_i2c_bus',
            default_value='2',
            description='Separate Pi I2C bus for the Mega, isolated from PCA9685/BNO085 on bus 1 (i2c2-pi5 overlay, GPIO4/5)'),
        Node(
            package='ankybot_i2c_bridge',
            executable='pca9685_commander_node',
            name='pca9685_commander',
            output='screen',
        ),
        Node(
            package='ankybot_i2c_bridge',
            executable='mega_feedback_reader_node',
            name='mega_feedback_reader',
            output='screen',
            parameters=[{'i2c_bus': LaunchConfiguration('mega_i2c_bus')}],
            condition=UnlessCondition(LaunchConfiguration('no_mega')),
        ),
    ])
