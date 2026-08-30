import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    default_policy_path = os.path.join(
        get_package_share_directory('ankybot_policy_runner'), 'policy', 'policy.onnx')

    imu_launch = os.path.join(
        get_package_share_directory('ankybot_imu'), 'launch', 'bno085_publisher.launch.py')
    i2c_bridge_launch = os.path.join(
        get_package_share_directory('ankybot_i2c_bridge'), 'launch', 'i2c_bridge.launch.py')
    policy_runner_launch = os.path.join(
        get_package_share_directory('ankybot_policy_runner'), 'launch', 'policy_runner.launch.py')
    speech_detection_launch = os.path.join(
        get_package_share_directory('speech_detection'), 'launch', 'speech_detection.launch.py')

    check_devices_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'check_i2c_devices.py')
    imu_diag_script = os.path.join(
        get_package_share_directory('ankybot_imu'), 'imu_diag.py')
    check_wifi_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'check_wifi.py')
    check_mic_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'check_microphone.py')
    check_dino_head_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'check_dino_head.py')
    spoof_joint_states_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'spoof_joint_states.py')
    fake_imu_script = os.path.join(
        get_package_share_directory('catbot_bringup'), 'scripts', 'fake_imu.py')

    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value=default_policy_path,
        description='Absolute path to the exported ONNX policy file')
    no_mega_arg = DeclareLaunchArgument(
        'no_mega',
        default_value='False',
        description='Skip the Arduino Mega (mega_feedback_reader_node) and spoof /joint_states instead')
    no_speech_recognizer_arg = DeclareLaunchArgument(
        'no_speech_recognizer',
        default_value='False',
        description='Skip speech_2_txt (e.g. running it on another machine instead) - command_interpreter/dino_head_controller still start and just listen on /speech_to_text')
    no_imu_arg = DeclareLaunchArgument(
        'no_imu',
        default_value='False',
        description='Skip the real BNO085 IMU node and spoof /imu/data instead (topic_pub_scripts/fake_imu.py) - combine with no_mega:=True for a full-spoof bringup that only needs the PCA9685 physically present')
    no_speech_detection_arg = DeclareLaunchArgument(
        'no_speech_detection',
        default_value='False',
        description='Skip the whole speech_detection package (not just the mic node - no_speech_recognizer only does that). command_interpreter continuously republishes /cmd_vel at 10Hz (zero by default) regardless of no_speech_recognizer, which fights a manual /cmd_vel test publisher on the same topic - set this True when driving /cmd_vel by hand (e.g. topic_pub_scripts/test_cmd_vel.py)')

    # Stage 1: confirm PCA9685/Mega/BNO085 all ack before starting anything - scans all three before reporting, names every missing device.
    check_devices = ExecuteProcess(
        cmd=['python3', check_devices_script,
             '--skip-mega', LaunchConfiguration('no_mega'),
             '--skip-imu', LaunchConfiguration('no_imu')],
        name='check_i2c_devices',
        output='screen',
    )

    # Stage 2 (gated on stage 1): confirms the BNO085 actually produces readable data, not just that its address acks - reuses imu_diag.py.
    check_imu_health = ExecuteProcess(
        cmd=['python3', imu_diag_script, '--quiet'],
        name='check_imu_health',
        output='screen',
    )

    # Non-blocking background monitors, not part of the pass/fail gate above - never delay or abort the launch, just run forever alongside everything else reporting once/3s while down.
    check_wifi = ExecuteProcess(
        cmd=['python3', check_wifi_script],
        name='check_wifi',
        output='screen',
    )
    check_microphone = ExecuteProcess(
        cmd=['python3', check_mic_script],
        name='check_microphone',
        output='screen',
    )
    check_dino_head = ExecuteProcess(
        cmd=['python3', check_dino_head_script],
        name='check_dino_head',
        output='screen',
    )

    # only runs with no_mega:=True - stands in for mega_feedback_reader_node's /joint_states so policy_runner still gets feedback.
    spoof_joint_states = ExecuteProcess(
        cmd=['python3', spoof_joint_states_script],
        name='spoof_joint_states',
        output='screen',
        condition=IfCondition(LaunchConfiguration('no_mega')),
    )

    # only runs with no_imu:=True - stands in for bno085_publisher's /imu/data.
    fake_imu = ExecuteProcess(
        cmd=['python3', fake_imu_script],
        name='fake_imu',
        output='screen',
        condition=IfCondition(LaunchConfiguration('no_imu')),
    )

    imu_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(imu_launch),
        condition=UnlessCondition(LaunchConfiguration('no_imu')))
    i2c_bridge_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(i2c_bridge_launch),
        launch_arguments={
            'no_mega': LaunchConfiguration('no_mega'),
        }.items())
    policy_runner_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(policy_runner_launch),
        launch_arguments={
            'policy_path': LaunchConfiguration('policy_path'),
        }.items())
    speech_detection_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(speech_detection_launch),
        launch_arguments={
            'no_speech_recognizer': LaunchConfiguration('no_speech_recognizer'),
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('no_speech_detection')))

    def on_devices_checked(event, context):
        if event.returncode != 0:
            return [
                LogInfo(msg='Launch Aborted'),
                EmitEvent(event=Shutdown(reason='I2C device presence check failed')),
            ]
        no_imu = LaunchConfiguration('no_imu').perform(context).strip().lower() in ('true', '1', 'yes')
        if no_imu:
            # no real BNO085 to health-check when it's spoofed - go straight to starting the stack.
            return [
                LogInfo(msg='I2C device presence check passed - IMU spoofed (no_imu:=True), skipping IMU health check.'),
            ] + start_stack_actions()
        return [
            LogInfo(msg='I2C device presence check passed - checking IMU sensor health...'),
            check_imu_health,
        ]

    def start_stack_actions():
        return [
            imu_include,
            i2c_bridge_include,
            policy_runner_include,
            speech_detection_include,
            spoof_joint_states,
            fake_imu,
        ]

    def on_imu_health_checked(event, context):
        if event.returncode != 0:
            return [
                LogInfo(msg=(
                    'IMU health check failed - see the FAIL line(s) above '
                    'for which stage failed. Aborting bringup, no nodes '
                    'were started.')),
                EmitEvent(event=Shutdown(reason='IMU health check failed')),
            ]
        return [
            LogInfo(msg='IMU health check passed - starting full stack.'),
        ] + start_stack_actions()

    return LaunchDescription([
        policy_path_arg,
        no_mega_arg,
        no_speech_recognizer_arg,
        no_imu_arg,
        no_speech_detection_arg,
        check_devices,
        check_wifi,
        check_microphone,
        check_dino_head,
        RegisterEventHandler(
            OnProcessExit(target_action=check_devices, on_exit=on_devices_checked)),
        RegisterEventHandler(
            OnProcessExit(target_action=check_imu_health, on_exit=on_imu_health_checked)),
    ])
