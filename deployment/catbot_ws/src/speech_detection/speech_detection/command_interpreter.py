#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.msg import String
from geometry_msgs.msg import Twist

# Linear speed bounds/step for incremental walk speed (m/s)
WALK_SPEED_DEFAULT = 0.4
WALK_SPEED_MIN = 0.2
WALK_SPEED_MAX = 0.8
WALK_SPEED_STEP = 0.1

# Fixed angular speed for turning (rad/s)
TURN_SPEED = 0.5

# How often to republish the current Twist while a motion is active (Hz)
PUBLISH_RATE_HZ = 10.0

ACTION_COMMANDS = {"roar", "sleep", "push up"}


class CommandInterpreter(Node):
    def __init__(self):
        super().__init__('command_interpreter')

        self.subscription = self.create_subscription(
            String,
            'speech_to_text',
            self.listener_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # /action publisher disabled 2026-07-21 - roar/sleep/push up deprioritized, not currently consumed by anything.
        # self.action_pub = self.create_publisher(String, 'action', 10)

        # Motion state: "stopped", "walking", "turning_left", "turning_right"
        self.motion = "stopped"
        self.linear_speed = 0.0
        self.angular_speed = 0.0

        # continuously republish cmd_vel so downstream consumers keep receiving commands until "stop" is heard.
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_current_twist)

        self.get_logger().info("Command interpreter: awaiting commands")

    def listener_callback(self, msg):
        command = msg.data.strip().lower()
        if not command:
            return

        self.get_logger().info(f"Command received: {command}")

        if command == "walk normal":
            self.motion = "walking"
            self.linear_speed = WALK_SPEED_DEFAULT
            self.angular_speed = 0.0

        elif command == "walk faster":
            if self.motion != "walking":
                self.motion = "walking"
                self.linear_speed = WALK_SPEED_DEFAULT
            else:
                self.linear_speed = min(self.linear_speed + WALK_SPEED_STEP, WALK_SPEED_MAX)
            self.angular_speed = 0.0
            self.get_logger().info(f"Walk speed now {self.linear_speed:.2f} m/s")

        elif command == "walk slower":
            if self.motion != "walking":
                self.motion = "walking"
                self.linear_speed = WALK_SPEED_DEFAULT
            else:
                self.linear_speed = max(self.linear_speed - WALK_SPEED_STEP, WALK_SPEED_MIN)
            self.angular_speed = 0.0
            self.get_logger().info(f"Walk speed now {self.linear_speed:.2f} m/s")

        elif command == "turn left":
            self.motion = "turning_left"
            self.linear_speed = 0.0
            self.angular_speed = TURN_SPEED

        elif command == "turn right":
            self.motion = "turning_right"
            self.linear_speed = 0.0
            self.angular_speed = -TURN_SPEED

        elif command == "stop":
            self.motion = "stopped"
            self.linear_speed = 0.0
            self.angular_speed = 0.0

        elif command in ACTION_COMMANDS:
            # actions (roar, sleep, push up) still halt movement; /action publish disabled 2026-07-21 (see __init__).
            self.motion = "stopped"
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            # self.publish_action(command)

        elif command == "not listening":
            self.get_logger().warn("Speech recognition not listening; stopping robot.")
            self.motion = "stopped"
            self.linear_speed = 0.0
            self.angular_speed = 0.0

        else:
            self.get_logger().warn(f"Unrecognized command: '{command}'")

    def publish_current_twist(self):
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist)

    # def publish_action(self, action):
    #     msg = String()
    #     msg.data = action
    #     self.action_pub.publish(msg)
    #     self.get_logger().info(f"Published action: '{action}'")


def main(args=None):
    rclpy.init(args=args)

    node = CommandInterpreter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
