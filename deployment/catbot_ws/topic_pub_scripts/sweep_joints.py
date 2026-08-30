#!/usr/bin/env python3
"""
Publishes a slow sine sweep of +-30 degrees on all joints to /joint_commands.
All joints sweep in phase together so motion is easy to see in the sim.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

JOINT_NAMES = [
    'BL_Hip_Joint', 'BR_Hip_Joint', 'FL_Hip_Joint', 'FR_Hip_Joint',
    'BL_Thigh_Joint', 'BR_Thigh_Joint', 'FL_Thigh_Joint', 'FR_Thigh_Joint',
    'BL_Foot_Joint', 'BR_Foot_Joint', 'FL_Foot_Joint', 'FR_Foot_Joint',
]

AMPLITUDE = math.radians(30)   # 30 degrees in radians
PERIOD    = 10.0                 # seconds per full sweep cycle
RATE_HZ   = 50                 # publish rate
# EMA smoothing on the published command, same idea as policy_runner.py's
# action_filter_alpha. This bridge/test script has no policy upstream to
# smooth for it, so it does its own. 1.0 = no smoothing; lower = smoother
# but more lag. Mostly a no-op for a slow sine like this one, but matters
# at startup (0 -> sine's initial value) and for any faster/steppier test
# pattern later.
SMOOTHING_ALPHA = 0.5


class JointSweep(Node):
    def __init__(self):
        super().__init__('joint_sweep')
        self.pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.t = 0.0
        self.dt = 1.0 / RATE_HZ
        self.filtered = [0.0] * len(JOINT_NAMES)
        self.create_timer(self.dt, self.publish)
        self.get_logger().info(
            f'Sweeping all joints +-30 deg, period={PERIOD}s at {RATE_HZ}Hz. Ctrl+C to stop.'
        )

    def publish(self):
        angle = AMPLITUDE * math.sin(2 * math.pi * self.t / PERIOD)
        self.t += self.dt

        targets = [angle] * 8 + [angle] * 4
        self.filtered = [
            SMOOTHING_ALPHA * target + (1.0 - SMOOTHING_ALPHA) * prev
            for target, prev in zip(targets, self.filtered)
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.filtered
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = JointSweep()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
