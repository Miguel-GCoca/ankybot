#!/usr/bin/env python3
"""
Prints /joint_states at a fixed, human-readable rate (default 25Hz),
decoupled from the actual publish rate. `ros2 topic echo` prints every
message (a scrolling wall of text at the real ~50Hz publish rate);
`ros2 topic hz` only reports timing stats once per second. This subscribes
normally but only prints on its own timer, always showing the latest
message, formatted compactly on one line (overwritten in place).
"""
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ThrottledJointStateEcho(Node):
    def __init__(self, topic: str, rate_hz: float):
        super().__init__('echo_throttled')
        self.latest = None
        self.create_subscription(JointState, topic, self.on_msg, 10)
        self.create_timer(1.0 / rate_hz, self.print_latest)
        self.get_logger().info(f'Printing {topic} at {rate_hz}Hz. Ctrl+C to stop.')

    def on_msg(self, msg):
        self.latest = msg

    def print_latest(self):
        if self.latest is None:
            return
        parts = [
            f'{name}:{pos:+.3f}'
            for name, pos in zip(self.latest.name, self.latest.position)
        ]
        print('\r' + ' '.join(parts) + '   ', end='', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/joint_states')
    parser.add_argument('--rate', type=float, default=25.0)
    args = parser.parse_args()

    rclpy.init()
    node = ThrottledJointStateEcho(args.topic, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
