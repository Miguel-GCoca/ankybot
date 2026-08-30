#!/usr/bin/env python3
import fcntl
import os
import time

import rclpy
from rclpy.node import Node
from example_interfaces.msg import String

I2C_SLAVE = 0x0703
DINO_ADDRESS = 0x5B
DINO_BUS = 3

CMD_IDLE = 1
CMD_HEARD_COMMAND = 2
CMD_WALKING = 3
CMD_TURNING = 4

ACTION_HOLD_S = 8.0

WAKE_COMMANDS = {"activated"}
TURN_COMMANDS = {"turn left", "turn right"}
WALK_COMMANDS = {"walk faster", "walk slower", "walk normal"}
IDLE_COMMANDS = {"stop", "not listening"}


class DinoHeadController(Node):
    def __init__(self):
        super().__init__('dino_head_controller')

        self.create_subscription(String, 'speech_to_text', self.speech_cb, 10)

        self.last_sent_mode = None
        self.action_hold_until = 0.0
        self.reported_failure = False

        self.get_logger().info('Dino head controller online.')

    def speech_cb(self, msg):
        command = msg.data.strip().lower()

        if command in WAKE_COMMANDS:
            self.action_hold_until = time.monotonic() + ACTION_HOLD_S
            if self._write_command(CMD_HEARD_COMMAND):
                self.last_sent_mode = CMD_HEARD_COMMAND
            return

        if time.monotonic() < self.action_hold_until:
            return

        if command in TURN_COMMANDS:
            mode = CMD_TURNING
        elif command in WALK_COMMANDS:
            mode = CMD_WALKING
        elif command in IDLE_COMMANDS:
            mode = CMD_IDLE
        else:
            return

        if mode == self.last_sent_mode:
            return
        if self._write_command(mode):
            self.last_sent_mode = mode

    def _write_command(self, mode):
        try:
            fd = os.open(f'/dev/i2c-{DINO_BUS}', os.O_RDWR)
            try:
                fcntl.ioctl(fd, I2C_SLAVE, DINO_ADDRESS)
                os.write(fd, bytes([mode]))
            finally:
                os.close(fd)
        except OSError as e:
            self.get_logger().warn(f'Dino head I2C write failed: {e}', throttle_duration_sec=3.0)
            self.reported_failure = True
            return False
        if self.reported_failure:
            self.get_logger().info('Dino head I2C write recovered.')
            self.reported_failure = False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DinoHeadController()
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
