#!/usr/bin/env python3
"""Drives the dino head prop (2-DOF yaw/pitch Arduino, I2C address 0x5B -
see arduino_ws/dino_head_control/dino_head_control.ino) directly from
speech_recognizer_node's /speech_to_text strings.

2026-07-21: switched from subscribing to command_interpreter's /cmd_vel +
/action to subscribing to /speech_to_text directly - /action was removed
(roar/sleep/push up deprioritized, see command_interpreter.py), and the
wake phrase ("hey anky" etc, published as "activated" once heard) needed
a way to trigger the head's nod that didn't depend on /action existing.
"activated" triggers a one-shot mode-2 "heard command" nod that takes
priority over walk/turn-derived mode for ACTION_HOLD_S seconds - long
enough to cover the sketch's own ~7s double-nod sequence (i2cSequence
timings in setup()) so a walk/turn command recognized moments later
doesn't cut the nod off early. Same stdlib-only os/fcntl I2C write
topic_pub_scripts/trigger_dino.py already uses for this exact device - no
smbus2 dependency needed for a single-byte write.

2026-07-23: moved to bus 3, hardcoded (was hardcoded bus 1) - same
isolation move as the Mega's split to bus 2 (see CLAUDE.md I2C
Migration/IMU Integration), keeping the occasional dino write off the
PCA9685/BNO085/Mega-shared bus 1. Needs a matching i2c-gpio (or
i2cN-pi5) dtoverlay added to the Pi's /boot/firmware/config.txt for a
third bus - not done from this container, pick free GPIO pins on the
robot and wire the dino head Arduino's SDA/SCL there.
"""
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
