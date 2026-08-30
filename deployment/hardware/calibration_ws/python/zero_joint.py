#!/usr/bin/env python3
"""
Commands a single joint's PCA9685 channel to 0 degrees and exits - for
parking a servo when you're done testing it. Works regardless of which
Mega firmware is currently flashed (arduino_mega_i2c_calibrate.ino or
arduino_mega_i2c_slave.ino) since it only talks to the PCA9685, not the
Mega - no feedback reading involved.

Usage:
    python3 zero_joint.py --joint BL_Hip_Joint
    python3 zero_joint.py --joint 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import PCA9685, ServoChannel, resolve_joint, JOINT_ORDER


def parse_args():
    parser = argparse.ArgumentParser(description="Park a joint at 0 degrees")
    parser.add_argument('--joint', required=True,
                         help=f"Joint name ({', '.join(JOINT_ORDER)}) or bare channel index 0-11")
    return parser.parse_args()


def main():
    args = parse_args()
    index = resolve_joint(args.joint)

    bus = SMBus(1)
    pca = PCA9685(bus)
    servo = ServoChannel(pca, index)
    servo.command_deg(0.0)

    print(f"Commanded channel {index} ({JOINT_ORDER[index]}) to 0 deg.")


if __name__ == '__main__':
    main()
