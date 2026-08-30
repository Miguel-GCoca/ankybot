#!/usr/bin/env python3
"""
I2C replacement for the old standalone-rig calibrate.ino/serial-monitor
workflow. Commands a single joint's PCA9685 channel and streams the Mega's
raw ADC reading for that same channel, so you can read off RAW_MIN/RAW_MAX
at the physical travel limits.

Requires the Mega to be flashed with arduino_mega_i2c_calibrate.ino (raw
ADC over I2C) - NOT arduino_mega_i2c_slave.ino, which applies calibration
on-device and won't give you raw counts. Reflash back to
arduino_mega_i2c_slave.ino (arduino_ws/arduino_mega_final/
arduino_mega_i2c_slave/) before running run_step_response_sweep.py.

Usage:
    python3 calibrate_i2c.py --joint FL_Thigh_Joint
    python3 calibrate_i2c.py --joint 4          # bare channel index also OK

At the prompt, enter a signed joint angle in degrees (-90..90) to move
there and stream raw ADC for a few seconds; blank input just re-streams the
current reading; 'q' quits.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import (
    PCA9685, MegaFeedback, ServoChannel, resolve_joint, JOINT_ORDER,
    I2C_BUS, MEGA_I2C_BUS,
)

STREAM_SECS = 3.0
STREAM_PERIOD_S = 0.2   # matches the old calibrate.ino's PRINT_INTERVAL_MS
MECH_SETTLE_S = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="I2C servo calibration helper")
    parser.add_argument('--joint', required=True,
                         help=f"Joint name ({', '.join(JOINT_ORDER)}) or bare channel index 0-11")
    return parser.parse_args()


def stream_channel(mega, index, duration_s):
    deadline = time.time() + duration_s
    while time.time() < deadline:
        try:
            raw = mega.read_channel(index)
            print(f"  raw_adc = {raw:.1f}")
        except OSError as e:
            print(f"  I2C read from Mega failed: {e}")
        time.sleep(STREAM_PERIOD_S)


def main():
    args = parse_args()
    index = resolve_joint(args.joint)

    # Mega is on a separate Pi I2C bus from the PCA9685 - see i2c_servo_common.MEGA_I2C_BUS.
    pca_bus = SMBus(I2C_BUS)
    mega_bus = SMBus(MEGA_I2C_BUS)
    pca = PCA9685(pca_bus)
    servo = ServoChannel(pca, index)
    mega = MegaFeedback(mega_bus)

    print(f"Commanding channel {index} ({JOINT_ORDER[index]}) via PCA9685, "
          f"reading raw ADC from the Mega (make sure it's flashed with "
          f"arduino_mega_i2c_calibrate.ino).")
    print("Enter a signed angle in degrees (-90..90), blank to re-stream, 'q' to quit.")

    while True:
        try:
            raw_in = input(f"[ch{index}] angle_deg> ").strip()
        except EOFError:
            break

        if raw_in.lower() == 'q':
            break
        if raw_in:
            try:
                deg = float(raw_in)
            except ValueError:
                print("  not a number, try again")
                continue
            servo.command_deg(deg)
            time.sleep(MECH_SETTLE_S)

        stream_channel(mega, index, STREAM_SECS)


if __name__ == '__main__':
    main()
