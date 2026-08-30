#!/usr/bin/env python3
"""Standalone BNO085 sanity check - no ROS involved. Run directly:
    python3 smoke_test.py
"""
import time

import board

# import digitalio
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = board.I2C()
# Reset pin temporarily disabled to isolate whether the OSError 121 during
# soft_reset() is caused by the reset-pin sequencing itself, since i2cdetect
# found the device fine on a plain address probe. Re-enable once confirmed:
#   reset_pin = digitalio.DigitalInOut(board.D17)  # physical pin 11
#   bno = BNO08X_I2C(i2c, address=0x4A, reset=reset_pin)
bno = BNO08X_I2C(i2c, address=0x4A)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

print("BNO085 online. Ctrl-C to stop.")
while True:
    ax, ay, az = bno.acceleration
    gx, gy, gz = bno.gyro
    qi, qj, qk, qr = bno.game_quaternion

    print(
        f"accel(m/s^2)= {ax:+.3f} {ay:+.3f} {az:+.3f}   "
        f"gyro(rad/s)= {gx:+.3f} {gy:+.3f} {gz:+.3f}   "
        f"quat(i,j,k,r)= {qi:+.3f} {qj:+.3f} {qk:+.3f} {qr:+.3f}"
    )
    time.sleep(0.2)
