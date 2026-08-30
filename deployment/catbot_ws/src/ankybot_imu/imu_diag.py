#!/usr/bin/env python3
"""BNO085 diagnostic
run on the pi with "python3 imu_diag.py"

exit code: 0 only if every stage that ran passed, 1 otherwise see sys.exit()
"""
import sys
import time

import board
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

I2C_ADDRESS = 0x4A

FEATURE_NAMES = {
    BNO_REPORT_ACCELEROMETER: "accelerometer",
    BNO_REPORT_GYROSCOPE: "gyroscope",
    BNO_REPORT_GAME_ROTATION_VECTOR: "game_rotation_vector",
}


def connect():
    i2c = board.I2C()
    return BNO08X_I2C(i2c, address=I2C_ADDRESS)


def stage(title, feature_ids, samples=5, delay=0.2):
    print(f"\n=== {title} ===")
    try:
        bno = connect()
    except Exception as e:
        print(f"FAIL: could not connect to sensor at {hex(I2C_ADDRESS)}: {e}")
        return False

    for fid in feature_ids:
        try:
            bno.enable_feature(fid)
        except Exception as e:
            print(f"FAIL: enable_feature({FEATURE_NAMES[fid]}) raised: {e}")
            return False

    for i in range(samples):
        try:
            if BNO_REPORT_ACCELEROMETER in feature_ids:
                accel = bno.acceleration
                print("  accel:", accel)
            if BNO_REPORT_GYROSCOPE in feature_ids:
                gyro = bno.gyro
                print("  gyro: ", gyro)
            if BNO_REPORT_GAME_ROTATION_VECTOR in feature_ids:
                quat = bno.game_quaternion
                print("  quat: ", quat)
        except Exception as e:
            print(f"FAIL: read raised on sample {i}: {e}")
            return False
        time.sleep(delay)

    print("PASS")
    return True


STAGES = [
    ("accelerometer", "Stage 1: connect + accelerometer", [BNO_REPORT_ACCELEROMETER]),
    ("gyroscope ", "Stage 2: connect + gyroscope ", [BNO_REPORT_GYROSCOPE]),
    ("game_rotation_vector ", "Stage 3: connect + game_rotation_vector", [BNO_REPORT_GAME_ROTATION_VECTOR]),
    ("all three together", "Stage 4: connect + all three",
     [BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_GAME_ROTATION_VECTOR]),
]


def main():
    results = {name: stage(title, features) for name, title, features in STAGES}

    print_summary(results)
    sys.stdout.flush()
    if not all(results.values()):
        print("\nIMU health check FAILED", file=sys.stderr, flush=True)
        sys.exit(1)
    print("\nIMU OK.", flush=True)


def print_summary(results):
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}, {name}")


if __name__ == "__main__":
    main()
