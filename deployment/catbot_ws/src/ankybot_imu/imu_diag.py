#!/usr/bin/env python3
"""Staged BNO085 diagnostic - no ROS, no colcon build needed.
Run directly on the Pi: python3 imu_diag.py

2026-07-20: also installed into share/ankybot_imu/ and invoked by
catbot_bringup's launch file as an automated pre-flight IMU health gate
(runs after check_i2c_devices.py confirms the address acks, before any
node starts) - a device can be present at 0x4A but functionally stuck
(see the 2026-07-14 gyroscope-stuck incident in CLAUDE.md's IMU Integration
notes, only found via a script like this one, not a bare address probe).
For that automated use, exit code matters: 0 only if every stage that ran
passed, 1 otherwise - see the sys.exit() calls below.

2026-07-20: added --quiet, used only by the launch invocation above - runs
the accelerometer/gyroscope/game_rotation_vector checks independently (no
combined stage, no per-sample values, no stage headers) and prints exactly
one [OK]/[FAIL] line per check. Manual/interactive use (no --quiet) is
unchanged: full 4-stage diagnostic with live sensor readings.
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


def stage(title, feature_ids, samples=5, delay=0.2, quiet=False):
    if not quiet:
        print(f"\n=== {title} ===")
    try:
        bno = connect()
    except Exception as e:
        if not quiet:
            print(f"FAIL: could not connect to sensor at {hex(I2C_ADDRESS)}: {e}")
        return False

    for fid in feature_ids:
        try:
            bno.enable_feature(fid)
        except Exception as e:
            if not quiet:
                print(f"FAIL: enable_feature({FEATURE_NAMES[fid]}) raised: {e}")
            return False

    for i in range(samples):
        try:
            if BNO_REPORT_ACCELEROMETER in feature_ids:
                accel = bno.acceleration
                if not quiet:
                    print("  accel:", accel)
            if BNO_REPORT_GYROSCOPE in feature_ids:
                gyro = bno.gyro
                if not quiet:
                    print("  gyro: ", gyro)
            if BNO_REPORT_GAME_ROTATION_VECTOR in feature_ids:
                quat = bno.game_quaternion
                if not quiet:
                    print("  quat: ", quat)
        except Exception as e:
            if not quiet:
                print(f"FAIL: read raised on sample {i}: {e}")
            return False
        time.sleep(delay)

    if not quiet:
        print("PASS")
    return True


def main_quiet():
    accel_ok = stage(
        "accelerometer", [BNO_REPORT_ACCELEROMETER], quiet=True)
    gyro_ok = stage(
        "gyroscope", [BNO_REPORT_GYROSCOPE], quiet=True)
    quat_ok = stage(
        "game_rotation_vector", [BNO_REPORT_GAME_ROTATION_VECTOR], quiet=True)

    print(f"  {'[OK]' if accel_ok else '[FAIL]'} - accelerometer")
    print(f"  {'[OK]' if gyro_ok else '[FAIL]'} - gyroscope")
    print(f"  {'[OK]' if quat_ok else '[FAIL]'} - game_rotation_vector")
    sys.exit(0 if (accel_ok and gyro_ok and quat_ok) else 1)


def main():
    results = {}

    results["accelerometer only"] = stage(
        "Stage 1: connect + accelerometer only", [BNO_REPORT_ACCELEROMETER]
    )
    if not results["accelerometer only"]:
        print(
            "\nSensor did not respond for the simplest possible case "
            "(connect + one feature). This points at wiring/power/address, "
            "not the multi-feature enable_feature() issue seen before."
        )
        print_summary(results)
        sys.exit(1)

    results["gyroscope only"] = stage(
        "Stage 2: connect + gyroscope only (fresh reset)", [BNO_REPORT_GYROSCOPE]
    )
    results["game_rotation_vector only"] = stage(
        "Stage 3: connect + game_rotation_vector only (fresh reset)",
        [BNO_REPORT_GAME_ROTATION_VECTOR],
    )
    results["all three together"] = stage(
        "Stage 4: connect + all three together (matches bno085_publisher.py)",
        [BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_GAME_ROTATION_VECTOR],
    )

    print_summary(results)
    sys.stdout.flush()
    if not all(results.values()):
        print("\nIMU health check FAILED - see FAIL lines above.", file=sys.stderr, flush=True)
        sys.exit(1)
    print("\nIMU health check passed.", flush=True)


def print_summary(results):
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} - {name}")


if __name__ == "__main__":
    if "--quiet" in sys.argv:
        main_quiet()
    else:
        main()
