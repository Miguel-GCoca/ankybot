#!/usr/bin/env python3
"""Pre-flight I2C device presence check for ankybot_bringup.launch.py.

Stdlib-only (os + fcntl), no smbus2/ROS dependency. Probes each address
with a zero-byte write, the same "quick write" method adafruit_bus_device
uses to detect the BNO085. A zero-length write triggers only the
address/ACK/NACK phase of the bus transaction, so it can't have any
protocol side effect regardless of a device's own register semantics.

Scans every device before reporting anything, even after an early
failure, so a single run always tells you about every missing device at
once rather than stopping at the first one. Exit code 0 if every device
acks, 1 if any is missing (named on stderr), ankybot_bringup.launch.py
gates the rest of the launch on this exit code.

--skip-mega and --skip-imu exclude those devices from the check, for
ankybot_bringup.launch.py's no_mega:=True/no_imu:=True bench-testing
modes. The Mega sits on its own Pi I2C controller (MEGA_I2C_BUS),
isolated from PCA9685/BNO085 on bus 1, so each DEVICES entry carries its
own bus number.
"""
import argparse
import fcntl
import os
import sys

I2C_SLAVE = 0x0703  # ioctl request code from linux/i2c-dev.h
I2C_BUS = 1        # PCA9685, BNO085
MEGA_I2C_BUS = 2   # Arduino Mega, separate Pi I2C controller
MEGA_ADDRESS = 0x08
IMU_ADDRESS = 0x4A

# the three devices this launch file depends on, deliberately not the dino head (0x5B), an unrelated prop.
DEVICES = [
    ("PCA9685 servo driver", 0x60, I2C_BUS),
    ("Arduino Mega feedback slave", MEGA_ADDRESS, MEGA_I2C_BUS),
    ("BNO085 IMU", IMU_ADDRESS, I2C_BUS),
]


def probe(bus, address):
    try:
        fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
    except OSError as e:
        print(f"  could not open /dev/i2c-{bus}: {e}", file=sys.stderr, flush=True)
        return False
    try:
        fcntl.ioctl(fd, I2C_SLAVE, address)
        os.write(fd, b"")
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mega", default="false")
    parser.add_argument("--skip-imu", default="false")
    args = parser.parse_args()
    skip_mega = args.skip_mega.strip().lower() in ("true", "1", "yes")
    skip_imu = args.skip_imu.strip().lower() in ("true", "1", "yes")

    devices = [
        d for d in DEVICES
        if not (skip_mega and d[1] == MEGA_ADDRESS)
        and not (skip_imu and d[1] == IMU_ADDRESS)
    ]

    missing = []
    for name, address, bus in devices:
        ok = probe(bus, address)
        status = "OK" if ok else "NOT DETECTED"
        print(f"[{status}] {name} at 0x{address:02X} (bus {bus})", flush=True)
        if not ok:
            missing.append(f"{name} (0x{address:02X}, bus {bus})")

    if missing:
        print(
            f"\nI2C device check FAILED, not detected: {', '.join(missing)}",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)

    print("\nI2C device check passed, all expected devices responded.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
