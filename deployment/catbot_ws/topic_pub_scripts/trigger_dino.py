#!/usr/bin/env python3
"""Fires the I2C-triggered action sequence on the dino head Arduino (0x5B).

Stdlib-only (os + fcntl) — no smbus2/smbus dependency needed.
"""
import argparse
import fcntl
import os

I2C_ADDRESS = 0x5B
I2C_SLAVE = 0x0703  # ioctl request code from linux/i2c-dev.h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bus', type=int, default=3, help='I2C bus number (default 3)')
    parser.add_argument('--address', type=lambda x: int(x, 0), default=I2C_ADDRESS,
                         help='I2C slave address (default 0x5B)')
    args = parser.parse_args()

    fd = os.open(f'/dev/i2c-{args.bus}', os.O_RDWR)
    try:
        fcntl.ioctl(fd, I2C_SLAVE, args.address)
        os.write(fd, bytes([0x01]))
    finally:
        os.close(fd)

    print(f'Triggered dino sequence at 0x{args.address:02X} on bus {args.bus}')


if __name__ == '__main__':
    main()
