#!/usr/bin/env python3
"""Continuous dino head prop presence monitor for ankybot_bringup.launch.py.

Never gates the launch, runs forever in parallel, printing once while not
detected and once on recovery. Probes I2C address 0x5B with a zero-byte
write, deliberately not part of the blocking device check, since the dino
head is an unrelated prop and its absence shouldn't hold up the
locomotion/sensing stack.
"""
import fcntl
import os
import sys
import time

I2C_SLAVE = 0x0703
DINO_ADDRESS = 0x5B
DINO_BUS = 3
POLL_PERIOD_S = 3.0


def dino_head_present():
    try:
        fd = os.open(f"/dev/i2c-{DINO_BUS}", os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.ioctl(fd, I2C_SLAVE, DINO_ADDRESS)
        os.write(fd, b"")
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def main():
    was_ok = None
    try:
        while True:
            ok = dino_head_present()
            if not ok:
                print("[NOT DETECTED] dino head not found at 0x5B", file=sys.stderr, flush=True)
            elif was_ok is False:
                print("[OK] dino head detected", flush=True)
            was_ok = ok
            time.sleep(POLL_PERIOD_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
