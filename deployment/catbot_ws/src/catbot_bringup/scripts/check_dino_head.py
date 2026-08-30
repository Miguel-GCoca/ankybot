#!/usr/bin/env python3
"""Continuous dino head prop presence monitor for ankybot_bringup.launch.py.

Same non-blocking pattern as check_wifi.py/check_microphone.py - never
gates the launch, just runs forever in parallel, printing once every
POLL_PERIOD_S while not detected and once on recovery. Probes I2C address
0x5B with a zero-byte write, the same technique check_i2c_devices.py uses
for the other three devices - deliberately not part of that blocking check
(the dino head is an unrelated prop, its absence shouldn't hold up the
locomotion/sensing stack).

2026-07-23: DINO_BUS moved 1 -> 3, hardcoded, same isolation move as the
Mega's split to bus 2 - see dino_head_controller.py's 2026-07-23 note.
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
