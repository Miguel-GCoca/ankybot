#!/usr/bin/env python3
"""Continuous WiFi/internet-connectivity monitor for ankybot_bringup.launch.py.

Unlike check_i2c_devices.py/imu_diag.py, this never gates the launch - it's
started once, in parallel with everything else, and just runs forever,
printing once every POLL_PERIOD_S while unreachable and once on recovery.
Tests actual internet reachability (a socket connect to 8.8.8.8:53, no DNS
lookup needed) rather than just WiFi link-layer association, since that's
what actually matters for speech_detection's Azure cloud STT calls.
"""
import socket
import sys
import time

REMOTE_HOST = "8.8.8.8"
REMOTE_PORT = 53
TIMEOUT_S = 3.0
POLL_PERIOD_S = 3.0


def has_internet():
    try:
        with socket.create_connection((REMOTE_HOST, REMOTE_PORT), timeout=TIMEOUT_S):
            return True
    except OSError:
        return False


def main():
    was_ok = None
    try:
        while True:
            ok = has_internet()
            if not ok:
                print("[NOT DETECTED] no WiFi/internet connection", file=sys.stderr, flush=True)
            elif was_ok is False:
                print("[OK] WiFi/internet connection restored", flush=True)
            was_ok = ok
            time.sleep(POLL_PERIOD_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
