#!/usr/bin/env python3
"""Continuous microphone-presence monitor for ankybot_bringup.launch.py.

Same non-blocking pattern as check_wifi.py - never gates the launch, just
runs forever in parallel, printing once every POLL_PERIOD_S while no
capture device is found and once on recovery. Uses `arecord -l` (lists
ALSA capture devices specifically, matching the SPXERR_MIC_NOT_AVAILABLE
failure mode actually seen from speech_detection's speech_2_txt node)
rather than just checking /proc/asound/cards, which would also match a
playback-only card.
"""
import subprocess
import sys
import time

POLL_PERIOD_S = 3.0


def has_microphone():
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "card" in result.stdout.lower()


def main():
    was_ok = None
    try:
        while True:
            ok = has_microphone()
            if not ok:
                print("[NOT DETECTED] no microphone/capture device found", file=sys.stderr, flush=True)
            elif was_ok is False:
                print("[OK] microphone detected", flush=True)
            was_ok = ok
            time.sleep(POLL_PERIOD_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
