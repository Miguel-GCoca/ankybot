#!/usr/bin/env python3
"""
Sweeps each of the 12 joints -deg -> +deg -> 0, one at a time, in
JOINT_ORDER (FL_Hip/Thigh/Foot, FR_Hip/Thigh/Foot, BL_..., BR_...). While
each joint is commanded, ALL 12 Mega feedback channels are read (not just
the commanded one) and diffed between the -deg/+deg extremes, so a pot
that reacts to some other joint's command shows up clearly.

2026-07-22, rewritten: reports every channel's delta as a PERCENTAGE of
the expected full-scale motion for the commanded sweep (2*--deg converted
to radians), not an absolute-unit threshold - the original absolute
CROSSTALK_THRESHOLD (first 3.0, then 0.15) was awkward to reason about
and firmware-unit-dependent. A channel reading near 100% means it's
tracking real motion at that scale; near 0% means it saw no motion; the
report below flags the split between the two so pin assignments can be
confirmed/corrected without guessing at raw units.

Physical/electrical follow-up on the BR_Hip_Joint/BR_Foot_Joint pair
(2026-07-22) established the actual root cause this script is now meant to
verify precisely: watching the robot directly, commanding BR_Hip_Joint
visibly moves ONLY the physical hip, and commanding BR_Foot_Joint visibly
moves ONLY the physical foot - actuation/PWM wiring is correct. But each
joint's OWN feedback channel reads ~0% (no motion) while the OTHER
channel's feedback reads ~100% - i.e. the two pots' signal wires are
swapped at the Mega's analog inputs, not the motor wiring. Fix decided:
correct this in the ADC channel <-> joint-name mapping in code
(arduino_mega_i2c_slave.ino / arduino_mega_i2c_calibrate.ino), NOT by
re-wiring the physical pot connectors. This script's job is to produce a
clear, quantified before/after report of exactly which channel reads which
joint's real motion, so that remap can be applied with confidence and
verified afterward.

Each joint is returned to 0 deg before moving on to the next, so the robot
isn't left with any servo parked at +-deg when the sweep ends.

Works against either Mega firmware - it only reads raw floats and diffs
them, same as i2c_servo_common.py's other consumers. The expected-full-
scale reference (see --expected_full_scale) assumes production firmware
(arduino_mega_i2c_slave.ino, calibrated radians) by default; pass
--expected_full_scale explicitly if running the raw-ADC calibrate firmware
instead, since raw counts aren't a fixed multiple of commanded degrees.

Usage:
    python3 check_wiring_crosstalk.py
    python3 check_wiring_crosstalk.py --joints BR_Hip_Joint,BR_Foot_Joint
    python3 check_wiring_crosstalk.py --significance_pct 15
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import (
    PCA9685, MegaFeedback, ServoChannel, resolve_joint, JOINT_ORDER, NUM_SERVOS, PI,
    I2C_BUS, MEGA_I2C_BUS,
)

DEFAULT_SWEEP_DEG = 30.0
DEFAULT_SETTLE_S = 1.0
DEFAULT_PAUSE_S = 1.5
# Report any channel whose |delta| exceeds this percent of the expected full-scale
# motion. Observed noise floor on healthy channels has been ~1-5%; 10% leaves margin
# above that while still catching a partial (not just full-swap) cross-talk signal.
DEFAULT_SIGNIFICANCE_PCT = 10.0
# If a joint's OWN channel reads below this percent of full-scale, its own feedback
# is suspect (either dead, or - as confirmed for BR_Hip/BR_Foot - reading someone
# else's pot instead).
OWN_CHANNEL_LOW_PCT = 50.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep each joint -deg..+deg sequentially and report which Mega feedback "
                    "channel(s) show significant motion, as a percent of expected full-scale, "
                    "to confirm/quantify pot wiring cross-talk."
    )
    parser.add_argument('--deg', type=float, default=DEFAULT_SWEEP_DEG,
                         help=f"Sweep to +-this many degrees (default {DEFAULT_SWEEP_DEG})")
    parser.add_argument('--settle', type=float, default=DEFAULT_SETTLE_S,
                         help=f"Seconds to wait after each command before reading feedback (default {DEFAULT_SETTLE_S})")
    parser.add_argument('--pause', type=float, default=DEFAULT_PAUSE_S,
                         help=f"Seconds to pause between joints for visual confirmation (default {DEFAULT_PAUSE_S})")
    parser.add_argument('--joints', default=None,
                         help="Comma-separated subset of joint names/indices to test, in the order given "
                              "(default: all 12, in JOINT_ORDER)")
    parser.add_argument('--expected_full_scale', type=float, default=None,
                         help="Override the 100% reference delta (radians for production firmware, raw "
                              "ADC counts for calibrate firmware). Default: 2*--deg converted to radians "
                              "(assumes production firmware).")
    parser.add_argument('--significance_pct', type=float, default=DEFAULT_SIGNIFICANCE_PCT,
                         help=f"Report a non-target channel if its delta exceeds this percent of the "
                              f"expected full-scale motion (default {DEFAULT_SIGNIFICANCE_PCT})")
    return parser.parse_args()


def sweep_joint(pca, mega, index, sweep_deg, settle_s):
    servo = ServoChannel(pca, index)

    servo.command_deg(-sweep_deg)
    time.sleep(settle_s)
    at_min = mega.read_all()

    servo.command_deg(sweep_deg)
    time.sleep(settle_s)
    at_max = mega.read_all()

    servo.command_deg(0.0)
    time.sleep(settle_s)

    return at_min, at_max


def main():
    args = parse_args()
    indices = ([resolve_joint(tok.strip()) for tok in args.joints.split(',')]
               if args.joints else list(range(NUM_SERVOS)))

    expected_full_scale = (args.expected_full_scale if args.expected_full_scale is not None
                            else 2.0 * args.deg * PI / 180.0)

    # Mega is on a separate Pi I2C bus from the PCA9685 - see i2c_servo_common.MEGA_I2C_BUS.
    pca_bus = SMBus(I2C_BUS)
    mega_bus = SMBus(MEGA_I2C_BUS)
    pca = PCA9685(pca_bus)
    mega = MegaFeedback(mega_bus)

    print(f"Sweeping +-{args.deg:.1f} deg sequentially: {', '.join(JOINT_ORDER[i] for i in indices)}")
    print(f"Expected full-scale delta (100% reference): {expected_full_scale:.3f} "
          f"(assumes production firmware/radians - pass --expected_full_scale to override)")
    print(f"Significance threshold: {args.significance_pct:.1f}% of full-scale\n")

    # (commanded_joint, responding_joint, pct, delta) for every significant non-target reading
    cross_readings = []

    try:
        for index in indices:
            name = JOINT_ORDER[index]
            print(f"--- [{index}] {name}: -{args.deg:.0f} -> +{args.deg:.0f} deg ---")
            at_min, at_max = sweep_joint(pca, mega, index, args.deg, args.settle)

            deltas = [at_max[i] - at_min[i] for i in range(NUM_SERVOS)]
            pcts = [abs(deltas[i]) / expected_full_scale * 100.0 for i in range(NUM_SERVOS)]

            own_note = ""
            if pcts[index] < OWN_CHANNEL_LOW_PCT:
                own_note = "  <-- LOW: own channel barely moved despite being commanded"
            print(f"  own channel: {pcts[index]:5.1f}% of full-scale ({deltas[index]:+.3f}){own_note}")

            ranked_others = sorted((i for i in range(NUM_SERVOS) if i != index),
                                    key=lambda i: -pcts[i])
            significant = [i for i in ranked_others if pcts[i] >= args.significance_pct]
            if significant:
                for i in significant:
                    print(f"  SIGNIFICANT: {JOINT_ORDER[i]:16s} {pcts[i]:5.1f}% of full-scale ({deltas[i]:+.3f})")
                    cross_readings.append((name, JOINT_ORDER[i], pcts[i], deltas[i]))
            else:
                print("  no other channel showed a significant reading.")

            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nInterrupted - servos left at last commanded position (not re-zeroed).")
        return

    print("\n=== Summary: significant cross-readings ===")
    if not cross_readings:
        print("None - every commanded joint's own channel was the only one to show significant motion.")
    else:
        for commanded, responded, pct, delta in cross_readings:
            print(f"  commanding {commanded:16s} -> {responded:16s} read {pct:5.1f}% of full-scale ({delta:+.3f})")
        print("\nA pair showing up in BOTH directions (A commanded -> B responds significantly, AND "
              "B commanded -> A responds significantly), each with its OWN channel reading LOW, is the "
              "signature of two pots' signal wires swapped at the Mega - fix by swapping their channel "
              "indices in the firmware's ADC-read mapping, not by re-wiring the connectors.")


if __name__ == '__main__':
    main()
