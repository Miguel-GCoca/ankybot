#!/usr/bin/env python3
"""
Automated two-point raw-ADC calibration. Commands the servo to a max and a
min degree value - a range this bench rig can actually reach, since the
real robot's full +/-90 deg travel usually isn't reachable in this test
configuration - waits for each end to mechanically/electrically settle,
and reports the settled raw ADC reading with a margin of error derived
from measured fluctuation at each end.

Servo commanding goes through the PCA9685 directly (i2c_servo_common.
ServoChannel) - the Mega is feedback-only, same split as the real robot.
Requires the Mega flashed with the RAW-ADC calibration firmware
(arduino_mega_i2c_calibrate.ino), NOT the production
arduino_mega_i2c_slave.ino - this script reads raw ADC counts, not
calibrated angle.

Writes calibration.json, consumed directly by run_step_response_targets.py
so the raw values / margins don't need to be re-typed by hand.

Also writes a CSV of every individual raw ADC sample taken (both during the
settle-wait and the final measurement window), so the reported mean/margin
can be checked by eye or recomputed against the underlying raw trace rather
than trusted blind - see settle_helpers.py's log_fn mechanism, which this
uses to capture every sample without settle_helpers needing to know about
CSV/file formats.

Usage:
    python3 find_calibration_extremes.py --joint FL_Thigh_Joint --val 30
    python3 find_calibration_extremes.py --joint FL_Thigh_Joint --min -20 --max 35
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import (
    PCA9685, MegaFeedback, ServoChannel, resolve_joint, JOINT_ORDER,
    I2C_BUS, MEGA_I2C_BUS,
)
from settle_helpers import wait_for_self_settle, sample_stats

MECH_SETTLE_S   = 1.0   # coarse pause after commanding, before self-settle polling starts
POLL_PERIOD_S   = 0.05
WINDOW_SIZE     = 8     # rolling-window sample count used to detect settle
SETTLE_TOL_RAW  = 5.0   # window max-min must drop below this raw-ADC-count spread...
HOLD_S          = 1.0   # ...and stay below it continuously for this long to call it settled
TIMEOUT_S       = 15.0
MEASURE_SECS    = 2.0   # post-settle window used for the reported mean/margin


def measure_extreme(servo, mega, index, signed_deg, csv_writer):
    servo.command_deg(signed_deg)
    time.sleep(MECH_SETTLE_S)

    read_fn = lambda: mega.read_channel(index)

    def log_settle(elapsed_s, raw):
        csv_writer.writerow([f"{elapsed_s:.3f}", f"{signed_deg:+.1f}", "settle", f"{raw:.2f}", "", ""])

    settled, elapsed_s, _ = wait_for_self_settle(
        read_fn, POLL_PERIOD_S, WINDOW_SIZE, SETTLE_TOL_RAW, HOLD_S, TIMEOUT_S, log_fn=log_settle
    )
    if not settled:
        print(f"  WARNING: did not settle within {TIMEOUT_S}s, measuring anyway")

    deadline = time.time() + MEASURE_SECS
    t_measure_start = time.time()
    samples = []
    while time.time() < deadline:
        raw = read_fn()
        samples.append(raw)
        csv_writer.writerow([
            f"{time.time() - t_measure_start:.3f}", f"{signed_deg:+.1f}", "measure", f"{raw:.2f}", "", ""
        ])
        time.sleep(POLL_PERIOD_S)

    mean, margin = sample_stats(samples)
    csv_writer.writerow(["", f"{signed_deg:+.1f}", "summary", "", f"{mean:.2f}", f"{margin:.2f}"])
    print(f"  {signed_deg:+.1f} deg -> raw = {mean:.2f} +/- {margin:.2f} "
          f"(settled after {elapsed_s:.2f}s, {len(samples)} samples)")
    return mean, margin


def parse_args():
    parser = argparse.ArgumentParser(description="Two-point raw-ADC calibration over I2C")
    parser.add_argument('--joint', required=True,
                         help=f"Joint name ({', '.join(JOINT_ORDER)}) or bare channel index 0-11")
    parser.add_argument('--val', type=float, default=None,
                         help="Symmetric extreme (commands +val then -val). "
                              "Mutually exclusive with --min/--max.")
    parser.add_argument('--min', type=float, default=None, dest='deg_min',
                         help="Explicit minimum (most negative) reachable extreme, "
                              "for an asymmetric range. Requires --max too.")
    parser.add_argument('--max', type=float, default=None, dest='deg_max',
                         help="Explicit maximum reachable extreme, for an asymmetric range. "
                              "Requires --min too.")
    parser.add_argument('--out', default='calibration.json',
                         help="Where to write the calibration JSON consumed by "
                              "run_step_response_targets.py (default: calibration.json)")
    parser.add_argument('--csv_out', default='calibration_samples.csv',
                         help="Where to write the per-sample raw ADC trace "
                              "(default: calibration_samples.csv)")
    return parser.parse_args()


def resolve_range(args):
    if args.val is not None:
        if args.deg_min is not None or args.deg_max is not None:
            raise SystemExit("--val is mutually exclusive with --min/--max")
        return -args.val, args.val
    if args.deg_min is None or args.deg_max is None:
        raise SystemExit("pass either --val, or both --min and --max")
    if args.deg_min >= args.deg_max:
        raise SystemExit(f"--min ({args.deg_min}) must be less than --max ({args.deg_max})")
    return args.deg_min, args.deg_max


def main():
    args = parse_args()
    index = resolve_joint(args.joint)
    deg_min, deg_max = resolve_range(args)

    # Mega is on a separate Pi I2C bus from the PCA9685 - see i2c_servo_common.MEGA_I2C_BUS.
    pca_bus = SMBus(I2C_BUS)
    mega_bus = SMBus(MEGA_I2C_BUS)
    pca = PCA9685(pca_bus)
    servo = ServoChannel(pca, index)
    mega = MegaFeedback(mega_bus)

    print(f"Calibrating channel {index} ({JOINT_ORDER[index]}) over [{deg_min:+.1f}, {deg_max:+.1f}] deg. "
          f"Make sure the Mega is flashed with arduino_mega_i2c_calibrate.ino.")

    with open(args.csv_out, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['t_s', 'target_deg', 'phase', 'raw', 'mean', 'margin'])

        print(f"\nCommanding {deg_max:+.1f} deg...")
        raw_at_max, margin_at_max = measure_extreme(servo, mega, index, deg_max, csv_writer)

        print(f"\nCommanding {deg_min:+.1f} deg...")
        raw_at_min, margin_at_min = measure_extreme(servo, mega, index, deg_min, csv_writer)

        # validation only, not a third calibration anchor: command_deg(0.0) already applies
        # this channel's JOINT_ZERO_TRIM_DEG, so this measures the raw ADC reading at the
        # already hardware-tuned true physical zero. Compared below against what the plain
        # 2-point (deg_min,raw_at_min)/(deg_max,raw_at_max) linear map would predict for 0 deg -
        # a large mismatch would mean the pot's response isn't linear across this channel's
        # full range, which the single trim constant can't fix. Not fed back into RAW_MIN/
        # RAW_MAX/the linear map itself.
        print("\nCommanding 0.0 deg (tuned zero, for validation only)...")
        raw_at_zero, margin_at_zero = measure_extreme(servo, mega, index, 0.0, csv_writer)

    print(f"\nPer-sample raw ADC trace -> {args.csv_out} "
          f"(t_s/'settle'/'measure' rows are individual samples; 'summary' rows carry the "
          f"reported mean/margin for that target_deg - recompute mean and 2*stdev over the "
          f"'measure' rows to cross-check the summary row by hand)")

    raw_span = abs(raw_at_max - raw_at_min)
    combined_margin = margin_at_max + margin_at_min
    if raw_span <= combined_margin:
        print(
            f"\nWARNING: raw readings at the two extremes are statistically indistinguishable "
            f"({raw_at_max:.2f}+/-{margin_at_max:.2f} vs {raw_at_min:.2f}+/-{margin_at_min:.2f}) - "
            f"this channel's pot doesn't appear to be responding to commands. This is the same "
            f"symptom as the known FL_Foot/BR_Foot wiring defect (see CLAUDE.md) - check for a "
            f"similar miswire on this channel before trusting/using this calibration."
        )

    # validation only - see the comment above the 0.0 deg measurement.
    predicted_raw_at_zero = raw_at_min + (0.0 - deg_min) * (raw_at_max - raw_at_min) / (deg_max - deg_min)
    zero_discrepancy = raw_at_zero - predicted_raw_at_zero
    combined_zero_margin = margin_at_zero + margin_at_min + margin_at_max
    print(
        f"\nZero-point check: measured raw@0deg = {raw_at_zero:.2f}+/-{margin_at_zero:.2f}, "
        f"linear-map prediction from the two extremes = {predicted_raw_at_zero:.2f}, "
        f"discrepancy = {zero_discrepancy:+.2f}"
    )
    if abs(zero_discrepancy) > combined_zero_margin:
        print(
            f"  NOTE: discrepancy exceeds combined measurement margin (+/-{combined_zero_margin:.2f}) - "
            f"this channel's raw-vs-angle response may not be linear across its full range. The "
            f"single JOINT_ZERO_TRIM_DEG constant can't correct for that; flag for a closer look "
            f"before trusting this channel's calibration away from its measured extremes."
        )

    calibration = {
        'joint': JOINT_ORDER[index],
        'channel': index,
        'deg_min': deg_min,
        'deg_max': deg_max,
        'raw_at_min': raw_at_min,
        'margin_at_min': margin_at_min,
        'raw_at_max': raw_at_max,
        'margin_at_max': margin_at_max,
        'raw_at_zero': raw_at_zero,
        'margin_at_zero': margin_at_zero,
        'predicted_raw_at_zero': predicted_raw_at_zero,
        'zero_discrepancy': zero_discrepancy,
    }
    with open(args.out, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nSaved calibration -> {args.out}")
    print(json.dumps(calibration, indent=2))


if __name__ == '__main__':
    main()
