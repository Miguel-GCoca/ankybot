#!/usr/bin/env python3
"""
I2C version of the multi-amplitude step response test - drives the servo
through the same physical path as the real robot (PCA9685 commanded
directly over I2C) and reads feedback from the Arduino Mega I2C slave,
instead of talking to a dedicated bench-test Arduino over USB serial.

Requires the Mega to be flashed with arduino_mega_i2c_slave.ino (production
feedback firmware - arduino_ws/arduino_mega_final/arduino_mega_i2c_slave/),
which reports calibrated joint angle in radians, NOT
arduino_mega_i2c_calibrate.ino (raw ADC, used only for calibrate_i2c.py).

There's no more GOTO/SETTLE/STEP serial protocol or on-device sample timer:
commanding a channel is an immediate I2C write (angle_to_pulse -> PCA9685),
and feedback is polled from the Mega as fast as I2C/its own ADC scan loop
reliably allow - unlike the old standalone rig's fixed 400Hz timer, this
just timestamps actual elapsed time per sample.

2026-07-19: RANGES_DEG extended down to 10/5/2 deg (was 45/30/15 only).
The original +-45/+-60 deg step data produced a stiffness/damping fit that
turned out to be dominated by the servo's own rated slew speed (600 deg/s)
rather than its small-signal spring stiffness - large steps mostly measure
"how fast can it slew this far", not "how stiff is the joint", and a linear
2nd-order fit can't tell those apart from position-vs-time data alone (see
/workspace/CLAUDE.md "Mechanical Values" and
calibration_ws/data/stiffness_damping_derivation.txt for the full
diagnosis). Keeping the original large ranges in the sweep (not replacing
them) is deliberate: having both ends in one sweep lets you check whether
the fitted natural frequency is roughly constant across amplitudes (good -
means you're in the small-signal linear regime) or keeps dropping as
amplitude increases (confirms saturation is still contaminating even the
smaller steps, and you'd need to go smaller still). The exact point where
that plateau happens hasn't been measured yet - 2/5/10 deg are reasoned
estimates (rough order-of-magnitude check against the datasheet-derived
wn~50 rad/s: peak velocity during a step scales roughly with
amplitude*wn, so a 2 deg step stays well under the 600 deg/s ceiling while
a 10 deg step is already getting closer to it) not confirmed-safe values -
treat the resulting data as the actual check, and add smaller ranges still
if 10/5/2 deg all show the same amplitude-dependent drop 45/30/15 did.

2026-07-22: 45/30 deg dropped from RANGES_DEG (user's explicit call, to cut
sweep time roughly in half and avoid unnecessary large-amplitude cycling
now that the datasheet-derived actuator config is already active/working -
this recharacterization is a refinement, not a from-scratch diagnosis like
the original +-45/60 attempt was). 15 deg kept as the one large-amplitude
anchor point, specifically so the amplitude-dependence cross-check above
still has something to compare 10/5/2 deg against - don't drop it too
without also dropping the reasoning for keeping any large point at all.

2026-07-22, same day: the project's earlier "raw CSV only, no on-device or
in-script computation of wn/zeta/amplitude ratio/phase lag" constraint was
explicitly lifted (user's request - wanted the result directly, not a raw
trace to fit externally). This script now also computes, per step, percent
overshoot / peak time -> zeta/omega_n via the standard closed-form
underdamped step-response relations (no curve-fitting library, no numpy/
scipy dependency added):
    zeta     = -ln(PO) / sqrt(pi^2 + ln(PO)^2)      (PO = overshoot fraction)
    omega_d  = pi / t_peak
    omega_n  = omega_d / sqrt(1 - zeta^2)
plus rise time (first crossing of target) and 2%-settling time (last sample
outside a +/-2%-of-step-size band around target). zeta/omega_n are only
computed when a genuine overshoot beyond the target was observed - a
critically/over-damped response (no overshoot) can't be resolved this way,
and is reported as null rather than a fabricated number. Caveat worth
remembering for the smallest steps (2/5 deg): this read path has no
deadband filtering (unlike the deployed ROS2 nodes), so real ADC jitter can
be a meaningful fraction of a 2deg step - a null zeta/omega_n on a small
step could mean "no overshoot" or "overshoot smaller than the noise floor",
and those look the same from this analysis alone; check the raw CSV instead
of trusting a suspicious result blind. Full raw per-sample CSVs are still
written (unchanged) alongside the new JSON summary, specifically so a
computed number can still be checked against the underlying trace.

2026-07-22, later same day: rewritten to run all 12 joints sequentially in
one invocation (was one joint per run via --joint) and report every joint's
results back in a single combined JSON, not 12 separate per-joint files -
user's explicit request ahead of a full-robot characterization pass. Each
joint's raw per-step CSVs are still written to their own
joint_<name>/ subdirectory exactly as before (unchanged, still the
underlying-trace-checking mechanism); only the JSON summary changed from
one-file-per-joint to one-file-for-everything, keyed by joint name. Use
--joints to restrict to a subset (comma-separated names/indices, same
convention as wiring_check/check_wiring_crosstalk.py) - default is all 12
in JOINT_ORDER. A run of all 12 takes roughly 25-35 minutes (4 ranges x 5
reps x 2 directions x ~4s per step, per joint) - if interrupted (Ctrl-C),
whatever joints completed so far are still written to the combined JSON
before exiting, rather than losing the whole run's data.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import (
    PCA9685, MegaFeedback, ServoChannel, resolve_joint, JOINT_ORDER, NUM_SERVOS, PI,
    I2C_BUS, MEGA_I2C_BUS,
)

RANGES_DEG      = [15.0, 10.0, 5.0, 2.0]   # signed half-range per sweep, largest first
REPEAT_COUNT    = 5                # up/down repetitions per range
RECORD_SECS     = 2.0              # recording window per step (rise + overshoot + settle)
MECH_SETTLE_S   = 2.0              # pause after commanding before recording the next step
# Resolved relative to this script's own location (not a fixed ~/Desktop path) so output always
# lands in step_response_sweep/servo_measurement/ regardless of the caller's cwd - matches where
# this session's data has actually been kept/transferred from throughout.
BASE_OUT_DIR    = str(Path(__file__).resolve().parent / "servo_measurement")


def run_step(servo, mega, index, target_deg):
    """Apply the step immediately, then poll the Mega for this channel's
    feedback for RECORD_SECS, timestamping actual elapsed time per sample."""
    servo.command_deg(target_deg)
    t0 = time.perf_counter()
    deadline = t0 + RECORD_SECS
    rows = []
    while time.perf_counter() < deadline:
        try:
            meas_rad = mega.read_channel(index)
        except OSError:
            continue
        t_us = int((time.perf_counter() - t0) * 1e6)
        meas_deg = meas_rad * 180.0 / PI
        rows.append([t_us, target_deg, meas_deg])
    return rows


def save_csv(out_dir, name, rows):
    path = os.path.join(out_dir, name)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_us', 'cmd_deg', 'meas_deg'])
        writer.writerows(rows)
    print(f"  {len(rows)} samples saved -> {path}")
    return path


def analyze_step(rows, initial_deg, target_deg):
    """Classical peak-overshoot step-response analysis - see module
    docstring's 2026-07-22 note for the formulas/caveats. Returns None if
    there's no data or a zero-size step (can't analyze either)."""
    step_size = target_deg - initial_deg
    if not rows or step_size == 0:
        return None

    t_s = [r[0] / 1e6 for r in rows]
    meas = [r[2] for r in rows]

    if step_size > 0:
        peak_val = max(meas)
        peak_idx = meas.index(peak_val)
    else:
        peak_val = min(meas)
        peak_idx = meas.index(peak_val)
    peak_time_s = t_s[peak_idx]

    overshoot_deg = (peak_val - target_deg) if step_size > 0 else (target_deg - peak_val)
    percent_overshoot = max(0.0, overshoot_deg / abs(step_size) * 100.0)

    zeta = None
    omega_n_rad_s = None
    if percent_overshoot > 0.0 and peak_time_s > 0.0:
        po_frac = percent_overshoot / 100.0
        ln_po = math.log(po_frac)
        zeta_val = -ln_po / math.sqrt(math.pi ** 2 + ln_po ** 2)
        if zeta_val < 1.0:
            zeta = zeta_val
            omega_d = math.pi / peak_time_s
            omega_n_rad_s = omega_d / math.sqrt(1.0 - zeta_val ** 2)

    tail_n = max(1, len(meas) // 10)
    final_value_deg = sum(meas[-tail_n:]) / tail_n

    rise_time_s = None
    for t, m in zip(t_s, meas):
        if (step_size > 0 and m >= target_deg) or (step_size < 0 and m <= target_deg):
            rise_time_s = t
            break

    tol = max(0.05, 0.02 * abs(step_size))
    last_outside_idx = None
    for i, m in enumerate(meas):
        if abs(m - target_deg) > tol:
            last_outside_idx = i
    if last_outside_idx is None:
        settle_time_s = 0.0
        settled_within_window = True
    elif last_outside_idx == len(meas) - 1:
        settle_time_s = None
        settled_within_window = False
    else:
        settle_time_s = t_s[last_outside_idx + 1]
        settled_within_window = True

    return {
        'step_size_deg': step_size,
        'final_value_deg': final_value_deg,
        'percent_overshoot': percent_overshoot,
        'peak_time_s': peak_time_s,
        'zeta': zeta,
        'omega_n_rad_s': omega_n_rad_s,
        'rise_time_s': rise_time_s,
        'settle_time_2pct_s': settle_time_s,
        'settled_within_window': settled_within_window,
        'n_samples': len(rows),
    }


def agg_stats(values):
    """min/mean/median/max over the non-None entries of values. Returns
    None if none are valid."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    valid_sorted = sorted(valid)
    n = len(valid_sorted)
    mid = n // 2
    median = valid_sorted[mid] if n % 2 else (valid_sorted[mid - 1] + valid_sorted[mid]) / 2.0
    return {
        'n': n,
        'min': valid_sorted[0],
        'max': valid_sorted[-1],
        'mean': sum(valid_sorted) / n,
        'median': median,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="I2C multi-amplitude servo step response data collection")
    parser.add_argument('--joints', default=None,
                         help="Comma-separated subset of joint names/indices to test, in the order given "
                              "(default: all 12, in JOINT_ORDER)")
    parser.add_argument('--out', default=None,
                         help="Path to the single combined JSON summary covering every tested joint "
                              f"(default: {os.path.join(BASE_OUT_DIR, 'step_response_all.json')})")
    return parser.parse_args()


def sweep_joint(pca, mega, index, joint_name):
    """Runs the full RANGES_DEG x REPEAT_COUNT x 2-directions sweep for one
    joint, writing its raw per-step CSVs to their own subdirectory, and
    returns that joint's result dict (same shape previously written as a
    standalone per-joint JSON file)."""
    out_dir = os.path.join(BASE_OUT_DIR, f"joint_{joint_name}")
    os.makedirs(out_dir, exist_ok=True)

    servo = ServoChannel(pca, index)

    print(f"\n########## [{index}] {joint_name} ##########")
    print(f"Commanding channel {index} ({joint_name}) via PCA9685, "
          f"reading calibrated feedback from the Mega (make sure it's flashed "
          f"with arduino_mega_i2c_slave.ino).")

    groups = []  # one entry per (range_label, direction)

    for half_range in RANGES_DEG:
        start = -half_range
        target = half_range
        range_label = f"range{int(2 * half_range)}"
        print(f"\n=== {joint_name} {range_label}: {start:+.1f}deg <-> {target:+.1f}deg, "
              f"{REPEAT_COUNT} repetitions, {RECORD_SECS}s per step ===")

        # get to the start point before the first repetition
        servo.command_deg(start)
        time.sleep(MECH_SETTLE_S)

        fwd_trials, rev_trials = [], []

        for rep in range(1, REPEAT_COUNT + 1):
            print(f"[{range_label} Rep {rep}] Step {start:+.1f} -> {target:+.1f}...")
            rows_fwd = run_step(servo, mega, index, target)
            csv_path = save_csv(out_dir, f'{range_label}_fwd_rep{rep}.csv', rows_fwd)
            analysis = analyze_step(rows_fwd, start, target)
            if analysis is not None:
                print(f"    overshoot={analysis['percent_overshoot']:.1f}% "
                      f"zeta={analysis['zeta']} omega_n={analysis['omega_n_rad_s']} "
                      f"rise={analysis['rise_time_s']}s settle={analysis['settle_time_2pct_s']}s")
                fwd_trials.append({'rep': rep, 'csv': csv_path, **analysis})
            time.sleep(MECH_SETTLE_S)

            print(f"[{range_label} Rep {rep}] Step {target:+.1f} -> {start:+.1f}...")
            rows_rev = run_step(servo, mega, index, start)
            csv_path = save_csv(out_dir, f'{range_label}_rev_rep{rep}.csv', rows_rev)
            analysis = analyze_step(rows_rev, target, start)
            if analysis is not None:
                print(f"    overshoot={analysis['percent_overshoot']:.1f}% "
                      f"zeta={analysis['zeta']} omega_n={analysis['omega_n_rad_s']} "
                      f"rise={analysis['rise_time_s']}s settle={analysis['settle_time_2pct_s']}s")
                rev_trials.append({'rep': rep, 'csv': csv_path, **analysis})
            time.sleep(MECH_SETTLE_S)
            print()

        for direction, trials, initial_deg, target_deg in [
            ('fwd', fwd_trials, start, target),
            ('rev', rev_trials, target, start),
        ]:
            groups.append({
                'range_label': range_label,
                'half_range_deg': half_range,
                'direction': direction,
                'initial_deg': initial_deg,
                'target_deg': target_deg,
                'trials': trials,
                'summary': {
                    key: agg_stats([t[key] for t in trials])
                    for key in ('zeta', 'omega_n_rad_s', 'rise_time_s',
                                'settle_time_2pct_s', 'percent_overshoot')
                },
            })

    servo.command_deg(0.0)  # park back at zero before moving to the next joint

    return {
        'joint': joint_name,
        'channel': index,
        'ranges_deg': RANGES_DEG,
        'repeat_count': REPEAT_COUNT,
        'record_secs': RECORD_SECS,
        'groups': groups,
    }


def main():
    args = parse_args()
    indices = ([resolve_joint(tok.strip()) for tok in args.joints.split(',')]
               if args.joints else list(range(NUM_SERVOS)))
    out_path = args.out or os.path.join(BASE_OUT_DIR, 'step_response_all.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Mega is on a separate Pi I2C bus from the PCA9685 - see i2c_servo_common.MEGA_I2C_BUS.
    pca_bus = SMBus(I2C_BUS)
    mega_bus = SMBus(MEGA_I2C_BUS)
    pca = PCA9685(pca_bus)
    mega = MegaFeedback(mega_bus)

    print(f"Sequentially characterizing {len(indices)} joint(s): "
          f"{', '.join(JOINT_ORDER[i] for i in indices)}")

    results = {}
    try:
        for index in indices:
            joint_name = JOINT_ORDER[index]
            results[joint_name] = sweep_joint(pca, mega, index, joint_name)
    except KeyboardInterrupt:
        print(f"\nInterrupted - writing results for the {len(results)} joint(s) completed so far.")
    finally:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCombined step-response summary for {len(results)} joint(s) -> {out_path}")


if __name__ == '__main__':
    main()
