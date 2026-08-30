#!/usr/bin/env python3
"""
Measures the real command-to-motion dead time of one joint's actuation
path (Pi I2C write -> PCA9685 -> servo -> mechanical response, observed via
the Mega's pot feedback) - the variable DelayedPDActuatorCfg's
min_delay/max_delay (physics time-steps, see isaaclab.actuators.
actuator_pd_cfg.DelayedPDActuatorCfg) are supposed to model.

Why this script exists: ankybot_v3.py's ANKYBOT_LEG_ACTUATOR_CFG has used
DelayedPDActuatorCfg with min_delay=0/max_delay=15 (0-30ms) since 2026-07-18,
but that range was only ever a byproduct of the stiffness/damping step-
response fit - the fit's t0 (onset-delay) parameter, extracted from data
collected at ~23ms sample resolution with the original (later reverted,
velocity-saturated) +/-45/60deg step test. Its own docstring says plainly:
"no dedicated latency test has been run on the real I2C command path" and
"this is a noisy order-of-magnitude estimate, not a precise round-trip
measurement." This script is that dedicated test - it isolates onset delay
directly (statistical threshold-crossing on raw ADC vs a per-trial noise
floor) instead of reading it out of a 2nd-order position fit, and samples
as fast as the I2C bus allows (~5-7ms/sample via single-channel
MegaFeedback.read_channel reads, per informal timing observed elsewhere in
this toolset) rather than the ~23ms poll rate the existing estimate is
limited by.

Uses small steps (--amplitude_deg, default 5.0) by design, matching the
2026-07-19 small-angle correction to run_step_response_targets.py/
run_step_response_sweep.py (see those files and CLAUDE.md "Mechanical
Values" for why: a large step's onset is contaminated by the servo's rated
slew time, which isn't communication/actuation dead time). Runs many trials
(--trials, alternating step direction) to get a real distribution instead
of a single noisy point.

Servo commanding goes through the PCA9685 directly (i2c_servo_common.
ServoChannel) - the Mega is feedback-only, same split as the real robot.
Requires the Mega flashed with the RAW-ADC calibration firmware
(arduino_mega_i2c_calibrate.ino), NOT the production
arduino_mega_i2c_slave.ino - onset detection works on raw ADC counts
directly, no degree calibration needed.

Writes a per-sample CSV (every baseline/return-settle/post-step reading,
plus one summary row per trial with the fitted onset bracket) so the
reported delay can be checked by eye against the raw trace, and a JSON
summary with the pooled delay distribution and a suggested min_delay/
max_delay in physics steps (using SIM_DT_S below - keep this in sync with
ankybot_v3_env_cfg_base.py's self.sim.dt).

Run once per joint you want data for; see aggregate_actuator_delay.py to
pool several joints' JSON summaries into one overall min_delay/max_delay
(DelayedPDActuatorCfg applies a single delay range to every joint, not a
per-joint-type dict like stiffness/damping).

Usage:
    python3 measure_actuation_delay.py --joint BL_Hip_Joint
    python3 measure_actuation_delay.py --joint BL_Thigh_Joint --amplitude_deg 5 --trials 30
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from smbus2 import SMBus

from i2c_servo_common import (
    PCA9685, MegaFeedback, ServoChannel, resolve_joint, JOINT_ORDER,
    I2C_BUS, MEGA_I2C_BUS,
)
from settle_helpers import wait_for_self_settle

# Must match ankybot_v3_env_cfg_base.py's self.sim.dt (physics step, seconds)
# - DelayedPDActuatorCfg.min_delay/max_delay count physics steps, not
# control steps or wall-clock ms. If sim.dt ever changes, re-derive
# min_delay/max_delay from the *_delay.json outputs rather than re-running
# the hardware test (the raw delay-in-seconds numbers don't change).
SIM_DT_S = 0.002

BASELINE_SETTLE_S = 0.5      # coarse pause after returning to baseline, before self-settle polling
RETURN_SETTLE_TOL_RAW = 5.0  # matches find_calibration_extremes.py's SETTLE_TOL_RAW
RETURN_SETTLE_WINDOW = 8
RETURN_SETTLE_HOLD_S = 0.3
RETURN_SETTLE_TIMEOUT_S = 3.0
RETURN_SETTLE_POLL_S = 0.02

BASELINE_SAMPLE_S = 0.15     # post-settle window used to characterize this trial's noise floor
POST_STEP_WINDOW_S = 0.4     # how long to keep sampling after the step looking for onset

THRESHOLD_SIGMA = 6.0        # onset threshold = baseline_mean +/- THRESHOLD_SIGMA * baseline_stdev
MIN_THRESHOLD_RAW = 2.0      # floor on the threshold so near-zero baseline noise can't make this
                              # oversensitive to single-count ADC dither
CONFIRM_SAMPLES = 3          # consecutive over-threshold samples required to call it real motion,
                              # not a single ADC glitch


def tight_sample(read_fn, duration_s, t0, phase, csv_writer, trial):
    """Samples read_fn() back-to-back (no sleep - only the I2C transaction
    itself paces this) for duration_s, timestamped relative to t0. Logs
    every sample to csv_writer and returns parallel (times, raws) lists."""
    times, raws = [], []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        raw = read_fn()
        elapsed = now - t0
        times.append(elapsed)
        raws.append(raw)
        csv_writer.writerow([trial, phase, f"{elapsed:.4f}", f"{raw:.1f}", "", "", "", "", ""])
    return times, raws


def find_onset(times, raws, baseline_mean, threshold_abs, confirm_n):
    """First index i where raws[i:i+confirm_n] are ALL >= threshold_abs away
    from baseline_mean. Returns (t_lower, t_upper, index) bounding the true
    onset - t_lower is the last known-not-yet-moved sample (0.0, i.e. the
    command-issue instant, if the very first post-command sample already
    qualifies), t_upper is the first confirmed-moved sample. Returns
    (None, None, None) if no such run occurs within the window."""
    n = len(raws)
    for i in range(n - confirm_n + 1):
        window = raws[i:i + confirm_n]
        if all(abs(v - baseline_mean) >= threshold_abs for v in window):
            t_upper = times[i]
            t_lower = times[i - 1] if i > 0 else 0.0
            return t_lower, t_upper, i
    return None, None, None


def run_trial(servo, mega, index, trial_num, baseline_deg, step_deg, csv_writer):
    read_fn = lambda: mega.read_channel(index)

    # Return to / confirm baseline before this trial's measurement.
    servo.command_deg(baseline_deg)
    time.sleep(BASELINE_SETTLE_S)

    def log_return(elapsed_s, raw):
        csv_writer.writerow([trial_num, "return_settle", f"{elapsed_s:.4f}", f"{raw:.1f}", "", "", "", "", ""])

    settled, _, _ = wait_for_self_settle(
        read_fn, RETURN_SETTLE_POLL_S, RETURN_SETTLE_WINDOW, RETURN_SETTLE_TOL_RAW,
        RETURN_SETTLE_HOLD_S, RETURN_SETTLE_TIMEOUT_S, log_fn=log_return
    )
    if not settled:
        print(f"  [trial {trial_num}] WARNING: did not settle at baseline within "
              f"{RETURN_SETTLE_TIMEOUT_S}s, measuring anyway")

    # Characterize this trial's noise floor immediately before the step.
    t_baseline0 = time.perf_counter()
    _, baseline_raws = tight_sample(
        read_fn, BASELINE_SAMPLE_S, t_baseline0, "baseline", csv_writer, trial_num
    )
    baseline_mean = statistics.fmean(baseline_raws)
    baseline_std = statistics.stdev(baseline_raws) if len(baseline_raws) > 1 else 0.0
    threshold_abs = max(THRESHOLD_SIGMA * baseline_std, MIN_THRESHOLD_RAW)

    # Issue the step and time-stamp the instant the I2C write returns.
    t_cmd = time.perf_counter()
    servo.command_deg(step_deg)
    times, raws = tight_sample(
        read_fn, POST_STEP_WINDOW_S, t_cmd, "post_step", csv_writer, trial_num
    )

    t_lower, t_upper, idx = find_onset(times, raws, baseline_mean, threshold_abs, CONFIRM_SAMPLES)

    note = "" if t_upper is not None else "NO_ONSET_DETECTED_IN_WINDOW"
    csv_writer.writerow([
        trial_num, "summary", "", "",
        f"{baseline_mean:.2f}", f"{baseline_std:.2f}", f"{threshold_abs:.2f}",
        f"{t_lower:.4f}" if t_lower is not None else "",
        f"{t_upper:.4f}" if t_upper is not None else "",
    ])

    if t_upper is None:
        print(f"  [trial {trial_num}] {baseline_deg:+.1f}->{step_deg:+.1f} deg: "
              f"no onset detected within {POST_STEP_WINDOW_S}s window "
              f"(baseline={baseline_mean:.1f}+/-{baseline_std:.2f}, threshold={threshold_abs:.2f}) {note}")
    else:
        print(f"  [trial {trial_num}] {baseline_deg:+.1f}->{step_deg:+.1f} deg: "
              f"onset in [{t_lower*1000:.1f}, {t_upper*1000:.1f}] ms "
              f"(baseline={baseline_mean:.1f}+/-{baseline_std:.2f}, threshold={threshold_abs:.2f}, "
              f"{len(raws)} post-step samples)")

    return t_lower, t_upper


def percentile(sorted_vals, p):
    """Simple linear-interpolation percentile, p in [0, 100]. Avoids a hard
    dependency on statistics.quantiles' exact binning/Python-version
    behavior for small N."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure command-to-motion actuation delay for DelayedPDActuatorCfg")
    parser.add_argument('--joint', required=True,
                         help=f"Joint name ({', '.join(JOINT_ORDER)}) or bare channel index 0-11")
    parser.add_argument('--baseline_deg', type=float, default=0.0,
                         help="Rest position between trials (default: 0.0)")
    parser.add_argument('--amplitude_deg', type=float, default=5.0,
                         help="Step size in degrees, alternating +/- from baseline_deg each trial "
                              "(default: 5.0, matching the small-angle correction used elsewhere "
                              "in this toolset - do not use a large amplitude here, see module "
                              "docstring)")
    parser.add_argument('--trials', type=int, default=20,
                         help="Number of step events to sample (default: 20)")
    parser.add_argument('--out', default=None,
                         help="JSON summary output path (default: actuation_delay_<joint>.json)")
    parser.add_argument('--csv_out', default=None,
                         help="Per-sample CSV output path (default: actuation_delay_<joint>.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    index = resolve_joint(args.joint)
    joint_name = JOINT_ORDER[index]
    out_path = args.out or f"actuation_delay_{joint_name}.json"
    csv_path = args.csv_out or f"actuation_delay_{joint_name}.csv"

    # Mega is on a separate Pi I2C bus from the PCA9685 - see i2c_servo_common.MEGA_I2C_BUS.
    pca_bus = SMBus(I2C_BUS)
    mega_bus = SMBus(MEGA_I2C_BUS)
    pca = PCA9685(pca_bus)
    servo = ServoChannel(pca, index)
    mega = MegaFeedback(mega_bus)

    print(f"Measuring actuation delay on channel {index} ({joint_name}) via PCA9685, "
          f"reading raw ADC from the Mega (make sure it's flashed with "
          f"arduino_mega_i2c_calibrate.ino).")
    print(f"{args.trials} trials, +/-{args.amplitude_deg:.1f} deg steps from "
          f"{args.baseline_deg:+.1f} deg baseline.\n")

    lowers, uppers = [], []
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'trial', 'phase', 't_s', 'raw',
            'baseline_mean', 'baseline_std', 'threshold', 't_lower_s', 't_upper_s'
        ])

        for trial_num in range(args.trials):
            sign = 1 if trial_num % 2 == 0 else -1
            step_deg = args.baseline_deg + sign * args.amplitude_deg
            t_lower, t_upper = run_trial(
                servo, mega, index, trial_num, args.baseline_deg, step_deg, csv_writer
            )
            if t_upper is not None:
                lowers.append(t_lower)
                uppers.append(t_upper)

    servo.command_deg(args.baseline_deg)

    n_valid = len(uppers)
    print(f"\n{n_valid}/{args.trials} trials produced a valid onset detection.")

    if n_valid == 0:
        print("No valid trials - can't compute delay stats. Check THRESHOLD_SIGMA/"
              "MIN_THRESHOLD_RAW against this joint's actual noise floor and amplitude_deg "
              "against its actual raw-ADC-per-degree sensitivity (see calibration.json from "
              "find_calibration_extremes.py for this joint if available).")
        return

    lowers.sort()
    uppers.sort()

    suggested_min_s = max(0.0, min(lowers))
    suggested_max_s = max(uppers)
    suggested_min_steps = int(suggested_min_s // SIM_DT_S)
    suggested_max_steps = -(-int(round(suggested_max_s / SIM_DT_S)))  # ceil, small safety margin

    summary = {
        'joint': joint_name,
        'channel': index,
        'amplitude_deg': args.amplitude_deg,
        'baseline_deg': args.baseline_deg,
        'n_trials': args.trials,
        'n_valid': n_valid,
        'sim_dt_s': SIM_DT_S,
        'delay_lower_s': {
            'min': lowers[0], 'median': statistics.median(lowers),
            'p95': percentile(lowers, 95), 'max': lowers[-1],
        },
        'delay_upper_s': {
            'min': uppers[0], 'median': statistics.median(uppers),
            'p95': percentile(uppers, 95), 'max': uppers[-1],
        },
        'suggested_min_delay_steps': suggested_min_steps,
        'suggested_max_delay_steps': suggested_max_steps,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-sample trace -> {csv_path} ('post_step' rows are the samples the onset bracket "
          f"for that trial was computed from; the 'summary' row's t_lower_s/t_upper_s should sit "
          f"right at a real jump in the raw column between two adjacent post_step rows)")
    print(f"Summary -> {out_path}")
    print(json.dumps(summary, indent=2))
    print(f"\nSuggested (this joint only - see aggregate_actuator_delay.py to pool multiple "
          f"joints before trusting a global value):\n"
          f"    min_delay={suggested_min_steps}, max_delay={suggested_max_steps},  "
          f"# {joint_name}, {suggested_min_s*1000:.1f}-{suggested_max_s*1000:.1f}ms, "
          f"measure_actuation_delay.py")


if __name__ == '__main__':
    main()
