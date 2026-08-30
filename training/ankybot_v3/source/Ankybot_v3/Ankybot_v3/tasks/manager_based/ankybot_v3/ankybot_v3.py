# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom Ankybot quadruped robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Assets.
##

ANKYBOT_USD_PATH = (
    "/workspace/custom_assets/Ankybot_v3_description/FinalRemakeURDF.SLDASM/usd_v3/World0.usd"
)
"""Path to the Ankybot USD asset."""

##
# Configuration - Actuators.
##

ANKYBOT_LEG_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
    effort_limit=3.92266,
    velocity_limit=10.4719755,
    effort_limit_sim=3.92266,
    velocity_limit_sim=10.4719755,
    stiffness={
        ".*_Hip_Joint": 10.6449,
        ".*_Thigh_Joint": 3.5125,
        ".*_Foot_Joint": 0.8102,
    },
    damping={
        ".*_Hip_Joint": 0.2334,
        ".*_Thigh_Joint": 0.1040,
        ".*_Foot_Joint": 0.0236,
    },
    min_delay=2,
    max_delay=7,
)
"""Position-servo PD configuration for the Ankybot leg joints.

2026-07-18: switched IdealPDActuatorCfg -> DelayedPDActuatorCfg. Randomly
delays the commanded joint position/velocity/effort setpoints by
min_delay-max_delay physics steps (re-rolled per env on every reset) before
they reach the PD law -- models real command-path latency (I2C write,
PCA9685 update, servo response lag) that a zero-delay actuator doesn't
capture, which is exactly the kind of sim-to-real gap that tends to make a
policy trained in perfect-latency sim brittle on real hardware.

max_delay=15 physics steps = 15*0.002s = 30ms (sim.dt=0.002, decimation=10
=> 50Hz/20ms control period) is derived from the same real BL-leg
step-response CSVs as the original (later-reverted) stiffness/damping
attempt, re-fit with the onset delay (t0) exposed as its own parameter
instead of discarded. Result across 13 valid step segments: t0 clustered
20-45ms (median 27ms) with one 89ms outlier and two negative values (-50,
-25ms -- physically impossible, a response can't start before the command,
confirming these are fit noise, not real delay). The ADC poll rate in this
dataset is ~23ms, close to the delay being measured, so t0 can't cleanly
separate true communication dead-time from the initial slow rise of the
2nd-order response -- this is a noisy order-of-magnitude estimate, not a
precise round-trip measurement (no dedicated latency test has been run on
the real I2C command path). min_delay=0/max_delay=15 (0-30ms) covers the
median with a bit of margin, clipping out the negative artifacts and the
single high outlier. Retune via min_delay/max_delay if a real latency
measurement becomes available.

2026-07-19: that real latency measurement now has a dedicated test --
calibration_ws/python/measure_actuation_delay.py (+ aggregate_actuator_delay.py
to pool multiple joints). Samples raw ADC as fast as the I2C bus allows
(~5-7ms/sample, vs. the ~23ms poll rate the estimate above was limited by)
and detects onset via a statistical threshold against each trial's own
noise floor, instead of reading delay out of a 2nd-order position fit.

2026-07-22: run on hardware, superseding the 0-30ms estimate above.
4 joints tested (FL_Hip, FL_Thigh, FL_Foot, BR_Thigh -- spanning all three
joint types plus a left/right check), 20 trials each, +/-5deg steps,
19-20/20 valid onset detections per joint. Onset was tight and consistent
across all four: lower bound ~4.04-4.06ms, upper bound ~14.95-14.96ms --
i.e. hip/thigh/foot and left/right do NOT meaningfully differ, unlike the
concern that motivated testing more than one joint. Pooled (min across
joints' lower bounds, max across joints' upper bounds): 4.04-14.96ms =>
min_delay=2, max_delay=7 physics steps (sim.dt=0.002). Meaningfully
tighter and shifted later than the old fit-based 0-30ms estimate -- that
estimate's t0 was contaminated by the servo's own slow-rise dynamics
(2nd-order position fit, ~23ms poll rate too coarse to isolate true
onset), not just noisy in magnitude.

Stiffness/damping below unchanged from the datasheet-derived values:

2026-07-18: derived from the Smraza 45KG coreless servo's datasheet rating
(stall torque 45 kgf*cm = 4.413 N*m, speed 600 deg/s @ 8.0V => 0.1s per 60deg
step), NOT from the real BL-leg step-response CSVs -- those were tried first
(see calibration_ws/data/stiffness_damping_derivation.txt) and produced
thigh/foot stiffness 20-45x too soft to hold the robot up (natural_frequency/
base1: ~100% joint_pos_out_of_bounds terminations, ~9-10 step episodes, zero
learning over 1000 iterations), root-caused to velocity/torque saturation
during the +-45/60 deg physical test steps.

Method here: same k=wn^2*J, c=2*zeta*wn*J relation (J = each joint's
physics:JointEquivalentInertia from the USD), but wn/zeta come from the
datasheet's single 60deg-step timing instead of the noisy real trace, via
the standard 2%-settling-time approximation t_s~=4/(zeta*wn), with
zeta=0.8 borrowed from the real data's own (still-plausible) fitted damping
ratio since a bare speed rating carries no shape/overshoot information:
zeta*wn = 4/0.1 = 40 rad/s, wn = 40/0.8 = 50 rad/s. One wn/zeta pair applies
to all three joint types (same physical servo model in all 12 channels);
per-joint k/c differ only because each joint's J differs. This still carries
the same saturation-contamination caveat as the original attempt (a rated
"time for a large step" is still amplitude-dependent, not a true small-signal
constant) but anchors to a manufacturer number instead of noisy ADC data, and
the resulting k values land within roughly the same order of magnitude as
the original provisional 20/20/18 (hip actually comes out stiffer, thigh/foot
softer but only ~2-5x, not 20-45x) -- verified with a short smoke test before
committing to a full training run.

2026-07-23: switched to real small-angle step-response data, superseding the
datasheet-derived values above. Unlike attempt 1 (2026-07-17/18, +-45/60deg
steps, reverted for being 20-45x too soft due to velocity/torque
saturation), this run used step_response_sweep/run_step_response_sweep.py's
deliberately smaller RANGES_DEG=[15,10,5,2] and a full-trace least-squares
fit of the classical 2nd-order step response against every raw sample in
each trial's CSV (not just peak-overshoot-derived zeta/omega_n, which proved
far too noise-dominated to use directly -- omega_n swung 3-200 rad/s across
nominally-equivalent small steps with no coherent trend). The fit uses each
trial's own measured start/end values rather than the nominal commanded
target, which also makes it immune to a separate hardware finding from the
same dataset: every joint shows a persistent 2-5deg steady-state calibration
offset (JOINT_ZERO_TRIM_DEG/RAW_MIN/RAW_MAX mismatch, not measurement noise)
that doesn't affect the fitted response *shape*.

k=J*wn^2, c=2*zeta*wn*J as before, with J now the real per-joint moment of
inertia about each joint's rotation (Z) axis from SolidWorks mass-properties
reports (foot assembly / thigh+foot / hip+thigh+foot, i.e. everything distal
to that joint) rather than USD-read inertia: J_hip=4.1452e-3, J_thigh=
1.9172e-3, J_foot=4.0315e-4 kg*m^2. wn/zeta are the median of per-trial fits
pooled across range10/20/30 (range4's 2deg steps had too few usable fits --
noise-floor limited), R^2 median 0.785 across 375/480 trials with R^2>=0.5:
hip zeta=0.556 wn=50.7 rad/s, thigh zeta=0.634 wn=42.8 rad/s, foot zeta=0.652
wn=44.8 rad/s.

Result: stiffness ~4-5x softer than the datasheet-derived values across all
three joint types (hip 44.66->10.6449, thigh 9.7725->3.5125, foot 4.1693->
0.8102), damping ~5-6x softer (hip 1.4291->0.2334, thigh 0.3127->0.1040,
foot 0.1334->0.0236) -- notably nowhere near attempt 1's 20-45x-too-soft
failure mode, but still meaningfully softer than the manufacturer rating.
Recomputing the damping ratio against real J also flips the qualitative
picture: the datasheet-derived config is actually overdamped once combined
with real inertia (zeta 1.14-1.66, no overshoot, but slow), while this fit
is underdamped (zeta 0.56-0.65) -- consistent with real overshoot actually
observed in the hardware traces. Net settling speed (zeta*wn, the decay
envelope rate) drops ~3-6x, so this is a genuine "softer AND slower to
settle" change, not just a rescaling -- see chat/CLAUDE_HISTORY.md for the
full before/after comparison.

Caveat: IQR across the pooled trials still spans roughly an order of
magnitude per joint, and wn/zeta aren't fully flat between range10/20/30
(some residual amplitude-dependence, i.e. saturation contamination, is
plausibly still present even at these smaller steps) -- treat as a
meaningfully-informed order-of-magnitude estimate, not a precise constant.
Given attempt 1's total training collapse on a much-too-soft real-data fit,
**run a short smoke test before committing to a full training run**, same
as attempt 2's own verification step.

Every existing checkpoint was trained under the datasheet-derived
stiffness/damping above -- same category of distributional shift as the
pose/delay-range changes elsewhere in this file; do not assume old
checkpoints transfer cleanly.
"""

##
# Configuration - Articulation.
##

ANKYBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANKYBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_Hip_Joint": 0.0,
            ".*_Thigh_Joint": 0.0,
            ".*_Foot_Joint": 0.5236,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": ANKYBOT_LEG_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of the custom Ankybot quadruped robot."""
