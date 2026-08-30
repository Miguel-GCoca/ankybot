# ankybot_policy_runner — fixing "too fast / steps too small"

Observed on hardware with `low_gait_v1_rec1`: gait cadence looks too fast and
individual steps are too short compared to what the sim trained. Root cause
diagnosed 2026-07-19: `action_filter_alpha` in `config/ankybot_policy.yaml`
heavily low-pass-filters the policy's raw output before it's scaled into a
joint target, and this filter has **no counterpart in the sim the policy was
trained in** — sim applies the raw scaled action directly, every step, with
no smoothing. See MANUAL.txt for how the filter math works.

Changes below, roughly ordered by how directly they target the diagnosed
cause. None of these have been tried on hardware yet — test one at a time so
you can tell what actually moved the needle.

## 1. Lower `action_filter_alpha` (most direct fix)

`config/ankybot_policy.yaml`, `action_filter_alpha: 0.8` -> try something
lower. At `alpha=0.8` the filtered action only moves 20% of the way to the
new raw output per 20ms tick (effective lag ~100ms, about 5 control periods)
- more than enough to clip the peak amplitude off a fast leg swing before it
completes. Suggested path: step it down incrementally (e.g. 0.8 -> 0.5 ->
0.3 -> 0.0) rather than jumping straight to 0, since there may be an
undocumented reason it was set this high (taming real servo/feedback jitter,
protecting mechanics from abrupt commands) that a sudden drop to 0 could
re-expose. Watch for: stride length recovering (good), buzzing/oscillating
joints or servo chatter appearing (bad, back off).

This is a runtime YAML value — no rebuild needed, just edit and relaunch.

## 2. Train the filter into the policy instead of removing it

If dropping `action_filter_alpha` reintroduces jitter/instability that it
was actually needed for, the better long-term fix is to add an equivalent
smoothing step inside the sim actuator/action pipeline during training, so
the policy learns a gait that already accounts for filtered output, instead
of learning an unfiltered gait and then filtering it after the fact at
deploy time. This is a training-side change (not in this package) — flag it
back to the Isaac Lab side if option 1 alone doesn't resolve it.

## 3. Re-check `action_scale` / `action_clip` after any retrain

Currently `action_scale: 0.4` and `action_clip: 1.4923` match
`ActionsCfg.joint_position` in `ankybot_v3.py` exactly, so they are not the
cause of the current symptom. But these are two independent files that have
to be kept in sync by hand — if `ankybot_v3.py`'s `scale`/`clip` ever change
in a future retrain, this yaml needs the same update or you get exactly this
class of bug again (silent amplitude mismatch, no error, no crash).

## 4. Verify real control-loop timing matches `control_dt`

`control_dt: 0.02` (50Hz) must match what the policy was actually trained
at (`decimation(10) * sim.dt(0.002)`). The joint-feedback path has
documented jitter (~2-36ms spread depending on sensor, see the main
project CLAUDE.md's "Hardware Feedback Layer"/IMU sections) — if the real
achieved loop rate drifts meaningfully from 50Hz, that's a second source of
timing mismatch between what the policy expects and what it gets, on top of
the filter. Not the primary suspect here, but worth ruling out if lowering
`action_filter_alpha` alone doesn't fully fix stride length.

## 5. Consider deploying a newer checkpoint

`low_gait_v1_rec1` was trained under the original provisional PD gains
(stiffness 20/20/18, damping 0.8), before the datasheet-derived stiffness/
damping and the measured command-delay actuator (`DelayedPDActuatorCfg`,
0-30ms) were added. The current best checkpoint in the same resume-chain
lineage is `low_gait/v2_recovery2_rand` (already exported to
`catbot_ws/policies/v2_recovery2_rand_2997/`). Because that policy was
trained with a modeled command delay already baked in, it may need *less*
deploy-side smoothing to behave well than `v1_rec1` did — worth testing
after trying option 1, not instead of it, since it's a different variable.
