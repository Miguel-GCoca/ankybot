# Engineering Notes

This project kept a very detailed internal dev log throughout — day-by-day
training runs, root-cause diagnoses, every fix and why it was made. This file
is a curated, human-readable summary of the parts of that log worth reading:
the real engineering problems that came up training and deploying a from-scratch
quadruped, and how they were tracked down.

## Training a policy that doesn't cheat

The reward function for this robot combines velocity tracking, a three-leg
gait-stance term, foot air-time shaping, foot-clearance shaping, and a set of
posture/safety penalties (joint limits, base orientation, base height).
Partway through one training run, playback showed the policy rocking back and
forth in place with periodic single-leg lifts — not walking at all, but still
climbing in reward.

Root cause: the gait-stance and foot-air-time terms rewarded "correct foot
contact pattern" with no reference to whether the robot was actually
translating. A stationary rocking motion satisfies "three feet grounded,
fourth foot lifts and lands" just as well as a real stride, so the policy
found the cheaper solution. The fix was to gate both reward terms on real
measured planar body velocity *projected onto the commanded direction* (not
just speed magnitude — an early version of the fix still let a fast enough
rock clear the bar on both halves of its cycle, since raw speed doesn't know
which way it's pointing). Combined with a new asymmetric penalty on backward
drift, this closed the exploit without hand-tuning weights back down.

## Sim-to-real: replacing datasheet numbers with measured ones

The actuator model started from datasheet PD gains. Once hardware was
available, several of the sim's physical assumptions turned out to be wrong
enough to matter:

- **Actuation delay** — measured directly (command-to-motion latency across
  multiple joints/types), giving a much narrower and later-starting delay
  window than the original curve-fit estimate, which had been contaminated by
  the servo's own slow-rise dynamics rather than pure delay.
- **Stiffness/damping** — re-derived from real step-response data (small-angle
  sweeps) combined with actual per-joint moment of inertia pulled from the
  CAD mass-properties reports, instead of datasheet torque/speed constants.
  The real numbers were substantially softer and, once combined with real
  inertia, revealed the datasheet-derived config had actually been
  *overdamped* relative to hardware — the fitted config is underdamped,
  matching the overshoot actually seen on the real servos.

Each of these was treated explicitly as a distributional shift for the RL
policy (not just a config tweak) — checkpoints trained under the old values
were not assumed to transfer, and were re-validated or retrained rather than
silently continued.

## Root-causing "the robot can't hold a stand-still command"

The most involved debugging thread: commanded to stand still, the robot
instead produced large continuous joint motion. Investigation ruled out
several plausible causes before landing on the real one:

1. **IMU data** — a bag showed `/imu/data` publishing at ~0.24 Hz instead of
   its configured 50 Hz. This looked like the obvious suspect, but directly
   replaying the trained actor network with the IMU input frozen vs. live
   showed only a ~0.3% effect on the output actions — the policy's trained
   `lin_accel` distribution already has enormous natural variance (footfall
   impact transients), so a stale reading isn't a meaningful outlier against
   it. Real, but not the cause.
2. **Joint velocity feedback** — replaying the same network with real
   hardware velocity *spikes* (15–21 rad/s, captured from live feedback)
   substituted in produced a ~36% swing in raw output actions — two orders of
   magnitude larger than the IMU test. The policy's trained joint-velocity
   distribution has a standard deviation of roughly 0.7–2.6 rad/s, so real
   spikes were landing 6–20+ standard deviations outside anything the policy
   had ever seen in simulation.

This is the same class of bug in both cases — a noisy real sensor feeding a
policy that was never trained on that noise — but only direct ablation against
the actual trained network (not just staring at the bag) separated the real
cause from the plausible-looking one. The eventual fix removed joint velocity
from the observation contract entirely rather than continuing to chase
filtering/smoothing on a signal the hardware couldn't deliver cleanly.

## I2C bus contention: why the IMU kept dying

Separately from the above, the IMU would run at a clean, correct rate when
launched alone, then collapse to near-zero messages once the rest of the
stack was running. Bringing every node up one at a time while logging to
`/rosout` and watching `/imu/data`'s rate over each 5-second window pinned
the exact trigger: the IMU ran perfectly for the ~10 seconds it was alone,
then collapsed the moment the microcontroller-feedback poller started issuing
its own I2C transactions at 200 Hz — both processes were opening the same
physical I2C bus with no shared arbitration. Fix: split the two devices onto
separate Pi I2C bus controllers (a hardware/wiring change, not just
software), which resolved the contention completely and was verified with
`i2cdetect` showing clean per-bus device isolation afterward.

## An iterative recovery curriculum

Rather than training a single flat locomotion policy, later stages of this
project trained a curriculum: a base walking policy, then successive
"recovery" stages that start from progressively wider randomized joint/pose
resets (so the policy learns to recover from being knocked over or
mis-started, not just to walk from a clean stand), then domain randomization
(mass, center-of-mass, friction), then external pushes. Each stage resumes
training from the previous stage's checkpoint rather than restarting from
scratch — except at the handful of points where the observation space, action
scale, or default pose itself changed, which are hard breaks a resumed
checkpoint can't load across, and were treated as explicit from-scratch
restarts.

## Result

Ankybot was awarded **"Faculty's Choice"** at the University of Central
Florida Summer '26 senior design showcase.
