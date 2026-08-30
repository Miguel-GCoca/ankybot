# Ankybot

A custom 12-DOF quadruped robot, trained in NVIDIA Isaac Lab (PPO) and deployed via ROS 2 to a Raspberry Pi 5 + Arduino Uno R3 servo system.

Training montage and demo clips are in [`media/`](media/) — see
`media/training_montage/ankybot_v2_training_montage/rl-video-step-6000.mp4` for
the latest checkpoint's gait, or `media/ankybot_v1.mp4` for a hardware demo.

See [`ENGINEERING.md`](ENGINEERING.md) for a write-up of the harder problems
that came up along the way — a reward-gaming exploit found via playback, a
sim-to-real actuator recalibration, and an ablation study that root-caused a
standing-instability bug to noisy real joint-velocity feedback.

## Repo layout

```
training/
  ankybot_v2/     Isaac Lab RL training project — 12-DOF locomotion policy (PPO, rsl_rl)
  ankybot_v3/     Next-generation training project — expanded gait/behavior experiments
                  (standing, walking, multiple gait variants); raw per-iteration
                  checkpoints and TensorBoard logs are gitignored, only final
                  exported policies + run metadata are kept
  custom_assets/  Robot USD descriptions (v1/v2/v3) referenced by the training
                  configs via hard-coded path — see note below

deployment/
  catbot_ws/      ROS 2 workspace: IMU driver, I2C servo bridge, and policy runner
                  nodes that run onboard the Raspberry Pi 5
  hardware/       Robot description (URDF/USD, SolidWorks exports), servo
                  characterization data, and standalone policy deployment scripts

media/            Training montage videos and hardware demo clips
```

## Pipeline

1. **Train** a locomotion policy in Isaac Lab (`training/`), using a 48-observation
   layout (IMU angular velocity, projected gravity, IMU linear acceleration,
   velocity command, joint positions/velocities, previous action) and 12
   joint-position actions.
2. **Validate** the exported policy (`exported/policy.pt` / `policy.onnx`) in sim
   across forward, lateral, and yaw motions.
3. **Deploy** onboard via the ROS 2 `ankybot_policy_runner` node
   (`deployment/catbot_ws/src/ankybot_policy_runner`), which loads the ONNX
   policy and drives the physical servos through the I2C bridge.

## Note on running the training projects

`ankybot_v2.py` and `ankybot_v3.py` hard-code an absolute path to their USD
asset (`/workspace/custom_assets/...`), matching the layout inside the
original `isaac-lab-base` Docker container. To run either project outside
that container, either recreate the same absolute path or update the path in
the task config to point at this repo's `training/custom_assets/`.

## Status

Complete! Awarded "Faculty's Choice" at University of Central Florida Summer '26 senior design showcase.

## License

MIT — see [`LICENSE`](LICENSE).
