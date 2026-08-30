# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

```
URDFs/
  ankybot_v2/          ← active robot description (SolidWorks → URDF export)
  ankybot_v3/          ← hardware revision WIP — NOT policy-ready (see issues below)
policy deployment/
  ankybot_policy_runner/   ← ROS2 Python package (ament_python)
  policy_log/
    ankybot_v2/
      mirror_loss_true/    ← preferred policy (symmetry-enforced during training)
      symmetry_baseline/   ← data-augmentation-only baseline
    test_policy.py         ← offline ONNX symmetry + response test (no ROS needed)
```

## Policy runner — build and run

This is a standard ROS2 ament_python package. From inside a sourced ROS2 workspace:

```bash
# Build
cd <ros2_ws>
colcon build --packages-select ankybot_policy_runner
source install/setup.bash

# Launch
ros2 launch ankybot_policy_runner policy_runner.launch.py

# Or run directly with a policy override
ros2 run ankybot_policy_runner policy_runner \
  --ros-args -p policy_path:=/abs/path/to/policy.onnx
```

## Test policy offline (no ROS, no robot)

```bash
pip install onnxruntime numpy
python "policy deployment/policy_log/test_policy.py" \
  "policy deployment/policy_log/ankybot_v2/mirror_loss_true/policy.onnx"
```

Runs four checks: default pose output, command response norms, double-transform identity, and symmetry error over 512 in-distribution observations. A `mean < 0.20 rad` on all three symmetry axes (LR, FB, diagonal) is the pass criterion.

## Architecture: policy ↔ hardware contract

Everything that touches joint ordering, default positions, or observation construction must respect this:

**Joint order** (confirmed from PhysX articulation — type-first, alphabetical by leg):
```
[0–3]   BL_Hip,   BR_Hip,   FL_Hip,   FR_Hip
[4–7]   BL_Thigh, BR_Thigh, FL_Thigh, FR_Thigh
[8–11]  BL_Foot,  BR_Foot,  FL_Foot,  FR_Foot
```

**Default positions:** Hip = 0.0 rad, Thigh = 0.5236 rad (30°), Foot = 0.3840 rad (22°). These match the URDF joint offsets baked into the thigh/foot `rpy` — the default is the zero-offset neutral stance.

**Action pipeline** (must match `ActionsCfg` in training):
```
raw = policy(obs)                          # 12-dim
raw = clip(raw, ±1.4923)
last_action ← raw                          # stored for next obs
filtered = 0.8·filtered + 0.2·raw         # EMA, α=0.8
target_pos = default_pos + 0.4 · filtered  # published to /joint_commands
```

**Observation vector (48-dim):**

| Dims | Term | Source |
|------|------|--------|
| 3 | `imu_ang_vel` | `/imu/data` |
| 3 | `projected_gravity` | `/imu/data` |
| 3 | `imu_lin_accel` | `/imu/data` |
| 3 | `velocity_commands` | `/cmd_vel` |
| 12 | `joint_pos` | `/joint_states` (encoder position, rad) |
| 12 | `joint_vel` | `/joint_states` (encoder velocity, rad/s — calculated on Arduino from potentiometer readings) |
| 12 | `last_action` | internal — raw clipped network output from previous step |

The runner subscribes to `/joint_states` (`sensor_msgs/JointState`) and remaps joints by name to the correct policy index, so the hardware bridge can publish in any order. Velocity is calculated on the Arduino (which samples faster than 50 Hz) and published in `msg.velocity`. The `joint_states_topic` parameter can be overridden in the yaml.

**Control rate:** 50 Hz (`decimation=10 × sim_dt=0.002s`). Changing this invalidates the trained policy.

## URDF axis convention (ankybot_v2)

Front legs (FR, FL): thigh and foot joints rotate around `+y`.
Back legs (BR, BL): thigh and foot joints rotate around `−y` (mirrored).
All hip joints rotate around `+x`.

This means a positive thigh command moves front and back legs in *physically opposite* directions — the URDF joint definitions absorb the sign flip. The policy training environment and the policy runner both rely on this convention being consistent.

## ankybot_v3 status

Joint limits, axis conventions, and link names now match ankybot_v2. Remaining blockers before v3 can be used for policy training or deployment:

- No Isaac Sim USD files
- No trained policy exists for this revision

## Keeping this file current

After any change to the URDF structure, policy pipeline, or deployment configuration, update this file to reflect the new state. Specifically: if joint limits, axis conventions, default positions, observation layout, or the v3 status change, edit the relevant section here before finishing the task.
