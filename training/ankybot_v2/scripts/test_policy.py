#!/usr/bin/env python3
"""Ankybot V2 policy inference and symmetry test — no Isaac Lab required.

Requirements: pip install onnxruntime numpy

Usage:
    python test_policy.py <path/to/policy.onnx>

Hardware deployment note
------------------------
ONNX output is the raw network action (delta, in radians).
Joint command sent to servos:  joint_cmd = default_pos + 0.4 * onnx_output
Default positions: Hip 0.0 rad, Thigh 0.5236 rad, Foot 0.3840 rad
Policy runs at 50 Hz (decimation 10 × sim dt 0.002 s).

Joint order (confirmed from PhysX articulation at runtime)
----------------------------------------------------------
Type-first, alphabetical by leg within each type:
  [0–3]  BL_Hip,   BR_Hip,   FL_Hip,   FR_Hip
  [4–7]  BL_Thigh, BR_Thigh, FL_Thigh, FR_Thigh
  [8–11] BL_Foot,  BR_Foot,  FL_Foot,  FR_Foot
"""

import sys
import numpy as np
import onnxruntime as ort

# ── Joint order (confirmed via print_joint_names.py headless run) ──────────────
JOINT_NAMES = [
    "BL_Hip_Joint",   "BR_Hip_Joint",   "FL_Hip_Joint",   "FR_Hip_Joint",
    "BL_Thigh_Joint", "BR_Thigh_Joint", "FL_Thigh_Joint", "FR_Thigh_Joint",
    "BL_Foot_Joint",  "BR_Foot_Joint",  "FL_Foot_Joint",  "FR_Foot_Joint",
]
assert len(JOINT_NAMES) == 12 and len(set(JOINT_NAMES)) == 12

# ── Observation layout (48-d) ──────────────────────────────────────────────────
# [0:3]   imu_ang_vel       rad/s   (axial vector)
# [3:6]   projected_gravity         (polar vector, body frame)
# [6:9]   imu_lin_accel     m/s²   (polar vector)
# [9:12]  velocity_commands  vx, vy, wyaw  (m/s, m/s, rad/s)
# [12:24] joint_pos          rad   (absolute)
# [24:36] joint_vel          rad/s (absolute)
# [36:48] prev_actions       (raw network output from previous step)

# Training ranges (used to generate in-distribution test observations)
_CMD_RANGE   = ([0.0, -0.2, -0.5], [1.0,  0.2,  0.5])  # vx forward-only
_HIP_NOISE   = 0.10   # reset randomization (rad)
_LIMB_NOISE  = 0.15   # typical walking variation for thigh/foot (rad)
_VEL_NOISE   = 2.0    # joint velocity range during walking (rad/s)
_ACT_NOISE   = 0.30   # prev action range (rad)

_DEFAULT_POS = {"Hip": 0.0, "Thigh": 0.5236, "Foot": 0.3840}
_NOISE_BY_TYPE = {"Hip": _HIP_NOISE, "Thigh": _LIMB_NOISE, "Foot": _LIMB_NOISE}


def _default_joint_pos():
    return np.array([
        _DEFAULT_POS[next(k for k in _DEFAULT_POS if k in name)]
        for name in JOINT_NAMES
    ], dtype=np.float32)


def _joint_noise():
    return np.array([
        _NOISE_BY_TYPE[next(k for k in _NOISE_BY_TYPE if k in name)]
        for name in JOINT_NAMES
    ], dtype=np.float32)


def make_obs(ang_vel=None, gravity=None, lin_accel=None, cmd=None,
             joint_pos=None, joint_vel=None, prev_action=None):
    obs = np.zeros(48, dtype=np.float32)
    obs[0:3]   = ang_vel     if ang_vel     is not None else [0., 0., 0.]
    obs[3:6]   = gravity     if gravity     is not None else [0., 0., -1.]
    obs[6:9]   = lin_accel   if lin_accel   is not None else [0., 0., -9.81]
    obs[9:12]  = cmd         if cmd         is not None else [0., 0., 0.]
    obs[12:24] = joint_pos   if joint_pos   is not None else _default_joint_pos()
    obs[24:36] = joint_vel   if joint_vel   is not None else np.zeros(12)
    obs[36:48] = prev_action if prev_action is not None else np.zeros(12)
    return obs


# ── Symmetry transforms ────────────────────────────────────────────────────────
_LR_LEG   = {"FR": "FL", "FL": "FR", "BR": "BL", "BL": "BR"}
_FB_LEG   = {"FR": "BR", "BR": "FR", "FL": "BL", "BL": "FL"}
_DIAG_LEG = {"FR": "BL", "BL": "FR", "FL": "BR", "BR": "FL"}

# (ang_vel_sign, grav/accel_sign, cmd_sign, negate_hips)
# ang_vel is axial; gravity and lin_accel are polar — different sign rules.
# Hips negated for LR and diagonal; nothing negated for FB.
_SYM_CFG = {
    "lr":   ([-1.,  1., -1.], [ 1., -1.,  1.], [ 1., -1., -1.], True),
    "fb":   ([ 1., -1., -1.], [-1.,  1.,  1.], [-1.,  1., -1.], False),
    "diag": ([-1., -1.,  1.], [-1., -1.,  1.], [-1., -1.,  1.], True),
}


def _build_perm_signs(leg_map, negate_hips):
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    perm, signs = [], np.ones(12, dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        leg, _, suffix = name.partition("_")
        src = f"{leg_map[leg]}_{suffix}"
        perm.append(idx[src])
        if negate_hips and "Hip" in name:
            signs[i] = -1.
    return np.array(perm), signs


_REMAPS = {
    sym: _build_perm_signs(
        {"lr": _LR_LEG, "fb": _FB_LEG, "diag": _DIAG_LEG}[sym],
        _SYM_CFG[sym][3],
    )
    for sym in ("lr", "fb", "diag")
}


def transform_obs(obs, sym):
    av_s, gv_s, cmd_s, _ = _SYM_CFG[sym]
    perm, js = _REMAPS[sym]
    out = obs.copy()
    out[0:3]   = obs[0:3]   * av_s
    out[3:6]   = obs[3:6]   * gv_s
    out[6:9]   = obs[6:9]   * gv_s
    out[9:12]  = obs[9:12]  * cmd_s
    out[12:24] = obs[12:24][perm] * js
    out[24:36] = obs[24:36][perm] * js
    out[36:48] = obs[36:48][perm] * js
    return out


def transform_action(action, sym):
    perm, js = _REMAPS[sym]
    return action[perm] * js


# ── Inference ──────────────────────────────────────────────────────────────────
def run(session, obs_batch):
    inp_name = session.get_inputs()[0].name
    if obs_batch.shape[0] == 1:
        return session.run(None, {inp_name: obs_batch})[0]
    return np.stack([
        session.run(None, {inp_name: obs_batch[i:i+1]})[0][0]
        for i in range(obs_batch.shape[0])
    ])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Model:  {path}")
    print(f"Input:  '{inp.name}'  {inp.shape}  {inp.type}")
    print(f"Output: '{out.name}'  {out.shape}  {out.type}")
    print()

    # ── Test 1: default pose, zero command ────────────────────────────────────
    a0 = run(session, make_obs()[None])[0]
    print("=== Default pose, zero command ===")
    for name, val in zip(JOINT_NAMES, a0):
        print(f"  {name:<22} {val:+.4f} rad  →  cmd {_DEFAULT_POS[next(k for k in _DEFAULT_POS if k in name)] + 0.4*val:+.4f} rad")
    print(f"  ||action||  {np.linalg.norm(a0):.4f}")
    print()

    # ── Test 2: command response ──────────────────────────────────────────────
    cmds = {
        "forward    vx=+0.5": [ 0.5,  0.,   0.],
        "lateral    vy=+0.2": [ 0.,   0.2,  0.],
        "yaw left  wz=+0.5":  [ 0.,   0.,   0.5],
        "yaw right wz=-0.5":  [ 0.,   0.,  -0.5],
    }
    print("=== Command response (||action||) ===")
    for label, cmd in cmds.items():
        a = run(session, make_obs(cmd=cmd)[None])[0]
        print(f"  {label}:  {np.linalg.norm(a):.4f}")
    print()

    # ── Test 3: double-transform identity ─────────────────────────────────────
    obs_fwd = make_obs(cmd=[0.5, 0., 0.])
    a_fwd   = run(session, obs_fwd[None])[0]
    print("=== Double-transform identity ===")
    ok = True
    for sym in ("lr", "fb", "diag"):
        obs_err = np.max(np.abs(transform_obs(transform_obs(obs_fwd, sym), sym) - obs_fwd))
        act_err = np.max(np.abs(transform_action(transform_action(a_fwd, sym), sym) - a_fwd))
        status = "OK" if obs_err < 1e-5 and act_err < 1e-5 else "FAIL"
        ok = ok and (status == "OK")
        print(f"  {sym:4s}: obs_err={obs_err:.2e}  action_err={act_err:.2e}  [{status}]")
    print(f"  {'Transform math correct.' if ok else 'FAIL — bug in transform or joint order.'}")
    print()

    # ── Test 4: symmetry error (in-distribution observations) ─────────────────
    rng  = np.random.default_rng(0)
    N    = 512
    lo, hi = np.array(_CMD_RANGE[0], np.float32), np.array(_CMD_RANGE[1], np.float32)
    noise   = _joint_noise()
    default = _default_joint_pos()

    batch = np.stack([make_obs(
        ang_vel     = rng.uniform(-0.5,  0.5,  3).astype(np.float32),
        gravity     = (np.array([0., 0., -1.], np.float32) + rng.uniform(-0.05, 0.05, 3).astype(np.float32)),
        lin_accel   = (np.array([0., 0., -9.81], np.float32) + rng.uniform(-0.3, 0.3, 3).astype(np.float32)),
        cmd         = rng.uniform(lo, hi).astype(np.float32),
        joint_pos   = default + rng.uniform(-noise, noise).astype(np.float32),
        joint_vel   = rng.uniform(-_VEL_NOISE, _VEL_NOISE, 12).astype(np.float32),
        prev_action = rng.uniform(-_ACT_NOISE, _ACT_NOISE, 12).astype(np.float32),
    ) for _ in range(N)])

    actions = run(session, batch)

    print(f"=== Symmetry error — {N} in-distribution observations ===")
    print(f"  (cmd vx∈[0,1], vy∈[-0.2,0.2], wz∈[-0.5,0.5]; hip noise ±{_HIP_NOISE} rad)")
    all_ok = True
    for sym in ("lr", "fb", "diag"):
        sym_obs  = np.stack([transform_obs(batch[i], sym) for i in range(N)])
        sym_acts = run(session, sym_obs)
        expected = np.stack([transform_action(actions[i], sym) for i in range(N)])
        err = np.linalg.norm(sym_acts - expected, axis=1)
        flag = "OK" if err.mean() < 0.20 else "HIGH"
        all_ok = all_ok and (flag == "OK")
        print(f"  {sym:4s}: mean={err.mean():.4f} rad  p95={np.percentile(err,95):.4f} rad  max={err.max():.4f} rad  [{flag}]")
    print()
    if all_ok:
        print("  Symmetry within acceptable range for data-augmentation training.")
    else:
        print("  Symmetry errors high. Consider retraining with use_mirror_loss=True.")


if __name__ == "__main__":
    main()
