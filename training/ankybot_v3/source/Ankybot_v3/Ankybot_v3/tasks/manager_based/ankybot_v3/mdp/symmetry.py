# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right, front-back, and diagonal symmetry augmentation for Ankybot V3.

Geometric basis (verified in sim against live USD articulation):
    - All four Hip joints rotate +X for +theta (unified across legs)
      -> Hip is negated for LEFT-RIGHT and DIAGONAL,
         but just swapped (no sign change) for FRONT-BACK.
    - Thigh and Foot joints share the same axis direction across all legs
      -> Thigh/Foot are just swapped (no sign change) for all three transforms
         (LEFT-RIGHT, FRONT-BACK, and DIAGONAL).

4x augmentation total (original + LR + FB + diag).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_POLICY_TERMS = (
    "imu_ang_vel",
    "projected_gravity",
    "imu_lin_accel",
    "velocity_commands",
    "joint_pos",
    "actions",
)

_POLICY_TERMS_HEIGHT = (
    "imu_ang_vel",
    "projected_gravity",
    "imu_lin_accel",
    "velocity_commands",
    "height_command",
    "joint_pos",
    "actions",
)

# 2026-07-24: added back for ankybot_v3_env_cfg_recovery3_random_push2_turn_frozen.py, a one-off
# reconstruction of a pre-2026-07-23 (joint_vel-included, 48-obs) checkpoint's exact training
# config -- that checkpoint's own agent.yaml confirms it trained with use_data_augmentation=True,
# so this restores support for its layout instead of dropping augmentation for that run. Purely
# additive: the 36/37-obs branches/checks below are unchanged.
_POLICY_TERMS_48 = (
    "imu_ang_vel",
    "projected_gravity",
    "imu_lin_accel",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)

# leg-label remap for each symmetry transform
_LR_LEG = {"FR": "FL", "FL": "FR", "BR": "BL", "BL": "BR"}   # left-right mirror
_FB_LEG = {"FR": "BR", "BR": "FR", "FL": "BL", "BL": "FL"}   # front-back mirror
_DIAG_LEG = {"FR": "BL", "BL": "FR", "FL": "BR", "BR": "FL"}  # diagonal (LR then FB)

_cache: dict = {}


def _remap_joint_name(name: str, leg_map: dict[str, str]) -> str:
    leg, sep, suffix = name.partition("_")
    if not sep or leg not in leg_map:
        raise ValueError(f"Unsupported Ankybot V3 joint name: {name!r}")
    return f"{leg_map[leg]}_{suffix}"


def _joint_type(name: str) -> str:
    if name.endswith("_Hip_Joint") or name.endswith("_Hip"):
        return "hip"
    return "thigh_foot"  # Thigh and Foot(calf) share the same sign rule


def _build_permutation_and_signs(joint_names: Sequence[str], leg_map: dict[str, str], sym_name: str) -> tuple[list[int], torch.Tensor]:
    if len(joint_names) != 12 or len(set(joint_names)) != 12:
        raise ValueError(f"Expected 12 unique Ankybot V3 joint names, received {list(joint_names)!r}")

    # which joint TYPE gets negated for this symmetry transform
    negate_types = {
        "lr": {"hip"},
        "fb": set(),
        "diag": {"hip"},
    }[sym_name]

    index_by_name = {name: i for i, name in enumerate(joint_names)}
    permutation = []
    signs = torch.ones(12, dtype=torch.float32)
    for out_i, name in enumerate(joint_names):
        src_name = _remap_joint_name(name, leg_map)
        if src_name not in index_by_name:
            raise ValueError(f"Joint {name!r} has no remapped counterpart {src_name!r}")
        permutation.append(index_by_name[src_name])
        if _joint_type(name) in negate_types:
            signs[out_i] = -1.0

    return permutation, signs


def _apply_joint_remap(joint_data: torch.Tensor, permutation: Sequence[int], signs: torch.Tensor) -> torch.Tensor:
    if joint_data.shape[-1] != len(permutation):
        raise ValueError(
            f"Joint tensor has width {joint_data.shape[-1]}, expected {len(permutation)}"
        )
    indices = torch.as_tensor(permutation, device=joint_data.device, dtype=torch.long)
    signs = signs.to(device=joint_data.device, dtype=joint_data.dtype)
    return joint_data.index_select(-1, indices) * signs


def _get_cached(env: ManagerBasedRLEnv, key: str, builder):
    env_id = id(env)
    if (env_id, key) not in _cache:
        _cache[(env_id, key)] = builder(env)
    return _cache[(env_id, key)]


def _policy_term_slices(env: ManagerBasedRLEnv) -> dict[str, slice]:
    def build(env):
        import math
        term_names = tuple(env.observation_manager.active_terms["policy"])
        term_dims = env.observation_manager.group_obs_term_dim["policy"]
        widths = [math.prod(dim) for dim in term_dims]
        if term_names == _POLICY_TERMS:
            expected_widths = [3, 3, 3, 3, 12, 12]
        elif term_names == _POLICY_TERMS_HEIGHT:
            expected_widths = [3, 3, 3, 3, 1, 12, 12]
        elif term_names == _POLICY_TERMS_48:
            expected_widths = [3, 3, 3, 3, 12, 12, 12]
        else:
            raise ValueError(f"Ankybot V3 symmetry: unrecognised policy terms {term_names}")
        if widths != expected_widths:
            raise ValueError(f"Ankybot V3 symmetry expected policy term widths {expected_widths}, received {widths}")
        slices = {}
        offset = 0
        for name, width in zip(term_names, widths):
            slices[name] = slice(offset, offset + width)
            offset += width
        return slices
    return _get_cached(env, "policy_term_slices", build)


def _runtime_remaps(env: ManagerBasedRLEnv) -> dict[str, dict[str, tuple[list[int], torch.Tensor]]]:
    """Resolve state/action permutations+signs for LR, FB, and diagonal."""
    def build(env):
        robot_joint_names = list(env.scene["robot"].joint_names)
        action_term = env.action_manager.get_term("joint_position")
        action_joint_names = getattr(action_term, "_joint_names", None) or getattr(action_term, "joint_names", None)
        if action_joint_names is None:
            raise AttributeError(
                "Cannot resolve action joint names from action term — check Isaac Lab API version"
            )
        action_joint_names = list(action_joint_names)

        out = {}
        for sym_name, leg_map in (("lr", _LR_LEG), ("fb", _FB_LEG), ("diag", _DIAG_LEG)):
            out[sym_name] = {
                "state": _build_permutation_and_signs(robot_joint_names, leg_map, sym_name),
                "action": _build_permutation_and_signs(action_joint_names, leg_map, sym_name),
            }
        return out
    return _get_cached(env, "runtime_remaps", build)


def _transform_policy_obs(env: ManagerBasedRLEnv, obs: torch.Tensor, sym_name: str) -> torch.Tensor:
    """Apply one of 'lr', 'fb', 'diag' symmetry transforms to the 36-d, 37-d, or 48-d policy obs."""
    slices = _policy_term_slices(env)
    if obs.shape[-1] not in (36, 37, 48):
        raise ValueError(f"Ankybot V3 symmetry expected 36, 37, or 48 policy observations, received {obs.shape[-1]}")

    remaps = _runtime_remaps(env)[sym_name]
    state_perm, state_signs = remaps["state"]
    action_perm, action_signs = remaps["action"]

    out = obs.clone()
    device = obs.device

    if sym_name == "lr":
        # left-right mirror: y and yaw-rate flip sign
        ang_vel_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
        grav_sign = torch.tensor([1.0, -1.0, 1.0], device=device)
        cmd_sign = torch.tensor([1.0, -1.0, -1.0], device=device)
    elif sym_name == "fb":
        # front-back mirror: x (forward) and yaw-rate flip sign
        ang_vel_sign = torch.tensor([1.0, -1.0, -1.0], device=device)
        grav_sign = torch.tensor([-1.0, 1.0, 1.0], device=device)
        cmd_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
    elif sym_name == "diag":
        # diagonal = LR then FB: x and y flip, yaw-rate unchanged (flipped twice)
        ang_vel_sign = torch.tensor([-1.0, -1.0, 1.0], device=device)
        grav_sign = torch.tensor([-1.0, -1.0, 1.0], device=device)
        cmd_sign = torch.tensor([-1.0, -1.0, 1.0], device=device)
    else:
        raise ValueError(f"Unknown symmetry name {sym_name!r}")

    out[..., slices["imu_ang_vel"]] = obs[..., slices["imu_ang_vel"]] * ang_vel_sign
    out[..., slices["imu_lin_accel"]] = obs[..., slices["imu_lin_accel"]] * grav_sign
    out[..., slices["projected_gravity"]] = obs[..., slices["projected_gravity"]] * grav_sign
    out[..., slices["velocity_commands"]] = obs[..., slices["velocity_commands"]] * cmd_sign
    out[..., slices["joint_pos"]] = _apply_joint_remap(obs[..., slices["joint_pos"]], state_perm, state_signs)
    if "joint_vel" in slices:
        # joint_vel is d/dt of joint_pos -- same per-joint remap/sign convention applies.
        out[..., slices["joint_vel"]] = _apply_joint_remap(obs[..., slices["joint_vel"]], state_perm, state_signs)
    out[..., slices["actions"]] = _apply_joint_remap(obs[..., slices["actions"]], action_perm, action_signs)
    return out


def _transform_actions(env: ManagerBasedRLEnv, actions: torch.Tensor, sym_name: str) -> torch.Tensor:
    action_perm, action_signs = _runtime_remaps(env)[sym_name]["action"]
    return _apply_joint_remap(actions, action_perm, action_signs)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original + left-right + front-back + diagonal batches (4x), matching ANYmal's scheme."""
    base_env = env.unwrapped

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(4)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:2 * batch_size] = _transform_policy_obs(base_env, obs["policy"], "lr")
        obs_aug["policy"][2 * batch_size:3 * batch_size] = _transform_policy_obs(base_env, obs["policy"], "fb")
        obs_aug["policy"][3 * batch_size:] = _transform_policy_obs(base_env, obs["policy"], "diag")
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty((batch_size * 4, actions.shape[1]), device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:2 * batch_size] = _transform_actions(base_env, actions, "lr")
        actions_aug[2 * batch_size:3 * batch_size] = _transform_actions(base_env, actions, "fb")
        actions_aug[3 * batch_size:] = _transform_actions(base_env, actions, "diag")
    else:
        actions_aug = None

    return obs_aug, actions_aug
