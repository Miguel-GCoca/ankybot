# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Ankybot V3 locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = wp.to_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    current_contact_time = wp.to_torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_b)[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - wp.to_torch(asset.data.root_ang_vel_b)[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - wp.to_torch(asset.data.root_lin_vel_b)[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


def _forward_progress_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_speed_for_full_credit: float,
    min_command_norm: float = 0.02,
) -> torch.Tensor:
    """Scale in [0, 1] for real motion *aligned with the commanded linear direction*.

    **2026-07-23 fix:** the original real-speed gate used raw planar speed magnitude
    (``norm(root_lin_vel_b[:, :2])``), which is direction-blind -- a rock that oscillates
    forward/backward clears the gate on both halves of the cycle (any commanded
    ``lin_vel_x`` in this project is strictly positive, so the backward half is pure
    exploit). This projects body-frame velocity onto the commanded linear-velocity unit
    vector and clamps to non-negative, so motion opposite the command scores zero instead
    of scoring identically to forward motion. Falls back to raw speed magnitude when the
    commanded linear norm is negligible (e.g. turn-in-place envs with ~zero linear
    command), where there is no linear direction to be "backward" relative to.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_vel = wp.to_torch(asset.data.root_lin_vel_b)[:, :2]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.norm(command, dim=1, keepdim=True)
    has_lin_command = command_norm.squeeze(1) > min_command_norm
    command_dir = command / torch.clamp(command_norm, min=1.0e-6)
    aligned_speed = torch.clamp(torch.sum(body_vel * command_dir, dim=1), min=0.0)
    raw_speed = torch.linalg.norm(body_vel, dim=1)
    progress_speed = torch.where(has_lin_command, aligned_speed, raw_speed)
    return torch.clamp(progress_speed / min_speed_for_full_credit, min=0.0, max=1.0)


def foot_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    min_speed_for_full_credit: float = 0.05,
) -> torch.Tensor:
    """Reward long steps taken by the feet using an L2 kernel.

    Same computation as ``isaaclab_tasks...velocity.mdp.feet_air_time``, but gated on the
    command term's explicit ``is_standing_env`` flag instead of a command-magnitude threshold,
    so only envs the command generator actually rolled to standing are zeroed out.

    **2026-07-23 real-speed gate (updated same day, see** :func:`_forward_progress_scale`
    **):** this reward only checks foot contact timing, with no reference to whether the base
    is actually translating *in the commanded direction* -- a policy can satisfy the raw
    contact-timing check with periodic single-foot lifts while rocking in place, never making
    net progress (found via playback of ``v5/base_scratch`` at iteration ~750, see
    CLAUDE_HISTORY.md). Scaled by :func:`_forward_progress_scale` so the reward ramps to zero
    as command-aligned body speed drops toward zero, independent of foot-contact bookkeeping.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = wp.to_torch(contact_sensor.compute_first_contact(env.step_dt))[:, sensor_cfg.body_ids]
    last_air_time = wp.to_torch(contact_sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    speed_scale = _forward_progress_scale(env, asset_cfg, command_name, min_speed_for_full_credit)
    reward = reward * speed_scale
    is_standing_env = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(is_standing_env, torch.zeros_like(reward), reward)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in
    :attr:`synced_feet_pair_names` to bias the policy towards a desired gait, i.e trotting,
    bounding, or pacing. Note that this reward is only for quadrupedal gaits with two pairs
    of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.command_term = env.command_manager.get_term(cfg.params["command_name"])
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_sensors(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_sensors(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        command_name: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        gait_reward = sync_reward * async_reward
        # only suppress gait for envs the command generator explicitly rolled to standing
        return torch.where(self.command_term.is_standing_env, torch.zeros_like(gait_reward), gait_reward)

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def three_leg_stance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    min_speed_for_full_credit: float = 0.05,
) -> torch.Tensor:
    """Reward having exactly three feet in ground contact at once.

    Simpler alternative to GaitReward: rather than shaping contact-timing
    synchronization between foot pairs, this directly rewards the tripod-style
    three-down-one-up stance pattern at each step. Suppressed for envs the
    command generator rolled to standing, matching GaitReward/foot_air_time's
    convention (a standing robot should plant all four feet, not hold a
    three-leg stance).

    **2026-07-23 real-speed gate (updated same day, see** :func:`_forward_progress_scale`
    **):** the raw contact-count check has no reference to whether the base is actually
    translating *in the commanded direction* -- a policy can max this term out with a
    perfectly stationary single-leg shuffle (lift one foot, set it back down, repeat), or with
    a rock that oscillates forward/backward, since a raw speed-magnitude gate can't tell that
    apart from real forward progress. This was the single largest reward term in the set
    (weight 3.0) and the best-supported driver of the "rocking + periodic leg lifts" exploit
    found via playback of ``v5/base_scratch`` at iteration ~750 (see CLAUDE_HISTORY.md). Scaled
    by :func:`_forward_progress_scale`, mirroring the same gate added to :func:`foot_air_time`
    -- reward requires real progress in the commanded direction, not just correct foot-contact
    bookkeeping or nonzero speed.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    num_grounded = torch.sum(is_contact, dim=1).float()
    reward = (num_grounded == 3.0).float()
    speed_scale = _forward_progress_scale(env, asset_cfg, command_name, min_speed_for_full_credit)
    reward = reward * speed_scale
    is_standing_env = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(is_standing_env, torch.zeros_like(reward), reward)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Shape swing-foot clearance while moving; penalize any foot losing ground contact while standing.

    Two regimes, selected per-env via the command term's ``is_standing_env`` flag (same
    convention as :class:`GaitReward`/:func:`foot_air_time`/:func:`three_leg_stance_reward` —
    not a raw command-magnitude threshold, which zeroed out a different, inconsistent set
    of envs):
      - moving envs: reward feet near ``target_height`` while swinging, weighted by
        horizontal foot velocity so a planted foot barely counts against the target.
      - standing envs: penalize any foot that has broken ground contact, via the contact
        sensor rather than raw body height (a foot link's at-rest world-z offset isn't
        verified, so a height-based standing check would risk a spurious penalty at rest).

    ``swing_reward`` is computed per-foot (each foot gets its own ``exp(-error/std)`` in
    ``(0, 1]``) and averaged, rather than summing every foot's error inside one shared
    exponential — the shared-exponential form let a few well-cleared feet mask a lagging
    one, and its gradient for any single foot shrank as total error grew, i.e. exactly when
    an underperforming foot needed the strongest signal. Per-foot averaging keeps each
    foot's contribution undiluted by the others.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(wp.to_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2], dim=2)
    )
    per_foot_reward = torch.exp(-foot_z_target_error * foot_velocity_tanh / std)
    swing_reward = torch.mean(per_foot_reward, dim=1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_airborne = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        <= contact_threshold
    )
    standing_lift_penalty = -torch.sum(is_airborne.float(), dim=1)

    is_standing_env = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(is_standing_env, standing_lift_penalty, swing_reward)


##
# Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def backward_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Quadratic penalty on negative (backward) base-frame x-velocity only.

    Added 2026-07-23 alongside the speed-gating fix to :func:`three_leg_stance_reward`/
    :func:`foot_air_time`, in response to a "rocking back and forth with periodic leg lifts"
    exploit found via playback (see CLAUDE_HISTORY.md). Every commanded ``lin_vel_x`` range in
    this project is strictly positive (the "slow meandering pace" convention), so any real
    backward excursion is already off-task -- this directly taxes the backward half of a
    forward/backward rocking oscillation without touching a robot that is simply standing still
    (zero velocity) or moving forward. Asymmetric by design, unlike ``lin_vel_z_l2``/
    ``ang_vel_xy_l2`` which penalize magnitude regardless of sign.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    backward_speed = torch.clamp(-wp.to_torch(asset.data.root_lin_vel_b)[:, 0], min=0.0)
    return torch.square(backward_speed)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = wp.to_torch(contact_sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    last_contact_time = wp.to_torch(contact_sensor.data.last_contact_time)[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def prolonged_airborne_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float = 0.75
) -> torch.Tensor:
    """Penalize any foot that has been continuously airborne longer than ``max_air_time``.

    Uses ``current_air_time`` (continuously updated every step while off the ground, reset to
    0 on contact) rather than the transition-snapshotted ``last_air_time`` that
    :func:`air_time_variance_penalty` relies on -- a foot that lifts once and is never put back
    down keeps accumulating here, so this does not go blind on that failure mode. Applied
    unconditionally (no ``is_standing_env`` gating, no foot-velocity gating) so it also catches a
    foot parked motionless mid-air during a moving env, which :func:`foot_clearance_reward`'s
    velocity-gated swing branch cannot see (a stationary airborne foot zeroes that term's
    velocity-tanh weighting, masking the height error entirely).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    current_air_time = wp.to_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    return torch.sum((current_air_time > max_air_time).float(), dim=1)


def feet_ungrounded_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize the number of feet (from sensor_cfg.body_ids) with no contact force above threshold.

    Direct per-step signal that all specified feet should be on the ground, complementing
    air_time_variance_penalty (which only penalizes asymmetry between feet, not an
    absolute "feet should be down" requirement).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return torch.sum(~is_contact, dim=1).float()


def leg_ground_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize any non-foot leg link (e.g. thigh) touching the ground.

    Direct contact-based signal for a "knees on the ground" collapse -- unlike
    :func:`base_height_l2`/:func:`base_orientation_penalty`, which only shape an indirect proxy
    (root height/orientation) and can be satisfied by a resting posture that still isn't standing
    on the feet, this fires only when the specified links (pass a ``sensor_cfg`` targeting e.g.
    ``.*_Thigh_Link``) actually contact the ground. Applied unconditionally (standing or moving) --
    a thigh should essentially never be in ground contact in either regime.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return torch.sum(is_contact.float(), dim=1)


def standing_leg_deviation_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Penalize selected joints deviating from their default (standing) angle, standing envs only.

    Gated by ``is_standing_env`` (same convention as :func:`GaitReward`/:func:`foot_air_time`/
    :func:`foot_clearance_reward`) so it only discourages resting in a collapsed leg angle (e.g.
    knees folded under, resting on the balls of the feet abandoned) while the command generator
    has actually rolled the env to standing -- zero effect on moving envs, so it cannot suppress
    the larger thigh/foot excursions genuine walking gait requires.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    default_joint_pos = wp.to_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
    is_standing_env = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(is_standing_env, deviation, torch.zeros_like(deviation))


def standing_joint_target_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: float, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Penalize selected joints deviating from a fixed ``target`` angle, standing envs only.

    Same ``is_standing_env`` gating as :func:`standing_leg_deviation_penalty`, but anchors to an
    explicit target instead of the default pose -- lets standing-only shaping (e.g. a wider hip
    stance) diverge from the walking baseline without touching the default/action-offset pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.square(joint_pos - target), dim=1)
    is_standing_env = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(is_standing_env, deviation, torch.zeros_like(deviation))


def min_joint_angle_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_angle: float
) -> torch.Tensor:
    """One-sided penalty for selected joints dropping below ``min_angle``, always active.

    Zero when a joint is at or above ``min_angle``; grows quadratically with the shortfall
    below it. Unlike :func:`standing_leg_deviation_penalty`, not gated by ``is_standing_env``
    and not symmetric -- a joint is free to go arbitrarily high, only penalized for sagging low.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    shortfall = torch.clamp(min_angle - joint_pos, min=0.0)
    return torch.sum(torch.square(shortfall), dim=1)


def max_joint_angle_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_angle: float
) -> torch.Tensor:
    """One-sided penalty for selected joints rising above ``max_angle``, always active.

    Mirrors :func:`min_joint_angle_penalty`: zero at or below ``max_angle``, grows
    quadratically with the excess above it. A joint is free to go arbitrarily low, only
    penalized for rising too high.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    excess = torch.clamp(joint_pos - max_angle, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(wp.to_torch(asset.data.root_lin_vel_b)[:, 2]) + 0.2 * torch.sum(
        torch.abs(wp.to_torch(asset.data.root_ang_vel_b)[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((wp.to_torch(asset.data.projected_gravity_b)[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground, via a squared-velocity (L2) kernel.

    Quadratic rather than linear so small stabilization jitter on a grounded foot stays
    cheap (avoiding an incentive to just lift a wobbly foot instead of correcting it) while
    sustained/large slip is still penalized hard.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    foot_planar_velocity_sq = torch.square(
        torch.linalg.norm(wp.to_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2], dim=2)
    )

    reward = is_contact * foot_planar_velocity_sq
    return torch.sum(reward, dim=1)


def joint_pos_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_limit: float,
    hard_limit: float,
) -> torch.Tensor:
    """Penalize selected joints only as they approach a soft-to-hard angular limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    range_width = hard_limit - soft_limit
    excess = torch.clamp(torch.abs(joint_pos) - soft_limit, min=0.0)
    normalized_excess = excess / range_width
    return torch.sum(torch.square(normalized_excess), dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((wp.to_torch(asset.data.joint_acc)), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
    reward = torch.linalg.norm((wp.to_torch(asset.data.joint_pos) - wp.to_torch(asset.data.default_joint_pos)), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((wp.to_torch(asset.data.applied_torque)), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((wp.to_torch(asset.data.joint_vel)), dim=1)


def stand_still_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocity when the full command (XY + yaw) is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = torch.linalg.norm(cmd, dim=1) < command_threshold
    return torch.sum(torch.square(wp.to_torch(asset.data.joint_vel)), dim=1) * standing


def turn_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Reward touchdown after long air time, but only for turning commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = wp.to_torch(contact_sensor.compute_first_contact(env.step_dt))[:, sensor_cfg.body_ids]
    last_air_time = wp.to_torch(contact_sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    return reward * (torch.abs(yaw_command) > yaw_threshold)


def straight_air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    linear_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Apply the air/contact-time equality penalty only while walking straight."""
    command = env.command_manager.get_command(command_name)
    straight_mode = torch.logical_and(
        torch.linalg.norm(command[:, :2], dim=1) > linear_threshold,
        torch.abs(command[:, 2]) <= yaw_threshold,
    )
    return air_time_variance_penalty(env, sensor_cfg) * straight_mode


def base_height_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise deviation from a commanded base height using an L2-squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(wp.to_torch(asset.data.root_pos_w)[:, 2] - target_height)


def single_support_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward having exactly one foot in the air at a time during locomotion."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    air_time = wp.to_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    num_in_air = (air_time > 0).float().sum(dim=1)

    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    body_vel = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
    reward = torch.exp(-torch.square(num_in_air - 1.0) / std)
    return torch.where(
        torch.logical_or(cmd > 0.1, body_vel > velocity_threshold),
        reward,
        torch.zeros_like(reward),
    )


class TurnGaitReward(GaitReward):
    """Apply the diagonal gait reward only for turning commands."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        yaw_threshold: float,
    ) -> torch.Tensor:
        reward = super().__call__(
            env,
            std,
            max_err,
            velocity_threshold,
            synced_feet_pair_names,
            asset_cfg,
            sensor_cfg,
        )
        yaw_command = env.command_manager.get_command("base_velocity")[:, 2]
        return reward * (torch.abs(yaw_command) > yaw_threshold)
