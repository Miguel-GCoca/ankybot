# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-free reward terms for the static-base (fixed-root) debug env.

:mod:`ankybot_v3_env_cfg_no_sensors.AnkybotV3StaticEnvCfg` has no velocity command
(the root is welded to the world, so there is nothing to track), but the goal is
still a continuous alternating foot-lift gait rather than idle standing. These are
copies of the corresponding functions/classes in :mod:`.rewards`, with the
``base_velocity`` command gating removed so the reward is always active — i.e. the
robot is always treated as "should be walking," never "should be standing still."
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


def air_time_reward_static(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
) -> torch.Tensor:
    """Reward feet for cycling between air and contact, capped at ``mode_time`` per phase.

    Unconditional version of :func:`.rewards.air_time_reward` — always rewards a fresh
    swing/stance toggle rather than only doing so while a velocity command is nonzero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    current_air_time = wp.to_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    current_contact_time = wp.to_torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    reward = torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min))
    return torch.sum(reward, dim=1)


def foot_clearance_reward_static(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height off the ground.

    Unconditional version of :func:`.rewards.foot_clearance_reward` — always active,
    since the static rig should always be lifting its feet, never standing flat.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(wp.to_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


class GaitRewardStatic(ManagerTermBase):
    """Command-free diagonal gait-sync reward for the static rig.

    Same synchronized/anti-synchronized foot-pair contact-timing reward as
    :class:`.rewards.GaitReward`, minus the ``base_velocity``/body-velocity gate — the
    static rig has no command and no floating base, so the gait pattern should be
    enforced unconditionally at every step.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
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
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        return sync_reward * async_reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)
