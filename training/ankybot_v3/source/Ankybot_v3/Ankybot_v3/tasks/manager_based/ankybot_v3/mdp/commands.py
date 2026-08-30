# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class UniformHeightCommand(CommandTerm):
    """Command term that samples a desired base height uniformly from a configured range.

    The command is a scalar (num_envs, 1) tensor holding the target base height in metres.
    It is resampled on the same schedule as velocity commands via the base CommandTerm logic.
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._height_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._height_command

    def _resample_command(self, env_ids: Sequence[int]):
        lo, hi = self.cfg.ranges.height
        self._height_command[env_ids, 0].uniform_(lo, hi)

    def _update_command(self):
        pass

    def _update_metrics(self):
        self.metrics["error_height"] = torch.abs(
            wp.to_torch(self._asset.data.root_pos_w)[:, 2] - self._height_command[:, 0]
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformHeightCommand`."""

    class_type: type = UniformHeightCommand
    asset_name: str = MISSING

    @configclass
    class Ranges:
        height: tuple[float, float] = (0.15, 0.24)

    ranges: Ranges = Ranges()


class TurnInPlaceVelocityCommand(CommandTerm):
    """Uniform SE(2) velocity command with a dedicated turn-in-place regime.

    Reimplements isaaclab's ``UniformVelocityCommand`` sampling logic locally rather than
    subclassing it, deliberately -- that implementation module imports
    :class:`isaaclab.markers.VisualizationMarkers`, which imports ``pxr`` at module level.
    ``mdp/commands.py`` is imported while Hydra builds ``env_cfg``, before
    ``AppLauncher``/Kit exists, so pulling that module in here reproduces the exact
    import-order poisoning failure documented in this project's CLAUDE.md ("Critical
    Warnings") -- confirmed by hitting it directly (``AttributeError: module
    'pxr.PhysxSchema' has no attribute 'Tokens'`` at sim startup) before switching to this
    self-contained version. No ``heading_command`` support (unused by every env cfg in this
    project) and no debug-vis markers (``debug_vis`` is force-disabled on every command term
    in every env cfg's ``__post_init__`` anyway).

    Independent uniform lin/ang sampling makes "rotate with ~zero linear velocity" a
    near-zero-probability event, so it's rarely trained on. For a configurable fraction of
    non-standing envs, this zeroes the sampled lin_vel_x/lin_vel_y after the fact while
    leaving the sampled ang_vel_z intact, guaranteeing that regime shows up at a fixed rate.
    """

    cfg: TurnInPlaceVelocityCommandCfg

    def __init__(self, cfg: TurnInPlaceVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_turn_in_place_env = torch.zeros_like(self.is_standing_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.linalg.norm(self.vel_command_b[:, :2] - wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - wp.to_torch(self.robot.data.root_ang_vel_b)[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.is_turn_in_place_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turn_in_place_envs

    def _update_command(self):
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        turn_env_ids = (self.is_turn_in_place_env & ~self.is_standing_env).nonzero(as_tuple=False).flatten()
        self.vel_command_b[turn_env_ids, 0] = 0.0
        self.vel_command_b[turn_env_ids, 1] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`TurnInPlaceVelocityCommand`."""

    class_type: type = TurnInPlaceVelocityCommand

    rel_turn_in_place_envs: float = 0.2
    """Sampled probability (independent of rel_standing_envs) that a non-standing env has its
    linear velocity command zeroed while keeping the sampled angular velocity command."""
