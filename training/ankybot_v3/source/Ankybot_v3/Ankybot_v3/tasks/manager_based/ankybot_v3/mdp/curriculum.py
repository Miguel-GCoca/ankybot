# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local curriculum functions for Ankybot V3 standing-recovery training.

Used with :func:`isaaclab.envs.mdp.curriculums.modify_term_cfg` to progressively widen
reset-event parameters (spawn joint fold / base roll / base height) within a single
continuous training run, in fixed-size steps gated on ``env.common_step_counter``.

Import discipline: this module (and its counterpart ``isaaclab.envs.mdp.curriculums``,
which houses ``modify_term_cfg``) must stay free of unconditional imports of anything
that touches ``omni``/``pxr`` (e.g. ``isaaclab.assets.Articulation``). ``env_cfg`` objects
are built by Hydra *before* ``AppLauncher`` starts Kit, so any such module-level import
here would execute pre-Kit and can corrupt later USD extension registration - this is
exactly what broke ``isaaclab.sh -p`` on 2026-07-09 (traced to the settle-gated action
term's unconditional import of ``isaaclab.envs.mdp.actions.joint_actions``, since removed).
Only touch ``env.common_step_counter`` (a plain int) and return plain Python values here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.curriculums import modify_term_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fold_stage_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data,
    stage_values: list[tuple[float, float]],
    stage_thresholds: list[int],
):
    """Step-function schedule over ``env.common_step_counter``.

    ``stage_values[0]`` applies from step 0 until ``env.common_step_counter`` reaches
    ``stage_thresholds[0]``, at which point ``stage_values[1]`` applies, and so on.
    ``len(stage_values)`` must equal ``len(stage_thresholds) + 1`` (the last stage has no
    upper threshold - it applies for the remainder of training).
    """
    step = env.common_step_counter
    stage_idx = 0
    for i, threshold in enumerate(stage_thresholds):
        if step >= threshold:
            stage_idx = i + 1
    target = stage_values[stage_idx]
    if tuple(data) == tuple(target):
        return modify_term_cfg.NO_CHANGE
    return target
